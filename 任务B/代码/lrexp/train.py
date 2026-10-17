"""Training / evaluation / diagnostics entry point.

Usage:
  python -m lrexp.train --out runs/NAME --set L=1024 d=256 n_q=8 steps=3000 pos=rope ...
Any field of DataCfg / ModelCfg / TrainCfg can be overridden with key=value.
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn.functional as F

from .data import DataCfg, Gen
from .model import ModelCfg, GPT


@dataclass
class TrainCfg:
    steps: int = 3000
    tokens_per_step: int = 131072
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup: int = 200
    sched: str = "const"     # const | cos
    wd: float = 0.1
    beta2: float = 0.95
    clip: float = 1.0
    loss: str = "all"        # all | targets
    loss_dmax: int = 0       # (targets loss) only events with d <= loss_dmax contribute (0 = all)
    eval_every: int = 100
    eval_ds: str = ""        # comma list of distances; default = training distances
    eval_Ls: str = ""        # extra context lengths to evaluate at (length transfer), comma list
    eval_ds_x: str = ""      # distances for the extra lengths (default: eval_ds)
    eval_B: int = 128
    log_every: int = 25
    seed: int = 0
    compile: int = 1
    compile_mode: str = "default"
    stop_acc: float = 0.0    # early stop when train-distance eval acc >= stop_acc for stop_patience evals
    stop_patience: int = 3
    diag_every: int = 0      # attention-mass diagnostics every N steps (0 = only at end)
    probe: int = 1           # linear probes at end
    init_from: str = ""      # checkpoint path to warm start
    save: int = 1


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    dc, mc, tc = DataCfg(), ModelCfg(), TrainCfg()
    for kv in a.set:
        k, v = kv.split("=", 1)
        done = False
        for obj in (dc, mc, tc):
            names = {f.name: f for f in fields(obj)}
            if k in names:
                t = type(getattr(obj, k))
                setattr(obj, k, (v.lower() in ("1", "true")) if t is bool else t(v))
                done = True
        if not done:
            raise KeyError(k)
    return a.out, dc, mc, tc


def lr_at(step, tc):
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    if tc.sched == "const":
        return tc.lr
    p = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
    return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def target_loss_acc(logits, x, tq):
    """logits (B,L,V); tq (B,n) positions whose next token is the queried value."""
    B = x.shape[0]
    bi = torch.arange(B, device=x.device)[:, None]
    lg = logits[bi, tq].float()          # (B,n,V)
    y = x[bi, tq + 1]
    loss = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    acc = (lg.argmax(-1) == y).float()
    return loss, acc


@torch.no_grad()
def attention_diag(raw, gen, d, B=8, L=None):
    """Mean attention mass from query position q to definition value (p+1) and key (p), per layer/head.
    Returns dict with lists [layer][head]. Chance level ~ 1/q."""
    x, tq, tp, _ = gen.sample(B, L=L, d=d)
    tq, tp = tq[:, :16], tp[:, :16]
    raw.set_diag(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw(x, raw.block_mask(x.shape[1], x.device))
    out = {"val": [], "key": [], "prev": [], "chance": float((1.0 / (tq.float() + 1)).mean())}

    def mass(P, pos):  # P (B,n,H,L), pos (B,n) -> (H,)
        Bn, n, H, _ = P.shape
        return P.gather(3, pos[:, :, None, None].expand(Bn, n, H, 1)).squeeze(-1).mean((0, 1))

    for blk in raw.blocks:
        P = blk.attn.attn_probs_q(tq)
        out["val"].append(mass(P, tp + 1).tolist())
        out["key"].append(mass(P, tp).tolist())
        P2 = blk.attn.attn_probs_q(tp + 1)  # previous-token head: value pos attends to its key
        out["prev"].append(mass(P2, tp).tolist())
        del P, P2
    raw.set_diag(False)
    return out


@torch.no_grad()
def collect_feats(raw, gen, d, n_batches, B, L=None):
    feats, ys = None, []
    for _ in range(n_batches):
        x, tq, tp, _ = gen.sample(B, L=L, d=d)
        raw.collect_resid = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw(x, raw.block_mask(x.shape[1], x.device))
        bi = torch.arange(B, device=x.device)[:, None]
        f = [r[bi, tq].float().flatten(0, 1) for r in raw.resid]
        ys.append(x[bi, tq + 1].flatten() - (gen.cfg.Vb + gen.cfg.K))
        feats = [[t] for t in f] if feats is None else [a + [t] for a, t in zip(feats, f)]
        raw.collect_resid = False
        raw.resid = []
    return [torch.cat(a) for a in feats], torch.cat(ys)


def linear_probe(Xtr, ytr, Xte, yte, n_cls, steps=400):
    with torch.enable_grad():
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        W = torch.zeros(Xtr.shape[1], n_cls, device=Xtr.device, requires_grad=True)
        b = torch.zeros(n_cls, device=Xtr.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(Xtr @ W + b, ytr).backward()
            opt.step()
    return float(((Xte @ W + b).argmax(-1) == yte).float().mean())


def probes(raw, gen, d, L=None):
    """Per-layer linear decodability of the queried value from the residual stream at the query position."""
    B = max(8, 16384 // (gen.cfg.n_q * 4))
    Xtr, ytr = collect_feats(raw, gen, d, 4, B, L)
    Xte, yte = collect_feats(raw, gen, d, 1, B, L)
    return [linear_probe(a, ytr, b, yte, gen.cfg.Vv) for a, b in zip(Xtr, Xte)]


def main():
    out, dc, mc, tc = parse()
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    mc.vocab = dc.vocab
    mc.max_len = max(mc.max_len, dc.L)
    json.dump({"data": asdict(dc), "model": asdict(mc), "train": asdict(tc)}, open(f"{out}/config.json", "w"), indent=1)

    gen = Gen(dc, dev, seed=tc.seed)
    egen = Gen(dc, dev, seed=10_000 + tc.seed)
    raw = GPT(mc).to(dev)
    if tc.init_from:
        sd = torch.load(tc.init_from, map_location=dev)
        raw.load_state_dict(sd["model"], strict=False)
    n_params = sum(p.numel() for p in raw.parameters())
    torch._dynamo.config.cache_size_limit = 64
    model = torch.compile(raw, mode=tc.compile_mode, dynamic=False) if tc.compile else raw

    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2 and "emb" not in n]
    nodecay = [p for n, p in raw.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=tc.lr, betas=(0.9, tc.beta2), fused=True)

    L = dc.L
    B = max(1, tc.tokens_per_step // L)
    bm = raw.block_mask(L, dev)
    # fixed eval sets
    if tc.eval_ds:
        eval_ds = [int(s) for s in tc.eval_ds.split(",")]
    else:
        eval_ds = sorted({dc.d} | ({dc.d_min, dc.d_short} if dc.dist == "mix" else set()))
    eval_ds = [d for d in eval_ds if d <= L - 2 - 2]
    eval_sets = {str(d): (egen.sample(tc.eval_B, d=d), bm) for d in eval_ds}
    for Lx in ([int(s) for s in tc.eval_Ls.split(",")] if tc.eval_Ls else []):
        for d in [int(s) for s in (tc.eval_ds_x or tc.eval_ds).split(",")]:
            if d <= Lx - 4:
                eval_sets[f"L{Lx}_d{d}"] = (egen.sample(max(1, tc.eval_B * L // Lx), L=Lx, d=d), raw.block_mask(Lx, dev))
    train_d_main = dc.d
    if dc.dist in ("uniform", "loguniform"):
        main_ds = [str(d) for d in eval_ds if dc.d_min <= d <= (tc.loss_dmax or dc.d)] or [str(eval_ds[-1])]
    elif dc.dist == "mix" and dc.d_long_min > 0:
        main_ds = [str(d) for d in eval_ds if dc.d_long_min <= d <= dc.d] or [str(eval_ds[-1])]
    else:
        main_ds = [str(dc.d)] if dc.d in eval_ds else [str(eval_ds[-1])]

    logf = open(f"{out}/metrics.jsonl", "w")
    print(f"params={n_params/1e6:.2f}M B={B} L={L} vocab={mc.vocab}", flush=True)

    def evaluate(step):
        model.eval()
        res = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k, ((x, tq, tp, _), bmx) in eval_sets.items():
                lg = model(x, bmx)
                l, a = target_loss_acc(lg, x, tq)
                res[k] = (float(l.mean()), float(a.mean()))
        model.train()
        return res

    t0 = time.time()
    tok_seen = 0
    hits = 0
    first = {}
    hist = []
    step = 0
    for step in range(tc.steps + 1):
        if step % tc.eval_every == 0 or step == tc.steps:
            ev = evaluate(step)
            main_acc = sum(ev[k][1] for k in main_ds) / len(main_ds)
            rec = {"step": step, "tokens": tok_seen, "time": time.time() - t0, "main_acc": main_acc,
                   "eval": {k: {"loss": l, "acc": a} for k, (l, a) in ev.items()}}
            if tc.diag_every and step % tc.diag_every == 0:
                rec["attn"] = attention_diag(raw, egen, int(main_ds[-1]))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            for thr in (0.5, 0.9):
                if main_acc >= thr and str(thr) not in first:
                    first[str(thr)] = step
            print(f"[eval {step}] main {main_acc:.3f} " + " ".join(f"{k}:{a:.3f}" for k, (l, a) in ev.items()), flush=True)
            hits = hits + 1 if (tc.stop_acc and main_acc >= tc.stop_acc) else 0
            if tc.stop_acc and hits >= tc.stop_patience:
                break
        if step == tc.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        if step % 16 == 0:
            buf = gen.sample(B * 16)
        j = (step % 16) * B
        x, tq, tp, dd = (t[j:j + B] for t in buf)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, bm)
            if tc.loss == "targets":
                tl, ta = target_loss_acc(logits, x, tq)
                if tc.loss_dmax > 0:
                    w = (dd <= tc.loss_dmax).float()
                    loss = (tl * w).sum() / w.sum().clamp_min(1.0)
                else:
                    loss = tl.mean()
            else:
                loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.clip)
        opt.step()
        tok_seen += B * L
        if step % tc.log_every == 0:
            with torch.no_grad():
                tl, ta = target_loss_acc(logits.detach(), x, tq)
            rec = {"step": step, "loss": loss.item(), "tgt_loss": float(tl.mean()), "tgt_acc": float(ta.mean()),
                   "gnorm": float(gn), "lr": lr_at(step, tc), "tok_s": tok_seen / (time.time() - t0)}
            logf.write(json.dumps(rec) + "\n")
            if step % (tc.log_every * 8) == 0:
                print(f"step {step} loss {rec['loss']:.4f} tgt {rec['tgt_loss']:.3f}/{rec['tgt_acc']:.3f} "
                      f"tok/s {rec['tok_s']/1e6:.2f}M", flush=True)

    elapsed = time.time() - t0
    summary = {"params": n_params, "steps_done": step, "tokens": tok_seen, "time": elapsed,
               "tok_s": tok_seen / elapsed, "first_step": first, "final_eval": {str(d): v for d, v in ev.items()}}
    # end-of-run diagnostics
    diag_ds = eval_ds
    summary["attn"] = {str(d): attention_diag(raw, egen, d) for d in diag_ds}
    if tc.probe:
        summary["probe"] = {str(d): probes(raw, egen, d) for d in diag_ds}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    if tc.save:
        torch.save({"model": raw.state_dict()}, f"{out}/ckpt.pt")
    print("DONE", json.dumps({k: summary[k] for k in ("steps_done", "time", "tok_s", "first_step")}), flush=True)


if __name__ == "__main__":
    main()
