"""Training entry point for the v2 recall task (lrexp.task).

  * loss only at the supervised query positions (logits[q] vs x[q+1]); the LM head is applied only there
  * `valid`: fixed set from the TRAINING distribution -> the only signal used for early stopping
  * `bench`: fixed, cached, distance-uniform benchmark (Delta ~ U[4, L-2], same context statistics) used for
    reporting only (learning curves on 512 sequences, full eval + per-query predictions at the end)
  * attention diagnostics are disabled by default; all run records stay local

Usage:
  python -m lrexp.train2 --out runs/NAME --set L=1024 n_q=32 n_dis=32 d_max=256 ...
Any field of TaskCfg / ModelCfg / TrainCfg2 can be overridden with key=value.
"""
import argparse
import contextlib
import json
import os
import time
import hashlib
import math
import shutil
import signal
import re
from pathlib import Path
from dataclasses import dataclass, asdict, fields, replace
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import TaskCfg, Task, make_benchmark
from .model import ModelCfg, GPT
from .train import lr_at
from .checkpoint_io import (save_complete, sha256_file, load_complete, save_model, write_json_atomic,
                            capture_local_artifacts, restore_local_artifacts)
from .sampling import TrainingSampler, make_validation
import random


@dataclass
class TrainCfg2:
    steps: int = 24000
    tokens_per_step: int = 131072   # batch B = tokens_per_step // L
    lr: float = 1e-3
    min_lr_frac: float = 0.1        # (sched=cos only)
    warmup: int = 200
    sched: str = "const"            # const | cos
    wd: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0               # 0 -> no clipping (grad norm still logged)
    precision: str = "bf16"         # bf16 (autocast, TF32 on) | fp32 (no autocast, TF32 off)
    log_every: int = 100
    eval_every: int = 250
    valid_n: int = 256              # sequences in the fixed training-distribution valid set
    bench_quick: int = 512          # bench sequences used for periodic (learning-curve) eval
    bench_queries: int = 131072     # benchmark size in queries (n_seq = bench_queries // n_q)
    bench_seed: int = 777
    stop_acc: float = 0.98          # early stop: both valid and development long accuracy reach this threshold
    stop_patience: int = 3
    diagnostics: bool = False
    diag_every: int = 25            # independent of the full validation interval
    diag_n: int = 16                # bench sequences used for attention diagnostics
    diag_dense_every: int = 5
    diag_dense_until: int = 1000
    diag_dense_window: int = 500
    diag_trigger_drop: float = 0.20
    diag_trigger_ref_steps: int = 100
    diag_trigger_cooldown: int = 1000
    loss_ema_beta: float = 0.98
    seed: int = 0
    data_seed: int = -1       # separate data stream from initialization when explicitly requested
    resume_from: str = ""
    deadline_unix: float = 0.0
    compile: int = 1
    compile_mode: str = "default"
    init_from: str = ""
    save: int = 1
    sampling_plan: str = "base"     # base | coverage | short_first | shuffled
    short_min: int = 4
    short_max: int = 96
    long_fraction: float = 0.0
    short_steps: int = 3000
    checkpoint_seconds: float = 600.0
    model_extra_steps: str = ""
    test_seed: int = 888
    test_queries: int = 131072


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    tk, mc, tc = TaskCfg(), ModelCfg(), TrainCfg2()
    for kv in a.set:
        k, v = kv.split("=", 1)
        done = False
        for obj in (tk, mc, tc):
            names = {f.name for f in fields(obj)}
            if k in names:
                t = type(getattr(obj, k))
                setattr(obj, k, (v.lower() in ("1", "true")) if t is bool else t(v))
                done = True
        if not done:
            raise KeyError(k)
    return a.out, tk, mc, tc


class QFwd(nn.Module):
    """GPT forward (same computation as GPT.forward) with the LM head applied only at positions tq (B,n)."""

    def __init__(self, raw: GPT):
        super().__init__()
        self.raw = raw

    def forward(self, x, tq, block_mask=None):
        r = self.raw
        h = r.emb(x)
        if r.cfg.smear:
            h = h + r.emb_prev(F.pad(x[:, :-1], (1, 0), value=0))
        for b in r.blocks:
            h = b(h, r.cos, r.sin, block_mask)
        h = h.gather(1, tq[..., None].expand(-1, -1, h.shape[-1]))
        return r.head(r.nf(h))                                   # (B, n, V)


# ---------------------------------------------------------------- benchmark
def bench_cfg(tk: TaskCfg):
    return TaskCfg(L=tk.L, n_q=tk.n_q, n_dis=tk.n_dis, dist="uniform", d_min=4, d_max=0, Vb=tk.Vb, K=tk.K,
                   Vv=tk.Vv, background=tk.background, branch=tk.branch)


def bench_path(tk: TaskCfg, seed, root=None):
    root = root or os.environ.get("LR_BENCH_DIR", "runs/_bench")
    return os.path.join(root, f"bench_L{tk.L}_nq{tk.n_q}_nd{tk.n_dis}_{tk.background}_K{tk.K}_V{tk.Vv}"
                              f"_Vb{tk.Vb}_s{seed}.pt")


def load_bench(tk: TaskCfg, tc: TrainCfg2):
    """Load (or create once) the cached benchmark. Generated on the GPU (CPU placement is far too slow); written to
    a temp file then os.replace'd, so concurrent creators are safe (same seed -> identical content on one box).
    Copy the cached file between boxes if bit-identical benchmarks across machines are needed."""
    path = bench_path(tk, tc.bench_seed)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.time()
        n_seq = tc.bench_queries // tk.n_q
        b = make_benchmark(bench_cfg(tk), n_seq, torch.device("cuda"), seed=tc.bench_seed)
        b = {k: v.to(torch.int16) for k, v in b.items()}
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(b, tmp)
        os.replace(tmp, path)
        print(f"created benchmark {path} n_seq={n_seq} in {time.time()-t0:.1f}s", flush=True)
    b = torch.load(path)
    if b["x"].shape != (tc.bench_queries // tk.n_q, tk.L):
        raise ValueError("Cached benchmark size differs from the configured query budget")
    return {k: v.long() for k, v in b.items()}, path


def test_path(tk: TaskCfg, tc: TrainCfg2):
    return bench_path(tk, tc.test_seed)


def test_config(tc: TrainCfg2):
    return replace(tc, bench_seed=tc.test_seed, bench_queries=tc.test_queries)


# ---------------------------------------------------------------- eval helpers
_PREC = {"amp": True}


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if _PREC["amp"] else contextlib.nullcontext()


@torch.no_grad()
def predict(qm, x, q, B, bm):
    """Run in fixed-size chunks of B (pads the last chunk -> no recompiles). Returns pred, loss (N, n)."""
    N = x.shape[0]
    preds, losses = [], []
    for i in range(0, N, B):
        xb, qb = x[i:i + B], q[i:i + B]
        nb = xb.shape[0]
        if nb < B:
            rep = (B + nb - 1) // nb
            xb, qb = xb.repeat(rep, 1)[:B], qb.repeat(rep, 1)[:B]
        with amp():
            lg = qm(xb, qb, bm).float()
        y = xb.gather(1, qb + 1)
        losses.append(F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none").view_as(y)[:nb])
        preds.append(lg.argmax(-1)[:nb])
    return torch.cat(preds), torch.cat(losses)


def acc_split(correct, d, L):
    lm = d >= L // 2
    f = lambda m: float(correct[m].float().mean()) if bool(m.any()) else None
    return {"short": f(~lm), "long": f(lm), "all": float(correct.float().mean())}


@torch.no_grad()
def attn_diag(raw, x, q, s, d, L, bm, include_snapshot=False):
    """Per layer / head: prev-token score (attention s+1 -> s), induction mass (q -> s+1) and key mass (q -> s),
    separately for short (Delta < L/2) and long queries. Chance ~ 1/(q+1)."""
    was_training = raw.training
    raw.eval()
    raw.set_diag(True)
    with amp():
        logits = QFwd(raw)(x, q, bm).float()  # Head only at query rows, also during diagnostics.
    target = x.gather(1, q + 1)
    correct_logit = logits.gather(-1, target[..., None]).squeeze(-1)
    probability = (correct_logit - logits.logsumexp(-1)).exp()
    wrong_logits = logits.clone()
    wrong_logits.scatter_(-1, target[..., None], float("-inf"))
    margin = correct_logit - wrong_logits.amax(-1)
    prediction = logits.argmax(-1)
    del logits, wrong_logits
    lm = d >= L // 2
    masks = {"short": ~lm, "long": lm}
    out = {k: {m: [] for m in masks} for k in ("prev", "ind", "key")}
    out["n"] = {m: int(v.sum()) for m, v in masks.items()}
    out["chance"] = {m: (float((1.0 / (q[v].float() + 1)).mean()) if out["n"][m] else None)
                     for m, v in masks.items()}
    out["relay"] = {m: [] for m in masks}
    out["entropy"] = {m: [] for m in masks}
    out["correct_value_probability"] = {m: float(probability[v].mean()) if out["n"][m] else None
                                         for m, v in masks.items()}
    out["correct_logit_margin"] = {m: float(margin[v].mean()) if out["n"][m] else None
                                   for m, v in masks.items()}

    def at(P, pos):  # P (B,n,H,L), pos (B,n) -> (B,n,H)
        return P.gather(3, pos[:, :, None, None].expand(-1, -1, P.shape[2], 1)).squeeze(-1)

    offsets = torch.arange(1, 65, device=s.device)
    positions = s[..., None] + offsets
    valid_offsets = (positions < L) & (positions <= q[..., None])
    try:
        for blk in raw.blocks:
            P = blk.attn.attn_probs_q(q)
            ind, key = at(P, s + 1), at(P, s)
            entropy = -(P * P.clamp_min(1e-30).log()).sum(-1)
            relay = P.gather(3, positions.clamp_max(L - 1)[:, :, None, :].expand(-1, -1, P.shape[2], -1))
            relay = relay * valid_offsets[:, :, None, :]
            del P
            P = blk.attn.attn_probs_q(s + 1)
            prev = at(P, s)
            del P
            for name, v in (("prev", prev), ("ind", ind), ("key", key)):
                for m, mk in masks.items():
                    out[name][m].append(v[mk].mean(0).tolist() if out["n"][m] else None)
            for m, mk in masks.items():
                if out["n"][m]:
                    average = relay[mk].mean(0)  # H,64; only aggregates leave the GPU.
                    mass, indices = average.topk(5, dim=-1)
                    out["relay"][m].append({"top5_offsets": (indices + 1).cpu().tolist(),
                                             "top5_mass": mass.cpu().tolist(),
                                             "sum64": average.sum(-1).cpu().tolist()})
                    out["entropy"][m].append(entropy[mk].mean(0).cpu().tolist())
                else:
                    out["relay"][m].append(None)
                    out["entropy"][m].append(None)
        if include_snapshot:
            # Positions refer to the immutable development benchmark.  No full
            # attention matrix or batch of activations is copied to the CPU.
            cut = (slice(0, min(4, q.shape[0])), slice(0, min(16, q.shape[1])))
            out["query_snapshot"] = {k: v[cut].cpu().tolist() for k, v in
                (("q", q), ("s", s), ("d", d), ("target", target), ("prediction", prediction),
                 ("correct_probability", probability), ("logit_margin", margin))}
    finally:
        raw.set_diag(False)
        raw.train(was_training)
    return out


def tables(correct, s, d, L, nb=16):
    W = L // nb
    db = (d // W).clamp_max(nb - 1)
    sb = (s // W).clamp_max(nb - 1)
    c = correct.float()
    cnt1 = torch.bincount(db, minlength=nb)
    hit1 = torch.bincount(db, weights=c, minlength=nb)
    idx = sb * nb + db
    cnt2 = torch.bincount(idx, minlength=nb * nb).view(nb, nb)
    hit2 = torch.bincount(idx, weights=c, minlength=nb * nb).view(nb, nb)
    acc = lambda h, n: [(float(a / b) if b > 0 else None) for a, b in zip(h.tolist(), n.tolist())]
    return ({"width": W, "count": cnt1.tolist(), "acc": acc(hit1, cnt1)},
            {"width": W, "rows": "s_bucket", "cols": "d_bucket", "count": cnt2.tolist(),
             "acc": [acc(h, n) for h, n in zip(hit2, cnt2)]})


class LeaseFence:
    """Cache a worker lease for at most five seconds; fail closed before updates."""
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.expires_unix = 0.0
        self.refreshed_monotonic = float("-inf")

    def failure(self):
        if self.path is None:
            return None
        now, monotonic = time.time(), time.monotonic()
        if monotonic - self.refreshed_monotonic >= 5 or now >= self.expires_unix:
            self.refreshed_monotonic = monotonic
            try:
                with self.path.open() as stream:
                    self.expires_unix = float(json.load(stream)["expires_unix"])
                if not math.isfinite(self.expires_unix):
                    raise ValueError("Nonfinite expiry")
            except (OSError, ValueError, TypeError, KeyError):
                self.expires_unix = 0.0
                return "lease_unreadable"
        return "lease_expired" if now >= self.expires_unix else None


# ---------------------------------------------------------------- main
def main():
    out, tk, mc, tc = parse()
    if tc.init_from:
        raise ValueError("This round starts from scratch; use only a complete night20261016 resume checkpoint")
    if tc.diagnostics and tc.diag_n != 16:
        raise ValueError("The next-round diagnostic protocol uses exactly diag_n=16")
    if tc.test_seed in (tc.bench_seed, 10_000 + tc.seed, tc.seed if tc.data_seed < 0 else tc.data_seed):
        raise ValueError("Final test seed must differ from training, valid and development seeds")
    if tc.test_queries <= 0 or tc.test_queries % tk.n_q:
        raise ValueError("test_queries must be a positive multiple of n_q")
    if tc.log_every <= 0 or tc.eval_every <= 0 or tc.checkpoint_seconds <= 0:
        raise ValueError("Logging, evaluation and checkpoint intervals must be positive")
    if not 0 <= tc.loss_ema_beta < 1 or not 0 < tc.diag_trigger_drop < 1:
        raise ValueError("Invalid loss EMA or diagnostic loss-drop threshold")
    if min(tc.diag_trigger_ref_steps, tc.diag_trigger_cooldown, tc.diag_dense_window) <= 0:
        raise ValueError("Diagnostic reference, cooldown and dense window must be positive")
    extra_model_steps = {int(v.strip()) for v in tc.model_extra_steps.split(",") if v.strip()}
    if extra_model_steps and (not re.search(r"_(C01|C02)_s", os.path.basename(out))
            or tc.seed not in (1000, 1001, 1002) or extra_model_steps != {3000, 3250, 4500}):
        raise ValueError("Only C01/C02 seeds 1000..1002 retain model points 3000,3250,4500")
    if tc.tokens_per_step < tk.L or tc.tokens_per_step % tk.L:
        raise ValueError("tokens_per_step must preserve an integral full batch")
    stop_requested = False
    pause_reason = None
    lease = LeaseFence(os.environ.get("NIGHT_LEASE_FILE"))
    def request_stop(signum, frame):
        nonlocal stop_requested, pause_reason
        stop_requested = True
        pause_reason = signal.Signals(signum).name
    for stop_signal in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(stop_signal, request_stop)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(tc.seed)
    assert tc.precision in ("bf16", "fp32"), tc.precision
    _PREC["amp"] = tc.precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = _PREC["amp"]
    torch.backends.cudnn.allow_tf32 = _PREC["amp"]
    torch.set_float32_matmul_precision("high" if _PREC["amp"] else "highest")
    dev = torch.device("cuda")
    mc.vocab = tk.vocab
    mc.max_len = max(mc.max_len, tk.L)
    cfg_all = {"task": asdict(tk), "model": asdict(mc), "train": asdict(tc),
               "execution":{"session":"night20261016", "full_checkpoint":True, "logging":"local_only"}}
    source_hashes = {p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    source_code = {p.name:p.read_text() for p in Path(__file__).parent.glob("*.py")}
    resumed = load_complete(tc.resume_from, cfg_all, source_hashes) if tc.resume_from else None
    if resumed is None and (Path(out) / "offline_run.json").exists():
        raise ValueError("Existing offline run requires its complete resume state")
    if resumed is not None:
        restore_local_artifacts(out, resumed)
    write_json_atomic(Path(out) / "config.json", cfg_all)

    L = tk.L
    B = max(1, tc.tokens_per_step // L)
    sampler = TrainingSampler(tk, tc, dev)
    valid = make_validation(tk, tc, dev)
    if resumed is not None:
        # Restore byte-identical evaluation assets on a fresh Colab filesystem.
        for target, data_key, hash_key in (
                (bench_path(tk, tc.bench_seed), "benchmark_file_bytes", "benchmark_sha256"),
                (bench_path(tk, tc.test_seed), "test_file_bytes", "test_sha256")):
            if not os.path.exists(target) and resumed.get(data_key):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                temporary = target + ".restore." + str(os.getpid())
                with open(temporary, "wb") as stream:
                    stream.write(resumed[data_key]); stream.flush(); os.fsync(stream.fileno())
                if sha256_file(temporary) != resumed[hash_key]:
                    raise ValueError("Saved evaluation asset bytes/hash differ")
                os.replace(temporary, target)
    bench, bpath = load_bench(tk, tc)
    bench = {k: v.to(dev) for k, v in bench.items()}
    nqk = min(tc.bench_quick, bench["x"].shape[0])
    quick = {k: v[:nqk] for k, v in bench.items()}
    # Prepare/check only the independent asset here; no test predictions or
    # scores are computed until the training loop has ended.
    prepared_test, tpath = load_bench(tk, test_config(tc))
    del prepared_test
    test_hash = sha256_file(tpath)

    raw = GPT(mc).to(dev)
    if tc.init_from:
        sd = torch.load(tc.init_from, map_location=dev)
        res = raw.load_state_dict(sd["model"], strict=False)
        print(f"init_from {tc.init_from}: loaded {len(sd['model'])} tensors; missing={res.missing_keys} "
              f"unexpected={res.unexpected_keys}", flush=True)
        assert not res.missing_keys, "init_from checkpoint does not cover the model"
    n_params = sum(p.numel() for p in raw.parameters())
    torch._dynamo.config.cache_size_limit = 64
    qraw = QFwd(raw)
    qm = torch.compile(qraw, mode=tc.compile_mode, dynamic=False) if tc.compile else qraw

    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2 and "emb" not in n]
    nodecay = [p for n, p in raw.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=tc.lr, betas=(tc.beta1, tc.beta2), fused=True)
    bm = raw.block_mask(L, dev)

    logf = open(f"{out}/metrics.jsonl", "a", buffering=1)
    print(f"params={n_params/1e6:.2f}M B={B} L={L} vocab={mc.vocab} bench={bpath} "
          f"n_bench_seq={bench['x'].shape[0]}", flush=True)

    def diag(include_snapshot=False):
        n = tc.diag_n
        return attn_diag(raw, bench["x"][:n], bench["q"][:n], bench["s"][:n], bench["d"][:n], L, bm, include_snapshot)

    def evaluate():
        qm.eval()
        vp, vl = predict(qm, valid["x"], valid["q"], B, bm)
        vc = vp == valid["x"].gather(1, valid["q"] + 1)
        bp, _ = predict(qm, quick["x"], quick["q"], B, bm)
        bc = bp == quick["x"].gather(1, quick["q"] + 1)
        qm.train()
        return float(vc.float().mean()), float(vl.mean()), acc_split(bc, quick["d"], L)

    # No W&B import, service, API, artifact, network gate or cache directory.
    if resumed is not None:
        raw.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["optimizer"])
        sampler.load_state_dict(resumed["sampling_state"])
        if sampler.updates != int(resumed["step_completed"]):
            raise ValueError("Saved sampler and optimizer update counts differ")
        valid = {k:v.to(dev) for k,v in resumed["valid"].items()}
    job_id = os.environ.get("NIGHT_JOB_ID", os.path.basename(out))
    identity = os.environ.get("NIGHT_RUN_ID") or hashlib.sha256(("night20261016:" + job_id).encode()).hexdigest()[:16]
    if resumed is not None and resumed.get("offline_run_id") != identity:
        raise ValueError("Resume run identity differs")
    write_json_atomic(Path(out) / "offline_run.json", {
        "id": identity, "job_id": job_id, "name": os.path.basename(out), "mode": "local_only",
        "source_hashes": source_hashes, "process_started_unix": time.time(),
        "resumed_at_step": int(resumed["step_completed"]) if resumed else 0})
    t0 = time.time()
    tok_seen, hits, step = 0, 0, 0
    first = {"valid": {}, "bench_long": {}}
    stopped = False
    budget_interrupted = False
    start_step = 0
    last_eval_step = -1
    updates_completed = 0
    elapsed_before = 0.0
    tokens_at_start = 0
    first_update_checkpoint_saved = bool(resumed.get("first_update_checkpoint_saved", False)) if resumed else False
    process_marker = time.time_ns() // 1000
    loss_ema = None
    loss_reference = deque()
    dense_until_step = tc.diag_dense_until
    last_loss_trigger_step = -tc.diag_trigger_cooldown
    last_diag_step = -1
    threshold_checkpoint_keys = set()
    overhead_ms = {"diag": 0.0, "eval": 0.0, "checkpoint": 0.0}
    if resumed is not None:
        stopped = bool(resumed.get("stopped_early", False))
        start_step = int(resumed["step_completed"])
        tok_seen = int(resumed["tokens"])
        hits = int(resumed["early_stop_hits"])
        first = resumed["first_threshold_steps"]
        last_eval_step = resumed.get("last_eval_step", -1)
        updates_completed = start_step
        tokens_at_start = tok_seen
        elapsed_before = float(resumed.get("elapsed",0))
        # Restore after all model/optimizer/data construction.
        torch.set_rng_state(resumed["cpu_rng"])
        torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        random.setstate(resumed["python_rng"])
        np.random.set_state(resumed["numpy_rng"])
        diagnostic_state = resumed["diagnostic_state"]
        loss_ema = diagnostic_state["loss_ema"]
        if loss_ema is not None:
            loss_ema = loss_ema.to(dev)
        loss_reference = deque(diagnostic_state["loss_reference"])
        dense_until_step = diagnostic_state["dense_until_step"]
        last_loss_trigger_step = diagnostic_state["last_loss_trigger_step"]
        last_diag_step = diagnostic_state["last_diag_step"]
        threshold_checkpoint_keys = set(diagnostic_state["threshold_checkpoint_keys"])
        overhead_ms = dict(diagnostic_state["overhead_ms"])
    benchmark_hash = sha256_file(bpath)
    with open(bpath,"rb") as stream:
        benchmark_file_bytes = stream.read()
    if resumed and benchmark_hash != resumed["benchmark_sha256"]:
        raise ValueError("Resume benchmark hash mismatch")
    if resumed and test_hash != resumed.get("test_sha256"):
        raise ValueError("Resume final-test hash mismatch")
    with open(tpath, "rb") as stream:
        test_file_bytes = stream.read()
    def complete_state():
        logf.flush()
        os.fsync(logf.fileno())
        local_artifacts = capture_local_artifacts(out, extra_model_steps, updates_completed)
        return {"model":raw.state_dict(), "optimizer":opt.state_dict(), "cfg":cfg_all,
                "step_completed":updates_completed, "tokens":tok_seen,
                "local_artifacts":local_artifacts, "model_extra_steps":sorted(extra_model_steps),
                "sampling_state":sampler.state_dict(),
                "training_generator_state":sampler.primary_stream.task.g.get_state(),
                "training_buffer":sampler.primary_stream.buffer,
                "buffer_next_batch":sampler.primary_stream.cursor - sampler.primary_stream.cache_start,
                "cpu_rng":torch.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state_all(),
                "python_rng":random.getstate(), "numpy_rng":np.random.get_state(),
                "valid":valid, "benchmark_file":bpath, "benchmark_sha256":benchmark_hash,
                "test_file":tpath, "test_sha256":test_hash,
                "benchmark_file_bytes":benchmark_file_bytes, "test_file_bytes":test_file_bytes,
                "early_stop_hits":hits, "first_threshold_steps":first,
                "last_eval_step":last_eval_step, "elapsed":elapsed_before+time.time()-t0,
                "offline_run_id":identity, "first_update_checkpoint_saved":first_update_checkpoint_saved, "stopped_early":stopped, "source_hashes":source_hashes, "source_code":source_code,
                "process_marker":process_marker,
                "diagnostic_state":{"loss_ema":loss_ema, "loss_reference":list(loss_reference),
                    "dense_until_step":dense_until_step, "last_loss_trigger_step":last_loss_trigger_step,
                    "last_diag_step":last_diag_step, "threshold_checkpoint_keys":sorted(threshold_checkpoint_keys),
                    "overhead_ms":dict(overhead_ms)},
                "scheduler":{"type":tc.sched,"completed":updates_completed,
                             "next_lr":lr_at(updates_completed,tc)}}
    last_checkpoint_time = time.monotonic()
    last_heartbeat_time = 0.0
    disk_state = {}
    night_root = Path(os.environ.get("NIGHT_ROOT", Path(__file__).resolve().parents[2]))
    min_free_bytes = int(float(os.environ.get("NIGHT_MIN_FREE_GIB", "2")) * 1024**3)
    limit_bytes = int(float(os.environ.get("NIGHT_MAX_GIB", "0")) * 1024**3)
    if min_free_bytes < 0 or limit_bytes < 0:
        raise ValueError("Disk limits cannot be negative")

    def heartbeat(force=False, status="running"):
        nonlocal last_heartbeat_time, disk_state, stop_requested, pause_reason
        if not force and time.monotonic() - last_heartbeat_time < 60:
            return
        # Only a minute-level filesystem scan; no GPU synchronization.
        size = 0
        for folder, _, names in os.walk(night_root):
            for name in names:
                try:
                    size += os.stat(os.path.join(folder, name), follow_symlinks=False).st_size
                except FileNotFoundError:
                    pass
        free = shutil.disk_usage(night_root).free
        disk_state = {"root":str(night_root), "bytes":size, "free_bytes":free,
                      "limit_bytes":limit_bytes, "minimum_free_bytes":min_free_bytes}
        if (limit_bytes > 0 and size >= limit_bytes) or free < min_free_bytes:
            stop_requested, pause_reason = True, "disk_limit"
        write_json_atomic(Path(out) / "status.json", {
            "id":identity, "name":os.path.basename(out), "pid":os.getpid(),
            "status":status, "step_completed":updates_completed, "target_steps":tc.steps,
            "tokens":tok_seen, "heartbeat_unix":time.time(), "pause_reason":pause_reason,
            "disk":disk_state, "elapsed":elapsed_before + time.time() - t0})
        last_heartbeat_time = time.monotonic()

    def pause_pending():
        nonlocal pause_reason
        lease_failure = lease.failure()
        if lease_failure:
            pause_reason = lease_failure
        pending = (lease_failure is not None or stop_requested or os.path.exists(os.path.join(out, "stop.request"))
                   or (tc.deadline_unix > 0 and time.time() >= tc.deadline_unix))
        if pending:
            pause_reason = pause_reason or "stop_request_or_deadline"
        return pending

    def finish_paused():
        checkpoint("paused")
        logf.flush(); os.fsync(logf.fileno()); logf.close()
        if attention_log is not None:
            attention_log.close()
        heartbeat(force=True, status="paused")
        print("PAUSED", json.dumps({"steps_done":updates_completed, "reason":pause_reason,
              "resume":"latest_resume.pt", "time":elapsed_before + time.time() - t0}), flush=True)

    def checkpoint(phase):
        nonlocal last_checkpoint_time
        checkpoint_started = time.perf_counter()
        snapshot = complete_state()
        record = save_complete(out, snapshot, phase)
        torch.set_rng_state(snapshot["cpu_rng"])
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        overhead_ms["checkpoint"] += 1000 * (time.perf_counter() - checkpoint_started)
        last_checkpoint_time = time.monotonic()
        return record

    attention_log = open(f"{out}/attention.jsonl", "a") if tc.diagnostics else None
    snapshot_dir = os.path.join(out, "attention_snapshots")
    if tc.diagnostics:
        os.makedirs(snapshot_dir, exist_ok=True)

    def emit_diag(step, reason, snapshot=False):
        nonlocal last_diag_step
        if not tc.diagnostics or (last_diag_step == step and not snapshot):
            return
        started = time.perf_counter()
        details = diag(include_snapshot=snapshot)
        duration_ms = 1000 * (time.perf_counter() - started)
        overhead_ms["diag"] += duration_ms
        row = {"step":step, "reason":reason, "diag_ms":duration_ms,
               "benchmark_sha256":benchmark_hash, "attn":details}
        attention_log.write(json.dumps(row) + "\n"); attention_log.flush()
        if snapshot:
            with open(os.path.join(snapshot_dir, f"step_{step:06d}_{reason}.json"), "w") as stream:
                json.dump(row, stream)
        last_diag_step = step


    torch.cuda.reset_peak_memory_stats()
    profile_step = start_step
    profile_time = time.perf_counter()
    profile_overhead_ms = sum(overhead_ms.values())
    profile_has_previous_log = False
    heartbeat(force=True)
    if resumed is None:
        checkpoint("initial")
    for step in range(start_step, tc.steps + 1):
        heartbeat()
        if pause_pending():
            budget_interrupted = True
            break
        if stopped:
            break
        if (step % tc.eval_every == 0 or step == tc.steps) and step != last_eval_step:
            eval_started = time.perf_counter()
            va, vl, ba = evaluate()
            eval_ms = 1000 * (time.perf_counter() - eval_started)
            overhead_ms["eval"] += eval_ms
            rec = {"step": step, "tokens": tok_seen, "time": elapsed_before + time.time() - t0, "valid_acc": va, "valid_loss": vl,
                   "bench_short": ba["short"], "bench_long": ba["long"], "bench_all": ba["all"],
                   "eval_ms":eval_ms, "sampling":sampler.statistics()}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            for thr in (0.5, 0.9, 0.98):
                if va >= thr and str(thr) not in first["valid"]:
                    first["valid"][str(thr)] = step
                if ba["long"] is not None and ba["long"] >= thr and str(thr) not in first["bench_long"]:
                    first["bench_long"][str(thr)] = step
            fm = lambda v: "nan" if v is None else f"{v:.3f}"
            print(f"[eval {step}] valid {va:.4f} (loss {vl:.3f}) bench short {fm(ba['short'])} "
                  f"long {fm(ba['long'])}", flush=True)
            last_eval_step = step
            hits = (hits + 1 if va >= tc.stop_acc and ba["long"] is not None
                    and ba["long"] >= tc.stop_acc else 0)
            if tc.stop_acc > 0 and hits >= tc.stop_patience:
                stopped = True
                break
        if pause_pending():
            budget_interrupted = True
            break
        dense = (step <= dense_until_step or abs(step - tc.short_steps) <= tc.diag_dense_window)
        interval = tc.diag_dense_every if dense else tc.diag_every
        if tc.diagnostics and interval > 0 and step % interval == 0:
            emit_diag(step, "dense" if dense else "periodic")
        if step == tc.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        batch = sampler.next_batch(step)
        x, tq = batch["x"], batch["q"]
        with amp():
            lg = qm(x, tq, bm)
        y = x.gather(1, tq + 1)
        loss = F.cross_entropy(lg.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.clip if tc.clip > 0 else float("inf"))
        # A slow forward/backward can outlive the lease. Do not apply its update;
        # rewind only the pending data cursor so the next process sees this batch.
        if pause_pending():
            sampler.discard_pending()
            opt.zero_grad(set_to_none=True)
            budget_interrupted = True
            break
        opt.step()
        sampler.record_consumed(batch)
        tok_seen += B * L
        updates_completed = step + 1
        detached_loss = loss.detach().float()
        loss_ema = (detached_loss.clone() if loss_ema is None else
                    loss_ema.mul(tc.loss_ema_beta).add(detached_loss, alpha=1-tc.loss_ema_beta))
        if updates_completed % tc.log_every == 0:
            ta = float((lg.detach().argmax(-1) == y).float().mean())
            loss_value, ema_value = float(loss.detach()), float(loss_ema)
            now = time.perf_counter()
            window_updates = updates_completed - profile_step
            window_ms = 1000 * (now - profile_time) / max(1, window_updates)
            window_overhead_ms = sum(overhead_ms.values()) - profile_overhead_ms
            training_window_ms = max(0.0, window_ms - window_overhead_ms / max(1, window_updates))
            el = time.time() - t0
            rec = {"step": updates_completed, "loss": loss_value, "loss_ema":ema_value, "tgt_acc": ta,
                   "gnorm": float(gn), "lr": lr_at(step, tc), "tok_s": (tok_seen-tokens_at_start) / el,
                   "step_ms_window":window_ms, "window_updates":window_updates,
                   "train_step_ms_window":training_window_ms,
                   "overhead_ms_window":window_overhead_ms,
                   "timing_window_valid":int(profile_has_previous_log and profile_step >= tc.warmup),
                   "tokens_per_second_window":B*L*1000/max(window_ms,1e-12),
                   "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                   "diag_ms_cumulative":overhead_ms["diag"], "eval_ms_cumulative":overhead_ms["eval"],
                   "checkpoint_ms_cumulative":overhead_ms["checkpoint"]}
            logf.write(json.dumps(rec) + "\n")
            profile_step, profile_time, profile_has_previous_log = updates_completed, now, True
            profile_overhead_ms = sum(overhead_ms.values())
            # Loss-triggered diagnostics and checkpoint bursts are intentionally disabled.
            if updates_completed % (tc.log_every * 8) == 0:
                print(f"step {updates_completed} loss {rec['loss']:.4f} acc {ta:.3f} tok/s {rec['tok_s']/1e6:.2f}M "
                      f"ms/step {1000*el/(step+1-start_step):.1f}", flush=True)
        if updates_completed == 1 and not first_update_checkpoint_saved:
            first_update_checkpoint_saved = True
            checkpoint("first_update")
            heartbeat(force=True)
        if updates_completed in extra_model_steps:
            save_model(Path(out) / "models" / f"step_{updates_completed:06d}.pt",
                       raw.state_dict(), cfg_all, updates_completed, source_hashes)
        if time.monotonic() - last_checkpoint_time >= tc.checkpoint_seconds:
            checkpoint("periodic")

    segment_elapsed = time.time() - t0
    elapsed = elapsed_before + segment_elapsed
    if budget_interrupted or pause_pending():
        finish_paused()
        return
    checkpoint("final")
    if "va" not in locals() or last_eval_step != updates_completed:
        eval_started = time.perf_counter()
        va, vl, ba = evaluate()
        overhead_ms["eval"] += 1000 * (time.perf_counter() - eval_started)
    if pause_pending():
        finish_paused()
        return
    # ---------------- final full benchmark eval
    qm.eval()
    bp, _ = predict(qm, bench["x"], bench["q"], B, bm)
    by = bench["x"].gather(1, bench["q"] + 1)
    bc = bp == by
    ba = acc_split(bc, bench["d"], L)
    dtab, sdtab = tables(bc.flatten(), bench["s"].flatten(), bench["d"].flatten(), L)
    np.savez_compressed(f"{out}/bench_preds.npz",
                        d=bench["d"].flatten().cpu().numpy().astype(np.int16),
                        s=bench["s"].flatten().cpu().numpy().astype(np.int16),
                        q=bench["q"].flatten().cpu().numpy().astype(np.int16),
                        correct=bc.flatten().cpu().numpy(),
                        pred=bp.flatten().cpu().numpy().astype(np.int16),
                        target=by.flatten().cpu().numpy().astype(np.int16),
                        n_q=np.int32(tk.n_q), L=np.int32(L))
    if pause_pending():
        finish_paused()
        return
    if not budget_interrupted:
        # This seed is never scored by evaluate(), early stopping or diagnostics.
        final_test, final_test_path = load_bench(tk, test_config(tc))
        if sha256_file(final_test_path) != test_hash:
            raise ValueError("Final-test asset changed during training")
        final_test = {k:v.to(dev) for k,v in final_test.items()}
        test_started = time.perf_counter()
        tp, tl = predict(qm, final_test["x"], final_test["q"], B, bm)
        ty = final_test["x"].gather(1, final_test["q"] + 1)
        test_correct = tp == ty
        test_scores = acc_split(test_correct, final_test["d"], L)
        test_eval_ms = 1000 * (time.perf_counter() - test_started)
        test_summary = {"seed":tc.test_seed, "queries":int(test_correct.numel()),
                        "benchmark_sha256":test_hash, "bench_file":final_test_path,
                        "accuracy":test_scores, "loss":float(tl.mean()), "eval_ms":test_eval_ms,
                        "used_for_training_selection":False}
        np.savez_compressed(f"{out}/test_preds.npz",
            **{k:final_test[k].flatten().cpu().numpy().astype(np.int16) for k in ("d","s","q")},
            correct=test_correct.flatten().cpu().numpy(), pred=tp.flatten().cpu().numpy().astype(np.int16),
            target=ty.flatten().cpu().numpy().astype(np.int16), n_q=np.int32(tk.n_q), L=np.int32(L))
        json.dump(test_summary, open(f"{out}/test_summary.json", "w"), indent=1)

        del final_test, tp, tl, ty, test_correct
    else:
        test_summary = {"seed":tc.test_seed, "used_for_training_selection":False,
                        "status":"not_evaluated_interrupted", "benchmark_sha256":test_hash}
        json.dump(test_summary, open(f"{out}/test_summary.json", "w"), indent=1)
    if pause_pending():
        finish_paused()
        return
    final_attention = None
    if tc.diagnostics:
        emit_diag(updates_completed, "final", snapshot=True)
    summary = {"params": n_params, "steps_done": updates_completed, "stopped_early": stopped,
               "budget_interrupted": budget_interrupted, "start_step": start_step, "tokens": tok_seen,
               "time": elapsed, "tok_s": tok_seen / elapsed, "first_step": first,
               "final_valid_acc": va, "final_valid_loss": vl,
               "bench": ba, "bench_file": bpath, "n_bench_queries": int(bc.numel()),
               "d_buckets": dtab, "sd_table": sdtab, "attn": final_attention,
               "sampling":sampler.statistics(), "test":test_summary,
               "bench_role":"development_only", "overhead_ms":overhead_ms,
               "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    if tc.save:
        save_model(Path(out) / "snapshot.pt", raw.state_dict(), cfg_all, updates_completed, source_hashes)
    write_json_atomic(Path(out) / "summary.json", summary)
    logf.flush(); os.fsync(logf.fileno()); logf.close()
    if attention_log is not None:
        attention_log.close()
    heartbeat(force=True, status="done")
    print("DONE", json.dumps({"steps_done": updates_completed, "time": round(elapsed, 1), "tok_s": round(tok_seen / elapsed),
                              "first_step": first, "valid": va, "bench": ba}), flush=True)


if __name__ == "__main__":
    main()
