"""MQAR-style long-range recall task (v2) with four independently controllable knobs.

Sequence of length L = filler tokens + task tokens.
  * n_q queried events j: key k_j at s_j, value v_j at s_j+1 (definition); key k_j again at q_j and
    v_j at q_j+1 (query). The model must predict x[q_j+1] = v_j at position q_j. Dependency distance
    Delta_j = q_j - s_j.
  * n_dis distractor pairs (k', v') placed uniformly, never queried.
Keys are unique within a sequence; values are sampled with replacement.

Knobs (orthogonal at fixed L):
  1. L                       context length
  2. Delta distribution      dist in {uniform, loguniform}, range [d_min, d_max]
  3. n_q                     supervised queries per sequence  (supervision density rho = n_q / L)
  4. n_dis                   non-queried distractor pairs     (competing keys = n_q + n_dis)

Filler ("background"): iid uniform over Vb filler tokens (default), all-zero, or order-1 Markov.
Everything is generated on the GPU.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class TaskCfg:
    L: int = 1024
    n_q: int = 32
    n_dis: int = 32
    dist: str = "uniform"      # uniform | loguniform
    d_min: int = 4
    d_max: int = 0             # 0 -> L - 2 (full range)
    Vb: int = 256              # filler vocab
    K: int = 512               # key vocab (must be >= n_q + n_dis)
    Vv: int = 128              # value vocab (chance = 1/Vv)
    background: str = "iid"    # iid | zero | markov
    branch: int = 4            # markov successors

    @property
    def vocab(self):
        return self.Vb + self.K + self.Vv

    @property
    def dmax(self):
        return self.d_max if self.d_max > 0 else self.L - 2


class Task:
    def __init__(self, cfg: TaskCfg, device, seed=0):
        assert cfg.n_q + cfg.n_dis <= cfg.K, "need n_q + n_dis <= K"
        assert cfg.dmax <= cfg.L - 2
        self.cfg, self.dev = cfg, device
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)
        if cfg.background == "markov":
            gc = torch.Generator().manual_seed(1234)
            self.succ = torch.randint(0, cfg.Vb, (cfg.Vb, cfg.branch), generator=gc).to(device)
            p = torch.tensor([0.5 ** i for i in range(cfg.branch)])
            self.cum = (p / p.sum()).cumsum(0).to(device)

    # ---------------- filler ----------------
    def _filler(self, B, L):
        c = self.cfg
        if c.background == "zero":
            return torch.zeros(B, L, dtype=torch.long, device=self.dev)
        if c.background == "iid":
            return torch.randint(0, c.Vb, (B, L), device=self.dev, generator=self.g)
        out = torch.empty(B, L, dtype=torch.long, device=self.dev)
        cur = torch.randint(0, c.Vb, (B,), device=self.dev, generator=self.g)
        u = torch.rand(B, L, device=self.dev, generator=self.g)
        for t in range(L):
            out[:, t] = cur
            j = torch.searchsorted(self.cum, u[:, t].contiguous()).clamp_max(c.branch - 1)
            cur = self.succ[cur, j]
        return out

    # ---------------- distances ----------------
    def sample_d(self, B, n, lo=None, hi=None, dist=None):
        c = self.cfg
        lo = c.d_min if lo is None else lo
        hi = c.dmax if hi is None else hi
        dist = dist or c.dist
        if dist == "uniform":
            return torch.randint(lo, hi + 1, (B, n), device=self.dev, generator=self.g)
        if dist == "loguniform":
            u = torch.rand(B, n, device=self.dev, generator=self.g)
            d = torch.exp(math.log(lo) + u * (math.log(hi + 1) - math.log(lo))).floor().long()
            return d.clamp(lo, hi)
        raise ValueError(dist)

    # ---------------- placement ----------------
    def _q_given_d(self, d, L):
        u = torch.rand(d.shape, device=self.dev, generator=self.g)
        return d + (u * (L - 2 - d + 1)).long()          # q in [d, L-2], s = q - d >= 0

    def _place(self, B, L, d, redraw=None):
        """Non-overlapping 2-token items. Settled-first repair: placed items never move; conflicting items
        are resampled keeping their distance. Distances very close to L have only a few feasible slots and two
        of them can be mutually exclusive; after 50 rounds such stuck queries redraw their distance via
        `redraw(shape)` (rare; the realised distance is always returned and used for scoring)."""
        nq, nd = d.shape[1], self.cfg.n_dis
        d = d.clone()
        q = self._q_given_d(d, L)
        s = q - d
        r = torch.randint(0, L - 1, (B, nd), device=self.dev, generator=self.g)
        sq = torch.zeros(B, nq, dtype=torch.bool, device=self.dev)
        sr = torch.zeros(B, nd, dtype=torch.bool, device=self.dev)
        for it in range(4000):
            starts = torch.cat([s, q, r], 1)
            sett = torch.cat([sq, sq, sr], 1)
            ws, wn = sett.int(), (~sett).int()
            occS = torch.zeros(B, L + 1, device=self.dev, dtype=torch.int32)
            occS.scatter_add_(1, starts, ws); occS.scatter_add_(1, starts + 1, ws)
            occN = torch.zeros(B, L + 1, device=self.dev, dtype=torch.int32)
            occN.scatter_add_(1, starts, wn); occN.scatter_add_(1, starts + 1, wn)
            hitS = (occS.gather(1, starts) + occS.gather(1, starts + 1)) > 0
            hitN = (occN.gather(1, starts) + occN.gather(1, starts + 1)) > 2
            ok = ~sett & ~hitS & ~hitN
            sq = sq | (ok[:, :nq] & ok[:, nq:2 * nq])
            sr = sr | ok[:, 2 * nq:]
            if bool(sq.all()) and bool(sr.all()):
                return s, q, r, d
            if it >= 50 and redraw is not None and it % 50 == 0:
                d = torch.where(sq, d, redraw(d.shape))
            q = torch.where(sq, q, self._q_given_d(d, L))
            s = q - d
            if nd:
                r = torch.where(sr, r, torch.randint(0, L - 1, (B, nd), device=self.dev, generator=self.g))
        raise RuntimeError("placement failed; too many items for L")

    @torch.no_grad()
    def sample(self, B, d=None, lo=None, hi=None, dist=None):
        """Returns dict: x (B,L), q (B,n_q) query positions (predict x[q+1]), s (B,n_q) source key positions,
        d (B,n_q) realised distances. If d is None it is drawn from the given (default: training) distribution."""
        c = self.cfg
        L = c.L
        redraw = lambda shape: self.sample_d(shape[0], shape[1], lo=lo, hi=hi, dist=dist)
        if d is None:
            d = redraw((B, c.n_q))
        s, q, r, d = self._place(B, L, d, redraw)
        x = self._filler(B, L)
        keys = torch.rand(B, c.K, device=self.dev, generator=self.g).argsort(1)[:, : c.n_q + c.n_dis] + c.Vb
        vals = torch.randint(0, c.Vv, (B, c.n_q + c.n_dis), device=self.dev, generator=self.g) + c.Vb + c.K
        kq, kd, vq, vd = keys[:, :c.n_q], keys[:, c.n_q:], vals[:, :c.n_q], vals[:, c.n_q:]
        x.scatter_(1, s, kq); x.scatter_(1, s + 1, vq)
        x.scatter_(1, q, kq); x.scatter_(1, q + 1, vq)
        if c.n_dis:
            x.scatter_(1, r, kd); x.scatter_(1, r + 1, vd)
        return {"x": x, "q": q, "s": s, "d": d}


def make_benchmark(cfg: TaskCfg, n_seq, device, seed, lo=4, hi=None):
    """Fixed benchmark: same context statistics as cfg (n_q, n_dis, filler) but every query's distance is
    uniform over the integers in [lo, hi] (default: the whole window [4, L-2]). Long-range = Delta >= L/2."""
    hi = hi or cfg.L - 2
    t = Task(cfg, device, seed=seed)
    out = {"x": [], "q": [], "s": [], "d": []}
    chunk = max(1, 262144 // cfg.L)
    for i in range(0, n_seq, chunk):
        b = min(chunk, n_seq - i)
        o = t.sample(b, lo=lo, hi=hi, dist="uniform")
        for k in out:
            out[k].append(o[k].cpu())
    return {k: torch.cat(v) for k, v in out.items()}
