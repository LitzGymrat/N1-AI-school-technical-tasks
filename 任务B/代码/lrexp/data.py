"""GPU-resident synthetic data generator: associative recall embedded in a local Markov stream.

Vocabulary layout:  [0, Vb) background | [Vb, Vb+K) keys | [Vb+K, Vb+K+Vv) values
A sequence of length L consists of a background stream (order-1 Markov chain with a few
likely successors per token -> strong *local* predictability, i.e. a "local shortcut") into which we
write
  * n_q query events: definition (key k, value v) at positions (p, p+1) and query (k, v) at
    (q, q+1) with q - p = d (the dependency distance). Predicting x[q+1] from position q requires
    retrieving v from d tokens back.
  * n_dis distractor definitions (k', v') whose key is never queried.
Everything is generated on the GPU: zero host->device traffic during training.
"""
from dataclasses import dataclass, field
import math

import torch


@dataclass
class DataCfg:
    Vb: int = 256            # background vocab
    K: int = 128             # key vocab
    Vv: int = 128            # value vocab
    branch: int = 4          # Markov successors per background token
    background: str = "markov"   # markov | iid
    L: int = 1024
    n_q: int = 8             # query events per sequence
    n_dis: int = 8           # distractor definitions per sequence
    dist: str = "fixed"      # fixed | uniform | loguniform | mix
    d: int = 256             # fixed distance (also used as d_max for ranges)
    d_min: int = 4
    mix_short_frac: float = 0.0  # for dist=mix: fraction of events with short distance
    d_short: int = 16            # short distance for mix (uniform in [d_min, d_short])
    d_long_min: int = 0          # for dist=mix: long part ~ U[d_long_min, d] (0 -> fixed d)
    q_place: str = "uniform"     # uniform | end (single query at the very end)
    chains: int = 4096
    chain_len: int = 1 << 15

    @property
    def vocab(self):
        return self.Vb + self.K + self.Vv


class Gen:
    def __init__(self, cfg: DataCfg, device, seed=0):
        self.cfg = cfg
        self.dev = device
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)
        c = cfg
        gcpu = torch.Generator().manual_seed(1234)  # task (Markov chain) fixed across runs
        self.succ = torch.randint(0, c.Vb, (c.Vb, c.branch), generator=gcpu).to(device)
        p = torch.tensor([0.5 ** i for i in range(c.branch)])
        self.cum = (p / p.sum()).cumsum(0).to(device)
        if c.background == "markov":
            self.buf = self._make_chain()
        else:
            self.buf = None

    def _make_chain(self):
        c = self.cfg
        out = torch.empty(c.chains, c.chain_len, dtype=torch.int16, device=self.dev)
        cur = torch.randint(0, c.Vb, (c.chains,), device=self.dev, generator=self.g)
        u = torch.rand(c.chains, c.chain_len, device=self.dev, generator=self.g)
        for t in range(c.chain_len):
            out[:, t] = cur
            j = torch.searchsorted(self.cum, u[:, t].contiguous()).clamp_max(c.branch - 1)
            cur = self.succ[cur, j]
        return out

    def background(self, B, L):
        c = self.cfg
        if self.buf is None:
            return torch.randint(0, c.Vb, (B, L), device=self.dev, generator=self.g)
        assert L <= c.chain_len
        ch = torch.randint(0, c.chains, (B,), device=self.dev, generator=self.g)
        off = torch.randint(0, c.chain_len - L + 1, (B,), device=self.dev, generator=self.g)
        idx = off[:, None] + torch.arange(L, device=self.dev)[None]
        return self.buf[ch[:, None], idx].long()

    def _dists(self, B, n):
        c = self.cfg
        g, dev = self.g, self.dev
        if c.dist == "fixed":
            return torch.full((B, n), c.d, device=dev, dtype=torch.long)
        if c.dist == "uniform":
            return torch.randint(c.d_min, c.d + 1, (B, n), device=dev, generator=g)
        if c.dist == "loguniform":
            u = torch.rand(B, n, device=dev, generator=g)
            return torch.exp(math.log(c.d_min) + u * (math.log(c.d) - math.log(c.d_min))).round().long()
        if c.dist == "mix":
            if c.d_long_min > 0:
                long_ = torch.randint(c.d_long_min, c.d + 1, (B, n), device=dev, generator=g)
            else:
                long_ = torch.full((B, n), c.d, device=dev, dtype=torch.long)
            short = torch.randint(c.d_min, c.d_short + 1, (B, n), device=dev, generator=g)
            m = torch.rand(B, n, device=dev, generator=g) < c.mix_short_frac
            return torch.where(m, short, long_)
        raise ValueError(c.dist)

    def _sample_q(self, d, L):
        c = self.cfg
        if c.q_place == "end":
            return torch.full_like(d, L - 2)
        u = torch.rand(d.shape, device=self.dev, generator=self.g)
        return d + (u * (L - 2 - d + 1).clamp_min(1)).long()   # q in [d, L-2] so p = q-d >= 0

    def _place(self, B, L, d):
        """Sample non-overlapping (p, q, r) start positions (each item occupies [s, s+1]).
        Settled-first repair: once an item is placed without conflict it never moves; unsettled items are
        resampled until they avoid settled items and each other."""
        c = self.cfg
        g, dev = self.g, self.dev
        nq, nd = d.shape[1], c.n_dis
        if c.q_place == "end":
            assert nq == 1, "q_place=end requires n_q=1"
        q = self._sample_q(d, L)
        p = q - d
        r = torch.randint(0, L - 1, (B, nd), device=dev, generator=g)
        sq = torch.zeros(B, nq, dtype=torch.bool, device=dev)
        sr = torch.zeros(B, nd, dtype=torch.bool, device=dev)
        for it in range(2000):
            starts = torch.cat([p, q, r], 1)
            sett = torch.cat([sq, sq, sr], 1)
            ws, wn = sett.int(), (~sett).int()
            occS = torch.zeros(B, L + 1, device=dev, dtype=torch.int32)
            occS.scatter_add_(1, starts, ws); occS.scatter_add_(1, starts + 1, ws)
            occN = torch.zeros(B, L + 1, device=dev, dtype=torch.int32)
            occN.scatter_add_(1, starts, wn); occN.scatter_add_(1, starts + 1, wn)
            hitS = (occS.gather(1, starts) + occS.gather(1, starts + 1)) > 0
            hitN = (occN.gather(1, starts) + occN.gather(1, starts + 1)) > 2
            ok = ~sett & ~hitS & ~hitN
            sq = sq | (ok[:, :nq] & ok[:, nq:2 * nq])
            sr = sr | ok[:, 2 * nq:]
            if it % 4 == 3 and bool(sq.all()) and bool(sr.all()):
                return p, q, r
            q = torch.where(sq, q, self._sample_q(d, L))
            p = q - d
            if nd:
                r = torch.where(sr, r, torch.randint(0, L - 1, (B, nd), device=dev, generator=g))
        raise RuntimeError("placement failed; too many events for L")

    @torch.no_grad()
    def sample(self, B, L=None, d=None):
        """Returns x (B,L) int64, tgt_pos (B,n_q) positions whose *next* token is the queried value,
        def_pos (B,n_q) key position of the matching definition, dist (B,n_q)."""
        c = self.cfg
        L = L or c.L
        n_ret = c.n_q
        if d is None:
            d = self._dists(B, c.n_q)
        elif isinstance(d, int):
            # probe a fixed distance: force the first n_ret events to distance d, others follow the
            # training distance distribution (keeps context statistics identical to training)
            n_ret = max(1, min(c.n_q, (L - 2 - d) // 16))
            dd = self._dists(B, c.n_q).clamp_max(L - 4)
            dd[:, :n_ret] = d
            d = dd
        p, q, r = self._place(B, L, d)
        nq, nd = c.n_q, c.n_dis
        assert nq + nd <= c.K, "need n_q + n_dis <= K (unique keys per sequence)"
        x = self.background(B, L)
        # unique keys per row
        keys = torch.rand(B, c.K, device=self.dev, generator=self.g).argsort(1)[:, : nq + nd] + c.Vb
        vals = torch.randint(0, c.Vv, (B, nq + nd), device=self.dev, generator=self.g) + c.Vb + c.K
        kq, kd = keys[:, :nq], keys[:, nq:]
        vq, vd = vals[:, :nq], vals[:, nq:]
        x.scatter_(1, p, kq); x.scatter_(1, p + 1, vq)
        x.scatter_(1, q, kq); x.scatter_(1, q + 1, vq)
        if nd:
            x.scatter_(1, r, kd); x.scatter_(1, r + 1, vd)
        return x, q[:, :n_ret], p[:, :n_ret], d[:, :n_ret]
