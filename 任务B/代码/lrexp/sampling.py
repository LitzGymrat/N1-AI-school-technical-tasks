"""Resumable sampling for the next round; no model or optimizer code lives here.

Each short/full stream owns its RNG and 16-update cache.  Reordering stream
choices therefore reorders exactly the same batches, including at a boundary
which is not divisible by 16.  Coverage mixtures instead assign every query a
fixed short/long identity, including while repairing placement collisions.
"""
from dataclasses import asdict, replace

import torch

from .task import Task


FORMAT = "next-round-sampling-v1"
MODES = ("base", "coverage", "short_first", "shuffled")


class MixtureTask(Task):
    """An exact long-query quota in each group of B sequences."""

    def __init__(self, cfg, device, seed, batch_size, short_min, short_max, long_fraction):
        super().__init__(cfg, device, seed)
        self.batch_size = batch_size
        self.short_min = short_min
        self.short_max = short_max
        self.long_fraction = long_fraction
        self._membership = None

    def _quota(self, queries):
        expected = queries * self.long_fraction
        count = round(expected)
        if abs(expected - count) > 1e-8:
            raise ValueError("long_fraction * queries per update must be an integer")
        return count

    def _identities(self, rows):
        masks = []
        for start in range(0, rows, self.batch_size):
            count = min(self.batch_size, rows - start) * self.cfg.n_q
            quota = self._quota(count)
            mask = torch.zeros(count, dtype=torch.bool, device=self.dev)
            if quota == count:
                mask.fill_(True)
            elif quota:
                order = torch.rand(count, device=self.dev, generator=self.g).argsort()
                mask[order[:quota]] = True
            masks.append(mask.reshape(-1, self.cfg.n_q))
        return torch.cat(masks)

    def sample_d(self, B, n, lo=None, hi=None, dist=None):
        # Task._place calls this again for colliding queries.  Their identities
        # must survive that redraw, rather than reverting to the base range.
        if self._membership is None or self._membership.shape != (B, n):
            raise RuntimeError("Mixture distance sampling requires fixed query identities")
        short = torch.randint(self.short_min, self.short_max + 1, (B, n), device=self.dev, generator=self.g)
        long = torch.randint(self.cfg.L // 2, self.cfg.L - 1, (B, n), device=self.dev, generator=self.g)
        return torch.where(self._membership, long, short)

    @torch.no_grad()
    def sample(self, B, d=None, lo=None, hi=None, dist=None):
        if d is not None or any(v is not None for v in (lo, hi, dist)):
            raise ValueError("MixtureTask owns its short/long ranges and query identities")
        self._membership = self._identities(B)
        try:
            d = self.sample_d(B, self.cfg.n_q)
            result = super().sample(B, d=d)
            result["is_long"] = self._membership
            return result
        finally:
            self._membership = None


class BatchStream:
    """One deterministic sequence of complete batches, cached 16 at a time."""

    def __init__(self, task, batch_size):
        self.task = task
        self.batch_size = batch_size
        self.cursor = 0
        self.cache_start = 0
        self.buffer = None

    def next(self):
        offset = self.cursor - self.cache_start
        if self.buffer is None or offset == 16:
            self.buffer = self.task.sample(self.batch_size * 16)
            self.cache_start = self.cursor
            offset = 0
        start = offset * self.batch_size
        result = {k: v[start:start + self.batch_size] for k, v in self.buffer.items()}
        self.cursor += 1
        return result

    def state_dict(self):
        return {"cursor": self.cursor, "cache_start": self.cache_start,
                "generator": self.task.g.get_state(), "buffer": self.buffer}

    def load_state_dict(self, state):
        cursor, start = int(state["cursor"]), int(state["cache_start"])
        if not (0 <= cursor - start <= 16) or cursor < 0:
            raise ValueError("Invalid saved stream cursor")
        self.task.g.set_state(state["generator"].cpu())
        self.buffer = ({k: v.to(self.task.dev) for k, v in state["buffer"].items()}
                       if state["buffer"] is not None else None)
        if self.buffer is None and cursor != 0:
            raise ValueError("A consumed stream requires its saved cache")
        if self.buffer is not None:
            expected = self.batch_size * 16
            if any(v.shape[0] != expected for v in self.buffer.values()):
                raise ValueError("Saved stream cache batch size differs")
        self.cursor, self.cache_start = cursor, start


class TrainingSampler:
    def __init__(self, cfg, train, device):
        self.cfg, self.train, self.device = cfg, train, device
        self.batch_size = max(1, train.tokens_per_step // cfg.L)
        self.mode = train.sampling_plan
        if self.mode not in MODES:
            raise ValueError(f"sampling_plan must be one of {MODES}")
        if self.mode != "base" and train.stop_acc != 0:
            raise ValueError("Coverage and curriculum comparisons require stop_acc=0")
        if self.mode != "base" and not (4 <= train.short_min <= train.short_max < cfg.L // 2):
            raise ValueError("short range must satisfy 4 <= short_min <= short_max < L//2")
        if not 0 <= train.long_fraction <= 1:
            raise ValueError("long_fraction must lie in [0,1]")
        if self.mode in ("short_first", "shuffled"):
            if not 0 <= train.short_steps <= train.steps:
                raise ValueError("short_steps must lie within the total update budget")
            if cfg.dist != "uniform" or cfg.d_min != 4 or cfg.dmax != cfg.L - 2:
                raise ValueError("Curriculum full stream must be U[4,L-2]")
        seed = train.seed if train.data_seed < 0 else train.data_seed
        self.seed = seed
        self.streams = {}
        if self.mode == "coverage":
            task = MixtureTask(cfg, device, seed, self.batch_size,
                               train.short_min, train.short_max, train.long_fraction)
            task._quota(self.batch_size * cfg.n_q)
            self.streams["coverage"] = BatchStream(task, self.batch_size)
        else:
            # The base run and both curricula use exactly this full stream.
            self.streams["full"] = BatchStream(Task(cfg, device, seed), self.batch_size)
            if self.mode != "base":
                short_cfg = replace(cfg, dist="uniform", d_min=train.short_min, d_max=train.short_max)
                self.streams["short"] = BatchStream(Task(short_cfg, device, seed + 10_000_000), self.batch_size)
        self.order = None
        if self.mode == "shuffled":
            g = torch.Generator().manual_seed(seed + 20_000_000)
            order = torch.zeros(train.steps, dtype=torch.bool)
            order[torch.randperm(train.steps, generator=g)[:train.short_steps]] = True
            self.order = order.tolist()  # One CPU conversion, never a per-update GPU sync.
        self.updates = 0
        self.total_queries = 0
        self.long_queries = torch.zeros((), dtype=torch.int64, device=device)
        self.histogram = torch.zeros(cfg.L, dtype=torch.int64, device=device)
        self._pending_stream = None

    def contract(self):
        return {"format": FORMAT, "task": asdict(self.cfg), "batch_size": self.batch_size,
                "mode": self.mode, "data_seed": self.seed, "steps": self.train.steps,
                "short_min": self.train.short_min,
                "short_max": self.train.short_max, "short_steps": self.train.short_steps,
                "long_fraction": self.train.long_fraction}

    def stream_name(self, step):
        if self.mode == "coverage":
            return "coverage"
        if self.mode == "short_first":
            return "short" if step < self.train.short_steps else "full"
        if self.mode == "shuffled":
            return "short" if self.order[step] else "full"
        return "full"

    def next_batch(self, step):
        if step != self.updates or self._pending_stream is not None:
            raise ValueError("Sampler update cursor differs or a batch is still pending")
        if step >= self.train.steps:
            raise ValueError("Sampling would exceed the configured total budget")
        name = self.stream_name(step)
        batch = self.streams[name].next()
        self._pending_stream = name
        return batch

    def discard_pending(self):
        """Rewind an unapplied batch while retaining its deterministic cache."""
        if self._pending_stream is not None:
            self.streams[self._pending_stream].cursor -= 1
            self._pending_stream = None

    @torch.no_grad()
    def record_consumed(self, batch):
        """Call after the optimizer update; counters remain on-device until logging."""
        if self._pending_stream is None:
            raise ValueError("No sampled batch to record")
        d = batch["d"].reshape(-1)
        self.histogram.scatter_add_(0, d, torch.ones_like(d, dtype=torch.int64))
        self.long_queries.add_((d >= self.cfg.L // 2).sum())
        self.total_queries += d.numel()
        self.updates += 1
        self._pending_stream = None

    def statistics(self):
        long_queries = int(self.long_queries.cpu())
        return {"updates": self.updates, "supervised_queries": self.total_queries,
                "long_queries": long_queries,
                "realized_long_fraction": long_queries / self.total_queries if self.total_queries else None,
                "distance_histogram": self.histogram.cpu().tolist(),
                "stream_updates": {k: s.cursor for k, s in self.streams.items()}}

    def state_dict(self):
        if self._pending_stream is not None:
            raise ValueError("Cannot checkpoint a sampled but unapplied update")
        return {"contract": self.contract(), "updates": self.updates,
                "total_queries": self.total_queries, "long_queries": self.long_queries,
                "histogram": self.histogram, "order": self.order,
                "streams": {k: s.state_dict() for k, s in self.streams.items()}}

    def load_state_dict(self, state):
        if state.get("contract") != self.contract():
            raise ValueError("Saved sampling contract differs from this run")
        if state.get("order") != self.order or set(state["streams"]) != set(self.streams):
            raise ValueError("Saved sampling order or stream identities differ")
        updates = int(state["updates"])
        if not 0 <= updates <= self.train.steps:
            raise ValueError("Saved sampling update is out of bounds")
        for k, stream in self.streams.items():
            stream.load_state_dict(state["streams"][k])
        if sum(s.cursor for s in self.streams.values()) != updates:
            raise ValueError("Saved stream cursors disagree with completed updates")
        if self.mode in ("short_first", "shuffled"):
            short_count = (min(updates, self.train.short_steps) if self.mode == "short_first"
                           else sum(self.order[:updates]))
            if (self.streams["short"].cursor != short_count
                    or self.streams["full"].cursor != updates-short_count):
                raise ValueError("Saved component cursors disagree with the sampling order")
        expected_queries = updates * self.batch_size * self.cfg.n_q
        if int(state["total_queries"]) != expected_queries:
            raise ValueError("Saved query count differs from completed updates")
        self.updates, self.total_queries = updates, expected_queries
        self.long_queries = state["long_queries"].to(self.device)
        self.histogram = state["histogram"].to(self.device)
        if self.histogram.shape != (self.cfg.L,):
            raise ValueError("Saved distance histogram has the wrong length")
        if (int(self.histogram.sum().cpu()) != expected_queries
                or int(self.histogram[self.cfg.L//2:].sum().cpu()) != int(self.long_queries.cpu())):
            raise ValueError("Saved realized distance counts disagree")

    @property
    def primary_stream(self):
        return self.streams.get("full", self.streams.get("coverage"))


def make_validation(cfg, train, device):
    """Coverage gets its own mixture; both curricula share fixed full validation."""
    seed = 10_000 + train.seed
    if train.sampling_plan == "coverage":
        batch_size = max(1, train.tokens_per_step // cfg.L)
        task = MixtureTask(cfg, device, seed, batch_size,
                           train.short_min, train.short_max, train.long_fraction)
    else:
        task = Task(cfg, device, seed)
    return task.sample(train.valid_n)
