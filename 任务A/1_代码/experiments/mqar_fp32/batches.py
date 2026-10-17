"""Convert the fixed cache once, then overlap pinned transfers with computation."""

import numpy as np
import torch

from experiments.mqar.data import load_split


class CachedPool:
    def __init__(self, directory, split):
        pool = load_split(directory, split)
        self.arrays = {name: torch.from_numpy(np.array(pool[name], dtype=np.int64))
                       for name in ("inputs", "positions", "targets")}
        self.size = len(self.arrays["inputs"])

    def batches(self, order, batch_size, device, start=0):
        if device.type != "cuda":
            for position in range(start, len(order), batch_size):
                indices = torch.from_numpy(order[position:position + batch_size])
                yield position + len(indices), tuple(value.index_select(0, indices) for value in self.arrays.values())
            return
        stream = torch.cuda.Stream(device=device)
        buffers, events = {}, {}

        def prepare(position, slot):
            indices = torch.from_numpy(order[position:position + batch_size])
            if slot in events:
                events[slot].synchronize()  # Only the previous copy, not the entire GPU.
            size = len(indices)
            key = (slot, size)
            if key not in buffers:
                buffers[key] = tuple(torch.empty((size, *value.shape[1:]), dtype=value.dtype,
                                                pin_memory=True) for value in self.arrays.values())
            host = buffers[key]
            for value, destination in zip(self.arrays.values(), host):
                torch.index_select(value, 0, indices, out=destination)
            with torch.cuda.stream(stream):
                values = tuple(value.to(device, non_blocking=True) for value in host)
                event = torch.cuda.Event()
                event.record(stream)
            events[slot] = event
            return position + size, values, event

        if start >= len(order):
            return
        pending = prepare(start, 0)
        slot = 0
        while pending is not None:
            end, values, event = pending
            current = torch.cuda.current_stream(device)
            current.wait_event(event)
            for value in values:
                value.record_stream(current)
            slot = 1 - slot
            pending = prepare(end, slot) if end < len(order) else None
            yield end, values
