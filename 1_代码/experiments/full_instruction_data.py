"""读取全量指令数据；token/mask 留在 mmap 中，每次只物化当前 batch。"""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class FullInstructionData:
    def __init__(self, directory, split, expected_manifest_hash=None):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if expected_manifest_hash and file_sha256(manifest_path) != expected_manifest_hash:
            raise ValueError("SFT data manifest changed after planning")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["format"] != "smoltalk_mmap_v1":
            raise ValueError("Unsupported SFT data format")
        paths = self.manifest["splits"][split]["shards"]
        stats = {row["path"]: row for row in self.manifest["shards"]}
        self.shards, indexes = [], []
        for number, relative_path in enumerate(paths):
            path = self.directory / relative_path
            for name, expected in stats[relative_path]["sha256"].items():
                if file_sha256(path / name) != expected:
                    raise ValueError(f"Data file changed: {relative_path}/{name}")
            tokens = np.memmap(path / "tokens.bin", mode="r", dtype="<u2")
            mask = np.memmap(path / "assistant_mask.bin", mode="r", dtype="u1")
            self.shards.append((tokens, mask))
            index = np.load(path / "windows.npy")
            # [shard_number, flat_token_offset, input_length, overlap_mask_prefix, source_row]
            indexes.append(np.column_stack([np.full(len(index), number, dtype="<u8"), index]))
        self.index = np.concatenate(indexes)
        self.lengths = self.index[:, 2].astype(np.int64)
        self.target_counts = np.empty(len(self), dtype=np.int64)
        for number, (shard, offset, length, prefix, _) in enumerate(self.index):
            mask = self.shards[int(shard)][1]
            start = int(offset) + 1 + int(prefix)
            self.target_counts[number] = int(mask[start:int(offset) + int(length) + 1].sum())
        if int(self.target_counts.sum()) != self.manifest["splits"][split]["supervised_tokens"]:
            raise ValueError("Supervised-token budget differs from data manifest")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, number):
        shard, offset, length, prefix, _ = map(int, self.index[number])
        tokens, mask = self.shards[shard]
        inputs = tokens[offset:offset + length].astype(np.int64)
        labels = tokens[offset + 1:offset + length + 1].astype(np.int64)
        labels[mask[offset + 1:offset + length + 1] == 0] = -100
        labels[:prefix] = -100
        return {"sft_input_ids": inputs, "sft_labels": labels}


def epoch_order(size, seed, epoch):
    return np.random.default_rng(seed + epoch).permutation(size)


def padded_positions(lengths, order, batch_size):
    full = len(order) // batch_size * batch_size
    positions = int(lengths[order[:full]].reshape(-1, batch_size).max(axis=1).sum()) * batch_size
    if full < len(order):
        positions += int(lengths[order[full:]].max()) * (len(order) - full)
    return positions
