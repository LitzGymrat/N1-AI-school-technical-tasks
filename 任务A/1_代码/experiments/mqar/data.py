"""Prepare fixed, disjoint original-layout MQAR splits without a text tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.tasks.mqar import generate_mqar


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def symbol(token, vocab_size):
    if token == 0:
        return "0"
    lower = token >= vocab_size // 2
    number = token - vocab_size // 2 + 1 if lower else token
    letters = ""
    while number:
        number, digit = divmod(number - 1, 26)
        letters = chr(ord("A") + digit) + letters
    return letters.lower() if lower else letters


def prepare(config, out_dir):
    task, settings = config["task"], config["data"]
    if task["random_non_queries"] or not task["labels_already_aligned"]:
        raise ValueError("This run uses original zero-filled MQAR and already aligned labels")
    length, vocab, pairs = task["input_seq_len"], task["vocab_size"], task["num_kv_pairs"]
    if vocab > 65536 or length > 65535:
        raise ValueError("uint16 cache requires vocabulary and positions within its range")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    hashes, records = {}, {}
    previews = []
    for split in ("train", "valid", "test"):
        count, seed = settings[f"{split}_samples"], settings[f"{split}_seed"]
        print(f"Generating {split}: {count} examples, seed={seed}", flush=True)
        data = generate_mqar(count, length, np.arange(vocab), seed,
                             num_kv_pairs=pairs, power_a=task["power_a"])
        inputs, positions = data["input_ids"], data["query_positions"]
        targets = np.take_along_axis(data["labels"], positions, axis=1)
        expected = np.take_along_axis(inputs, data["source_positions"] + 1, axis=1)
        if not np.array_equal(targets, expected):
            raise ValueError("A query answer differs from its original symbol association")
        if not np.array_equal(np.take_along_axis(inputs, positions, axis=1), inputs[:, :2 * pairs:2]):
            raise ValueError("Repeated query symbols differ from the original symbols")
        if not np.all(np.count_nonzero(data["labels"] != -100, axis=1) == pairs):
            raise ValueError("Unexpected supervised positions")
        if not np.all(np.take_along_axis(inputs, positions + 1, axis=1) == 0):
            raise ValueError("An answer was inserted into the query input")
        arrays = {"inputs": inputs.astype(np.uint16), "positions": positions.astype(np.uint16),
                  "targets": targets.astype(np.uint16),
                  "distances": data["dependency_distances"].astype(np.uint16)}
        hashes[split] = {hashlib.sha256(row.tobytes()).digest() for row in arrays["inputs"]}
        files = {}
        for name, array in arrays.items():
            filename = f"{split}_{name}.npy"
            np.save(out_dir / filename, array)
            files[name] = {"path": filename, "shape": list(array.shape), "sha256": sha256(out_dir / filename)}
        records[split] = {"samples": count, "seed": seed, "unique_inputs": len(hashes[split]),
                          "far_queries": int((arrays["distances"] >= config["evaluation"]["far_distance_min"]).sum()),
                          "files": files}
        previews.append(f"{split} sample 0; each displayed symbol is one token\n" +
                        " ".join(symbol(int(token), vocab) for token in arrays["inputs"][0]) + "\n" +
                        "\n".join(f"query position {int(q)}: {symbol(int(inputs[0,q]),vocab)} -> "
                                  f"{symbol(int(t),vocab)}, distance={int(d)}"
                                  for q, t, d in zip(positions[0], targets[0], data["dependency_distances"][0])))
        del data, inputs, positions, targets, expected, arrays
    overlap = {f"{left}_{right}": len(hashes[left] & hashes[right])
               for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    if any(overlap.values()):
        raise ValueError(f"Cross-split full input overlap: {overlap}")
    (out_dir / "preview.txt").write_text("\n\n".join(previews) + "\n")
    manifest = {"status": "verified", "task": task, "data_config": settings,
                "reference": config["reference"], "splits": records, "cross_split_overlap": overlap,
                "layout": "first symbol associations followed by scattered queries; labels aligned at query positions"}
    atomic_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"data_dir": str(out_dir), "cross_split_overlap": overlap,
                      "samples": {k: v["samples"] for k, v in records.items()}}), flush=True)
    return manifest


def load_split(data_dir, split):
    return {name: np.load(Path(data_dir) / f"{split}_{name}.npy", mmap_mode="r")
            for name in ("inputs", "positions", "targets", "distances")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out_dir)


if __name__ == "__main__":
    main()
