"""Evaluate a frozen checkpoint on fresh, exactly balanced, random far layouts."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mqar.data import atomic_json, sha256
from .model import build_computation, build_model


def random_positions(seed, attempts=2000000):
    """Randomize a feasible state by trades preserving the full distance histogram.

    The initial state is discarded. Trades exchange four distances across two
    rows and two source indices; accept only in-range, collision-free layouts.
    This is constrained randomization, not independent uniform position draws.
    """
    rng = np.random.RandomState(seed)
    source = np.arange(64)
    distances = np.tile(np.stack([510 - 4 * source, 508 - 4 * source]), (50, 1)) // 2
    distances = distances.tolist()
    occupied = [sum(1 << (d + i) for i, d in enumerate(row)) for row in distances]
    permutations = list(itertools.permutations(range(4)))
    accepted = 0
    for start in range(0, attempts, 100000):
        draws = rng.randint(0, 1000000, size=(min(100000, attempts - start), 5))
        for aa, bb, ii, jj, pp in draws.tolist():
            a, b, i, j = aa % 100, bb % 100, ii % 64, jj % 64
            if a == b or i == j:
                continue
            old = (distances[a][i], distances[a][j], distances[b][i], distances[b][j])
            perm = permutations[pp % 24]
            new = tuple(old[k] for k in perm)
            if new == old:
                continue
            query = (new[0] + i, new[1] + j, new[2] + i, new[3] + j)
            if max(query) > 255 or min(query) < 128 or query[0] == query[1] or query[2] == query[3]:
                continue
            free_a = occupied[a] ^ (1 << (old[0] + i)) ^ (1 << (old[1] + j))
            free_b = occupied[b] ^ (1 << (old[2] + i)) ^ (1 << (old[3] + j))
            bits_a, bits_b = (1 << query[0]) | (1 << query[1]), (1 << query[2]) | (1 << query[3])
            if free_a & bits_a or free_b & bits_b:
                continue
            distances[a][i], distances[a][j], distances[b][i], distances[b][j] = new
            occupied[a], occupied[b] = free_a | bits_a, free_b | bits_b
            accepted += 1
    distances = 2 * np.asarray(distances, dtype=np.int64)
    positions = distances + 2 * source
    assert np.array_equal(np.unique(distances, return_counts=True)[1], np.full(128, 50))
    assert np.all([len(set(row)) == 64 for row in positions])
    assert len({row.tobytes() for row in positions}) == 100
    initial = (510 - 2 * source, 508 - 2 * source)
    assert all(not any(np.array_equal(row, base) for base in initial) for row in positions)
    return positions, distances, {"seed": seed, "attempts": attempts, "accepted_trades": accepted,
        "unique_position_layouts": 100,
        "method": "constraint-preserving random trades; initial feasible layouts discarded",
        "boundary": "Random under exact histogram, prefix KV, even query slots and no-collision constraints; not an unbiased uniform sampler."}


def binding_bytes(row):
    pairs = row[:128].reshape(64, 2)
    return pairs[np.argsort(pairs[:, 0])].astype("<u2").tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--position-seed", type=int, default=20261014)
    parser.add_argument("--kv-seed", type=int, default=20261015)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.run_dir / "best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    assert config == json.loads((args.run_dir / "config.json").read_text())
    assert config["task"]["input_seq_len"] == 512 and config["task"]["num_kv_pairs"] == 64
    assert config["task"]["vocab_size"] == 8192
    assert not config["task"]["random_non_queries"] and config["task"]["labels_already_aligned"]
    assert sha256(args.data_dir / "manifest.json") == state["manifest_hash"]
    root = Path(__file__).resolve().parents[2]
    for name, digest in state["source_sha256"].items():
        assert sha256(root / name) == digest, name
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    print("Checkpoint, training configuration and source hashes verified", flush=True)
    positions, distances, sampling = random_positions(args.position_seed)
    print(json.dumps(sampling), flush=True)
    rng = np.random.RandomState(args.kv_seed)
    keys = np.array([rng.choice(np.arange(1, 4096), 64, replace=False) for _ in range(100)])
    values = np.array([rng.choice(np.arange(4096, 8192), 64, replace=False) for _ in range(100)])
    inputs = np.zeros((100, 512), dtype=np.uint16)
    inputs[:, :128:2], inputs[:, 1:128:2] = keys, values
    np.put_along_axis(inputs, positions, keys, axis=1)
    labels = np.full((100, 512), -100, dtype=np.int64)
    np.put_along_axis(labels, positions, values, axis=1)
    assert np.all(np.take_along_axis(inputs, positions + 1, axis=1) == 0)
    assert np.all(np.sum(labels != -100, axis=1) == 64)
    assert np.all([np.array_equal(np.array([np.flatnonzero(row == key)[0] for key in row[:128:2]]),
                                 np.arange(0, 128, 2)) for row in inputs])
    full_hashes = {hashlib.sha256(row.astype("<u2").tobytes()).digest() for row in inputs}
    binding_hashes = {hashlib.sha256(binding_bytes(row)).digest() for row in inputs}
    assert len(full_hashes) == len(binding_hashes) == 100
    independence = {}
    training_far = None
    for split, record in manifest["splits"].items():
        for file in record["files"].values():
            assert sha256(args.data_dir / file["path"]) == file["sha256"]
        original = np.load(args.data_dir / record["files"]["inputs"]["path"], mmap_mode="r")
        full_overlap = binding_overlap = 0
        for row in original:
            full_overlap += hashlib.sha256(row.astype("<u2").tobytes()).digest() in full_hashes
            binding_overlap += hashlib.sha256(binding_bytes(row)).digest() in binding_hashes
        independence[split] = {"samples_checked": len(original), "full_input_overlap": full_overlap,
                               "full_kv_binding_overlap_order_independent": binding_overlap}
        assert full_overlap == binding_overlap == 0
        if split == "train":
            original_distances = np.load(args.data_dir / record["files"]["distances"]["path"], mmap_mode="r")
            far = int(((original_distances >= 256) & (original_distances < 512)).sum())
            training_far = {"queries": int(original_distances.size), "far_queries": far,
                            "fraction": far / original_distances.size}
    print(json.dumps({"independence": independence, "training_far": training_far}), flush=True)
    np.savez(args.out_dir / "samples.npz", inputs=inputs, positions=positions.astype(np.uint16),
             targets=values.astype(np.uint16), distances=distances.astype(np.uint16), labels=labels)
    device = torch.device("cuda")
    model = build_model(config, device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    computation = build_computation(model, config, compiled=False)
    with torch.inference_mode():
        logits, loss = computation(torch.as_tensor(inputs.astype(np.int64), device=device),
            torch.as_tensor(positions, device=device), torch.as_tensor(values, device=device))
        predictions = logits.argmax(-1).cpu().numpy()
        loss_value = loss.item()
    correct = predictions == values
    np.save(args.out_dir / "predictions.npy", predictions.astype(np.uint16))
    bands = {}
    for lower in (256, 320, 384, 448):
        mask = (distances >= lower) & (distances < lower + 64)
        count = int(mask.sum())
        hits = int(correct[mask].sum())
        assert count == 1600
        bands[f"[{lower},{lower + 64})"] = {"correct": hits, "queries": count, "accuracy": hits / count}
    result = {"status": "complete", "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": state["step"], "checkpoint_epoch": state["epoch"], "frozen_model": True,
        "optimizer_updates": 0, "sequence_length": 512, "model_d_model": config["model"]["d_model"],
        "samples": 100, "kv_pairs_per_sample": 64, "queries_per_sample": 64,
        "correct": int(correct.sum()), "queries": 6400, "query_accuracy": float(correct.mean()),
        "all_correct_samples": int(correct.all(axis=1).sum()), "all_queries_correct_sample_fraction": float(correct.all(axis=1).mean()),
        "distance_bands": bands, "query_loss": loss_value, "sampling": sampling,
        "distance_histogram": {str(d): int((distances == d).sum()) for d in range(256, 512, 2)},
        "independence": independence, "training_far": training_far, "new_far_fraction": 1.0,
        "distance_definition": "query position minus corresponding key first occurrence position; zero-based",
        "label_alignment": "value target at query position; no shift; answer slot stays zero",
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "precision": "fp32", "tf32": False, "attention": config["attention"], "model_eval": True},
        "evaluation_script_sha256": sha256(__file__), "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Within-512-token far-distance distribution transfer with jointly randomized position and distance allocation."}
    atomic_json(args.out_dir / "results.json", result)
    with (args.out_dir / "query_predictions.jsonl").open("w") as stream:
        for s in range(100):
            stream.write(json.dumps({"sample": s, "source_positions": list(range(0, 128, 2)),
                "query_positions": positions[s].tolist(), "distances": distances[s].tolist(),
                "targets": values[s].tolist(), "predictions": predictions[s].tolist(),
                "correct_queries": int(correct[s].sum())}) + "\n")
    assert sha256(checkpoint_path) == checkpoint_hash
    print(json.dumps({k: result[k] for k in ("query_accuracy", "correct", "queries", "all_correct_samples",
        "all_queries_correct_sample_fraction", "distance_bands", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
