"""Paired 1024-token near/far evaluation of the frozen 512-token checkpoint."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mqar.data import atomic_json, sha256
from .evaluate_far import binding_bytes
from .model import build_computation, build_model


def metrics(predictions, targets, distances, lower, width):
    correct = predictions == targets
    bands = {}
    for lo in range(lower, lower + 4 * width, width):
        mask = (distances >= lo) & (distances < lo + width)
        hits, count = int(correct[mask].sum()), int(mask.sum())
        assert count == 1600
        bands[f"[{lo},{lo + width})"] = {"correct": hits, "queries": count, "accuracy": hits / count}
    return {"samples": 100, "correct": int(correct.sum()), "queries": 6400,
            "query_accuracy": float(correct.mean()), "all_correct_samples": int(correct.all(axis=1).sum()),
            "all_queries_correct_sample_fraction": float(correct.all(axis=1).mean()), "distance_bands": bands}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.base_eval_dir / "results.json").read_text())
    checkpoint = args.run_dir / "best.pt"
    digest = sha256(checkpoint)
    assert digest == previous["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    assert config == json.loads((args.run_dir / "config.json").read_text())
    assert sha256(args.data_dir / "manifest.json") == state["manifest_hash"]
    root = Path(__file__).resolve().parents[2]
    for name, expected in state["source_sha256"].items():
        assert sha256(root / name) == expected, name
    base = np.load(args.base_eval_dir / "samples.npz")
    source = np.arange(0, 128, 2)
    near_positions = base["positions"].astype(np.int64)
    near_distances = near_positions - source
    targets = base["targets"].astype(np.int64)
    assert np.array_equal(near_distances, base["distances"])
    near_inputs = np.pad(base["inputs"], ((0, 0), (0, 512)))
    rng = np.random.RandomState(args.seed)
    offsets = np.full((100, 64), 256, dtype=np.int64)
    for distance in range(256, 512, 2):
        where = np.flatnonzero(near_distances.ravel() == distance)
        assert len(where) == 50
        offsets.ravel()[rng.permutation(where)[:25]] = 512
    far_positions = near_positions + offsets
    far_distances = far_positions - source
    far_inputs = np.zeros((100, 1024), dtype=np.uint16)
    far_inputs[:, :128] = base["inputs"][:, :128]
    np.put_along_axis(far_inputs, far_positions, base["inputs"][:, :128:2], axis=1)
    assert np.array_equal(np.unique(far_distances), np.arange(512, 1024, 2))
    assert np.all(np.unique(far_distances, return_counts=True)[1] == 25)
    assert np.array_equal(near_inputs[:, :128], far_inputs[:, :128])
    assert np.all(far_positions < 1024)
    assert len({row.tobytes() for row in far_positions}) == 100
    for inputs, positions in ((near_inputs, near_positions), (far_inputs, far_positions)):
        assert all(len(set(row)) == 64 for row in positions)
        assert np.array_equal(np.take_along_axis(inputs, positions, axis=1), inputs[:, :128:2])
        assert np.array_equal(targets, inputs[:, 1:128:2])
        assert np.all(np.take_along_axis(inputs, positions + 1, axis=1) == 0)
        assert np.all(np.count_nonzero(inputs, axis=1) == 192)
    print("Paired data verified: same KV, 64 queries, zero filling, exact distance histograms", flush=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    input_hashes = {group: {hashlib.sha256(row.astype("<u2").tobytes()).digest() for row in inputs}
                    for group, inputs in (("near", near_inputs), ("far", far_inputs))}
    binding_hashes = {hashlib.sha256(binding_bytes(row)).digest() for row in near_inputs}
    independence = {}
    for split, record in manifest["splits"].items():
        for file in record["files"].values():
            assert sha256(args.data_dir / file["path"]) == file["sha256"]
        original = np.load(args.data_dir / record["files"]["inputs"]["path"], mmap_mode="r")
        counts = {"near_full_input_overlap": 0, "far_full_input_overlap": 0, "full_kv_binding_overlap_order_independent": 0}
        for row in original:
            normalized_hash = hashlib.sha256(row.astype("<u2").tobytes() + bytes(1024)).digest()
            for group in ("near", "far"):
                counts[f"{group}_full_input_overlap"] += normalized_hash in input_hashes[group]
            counts["full_kv_binding_overlap_order_independent"] += hashlib.sha256(binding_bytes(row)).digest() in binding_hashes
        assert not any(counts.values())
        independence[split] = {"samples_checked": len(original), **counts}
    print(json.dumps({"independence": independence}), flush=True)
    for group, inputs, positions, distances in (("near", near_inputs, near_positions, near_distances),
                                               ("far", far_inputs, far_positions, far_distances)):
        labels = np.full((100, 1024), -100, dtype=np.int64)
        np.put_along_axis(labels, positions, targets, axis=1)
        np.savez(args.out_dir / f"{group}_samples.npz", inputs=inputs, positions=positions.astype(np.uint16),
                 targets=targets.astype(np.uint16), distances=distances.astype(np.uint16), labels=labels)
    runtime_config = copy.deepcopy(config)
    runtime_config["model"]["context_len"] = 1024
    model = build_model(runtime_config, torch.device("cuda"))
    model.load_state_dict(state["model"], strict=True)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), state["model"][name]), name
    model.eval()
    computation = build_computation(model, runtime_config, compiled=False)
    def infer(inputs, positions):
        with torch.inference_mode():
            logits, loss = computation(torch.as_tensor(inputs.astype(np.int64), device="cuda"),
                torch.as_tensor(positions, device="cuda"), torch.as_tensor(targets, device="cuda"))
            return logits, logits.argmax(-1).cpu().numpy(), loss.item()
    baseline_logits, baseline_predictions, baseline_loss = infer(base["inputs"], near_positions)
    assert np.array_equal(baseline_predictions, np.load(args.base_eval_dir / "predictions.npy"))
    near_logits, near_predictions, near_loss = infer(near_inputs, near_positions)
    padding_max_logit_error = (near_logits - baseline_logits).abs().max().item()
    padding_prediction_changes = int((near_predictions != baseline_predictions).sum())
    del baseline_logits, near_logits
    far_logits, far_predictions, far_loss = infer(far_inputs, far_positions)
    del far_logits
    near_result = {**metrics(near_predictions, targets, near_distances, 256, 64), "loss": near_loss}
    far_result = {**metrics(far_predictions, targets, far_distances, 512, 128), "loss": far_loss}
    for group, prediction in (("near", near_predictions), ("far", far_predictions)):
        np.save(args.out_dir / f"{group}_predictions.npy", prediction.astype(np.uint16))
    near_correct, far_correct = near_predictions == targets, far_predictions == targets
    paired = {"both_correct": int((near_correct & far_correct).sum()),
              "near_only_correct": int((near_correct & ~far_correct).sum()),
              "far_only_correct": int((~near_correct & far_correct).sum()),
              "both_incorrect": int((~near_correct & ~far_correct).sum())}
    result = {"status": "complete", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "checkpoint_step": state["step"], "training_context_length": 512, "evaluation_context_length": 1024,
        "model_d_model": config["model"]["d_model"], "optimizer_updates": 0,
        "controlled": {"same_weights_verified_tensor_by_tensor": True, "same_kv_per_paired_sample": True,
            "same_64_query_keys_and_targets": True, "same_prefix_kv_positions": True,
            "same_zero_count_per_1024_sample": 832, "same_label_alignment": "target at query; no shift",
            "same_precision": "fp32", "same_rope_theta": config["model"]["rope_theta"],
            "runtime_config_changes": {"model.context_len": [512, 1024]},
            "rope_extension": "original formula and theta; only non-learned position caches extended; no scaling"},
        "baseline_512": metrics(baseline_predictions, targets, near_distances, 256, 64),
        "near_1024": near_result, "far_1024": far_result, "paired_query_outcomes": paired,
        "padding_control": {"prediction_changes_vs_512": padding_prediction_changes,
                            "maximum_absolute_logit_difference": padding_max_logit_error},
        "far_minus_near_accuracy_percentage_points": (far_result["query_accuracy"] - near_result["query_accuracy"]) * 100,
        "independence": independence, "independence_normalization": "original 512-token inputs padded to 1024 before full-input comparison",
        "base_samples_sha256": sha256(args.base_eval_dir / "samples.npz"),
        "position_randomization": {"seed": args.seed, "method": "randomly select 25 of 50 occurrences per near distance for +512; other 25 get +256",
                                   "far_unique_layouts": 100, "each_even_far_distance_count": 25},
        "distance_histograms": {group: {str(d): int((distances == d).sum()) for d in np.unique(distances)}
            for group, distances in (("near", near_distances), ("far", far_distances))},
        "limitation": "With fixed prefix KV, increasing distance also moves query absolute positions beyond training. The padding control cannot separate distance extrapolation from absolute-position/RoPE extrapolation; intermediate query context also changes when query positions move.",
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "tf32": False,
                        "attention": config["attention"], "model_eval": True},
        "elapsed_seconds": time.perf_counter() - started, "evaluation_script_sha256": sha256(__file__)}
    assert sha256(checkpoint) == digest
    atomic_json(args.out_dir / "results.json", result)
    print(json.dumps({k: result[k] for k in ("near_1024", "far_1024", "padding_control", "paired_query_outcomes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
