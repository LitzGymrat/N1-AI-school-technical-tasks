"""Run the exact dense held-out evaluations on the MoE best checkpoint."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mqar.data import atomic_json, sha256
from .model import build_computation, build_model


def evaluate(model, config, sample_path, prediction_path, lower, width):
    data = np.load(sample_path)
    inputs, positions, targets, distances = (data[name] for name in ("inputs", "positions", "targets", "distances"))
    assert inputs.shape[0] == 100 and positions.shape == targets.shape == (100, 64)
    assert np.array_equal(distances.astype(int), positions.astype(int) - np.arange(0, 128, 2))
    computation = build_computation(model, config, compiled=False)
    model.eval()
    with torch.inference_mode():
        values = tuple(torch.as_tensor(array.astype(np.int64), device="cuda") for array in (inputs, positions, targets))
        logits, ce, balance, loads = computation(*values)
        predictions = logits.argmax(-1).cpu().numpy()
    correct = predictions == targets
    bands = {}
    for lo in range(lower, lower + 4 * width, width):
        mask = (distances >= lo) & (distances < lo + width)
        count, hits = int(mask.sum()), int(correct[mask].sum())
        assert count == 1600
        bands[f"[{lo},{lo + width})"] = {"queries": count, "correct": hits, "accuracy": hits / count}
    np.save(prediction_path, predictions.astype(np.uint16))
    return {"samples": 100, "queries": 6400, "correct": int(correct.sum()), "query_accuracy": float(correct.mean()),
            "all_correct_samples": int(correct.all(axis=1).sum()), "all_queries_correct_sample_fraction": float(correct.all(axis=1).mean()),
            "distance_bands": bands, "ce_loss": ce.item(), "balance_loss": balance.item(), "expert_loads": loads.tolist(),
            "samples_sha256": sha256(sample_path), "predictions_sha256": sha256(prediction_path)}, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dense-far-dir", type=Path, required=True)
    parser.add_argument("--dense-transfer-dir", type=Path, required=True)
    args = parser.parse_args()
    final = json.loads((args.run_dir / "final_results.json").read_text())
    assert final["status"] == "complete"
    state = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    root = Path(__file__).resolve().parents[2]
    for name, digest in state["source_sha256"].items():
        assert sha256(root / name) == digest, name
    dense_far = json.loads((args.dense_far_dir / "results.json").read_text())
    dense_transfer = json.loads((args.dense_transfer_dir / "results.json").read_text())
    assert sha256(args.dense_far_dir / "samples.npz") == dense_transfer["base_samples_sha256"]
    out = args.run_dir / "controlled_evaluation"
    out.mkdir(exist_ok=False)
    model = build_model(config, torch.device("cuda"))
    model.load_state_dict(state["model"], strict=True)
    digest = sha256(args.run_dir / "best.pt")
    far512, _ = evaluate(model, config, args.dense_far_dir / "samples.npz", out / "far_512_predictions.npy", 256, 64)
    runtime = copy.deepcopy(config)
    runtime["model"]["context_len"] = 1024
    del model
    model = build_model(runtime, torch.device("cuda"))
    model.load_state_dict(state["model"], strict=True)
    near1024, near_correct = evaluate(model, runtime, args.dense_transfer_dir / "near_samples.npz", out / "near_1024_predictions.npy", 256, 64)
    far1024, far_correct = evaluate(model, runtime, args.dense_transfer_dir / "far_samples.npz", out / "far_1024_predictions.npy", 512, 128)
    assert sha256(args.run_dir / "best.pt") == digest
    comparisons = {}
    for name, dense, moe in (("far_512", dense_far, far512), ("near_1024", dense_transfer["near_1024"], near1024),
                             ("far_1024", dense_transfer["far_1024"], far1024)):
        comparisons[name] = {"dense": {key: dense[key] for key in ("query_accuracy", "correct", "all_correct_samples", "distance_bands")},
            "moe": moe, "moe_minus_dense_percentage_points": 100 * (moe["query_accuracy"] - dense["query_accuracy"])}
    result = {"status": "complete", "training": final, "checkpoint_step": state["step"], "best_sha256": digest,
        "moe": config["moe"], "comparisons": comparisons, "same_exact_dense_samples": True,
        "task1_independence": dense_far["independence"], "task2_independence": dense_transfer["independence"],
        "controlled": "same data, seed, full batch, optimizer, LR schedule, regularization, precision and stopping rule; only FFN topology/router/auxiliary loss differ",
        "limitation": dense_transfer["limitation"],
        "near_far_paired_outcomes": {"both_correct": int((near_correct & far_correct).sum()),
            "near_only_correct": int((near_correct & ~far_correct).sum()), "far_only_correct": int((~near_correct & far_correct).sum()),
            "both_incorrect": int((~near_correct & ~far_correct).sum())}}
    atomic_json(out / "comparison.json", result)
    lines = ["MQAR Dense / 4-expert Top-2 MoE 受控对比", "", "每组100条、每条64 KV/query，共6400次召回；使用与dense完全相同的已验重评测样本。",
             "每专家宽度672，Top-2激活宽度1344；均衡损失系数0.01，其余训练设置沿用dense。",
             f"训练状态：{final['status']}；步数{final['step']}；轮数{final['epochs_completed']}；验证准确率{final['best_valid_accuracy']*100:.4f}%；独立原test准确率{final['test']['query_accuracy']*100:.4f}%。", ""]
    for name, entry in comparisons.items():
        dense, moe = entry["dense"], entry["moe"]
        lines += [name, f"Dense query准确率：{dense['query_accuracy']*100:.4f}%；整条全对{dense['all_correct_samples']}/100。",
                  f"MoE query准确率：{moe['query_accuracy']*100:.4f}%（{moe['correct']}/6400）；整条全对{moe['all_correct_samples']}/100。",
                  f"MoE - Dense：{entry['moe_minus_dense_percentage_points']:+.4f}个百分点。"]
        for band, values in moe["distance_bands"].items():
            lines.append(f"{band}：Dense {dense['distance_bands'][band]['accuracy']*100:.4f}%，MoE {values['accuracy']*100:.4f}%（{values['correct']}/{values['queries']}）。")
        lines.append("")
    lines += ["1024评测仅按原公式扩展RoPE缓存，theta不变。", "边界：远距query的绝对位置也后移，本评测不能分离相对距离与绝对位置外推的影响。", f"best.pt SHA256: {digest}"]
    (out / "comparison.txt").write_text("\n".join(lines) + "\n")
    import wandb
    info = json.loads((args.run_dir / "wandb_run.json").read_text())
    run = wandb.Api().run(f"{info['entity']}/{info['project']}/{info['id']}")
    for name, entry in comparisons.items():
        for key in ("query_accuracy", "all_queries_correct_sample_fraction"):
            run.summary[f"controlled/{name}/{key}"] = entry["moe"][key]
    run.summary["controlled/status"] = "complete"
    run.summary.update()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
