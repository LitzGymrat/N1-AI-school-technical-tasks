"""生成 Dense/MoE 全量 SFT 配置、实际 token 预算和已有日志估时；不训练。"""

import json
import math
from pathlib import Path
import statistics

import numpy as np

from experiments.full_instruction_data import epoch_order, file_sha256, padded_positions
from experiments.prepare_full_instruction_data import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "../2_实验和训练数据/training_data/smol_smoltalk_parent"
PARENTS = {
    "dense": ("owt-b1-full-20261013-130555", "1c02481106a30b3db490c6617cf564fcf716d50bfd09e85732208db1d283d39a"),
    "moe": ("owt-moe-full-20261013-185250", "fa410208b22a6aca870fe0a7efc5d185799f44ba406023626eeeade0848b1610"),
}


def lengths_for(manifest, split):
    return np.concatenate([np.load(DATA / path / "windows.npy")[:, 1].astype(np.int64)
                           for path in manifest["splits"][split]["shards"]])


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    evidence = json.loads((ROOT / "experiments/outputs/setup/full-sft-parent-evidence-20261014.json").read_text())
    if evidence["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("Remote tokenizer differs from prepared-data tokenizer")
    batch_size, epochs, seed = 16, 2, 42
    lengths = lengths_for(manifest, "train")
    valid_lengths = lengths_for(manifest, "validation")
    steps_per_epoch = math.ceil(len(lengths) / batch_size)
    epoch_budgets = [{"epoch": epoch, "updates": steps_per_epoch,
        "padded_input_positions": padded_positions(lengths, epoch_order(len(lengths), seed, epoch), batch_size),
        "useful_input_positions": int(lengths.sum()),
        "supervised_tokens": manifest["splits"]["train"]["supervised_tokens"]} for epoch in range(1, epochs + 1)]
    positions = sum(row["padded_input_positions"] for row in epoch_budgets)
    valid_positions = padded_positions(valid_lengths, np.arange(len(valid_lengths)), batch_size) * epochs
    dense_measured = [json.loads(line) for line in (ROOT / "experiments/outputs/microbatch-20261013/micro-16-accum-1/metrics.jsonl").read_text().splitlines()]
    dense_batch16 = [row for row in dense_measured if "tokens_per_second" in row][-1]
    dense_b1_rate = statistics.median(row["tokens_per_second"] for row in evidence["runs"][PARENTS["dense"][0]]["recent_metrics"])
    moe_b1_rate = statistics.median(row["tokens_per_second"] for row in evidence["runs"][PARENTS["moe"][0]]["recent_metrics"])
    slowdown = dense_b1_rate / moe_b1_rate
    dense_b1_peak = max(row["peak_allocated_gib"] for row in evidence["runs"][PARENTS["dense"][0]]["recent_metrics"])
    moe_b1_peak = max(row["peak_allocated_gib"] for row in evidence["runs"][PARENTS["moe"][0]]["recent_metrics"])
    extra_fixed_gib = 16 * (63851008 - 51452416) / 1024**3
    rough_moe_peak = dense_batch16["peak_allocated_gib"] + extra_fixed_gib + batch_size * (moe_b1_peak - dense_b1_peak - extra_fixed_gib)
    estimates = {}
    for kind, (run, parent_hash) in PARENTS.items():
        parent = evidence["runs"][run]
        if parent["last_step"] != 160000:
            raise ValueError(f"Parent training incomplete: {run}")
        ratio = 1.0 if kind == "dense" else slowdown
        estimated_rate = dense_batch16["tokens_per_second"] / ratio
        token_hours = positions / estimated_rate / 3600
        step_hours = steps_per_epoch * epochs * dense_batch16["seconds_per_step"] * ratio / 3600
        estimates[kind] = {"dense_full_batch_measurement": "outputs/microbatch-20261013/micro-16-accum-1/metrics.jsonl",
            "measured_dense_batch_size": 16, "measured_dense_context_len": 2048,
            "measured_dense_positions_per_second": dense_batch16["tokens_per_second"],
            "measured_dense_seconds_per_step": dense_batch16["seconds_per_step"],
            "historical_same_batch1_moe_slowdown_ratio": slowdown,
            "estimated_batch16_positions_per_second": estimated_rate,
            "pure_training_hours_by_positions": token_hours, "pure_training_hours_by_fixed_step": step_hours,
            "planning_wall_hours_low": (positions + valid_positions) / estimated_rate * 1.10 / 3600,
            "planning_wall_hours_high": (step_hours + valid_positions / estimated_rate / 3600) * 1.15,
            "boundary": "Dense uses existing full-batch16 measurement. MoE batch16 has no prior measurement: extrapolate the observed same-batch1 Dense/MoE slowdown ratio, which may change with batch size. Token and fixed-step estimates are planning scenarios, not bounds. Validation allowance and 10-15% margin are estimates."}
        model = json.loads((ROOT / f"experiments/outputs/{run}/config.json").read_text())["model"]
        config = {"status": "prepared_training_not_started", "model_type": kind,
            "initial_checkpoint": f"experiments/outputs/{run}/checkpoint.pt",
            "initial_checkpoint_sha256": parent_hash, "parent_pretraining_step": 160000,
            "model": model, "attention": "flash2", "precision": "bf16", "seed": seed,
            "data": {"directory": str(Path("..") / DATA.relative_to(ROOT.parent)), "format": manifest["format"],
                "manifest_sha256": file_sha256(DATA / "manifest.json"), "max_length": 2048, "overlap": 0,
                "batching": "Same deterministic np.random.default_rng(seed+epoch) permutation for both models; full batch16 right padding; no packing; keep final partial batch.",
                "loss": manifest["loss"], "selection": manifest["selection"], "windowing": manifest["windowing"]},
            "training": {"batch_size": batch_size, "grad_accum_steps": 1, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
                "max_steps": steps_per_epoch * epochs, "max_lr": 1e-4, "min_lr": 1e-5,
                "warmup_steps": math.ceil(steps_per_epoch * epochs * .1), "betas": [.9, .95],
                "eps": 1e-8, "weight_decay": .1, "max_grad_norm": 1.0, "log_interval": 100,
                "checkpoint_interval": 2000, "eval_interval": steps_per_epoch,
                "balance_loss_coef": .01 if kind == "moe" else 0.0,
                "loss_reduction": "Mean CE over all supervised tokens in the full batch; MoE adds balance coefficient times the full-batch auxiliary loss; one forward/backward/update per batch.",
                "checkpoint_selection": "Lowest assistant-token-averaged CE on full original test split after each epoch; no RULER test or base reevaluation.",
                "optimizer_state": "Load original pretraining weights only; fresh FP32 AdamW state and SFT schedule."},
            "reference": "https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config_smol.yaml",
            "recipe_boundary": "2 epochs, cosine and 10% warmup follow the author recipe. LR=1e-4, assistant-only loss, existing tokenizer/plain-text roles, complete conversations <2048 and full batch16 without accumulation are our experiment choices.",
            "budget": str(Path("..") / (DATA / "training_budget.json").relative_to(ROOT.parent))}
        if kind == "moe":
            config["moe_backend"] = "grouped"
        write_json(ROOT / f"experiments/configs/instruction_sft_full_{kind}.json", config)
    budget = {"training_started": False, "train_original_conversations": manifest["splits"]["train"]["original_conversations"],
        "train_examples_after_length_filter": len(lengths), "validation_original_conversations": manifest["splits"]["validation"]["original_conversations"],
        "validation_examples_after_length_filter": len(valid_lengths), "epochs": epochs, "effective_batch_size": batch_size,
        "batch_size": batch_size, "grad_accum_steps": 1, "epoch_budgets": epoch_budgets,
        "total_padded_training_positions_per_model": positions, "total_validation_positions_per_model": valid_positions,
        "estimates": estimates,
        "memory": {"dense_parameter_count": 51452416, "moe_parameter_count": 63851008,
            "dense_fp32_weights_gradients_adam_bytes": 16 * 51452416,
            "moe_fp32_weights_gradients_adam_bytes": 16 * 63851008,
            "max_actual_input_length": int(lengths.max()),
            "full_batch_bf16_logits_bytes": 16 * int(lengths.max()) * 32000 * 2,
            "existing_dense_batch16_context2048_peak_allocated_gib": dense_batch16["peak_allocated_gib"],
            "rough_moe_batch16_peak_extrapolated_gib": rough_moe_peak,
            "rough_moe_formula": "Dense measured batch16 peak + extra weights/grad/Adam bytes +16*(observed MoE-Dense batch1 peak difference minus extra weights/grad/Adam bytes).",
            "boundary": "Static tensor bytes and existing Dense full-batch measurement. MoE peak assumes batch-dependent memory scales linearly and is only a rough planning estimate; no new MoE batch16 runtime-memory measurement."},
        "sequential_planning_wall_hours_low": sum(row["planning_wall_hours_low"] for row in estimates.values()),
        "sequential_planning_wall_hours_high": sum(row["planning_wall_hours_high"] for row in estimates.values())}
    write_json(DATA / "training_budget.json", budget)
    print(json.dumps(budget, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
