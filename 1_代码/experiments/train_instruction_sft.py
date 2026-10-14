"""完整 batch 的通用指令 SFT；保留原模型，只监督回答和 EOS。"""

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from optimizer.AdamW import AdamW
from training_utils.Crossentropyloss import cross_entropy
from training_utils.Gradient_clipping import gradient_clipping
from training_utils.Learning_rate_schedule import get_lr_cosine_schedule
from experiments.model import build_model
from experiments.train_owt import ROOT, TOKENIZER_DIR, save_checkpoint, synchronize


def read_data(path, expected_hash, max_length):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    rows = [json.loads(line) for line in path.open()]
    for row in rows:
        assert len(row["sft_input_ids"]) == len(row["sft_labels"]) < max_length
        assert row["sft_labels"][-1] == 256  # 现有 tokenizer 的 EOS。
        assert any(label != -100 for label in row["sft_labels"])
    return rows


def make_batch(rows, device):
    length = max(len(row["sft_input_ids"]) for row in rows)
    x = torch.full((len(rows), length), 256, dtype=torch.long)
    y = torch.full_like(x, -100)
    for index, row in enumerate(rows):
        size = len(row["sft_input_ids"])
        x[index, :size] = torch.tensor(row["sft_input_ids"])
        y[index, :size] = torch.tensor(row["sft_labels"])
    targets = int((y != -100).sum())
    useful = sum(len(row["sft_input_ids"]) for row in rows)
    return x.to(device), y.to(device), targets, useful


def compute_loss(model, x, y, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x)
    valid = y != -100
    # 先取回答位置，再转 FP32。标签已错位一次，这里不再 shift。
    return cross_entropy(logits[valid].float(), y[valid])


@torch.no_grad()
def evaluate(model, rows, batch_size, device):
    model.eval()
    synchronize(device)
    start = time.perf_counter()
    weighted_loss = torch.zeros((), device=device)
    targets = 0
    for index in range(0, len(rows), batch_size):
        x, y, count, _ = make_batch(rows[index:index + batch_size], device)
        weighted_loss += compute_loss(model, x, y, device) * count
        targets += count
    synchronize(device)
    result = {"valid_loss": weighted_loss.item() / targets,
              "validation_seconds": time.perf_counter() - start,
              "validation_samples": len(rows), "validation_supervised_tokens": targets}
    model.train()
    return result


def main(args):
    wall_start = time.perf_counter()
    config = json.loads(args.config.read_text())
    training, data = config["training"], config["data"]
    assert training["grad_accum_steps"] == 1, "本入口只执行完整 batch，不做梯度累积。"
    device = torch.device(args.device)
    assert device.type == "cuda" and config["precision"] == "bf16" and config["attention"] == "flash2"
    train_rows = read_data(ROOT / data["train"], data["data_sha256"]["train"], data["max_length"])
    valid_rows = read_data(ROOT / data["validation"], data["data_sha256"]["validation"], data["max_length"])
    batch_size = training["batch_size"]
    assert math.ceil(len(train_rows) / batch_size) == training["steps_per_epoch"]
    assert training["epochs"] * training["steps_per_epoch"] == training["max_steps"]
    rng = random.Random(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    run_dir = args.out_dir or ROOT.parent / "2_实验和训练数据/runs" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("instruction-sft-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        (run_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    write_json("status.json", {"status": "running", "training_started": True})
    write_json("config.json", config)
    shutil.copy2(ROOT / data["manifest"], run_dir / "data_manifest.json")
    (run_dir / "tokenizer").mkdir()
    for name in ["vocab.json", "merges.json"]:
        shutil.copy2(TOKENIZER_DIR / name, run_dir / "tokenizer" / name)
    wandb_run = None
    try:
        initial = ROOT / config["initial_checkpoint"]
        parent_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
        assert parent_hash == config["initial_checkpoint_sha256"]
        state = torch.load(initial, map_location="cpu", weights_only=False)
        assert state["config"]["model"] == config["model"]
        model = build_model(config, device)
        model.load_state_dict(state["model_state"], strict=True)
        parent_step = state["step"]
        del state
        optimizer = AdamW(model.parameters(), lr=training["max_lr"], betas=tuple(training["betas"]),
                          weight_decay=training["weight_decay"], eps=training["eps"])
        assert not optimizer.state  # SFT 的动量从零开始。
        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(device), "command": sys.argv,
                       "parent_checkpoint": str(initial), "parent_checkpoint_sha256": parent_hash,
                       "parent_step": parent_step, "parameter_count": sum(p.numel() for p in model.parameters()),
                       "precision": "bf16 autocast, fp32 parameters and optimizer state",
                       "batch_size": batch_size, "grad_accum_steps": 1,
                       "train_samples": len(train_rows), "validation_samples": len(valid_rows)}
        write_json("environment.json", environment)
        if args.wandb_mode != "disabled":
            import wandb
            wandb_run = wandb.init(project="n1-long-range", name=run_dir.name, dir=str(run_dir),
                                   mode=args.wandb_mode, config={**config, **environment})
            if args.wandb_mode == "online":
                write_json("wandb_run.json", {"id": wandb_run.id, "url": wandb_run.url})
        print(f"run_dir={run_dir}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        step, training_seconds, save_seconds = 0, 0.0, 0.0
        valid_results, checkpoints = [], []
        with (run_dir / "metrics.jsonl").open("w") as metrics:
            def record(row):
                row["wall_seconds"] = time.perf_counter() - wall_start
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
                if wandb_run is not None:
                    wandb_run.log(row, step=row["step"])

            base = {"step": 0, "epoch": 0, **evaluate(model, valid_rows, batch_size, device)}
            assert math.isfinite(base["valid_loss"])
            record(base)
            valid_results.append(base)
            best_loss, best_checkpoint, best_epoch = base["valid_loss"], str(initial), 0
            for epoch in range(1, training["epochs"] + 1):
                ordered = train_rows.copy()
                rng.shuffle(ordered)
                synchronize(device)
                window_start = time.perf_counter()
                window_loss = torch.zeros((), device=device)
                window_targets = window_positions = window_useful = window_samples = 0
                for index in range(0, len(ordered), batch_size):
                    batch = ordered[index:index + batch_size]
                    x, y, targets, useful = make_batch(batch, device)
                    step += 1
                    lr = get_lr_cosine_schedule(step, training["max_lr"], training["min_lr"],
                                               training["warmup_steps"], training["max_steps"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss(model, x, y, device)
                    window_loss += loss.detach() * targets
                    loss.backward()
                    gradient_clipping(model.parameters(), training["max_grad_norm"])
                    optimizer.step()
                    del loss
                    window_targets += targets
                    window_positions += x.numel()
                    window_useful += useful
                    window_samples += len(batch)
                    last_batch = index + batch_size >= len(ordered)
                    if step == 1 or step % training["log_interval"] == 0 or last_batch:
                        synchronize(device)
                        elapsed = time.perf_counter() - window_start
                        training_seconds += elapsed
                        train_loss = window_loss.item() / window_targets
                        assert math.isfinite(train_loss), f"Non-finite training loss at step {step}"
                        record({"step": step, "epoch": epoch, "train_loss": train_loss, "lr": lr,
                                "window_samples": window_samples, "window_supervised_tokens": window_targets,
                                "window_padded_positions": window_positions, "window_useful_positions": window_useful,
                                "padded_positions_per_second": window_positions / elapsed,
                                "useful_positions_per_second": window_useful / elapsed,
                                "training_seconds": training_seconds,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
                        window_loss.zero_()
                        window_targets = window_positions = window_useful = window_samples = 0
                        synchronize(device)
                        window_start = time.perf_counter()
                result = {"step": step, "epoch": epoch, **evaluate(model, valid_rows, batch_size, device)}
                assert math.isfinite(result["valid_loss"])
                record(result)
                valid_results.append(result)
                best_loss = min(best_loss, result["valid_loss"])
                checkpoint = run_dir / f"epoch{epoch}.pt"
                start = time.perf_counter()
                save_checkpoint(checkpoint, model, optimizer, config, step, best_loss, training_seconds, device)
                synchronize(device)
                save_seconds += time.perf_counter() - start
                checkpoints.append(str(checkpoint))
                if result["valid_loss"] < valid_results[best_epoch]["valid_loss"]:
                    best_epoch, best_checkpoint = epoch, str(checkpoint)
            assert step == training["max_steps"]
        summary = {"status": "complete", "step": step, "epochs": training["epochs"],
                   "best_epoch": best_epoch, "best_valid_loss": best_loss, "best_checkpoint": best_checkpoint,
                   "validation": valid_results, "checkpoints": checkpoints,
                   "training_seconds": training_seconds, "checkpoint_save_seconds": save_seconds,
                   "validation_seconds": sum(row["validation_seconds"] for row in valid_results),
                   "wall_seconds": time.perf_counter() - wall_start,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                   "training_started": True}
        write_json("summary.json", summary)
        write_json("status.json", {"status": "complete", "step": step})
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json("status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/instruction_sft_proposed.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    main(parser.parse_args())
