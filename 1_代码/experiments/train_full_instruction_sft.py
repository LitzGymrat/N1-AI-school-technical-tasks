"""Dense/MoE 共用完整 batch SFT 入口；固定数据顺序，支持断点恢复。"""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from optimizer.AdamW import AdamW
from training_utils.Crossentropyloss import cross_entropy
from training_utils.Gradient_clipping import gradient_clipping
from training_utils.Learning_rate_schedule import get_lr_cosine_schedule
from experiments.full_instruction_data import FullInstructionData, epoch_order, file_sha256
from experiments.train_instruction_sft import make_batch
from experiments.train_owt import ROOT, TOKENIZER_DIR, synchronize


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def forward_loss(model, x, y, device, model_type):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(x)
    if model_type == "moe":
        logits, balance, loads = output
    else:
        logits, balance, loads = output, None, None
    valid = y != -100
    ce = cross_entropy(logits[valid].float(), y[valid])
    return ce, balance, loads


@torch.no_grad()
def evaluate(model, data, batch_size, device, model_type):
    model.eval()
    synchronize(device)
    start = time.perf_counter()
    weighted_loss = torch.zeros((), device=device)
    for begin in range(0, len(data), batch_size):
        rows = [data[index] for index in range(begin, min(begin + batch_size, len(data)))]
        x, y, targets, _ = make_batch(rows, device)
        ce, _, _ = forward_loss(model, x, y, device, model_type)
        weighted_loss += ce * targets
    synchronize(device)
    result = {"valid_loss": weighted_loss.item() / int(data.target_counts.sum()),
              "validation_seconds": time.perf_counter() - start,
              "validation_examples": len(data), "validation_supervised_tokens": int(data.target_counts.sum())}
    model.train()
    return result


def save(path, model, optimizer, config, step, best_loss, training_seconds):
    state = {"config": config, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "step": step, "best_valid_loss": best_loss, "training_seconds": training_seconds,
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(args):
    wall_start = time.perf_counter()
    config = json.loads(args.config.read_text())
    training, data_config = config["training"], config["data"]
    device = torch.device(args.device)
    if device.type != "cuda" or config["precision"] != "bf16" or config["attention"] != "flash2":
        raise ValueError("This run requires CUDA, BF16 and FA2")
    if device.index is not None:
        torch.cuda.set_device(device)
    batch_size = training["batch_size"]
    if training["grad_accum_steps"] != 1 or "micro_batch_size" in training:
        raise ValueError("This entry uses complete batches without splitting or accumulation")
    directory = ROOT / data_config["directory"]
    train_data = FullInstructionData(directory, "train", data_config["manifest_sha256"])
    valid_data = FullInstructionData(directory, "validation", data_config["manifest_sha256"])
    if math.ceil(len(train_data) / batch_size) != training["steps_per_epoch"]:
        raise ValueError("Training step budget differs from prepared data")
    for name, expected in train_data.manifest["tokenizer_sha256"].items():
        if file_sha256(TOKENIZER_DIR / name) != expected:
            raise ValueError(f"Tokenizer changed: {name}")
    initial = ROOT / config["initial_checkpoint"]
    if file_sha256(initial) != config["initial_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["model_type"] == "moe":
        from experiments.model_moe import build_model
        model = build_model(config, device, config["moe_backend"])
    else:
        from experiments.model import build_model
        model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=training["max_lr"], betas=tuple(training["betas"]),
                      eps=training["eps"], weight_decay=training["weight_decay"])
    run_dir = args.out_dir
    state = torch.load(args.resume or initial, map_location="cpu", weights_only=False)
    if state["config"]["model"] != config["model"]:
        raise ValueError("Checkpoint model differs from config")
    model.load_state_dict(state["model_state"], strict=True)
    step, best_loss, training_seconds = 0, math.inf, 0.0
    if args.resume:
        if state["config"] != config or not run_dir.is_dir():
            raise ValueError("Resume requires the original config and output directory")
        optimizer.load_state_dict(state["optimizer_state"])
        step, best_loss = state["step"], state["best_valid_loss"]
        training_seconds = state["training_seconds"]
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "config.json", config)
        shutil.copy2(directory / "manifest.json", run_dir / "data_manifest.json")
        (run_dir / "tokenizer").mkdir()
        for name in ["vocab.json", "merges.json"]:
            shutil.copy2(TOKENIZER_DIR / name, run_dir / "tokenizer" / name)
        write_json(run_dir / "environment.json", {"gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "parent_step": state["step"], "parent_sha256": config["initial_checkpoint_sha256"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_type": config["model_type"], "moe_backend": config.get("moe_backend"),
            "train_examples": len(train_data), "validation_examples": len(valid_data),
            "batch_size": batch_size, "grad_accum_steps": 1})
    del state
    wandb_run = None
    try:
        if args.wandb_mode != "disabled":
            import wandb
            previous = run_dir / "wandb_run.json"
            resume_options = {"id": json.loads(previous.read_text())["id"], "resume": "must"} if args.resume and previous.exists() else {}
            wandb_run = wandb.init(project="n1-long-range", name=run_dir.name, dir=str(run_dir),
                                  mode=args.wandb_mode, config=config, **resume_options)
            if args.wandb_mode == "online":
                write_json(previous, {"id": wandb_run.id, "url": wandb_run.url})
        write_json(run_dir / "status.json", {"status": "running", "step": step, "training_started": True})
        torch.cuda.reset_peak_memory_stats(device)
        validations = []
        with (run_dir / "metrics.jsonl").open("a" if args.resume else "w") as metrics:
            def record(row):
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
                if wandb_run:
                    wandb_run.log(row, step=row["step"])

            for epoch in range(step // training["steps_per_epoch"] + 1, training["epochs"] + 1):
                order = epoch_order(len(train_data), config["seed"], epoch)
                first_update = step % training["steps_per_epoch"]
                synchronize(device)
                window_start = time.perf_counter()
                window_ce = torch.zeros((), device=device)
                window_balance = torch.zeros((), device=device)
                window_targets = window_padded = window_useful = window_samples = window_updates = 0
                for update in range(first_update, training["steps_per_epoch"]):
                    batch_indexes = order[update * batch_size:(update + 1) * batch_size]
                    step += 1
                    lr = get_lr_cosine_schedule(step, training["max_lr"], training["min_lr"],
                                                training["warmup_steps"], training["max_steps"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    rows = [train_data[int(index)] for index in batch_indexes]
                    x, y, targets, useful = make_batch(rows, device)
                    ce, balance, loads = forward_loss(model, x, y, device, config["model_type"])
                    loss = ce if balance is None else ce + training["balance_loss_coef"] * balance
                    loss.backward()
                    window_ce += ce.detach() * targets
                    if balance is not None:
                        window_balance += balance.detach() * targets
                    window_targets += targets
                    window_padded += x.numel()
                    window_useful += useful
                    window_samples += len(rows)
                    del ce, loss, balance, loads, x, y
                    gradient_clipping(model.parameters(), training["max_grad_norm"])
                    optimizer.step()
                    window_updates += 1
                    end_epoch = update + 1 == training["steps_per_epoch"]
                    if step == 1 or step % training["log_interval"] == 0 or end_epoch:
                        synchronize(device)
                        elapsed = time.perf_counter() - window_start
                        training_seconds += elapsed
                        ce_value = window_ce.item() / window_targets
                        if not math.isfinite(ce_value):
                            raise ValueError(f"Non-finite CE at step {step}")
                        row = {"step": step, "epoch": epoch, "train_loss": ce_value, "lr": lr,
                            "window_samples": window_samples, "window_supervised_tokens": window_targets,
                            "window_padded_positions": window_padded, "window_useful_positions": window_useful,
                            "padded_positions_per_second": window_padded / elapsed,
                            "seconds_per_step": elapsed / window_updates, "training_seconds": training_seconds,
                            "estimated_remaining_train_seconds": (training["max_steps"] - step) * elapsed / window_updates,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
                        if config["model_type"] == "moe":
                            row["balance_loss"] = window_balance.item() / window_targets
                        record(row)
                        window_ce.zero_()
                        window_balance.zero_()
                        window_targets = window_padded = window_useful = window_samples = window_updates = 0
                    if end_epoch:
                        result = {"step": step, "epoch": epoch,
                            **evaluate(model, valid_data, batch_size, device, config["model_type"])}
                        if not math.isfinite(result["valid_loss"]):
                            raise ValueError("Non-finite validation CE")
                        validations.append(result)
                        record(result)
                        improved = result["valid_loss"] < best_loss
                        best_loss = min(best_loss, result["valid_loss"])
                        save(run_dir / f"epoch{epoch}.pt", model, optimizer, config, step, best_loss, training_seconds)
                        if improved:
                            shutil.copy2(run_dir / f"epoch{epoch}.pt", run_dir / "best.pt")
                    if step % training["checkpoint_interval"] == 0 or end_epoch:
                        save(run_dir / "last.pt", model, optimizer, config, step, best_loss, training_seconds)
                        write_json(run_dir / "status.json", {"status": "running", "step": step, "epoch": epoch})
                    if window_updates == 0:
                        synchronize(device)
                        window_start = time.perf_counter()
        if step != training["max_steps"]:
            raise ValueError("Final step differs from planned budget")
        write_json(run_dir / "summary.json", {"status": "complete", "step": step, "validation_this_session": validations,
            "best_valid_loss": best_loss, "best_checkpoint": str(run_dir / "best.pt"),
            "training_seconds": training_seconds, "wall_seconds_this_session": time.perf_counter() - wall_start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
        write_json(run_dir / "status.json", {"status": "complete", "step": step})
    except BaseException as error:
        write_json(run_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                             "step": step, "error": str(error)})
        raise
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="online")
    main(parser.parse_args())
