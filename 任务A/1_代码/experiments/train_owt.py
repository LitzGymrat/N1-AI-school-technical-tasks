"""OWT 预训练：BF16、FA2、梯度累积、验证和断点续训。"""

import argparse
import json
import math
import platform
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "2_实验和训练数据/training_data/owt"
TOKENIZER_DIR = ROOT.parent / "2_实验和训练数据/training_data/tokenizer"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/owt_51m.json")
    parser.add_argument("--train-data", type=Path, default=DATA_DIR / "owt_train.npy")
    parser.add_argument("--valid-data", type=Path, default=DATA_DIR / "owt_valid.npy")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=["torch", "flash2"])
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--steps", type=int, help="覆盖配置中的总更新次数，供短跑使用")
    parser.add_argument("--warmup-steps", type=int, help="学习率预热次数，不是测速 warm-up")
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="恢复 checkpoint；训练配置必须与原运行一致")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="offline")
    return parser.parse_args()


def sample_batch(token_ids, batch_size, seq_len, device):
    starts = np.random.randint(0, len(token_ids) - seq_len, size=batch_size)
    # 一次取 seq_len+1 个 token，输入和目标恰好错开一位。
    rows = np.stack([token_ids[i : i + seq_len + 1] for i in starts]).astype(np.int64)
    batch = torch.from_numpy(rows)
    if device.type == "cuda":
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)
    return batch[:, :-1], batch[:, 1:]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_loss(model, x, y, device, precision):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(x)
    # 自己实现的交叉熵在 FP32 下计算；backward 放在 autocast 外面。
    return cross_entropy(logits.float(), y)


@torch.no_grad()
def evaluate(model, batches, device, precision):
    model.eval()
    losses = [compute_loss(model, x, y, device, precision).item() for x, y in batches]
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, config, step, best_loss, training_seconds, device):
    state = {
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    # 完整写完再替换，避免中断时破坏上一次 checkpoint。
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("attention", "precision"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    training = config["training"]
    for key in ("steps", "warmup_steps", "micro_batch_size", "grad_accum_steps"):
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)

    device = torch.device(args.device)
    if config["attention"] == "flash2" and (device.type != "cuda" or config["precision"] != "bf16"):
        raise ValueError("这次 FA2 训练入口要求 CUDA + BF16；CPU 检查请显式选择 torch + fp32。")
    if not 0 <= training["warmup_steps"] < training["steps"]:
        raise ValueError("warmup_steps 必须小于总更新次数；短跑可指定 --warmup-steps 5。")
    positive_keys = (
        "micro_batch_size", "grad_accum_steps", "log_interval", "eval_interval", "eval_batches", "checkpoint_interval"
    )
    if any(training[key] <= 0 for key in positive_keys):
        raise ValueError("batch、累积次数和记录间隔必须为正数。")

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # "cuda" 表示当前 GPU；set_device 只接受显式编号（如 cuda:0）。
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")
    seq_len = config["model"]["context_len"]
    batch_size = training["micro_batch_size"]
    accum_steps = training["grad_accum_steps"]
    tokens_per_step = batch_size * accum_steps * seq_len

    # 固定验证样本，但不消耗训练采样器的随机数序列。
    train_rng = np.random.get_state()
    np.random.seed(config["seed"] + 1)
    valid_batches = [sample_batch(valid_data, batch_size, seq_len, device) for _ in range(training["eval_batches"])]
    np.random.set_state(train_rng)

    model = build_model(config, device)
    optimizer = AdamW(
        model.parameters(), lr=training["max_lr"], betas=tuple(training["betas"]),
        weight_decay=training["weight_decay"], eps=training["eps"],
    )
    start_step = 0
    best_loss = math.inf
    training_seconds = 0.0
    if args.resume is not None:
        # 仅读取本项目自己保存的 checkpoint，其中含 NumPy 随机数状态。
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("恢复配置与 checkpoint 不一致；请使用原配置和相同的短跑覆盖参数。")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_step = state["step"]
        best_loss = state["best_valid_loss"]
        training_seconds = state["training_seconds"]
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state

    if args.resume is not None:
        run_dir = args.resume.parent
        if args.out_dir is not None and args.out_dir.resolve() != run_dir.resolve():
            raise ValueError("续训记录追加到原运行目录，不另设 out-dir。")
    else:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
        run_dir = args.out_dir or ROOT.parent / "2_实验和训练数据/runs" / f"owt-51m-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        tokenizer_out = run_dir / "tokenizer"
        tokenizer_out.mkdir()
        for name in ("vocab.json", "merges.json"):
            shutil.copy2(args.tokenizer_dir / name, tokenizer_out / name)

    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_source": str(ROOT / "transformer"), "command": sys.argv,
        "attention_backend": "pytorch_sdpa_flash2" if config["attention"] == "flash2" else "user_pytorch",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokens_per_step": tokens_per_step, "planned_tokens": tokens_per_step * training["steps"],
        "train_data": str(args.train_data), "train_tokens": len(train_data),
        "valid_data": str(args.valid_data), "valid_tokens": len(valid_data),
        "eval_tokens": batch_size * seq_len * training["eval_batches"],
    }
    environment_file = "environment.json" if args.resume is None else f"environment-resume-{start_step}.json"
    (run_dir / environment_file).write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)
    print(json.dumps(environment, indent=2), flush=True)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_options = {}
        wandb_info_path = run_dir / "wandb_run.json"
        if args.wandb_mode == "online" and args.resume is not None:
            wandb_info = json.loads(wandb_info_path.read_text(encoding="utf-8"))
            wandb_options = {key: wandb_info[key] for key in ("id", "entity")}
            wandb_options["resume"] = "must"
        wandb_run = wandb.init(project="n1-long-range", name=run_dir.name, dir=str(run_dir),
                               mode=args.wandb_mode, config={**config, **environment}, **wandb_options)
        if args.wandb_mode == "online":
            wandb_info = {"id": wandb_run.id, "entity": wandb_run.entity, "url": wandb_run.url}
            wandb_info_path.write_text(json.dumps(wandb_info, indent=2), encoding="utf-8")
            print(f"wandb_url={wandb_run.url}", flush=True)

    wall_start = time.perf_counter()
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        def record(row):
            row["tokens_seen"] = row["step"] * tokens_per_step
            row["wall_seconds_this_session"] = time.perf_counter() - wall_start
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
            if wandb_run is not None:
                wandb_run.log(row, step=row["step"])

        if args.resume is None:
            valid_loss = evaluate(model, valid_batches, device, config["precision"])
            record({"step": 0, "valid_loss": valid_loss})

        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        window_start = time.perf_counter()
        window_loss = torch.zeros((), device=device)
        window_steps = 0
        for step in range(start_step + 1, training["steps"] + 1):
            # step 是已经完成的更新次数：首次更新使用 warmup 的第 1 步学习率。
            lr = get_lr_cosine_schedule(
                step, training["max_lr"], training["min_lr"], training["warmup_steps"], training["steps"]
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            for _ in range(accum_steps):
                x, y = sample_batch(train_data, batch_size, seq_len, device)
                loss = compute_loss(model, x, y, device, config["precision"])
                window_loss += loss.detach() / accum_steps
                (loss / accum_steps).backward()
                del loss
            # 累积完再截断，并且只更新一次参数；参数与两个动量保持 FP32。
            gradient_clipping(model.parameters(), training["max_grad_norm"])
            optimizer.step()
            window_steps += 1

            is_last = step == training["steps"]
            do_eval = step % training["eval_interval"] == 0 or is_last
            do_save = step % training["checkpoint_interval"] == 0 or is_last
            do_log = step % training["log_interval"] == 0 or do_eval or do_save
            if do_log:
                synchronize(device)
                elapsed = time.perf_counter() - window_start
                training_seconds += elapsed
                train_loss = window_loss.item() / window_steps
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"step={step}: 训练 loss 非有限值，停止运行。")
                throughput = window_steps * tokens_per_step / elapsed
                row = {
                    "step": step, "train_loss": train_loss, "lr": lr,
                    "tokens_per_second": throughput, "seconds_per_step": elapsed / window_steps,
                    "training_seconds": training_seconds,
                    "estimated_remaining_train_seconds": (training["steps"] - step) * tokens_per_step / throughput,
                }
                if do_eval:
                    row["valid_loss"] = evaluate(model, valid_batches, device, config["precision"])
                    if not math.isfinite(row["valid_loss"]):
                        raise FloatingPointError(f"step={step}: 验证 loss 非有限值，停止运行。")
                    if row["valid_loss"] < best_loss:
                        best_loss = row["valid_loss"]
                        save_checkpoint(run_dir / "best.pt", model, optimizer, config, step,
                                        best_loss, training_seconds, device)
                if device.type == "cuda":
                    row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                record(row)
                if do_save:
                    save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, config, step,
                                    best_loss, training_seconds, device)
                # 吞吐窗口不包含验证、日志写盘、checkpoint 保存。
                synchronize(device)
                window_loss.zero_()
                window_steps = 0
                window_start = time.perf_counter()

    if wandb_run is not None:
        wandb_run.finish()
    print(f"完成：{run_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
