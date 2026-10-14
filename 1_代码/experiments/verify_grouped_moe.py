"""在 CUDA 上核验 grouped 算子；可用现有 checkpoint 对照完整训练 step。"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from experiments.model_moe import build_model
from moe.MoE_module import MoE
from experiments.train_owt_moe import AdamW, cross_entropy, gradient_clipping, ROOT


def relative_error(actual, expected):
    difference = (actual.float() - expected.float()).norm()
    return (difference / expected.float().norm().clamp_min(1e-12)).item()


def check_operator(empty_experts=False):
    torch.manual_seed(17)
    original = MoE(64, 64, 4, 2, device="cuda", dtype=torch.float32)
    grouped = copy.deepcopy(original)
    grouped.expert_backend = "grouped"
    x = torch.randn(1, 33, 64, device="cuda", requires_grad=True)
    if empty_experts:
        with torch.no_grad():
            original.router.weights.zero_()
            original.router.weights[:, 0] = torch.tensor([4., 2., -2., -4.], device="cuda")
            x[:, :, 0] = 1
        grouped.load_state_dict(original.state_dict())
    other_x = x.detach().clone().requires_grad_()
    direction = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, balance, load = original(x)
        other_output, other_balance, other_load = grouped(other_x)
    torch.testing.assert_close(load, other_load, rtol=0, atol=0)
    torch.testing.assert_close(balance, other_balance, rtol=0, atol=0)
    output_error = relative_error(other_output, output)
    assert output_error < 0.01, output_error
    ((output * direction).mean() + 0.01 * balance).backward()
    ((other_output * direction).mean() + 0.01 * other_balance).backward()
    errors = {"input": relative_error(other_x.grad, x.grad)}
    for (name, parameter), (other_name, other) in zip(
            original.named_parameters(), grouped.named_parameters(), strict=True):
        assert name == other_name and (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            errors[name] = relative_error(other.grad, parameter.grad)
    assert max(errors.values()) < 0.02, errors
    if empty_experts:
        assert all(p.grad is None for expert in grouped.experts[2:] for p in expert.parameters())
    print(json.dumps({"check": "empty_experts" if empty_experts else "normal_routing",
                      "output_relative_error": output_error, "gradient_relative_errors": errors}), flush=True)


def training_step(model, optimizer, x, y, config):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, balance, _ = model(x)
    ce = cross_entropy(logits.float(), y)
    loss = ce + config["training"]["balance_loss_coef"] * balance
    loss.backward()
    gradient_clipping(model.parameters(), config["training"]["max_grad_norm"])
    optimizer.step()
    return ce.detach()


def benchmark(checkpoint, steps, warmup):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = state["step"]
    config = state["config"]
    assert config["training"]["micro_batch_size"] == config["training"]["grad_accum_steps"] == 1
    models, optimizers = {}, {}
    for backend in ("torch", "grouped"):
        model = build_model(config, torch.device("cuda"), backend)
        model.load_state_dict(state["model_state"])
        settings = config["training"]
        optimizer = AdamW(model.parameters(), lr=settings["max_lr"], betas=tuple(settings["betas"]),
                          weight_decay=settings["weight_decay"], eps=settings["eps"])
        optimizer.load_state_dict(state["optimizer_state"])
        # 使用 checkpoint 当时的实际 lr，保持两个后端的优化器状态一致。
        models[backend], optimizers[backend] = model, optimizer
    del state
    data = np.load(ROOT.parent / "2_实验和训练数据/training_data/owt/owt_train.npy", mmap_mode="r")
    seq_len = config["model"]["context_len"]
    row = torch.tensor(np.array(data[12345:12345 + seq_len + 1], dtype=np.int64), device="cuda")[None, :]
    x, y = row[:, :-1], row[:, 1:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = models["torch"](x)[0]
        other_logits = models["grouped"](x)[0]
        ce = cross_entropy(logits.float(), y).item()
        other_ce = cross_entropy(other_logits.float(), y).item()
    assert abs(ce - other_ce) < 0.01, (ce, other_ce)
    del logits, other_logits
    for backend in models:
        for _ in range(warmup):
            training_step(models[backend], optimizers[backend], x, y, config)
    records = {backend: [] for backend in models}
    peaks = {}
    # 交替测量，避免始终让某个后端在 GPU 更热/更冷时运行。
    for backend in ("torch", "grouped", "grouped", "torch"):
        model, optimizer = models[backend], optimizers[backend]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            last_ce = training_step(model, optimizer, x, y, config)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.isfinite(last_ce), last_ce
        records[backend].append(elapsed / steps)
        # 两个模型同时驻留；只比较训练 step 超出驻留基线的增量峰值。
        peaks[backend] = (torch.cuda.max_memory_allocated() - baseline_memory) / 1024**3
    result = {
        "checkpoint": str(checkpoint), "checkpoint_step": checkpoint_step,
        "batch_size": 1, "context_len": seq_len, "precision": "bf16", "attention": "flash2",
        "steps_per_window": steps, "warmup_steps": warmup, "initial_ce": {"torch": ce, "grouped": other_ce},
        "windows_seconds_per_step": records, "incremental_peak_gib": peaks,
    }
    for backend, values in records.items():
        seconds = statistics.mean(values)
        result[backend] = {"milliseconds_per_step": seconds * 1000, "tokens_per_second": seq_len / seconds}
    result["speedup"] = result["torch"]["milliseconds_per_step"] / result["grouped"]["milliseconds_per_step"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="给出时额外测完整训练 step，包含原来的 AdamW 和截断")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/outputs/setup/grouped-moe-check.json")
    args = parser.parse_args()
    check_operator()
    check_operator(empty_experts=True)
    if args.checkpoint is not None:
        result = benchmark(args.checkpoint, args.steps, args.warmup)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
