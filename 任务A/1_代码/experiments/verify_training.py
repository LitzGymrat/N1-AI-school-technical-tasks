"""检查真实数据入口、梯度累积和断点续训；CUDA 下另外比较 FA2。"""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from optimizer.AdamW import AdamW
from experiments import train_owt
from experiments.model import build_model


def tiny_config():
    config = json.loads((train_owt.ROOT / "experiments/configs/owt_51m.json").read_text())
    config["model"].update(vocab_size=64, d_model=32, num_heads=2, d_ff=64, num_layers=2, context_len=16)
    config["training"].update(steps=4, warmup_steps=1, micro_batch_size=2, grad_accum_steps=2,
                                log_interval=2, eval_interval=2, eval_batches=2, checkpoint_interval=2)
    config["attention"] = "torch"
    config["precision"] = "fp32"
    return config


def check_accumulation(config):
    device = torch.device("cpu")
    torch.manual_seed(7)
    whole_batch_model = build_model(config, device)
    accumulated_model = copy.deepcopy(whole_batch_model)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    whole_optimizer = AdamW(whole_batch_model.parameters(), lr=0.001)
    accumulated_optimizer = AdamW(accumulated_model.parameters(), lr=0.001)
    train_owt.compute_loss(whole_batch_model, x, y, device, "fp32").backward()
    for start in (0, 2):
        loss = train_owt.compute_loss(accumulated_model, x[start:start + 2], y[start:start + 2], device, "fp32")
        (loss / 2).backward()
    for whole, accumulated in zip(whole_batch_model.parameters(), accumulated_model.parameters(), strict=True):
        torch.testing.assert_close(whole.grad, accumulated.grad, rtol=1e-4, atol=1e-6)
    whole_optimizer.step()
    accumulated_optimizer.step()
    for whole, accumulated in zip(whole_batch_model.parameters(), accumulated_model.parameters(), strict=True):
        torch.testing.assert_close(whole, accumulated, rtol=1e-4, atol=1e-6)
    print("PASS: 梯度累积与等效大 batch 的梯度、参数更新一致")


def check_resume(config):
    with tempfile.TemporaryDirectory(prefix="n1-training-check-") as temporary:
        root = Path(temporary)
        (root / "config.json").write_text(json.dumps(config))
        tokens = np.random.default_rng(7).integers(0, 64, size=1024, dtype=np.uint16)
        np.save(root / "train.npy", tokens)
        np.save(root / "valid.npy", tokens[::-1])
        args = argparse.Namespace(
            config=root / "config.json", train_data=root / "train.npy", valid_data=root / "valid.npy",
            tokenizer_dir=train_owt.TOKENIZER_DIR, device="cpu", attention=None, precision=None,
            moe_backend="torch",
            steps=None, warmup_steps=None, micro_batch_size=None, grad_accum_steps=None,
            out_dir=root / "continuous", resume=None, wandb_mode="disabled",
        )
        train_owt.main(args)
        continuous = torch.load(args.out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)

        args.out_dir = root / "interrupted"
        original_save = train_owt.save_checkpoint

        class PlannedInterruption(Exception):
            pass

        def interrupt_after_save(path, *save_args):
            original_save(path, *save_args)
            if path.name == "checkpoint.pt":
                raise PlannedInterruption

        # 模拟写完第 2 步 checkpoint 后中断；配置仍然是总共 4 步。
        with patch.object(train_owt, "save_checkpoint", side_effect=interrupt_after_save):
            try:
                train_owt.main(args)
            except PlannedInterruption:
                pass
        args.resume = args.out_dir / "checkpoint.pt"
        train_owt.main(args)
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert continuous["step"] == resumed["step"] == 4
        for name, tensor in continuous["model_state"].items():
            torch.testing.assert_close(tensor, resumed["model_state"][name], rtol=0, atol=0)
        for key, state in continuous["optimizer_state"]["state"].items():
            for name, value in state.items():
                other = resumed["optimizer_state"]["state"][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    assert value == other
        np.testing.assert_array_equal(continuous["numpy_rng"][1], resumed["numpy_rng"][1])
        assert continuous["numpy_rng"][2:] == resumed["numpy_rng"][2:]
        torch.testing.assert_close(continuous["torch_rng"], resumed["torch_rng"], rtol=0, atol=0)
    print("PASS: 中断续训与连续训练的最终权重、优化器及随机数状态完全一致")


def check_flash2(config):
    device = torch.device("cuda")
    reference = build_model(config, device)
    flash_config = copy.deepcopy(config)
    flash_config.update(attention="flash2", precision="bf16")
    flash_model = build_model(flash_config, device)
    # 同名权重可以直接加载，验证适配层没有改变 checkpoint 的格式。
    flash_model.load_state_dict(reference.state_dict())
    x = torch.randint(0, 64, (2, 16), device=device)
    y = torch.randint(0, 64, (2, 16), device=device)
    reference_logits = reference(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flash_logits = flash_model(x)
    assert flash_logits.dtype == torch.bfloat16
    pending = [flash_logits.grad_fn]
    visited = set()
    while pending:
        node = pending.pop()
        if node is not None and node not in visited:
            visited.add(node)
            pending.extend(next_node for next_node, _ in node.next_functions)
    flash_nodes = [node for node in visited if type(node).__name__ == "ScaledDotProductFlashAttentionBackward0"]
    assert len(flash_nodes) == config["model"]["num_layers"]
    print(f"Confirmed FlashAttention backward nodes: {len(flash_nodes)}")
    torch.testing.assert_close(flash_logits.float(), reference_logits, rtol=0.03, atol=0.03)
    train_owt.cross_entropy(reference_logits, y).backward()
    train_owt.cross_entropy(flash_logits.float(), y).backward()
    for name, parameter in flash_model.named_parameters():
        assert parameter.dtype == torch.float32 and parameter.grad.dtype == torch.float32
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=0.05, atol=0.002)
    optimizer = AdamW(flash_model.parameters(), lr=0.001)
    optimizer.step()
    for state in optimizer.state.values():
        assert state["m"].dtype == state["v"].dtype == torch.float32
    # 改变后半段 token，前半段 logits 应保持不变，排除未来信息泄漏。
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 64
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = flash_model(x)
        after = flash_model(changed)
    torch.testing.assert_close(before[:, :8], after[:, :8], rtol=0, atol=0)
    print("PASS: CUDA FA2+BF16 前向/梯度匹配参考，动量为 FP32，因果掩码无未来泄漏")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    # 小模型检查不受 CPU 线程池启动开销支配。
    torch.set_num_threads(1)
    config = tiny_config()
    tokens = np.arange(100, dtype=np.uint16)
    x, y = train_owt.sample_batch(tokens, 2, 8, torch.device("cpu"))
    assert x.shape == y.shape == (2, 8) and x.dtype == torch.long
    torch.testing.assert_close(y, x + 1)
    actual = np.load(train_owt.DATA_DIR / "owt_train.npy", mmap_mode="r")
    x, y = train_owt.sample_batch(actual, 2, 16, torch.device("cpu"))
    assert 0 <= x.min().item() <= x.max().item() < 32000
    assert 0 <= y.min().item() <= y.max().item() < 32000
    print(f"PASS: OWT memmap 入口，{len(actual):,} tokens，dtype={actual.dtype}，样本 ID 在词表内")
    check_accumulation(config)
    check_resume(config)
    if args.device == "cuda":
        check_flash2(config)
    else:
        print("本次未执行 CUDA/FA2 检查；服务器就绪后用 --device cuda。")


if __name__ == "__main__":
    main()
