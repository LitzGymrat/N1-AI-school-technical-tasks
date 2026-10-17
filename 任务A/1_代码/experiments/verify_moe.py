"""核验稀疏分发、梯度流、参数统计、续训；CUDA 下检查实际 FA2/BF16。"""

import argparse
import copy
import json
from unittest.mock import patch

import torch

from optimizer.AdamW import AdamW
from transformer.SoftMax_function import softmax
from experiments import train_owt_moe, verify_training
from experiments.model_moe import build_model
from moe.MoE_module import MoE


def tiny_config():
    config = json.loads((train_owt_moe.ROOT / "experiments/configs/owt_moe.json").read_text())
    config["model"].update(vocab_size=64, d_model=32, num_heads=2, d_ff=32, num_layers=2, context_len=16)
    config["training"].update(steps=4, warmup_steps=1, micro_batch_size=1, grad_accum_steps=1,
                              log_interval=2, eval_interval=2, eval_batches=2, checkpoint_interval=2)
    config.update(attention="torch", precision="fp32")
    return config


def check_sparse_reference():
    torch.manual_seed(7)
    sparse = MoE(4, 6, 4, 2, dtype=torch.float32)
    reference = copy.deepcopy(sparse)
    x = torch.randn(2, 3, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output, balance, load = sparse(x)

    # 独立参考：对每个 token 计算全部四个专家，再用稀疏 gate 加权求和。
    tokens = reference_x.reshape(-1, 4)
    scores = softmax(reference.router(tokens), dim=-1)
    weights, indices = torch.topk(scores, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    gates = torch.zeros_like(scores).scatter(1, indices, weights)
    all_outputs = torch.stack([expert(tokens) for expert in reference.experts], dim=1)
    expected = (all_outputs * gates[:, :, None]).sum(dim=1).reshape(2, 3, 4)
    expected_load = torch.nn.functional.one_hot(indices, 4).float().mean(dim=(0, 1))
    expected_balance = 4 * (expected_load * scores.mean(dim=0)).sum()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(load, expected_load)
    torch.testing.assert_close(balance, expected_balance)

    direction = torch.randn_like(output)
    ((output * direction).sum() + 0.01 * balance).backward()
    ((expected * direction).sum() + 0.01 * expected_balance).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-6)
    for actual, wanted in zip(sparse.parameters(), reference.parameters(), strict=True):
        if actual.grad is None:
            # 稀疏版跳过未选专家；全计算参考中同一专家的梯度为零。
            torch.testing.assert_close(wanted.grad, torch.zeros_like(wanted))
        else:
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-6)
    print("PASS: 稀疏分发的输出、输入梯度、router 与专家梯度匹配全计算参考")


def check_gradient_flow():
    moe = MoE(2, 2, 4, 2, dtype=torch.float32)
    with torch.no_grad():
        moe.router.weights.copy_(torch.tensor([[4., 0.], [2., 0.], [-2., 0.], [-4., 0.]]))
        for idx, expert in enumerate(moe.experts):
            expert.w1.fill_(1.)
            expert.w3.fill_(1.)
            expert.w2.fill_(idx + 1.)
    x = torch.tensor([[[1., 0.], [1., 0.]]], requires_grad=True)
    output, balance, load = moe(x)
    torch.testing.assert_close(load, torch.tensor([0.5, 0.5, 0., 0.]))
    output.sum().backward()
    task_gradient = moe.router.weights.grad.detach().clone()
    assert task_gradient[:2].abs().sum() > 0
    # Top-2 内归一化后，任务损失对未选中 router logits 的梯度抵消为零。
    torch.testing.assert_close(task_gradient[2:], torch.zeros(2, 2), rtol=0, atol=1e-6)
    assert all(p.grad is not None for expert in moe.experts[:2] for p in expert.parameters())
    assert all(p.grad is None for expert in moe.experts[2:] for p in expert.parameters())
    assert x.grad.abs().sum() > 0

    moe.zero_grad(set_to_none=True)
    _, balance, _ = moe(x)
    balance.backward()
    balance_gradient = moe.router.weights.grad.detach()
    assert balance_gradient[2:, 0].abs().min() > 0
    print("Gradient example:", json.dumps({
        "task_router_gradient": task_gradient[:, 0].tolist(),
        "balance_router_gradient": balance_gradient[:, 0].tolist(),
    }))
    print("PASS: 硬选择无梯度；任务通过选中权重训练 router，均衡损失也影响未选专家的 router 分数")


def check_parameter_counts():
    config = json.loads((train_owt_moe.ROOT / "experiments/configs/owt_moe.json").read_text())
    model = build_model(config, torch.device("cpu"))
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for layer in model.layers for p in layer.moe.experts.parameters())
    active = total - experts + experts * 2 // 4
    assert total == 63851008 and active == 51464704
    print(f"PASS: 实际参数 total={total:,}, active={active:,}；FFN 激活量与 dense 一致")


def check_cuda(config):
    config = copy.deepcopy(config)
    config.update(attention="flash2", precision="bf16")
    device = torch.device("cuda")
    model = build_model(config, device)
    router_dtypes = []
    hooks = [layer.moe.router.register_forward_hook(
        lambda module, inputs, output: router_dtypes.append(output.dtype)
    ) for layer in model.layers]
    x = torch.randint(0, 64, (1, 16), device=device)
    y = torch.randint(0, 64, (1, 16), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, balance, loads = model(x)
    assert logits.dtype == torch.bfloat16 and balance.dtype == torch.float32
    assert router_dtypes == [torch.float32] * config["model"]["num_layers"]
    pending, visited = [logits.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is not None and node not in visited:
            visited.add(node)
            pending.extend(next_node for next_node, _ in node.next_functions)
    flash_nodes = [node for node in visited if type(node).__name__ == "ScaledDotProductFlashAttentionBackward0"]
    assert len(flash_nodes) == config["model"]["num_layers"]
    loss = train_owt_moe.cross_entropy(logits.float(), y) + 0.01 * balance
    loss.backward()
    for layer in model.layers:
        gradient = layer.moe.router.weights.grad
        assert gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
    optimizer = AdamW(model.parameters(), lr=0.001)
    optimizer.step()
    assert all(state["m"].dtype == state["v"].dtype == torch.float32 for state in optimizer.state.values())
    for hook in hooks:
        hook.remove()

    # 改后半段 token，前半段输出应完全不变，排除 MoE 接入后的未来信息泄漏。
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 64
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = model(x)[0]
        after = model(changed)[0]
    torch.testing.assert_close(before[:, :8], after[:, :8], rtol=0, atol=0)
    print("PASS: CUDA FA2 backward、BF16 logits、FP32 router/动量、因果掩码均通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_sparse_reference()
    check_gradient_flow()
    check_parameter_counts()
    config = tiny_config()
    # 复用现有续训核验，只把它调用的训练入口换成独立 MoE 版本。
    with patch.object(verify_training, "train_owt", train_owt_moe):
        verify_training.check_resume(config)
    if args.device == "cuda":
        check_cuda(config)


if __name__ == "__main__":
    main()
