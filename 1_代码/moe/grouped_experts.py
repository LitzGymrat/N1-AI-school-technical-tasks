"""使用 fanshiqing/grouped_gemm 的分发、BF16 GEMM 和合并算子。

保留原专家 Parameter，stack/cast 的梯度会回传给原参数，checkpoint 无需转换。
此后端用于 CUDA + BF16 训练；router、均衡损失仍由 MoE_module 计算。
"""

import torch
from grouped_gemm import ops


def compute_grouped_experts(experts, tokens, weights, expert_indices):
    counts = torch.bincount(expert_indices.reshape(-1), minlength=len(experts))
    # 该库的 GEMM 接口要求 CPU int64 的各组行数；只传输这一小段计数。
    cpu_counts = counts.cpu()
    active_experts = [expert for expert, count in zip(experts, cpu_counts.tolist(), strict=True) if count > 0]
    batch_sizes = cpu_counts[cpu_counts > 0]
    # 不把空专家放进 stack，保持原实现的 grad=None 和 AdamW 跳过行为。

    # Top-2 token 会展开成两行，再按专家排序；不丢 token、不添加 capacity 限制。
    grouped_tokens, row_map = ops.permute(tokens.contiguous(), expert_indices.to(torch.int32))
    grouped_tokens = grouped_tokens.to(torch.bfloat16)

    # 将 SwiGLU 的两次输入投影合在同一次 grouped GEMM 中，随后拆回 gate/up。
    input_weights = torch.stack([torch.cat((expert.w1, expert.w3), dim=0) for expert in active_experts])
    output_weights = torch.stack([expert.w2 for expert in active_experts])
    input_weights = input_weights.to(torch.bfloat16)
    output_weights = output_weights.to(torch.bfloat16)
    projections = ops.gmm(grouped_tokens, input_weights, batch_sizes, trans_b=True)
    gate, up = projections.chunk(2, dim=-1)
    hidden = (gate * torch.sigmoid(gate)) * up
    expert_output = ops.gmm(hidden, output_weights, batch_sizes, trans_b=True)

    # 与原实现一样，在 FP32 中按路由权重加权并累加，再由调用方转回 x.dtype。
    output = ops.unpermute(expert_output.float(), row_map, weights)
    return output, counts.float()
