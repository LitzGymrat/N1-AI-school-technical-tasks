"""线性 router + Softmax + Top-k；按专家批量计算 SwiGLU。

分发参考 MiniMind/Mixtral，均衡损失参考 DeepSeekMoE；Linear/SwiGLU 复用自己的实现。
具体来源及与参考的差异见 experiments/README.md 的 MoE 部分。
"""

import torch
import torch.nn as nn
from einops import rearrange

from transformer.Linear_module import Linear
from transformer.SoftMax_function import softmax
from transformer.Swiglu_module import SWIGLU


class MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in = d_model
        self.d_ff = d_ff  # 一个专家的中间宽度，不是所有选中专家的宽度之和。
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_backend = "torch"
        self.router = Linear(d_model, num_experts, device=device, dtype=dtype)
        self.experts = nn.ModuleList([
            SWIGLU(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        tokens = rearrange(x, "b s d -> (b s) d")

        # 外层训练使用 BF16 autocast；router 的线性计算和 softmax 保持 FP32。
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_input = tokens.to(torch.float32)
            router_logits = self.router(router_input)  # (tokens, experts)
            scores = softmax(router_logits, dim=-1)
            weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
            weight_sum = weights.sum(dim=-1, keepdim=True)
            weights = weights / weight_sum  # (tokens, top_k)，每行权重之和为 1。

        if self.expert_backend == "grouped":
            from moe.grouped_experts import compute_grouped_experts

            output, expert_counts = compute_grouped_experts(self.experts, tokens, weights, expert_indices)
        else:
            output, expert_counts = self.compute_experts(tokens, weights, expert_indices)

        # 每个 token 有 top_k 次分配；load 是所有分配中各专家所占的比例。
        total_assignments = tokens.shape[0] * self.top_k
        load = expert_counts / total_assignments
        mean_scores = scores.mean(dim=0)
        # 索引及 load 不可微；scores 保留梯度，使均衡损失能训练 router。
        balance_loss = self.num_experts * (load * mean_scores).sum()
        output = rearrange(output, "(b s) d -> b s d", b=batch_size, s=seq_len)
        return output.to(x.dtype), balance_loss, load

    def compute_experts(self, tokens, weights, expert_indices):
        # 专家在外层 BF16 context 内计算，随后在 FP32 中加权累加。
        output = torch.zeros(tokens.shape, device=tokens.device, dtype=torch.float32)
        expert_counts = torch.zeros(self.num_experts, device=tokens.device, dtype=torch.float32)
        for expert_idx, expert in enumerate(self.experts):
            # 相等的位置是 True；sum 将 True 按 1 计数，得到该专家收到的 token 数。
            expert_mask = expert_indices == expert_idx  # (tokens, top_k)
            expert_counts[expert_idx] = expert_mask.sum()
            # 单参数 where 返回 True 位置的坐标：行是 token，列是 Top-k 内的位置。
            # 例如 [[0, 2], [2, 3]] 中找专家 2，得到行 [0, 1]、列 [1, 0]。
            token_indices, slot_indices = torch.where(expert_mask)
            if token_indices.numel() == 0:
                continue  # 这个专家没有收到 token，本轮不计算，也没有专家参数梯度。
            expert_input = tokens[token_indices]
            expert_output = expert(expert_input)
            token_weights = weights[token_indices, slot_indices]  # (收到的 tokens,)
            token_weights = rearrange(token_weights, "t -> t 1")  # 广播到每个 token 的所有分量。
            expert_output = expert_output.to(torch.float32)
            contributions = expert_output * token_weights
            # 同一专家内 token_indices 不重复；旧值包含此前专家的贡献，继续相加。
            previous_output = output[token_indices]
            output[token_indices] = previous_output + contributions

        return output, expert_counts
