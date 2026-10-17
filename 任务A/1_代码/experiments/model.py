"""保留自己的模型结构，只替换注意力内部的计算。"""

import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformer.Multihead_self_attention_module import Multihead_self_attention
from transformer.Transformer_lm import Transformer_lm


class FlashAttention2(nn.Module):
    def __init__(self, attention: Multihead_self_attention):
        super().__init__()
        # 引用已有权重，保留参数名和 state_dict 格式，不重新初始化。
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.num_heads = attention.num_heads
        self.use_rope = attention.use_rope
        if self.use_rope:
            self.rope = attention.rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        q = einsum(self.q_proj, x, "out d, b s d -> b s out")
        k = einsum(self.k_proj, x, "out d, b s d -> b s out")
        v = einsum(self.v_proj, x, "out d, b s d -> b s out")
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(x.shape[1], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # PyTorch 自带 FA2，接收 (batch, heads, sequence, head_dim)。
        # 明确只启用 FlashAttention 后端，不让 SDPA 自行选择其他算法。
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        output = rearrange(output, "b h s d -> b s (h d)")
        return einsum(self.o_proj, output, "out d, b s d -> b s out")


def build_model(config: dict, device: torch.device) -> Transformer_lm:
    model_config = config["model"]
    d_model = model_config["d_model"]
    num_heads = model_config["num_heads"]
    if d_model % num_heads != 0 or (d_model // num_heads) % 2 != 0:
        raise ValueError("d_model 必须能被 num_heads 整除，RoPE 的 head_dim 必须是偶数。")

    # 在 CPU 初始化，避免 RoPE 的逐元素建表在 CUDA 上提交大量小操作。
    model = Transformer_lm(**model_config, device="cpu", dtype=torch.float32)
    if config["attention"] == "flash2":
        for layer in model.layers:
            layer.msa = FlashAttention2(layer.msa)
    return model.to(device)
