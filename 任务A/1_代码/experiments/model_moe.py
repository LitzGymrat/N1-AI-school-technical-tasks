"""构建 MoE Transformer；注意力复用已经验证的 FA2 适配层。"""

import torch

from experiments.model import FlashAttention2
from moe.Transformer_lm import Transformer_lm


def build_model(config: dict, device: torch.device, moe_backend: str = "torch") -> Transformer_lm:
    if moe_backend == "grouped" and (device.type != "cuda" or config["precision"] != "bf16"):
        raise ValueError("grouped_gemm 后端要求 CUDA + BF16。")
    # 与 dense 入口一样先在 CPU 初始化，再统一移动到 GPU。
    model = Transformer_lm(**config["model"], device="cpu", dtype=torch.float32)
    for layer in model.layers:
        layer.moe.expert_backend = moe_backend
    if config["attention"] == "flash2":
        for layer in model.layers:
            layer.msa = FlashAttention2(layer.msa)
    return model.to(device)
