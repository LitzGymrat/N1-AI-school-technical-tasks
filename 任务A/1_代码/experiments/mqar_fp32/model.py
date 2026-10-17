"""Keep the user's Transformer; add initialization, tying, dropout and SDPA."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformer.Transformer_lm import Transformer_lm


class EmbeddingDropout(nn.Module):
    def __init__(self, embedding, probability):
        super().__init__()
        self.weights = embedding.weights
        self.probability = probability

    def forward(self, indices):
        return F.dropout(self.weights[indices], self.probability, self.training)


class Attention(nn.Module):
    def __init__(self, original, dropout, backend):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(original, name))
        self.num_heads = original.num_heads
        self.rope = original.rope
        self.dropout = dropout
        self.backend = {"efficient": SDPBackend.EFFICIENT_ATTENTION,
                        "math": SDPBackend.MATH}[backend]

    def forward(self, hidden, positions=None):
        batch, length, width = hidden.shape
        def project(weight):
            return F.linear(hidden, weight).view(batch, length, self.num_heads, width // self.num_heads).transpose(1, 2)
        q, k, v = [project(weight) for weight in (self.q_proj, self.k_proj, self.v_proj)]
        if positions is None:
            positions = torch.arange(length, device=hidden.device)
        q, k = self.rope(q, positions), self.rope(k, positions)
        with sdpa_kernel(self.backend):
            output = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return F.linear(output.transpose(1, 2).reshape(batch, length, width), self.o_proj)


def build_model(config, device, backend=None):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config["precision"] != "fp32":
        raise ValueError("This entry requires FP32 without autocast or TF32")
    model = Transformer_lm(**config["model"], device="cpu", dtype=torch.float32)
    regularization = config["regularization"]
    model.embedding = EmbeddingDropout(model.embedding, regularization["embedding_dropout"])
    model.lm_head.weights = model.embedding.weights
    for layer in model.layers:
        layer.msa = Attention(layer.msa, regularization["attention_dropout"],
                              backend or config["attention"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                std = config["initialization"]["std"]
                if name.endswith("msa.o_proj") or name.endswith("swiglu.w2"):
                    std /= math.sqrt(2 * config["model"]["num_layers"])
                nn.init.normal_(parameter, std=std)
    model = model.to(device)
    model.lm_head.weights = model.embedding.weights
    return model


class QueryComputation(nn.Module):
    def __init__(self, model, query_head_only=True):
        super().__init__()
        self.model = model
        self.query_head_only = query_head_only

    def forward(self, inputs, positions, targets):
        hidden = self.model.embedding(inputs)
        token_positions = torch.arange(inputs.shape[1], device=inputs.device)
        for layer in self.model.layers:
            hidden = layer(hidden, token_positions)
        rows = torch.arange(inputs.shape[0], device=inputs.device)[:, None]
        if self.query_head_only:
            logits = self.model.lm_head(self.model.final_norm(hidden[rows, positions]))
        else:
            logits = self.model.lm_head(self.model.final_norm(hidden))[rows, positions]
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def build_computation(model, config, compiled=True):
    computation = QueryComputation(model, config["optimization"]["query_head_only"])
    if compiled and config["optimization"]["compile"]:
        computation = torch.compile(computation, mode=config["optimization"]["compile_mode"],
                                    fullgraph=True, dynamic=False)
    return computation
