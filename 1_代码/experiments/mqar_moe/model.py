"""Keep the checked dense initialization and attention; replace only the FFN."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.mqar_fp32.model import build_model as build_dense
from moe.MoE_module import MoE


class SparseExperts(MoE):
    def compute_experts(self, tokens, weights, expert_indices):
        # Empty matrices work too; avoiding a Python branch on a dynamic token
        # count lets Inductor compile genuine sparse, capacity-free dispatch.
        output = torch.zeros_like(tokens, dtype=torch.float32)
        counts = []
        for index, expert in enumerate(self.experts):
            mask = expert_indices == index
            counts.append(mask.sum())
            rows, slots = torch.where(mask)
            # A zero-token expert otherwise creates a zero-size Triton launch.
            # One zero-weight dummy assignment keeps GEMM/scatter nonempty and
            # has exactly zero contribution to outputs and gradients.
            selected_weights = F.pad(weights[rows, slots], (0, 1), value=0)
            rows = F.pad(rows, (0, 1), value=0)
            # Merge the SwiGLU gate/up projections into one FP32 GEMM, keeping
            # the original expert Parameters and the sparse token assignment.
            projections = F.linear(tokens[rows], torch.cat((expert.w1, expert.w3), dim=0))
            gate, up = projections.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, expert.w2)
            contributions = expert_output.float() * selected_weights[:, None]
            output = output.index_add(0, rows, contributions)
        return output, torch.stack(counts).float()


class Block(nn.Module):
    def __init__(self, original, moe):
        super().__init__()
        self.msa = original.msa
        self.rmsnorm_1 = original.rmsnorm_1
        self.rmsnorm_2 = original.rmsnorm_2
        self.moe = moe

    def forward(self, hidden, positions):
        attended = hidden + self.msa(self.rmsnorm_1(hidden), positions)
        output, balance, load = self.moe(self.rmsnorm_2(attended))
        return attended + output, balance, load


def build_model(config, device, backend=None):
    if config["model"]["norm_position"] != "pre":
        raise ValueError("The controlled baseline uses pre-norm")
    # Dense shared parameters have exactly the dense seed-123 initialization.
    model = build_dense(config, torch.device("cpu"), backend)
    settings = config["moe"]
    if settings["num_experts"] != 4 or settings["top_k"] != 2:
        raise ValueError("This experiment uses four experts and Top-2")
    if settings["expert_d_ff"] * settings["top_k"] != config["model"]["d_ff"]:
        raise ValueError("Active FFN width must equal the dense FFN width")
    for index, layer in enumerate(model.layers):
        moe = SparseExperts(config["model"]["d_model"], settings["expert_d_ff"],
                            settings["num_experts"], settings["top_k"], device="cpu", dtype=torch.float32)
        with torch.no_grad():
            for name, parameter in moe.named_parameters():
                std = config["initialization"]["std"]
                if name.endswith("w2"):
                    std /= math.sqrt(2 * config["model"]["num_layers"])
                nn.init.normal_(parameter, std=std)
        model.layers[index] = Block(layer, moe)
    return model.to(device)


class QueryComputation(nn.Module):
    def __init__(self, model, query_head_only=True):
        super().__init__()
        self.model = model
        self.query_head_only = query_head_only

    def forward(self, inputs, positions, targets):
        hidden = self.model.embedding(inputs)
        token_positions = torch.arange(inputs.shape[1], device=inputs.device)
        balances, loads = [], []
        for layer in self.model.layers:
            hidden, balance, load = layer(hidden, token_positions)
            balances.append(balance)
            loads.append(load)
        rows = torch.arange(inputs.shape[0], device=inputs.device)[:, None]
        if self.query_head_only:
            logits = self.model.lm_head(self.model.final_norm(hidden[rows, positions]))
        else:
            logits = self.model.lm_head(self.model.final_norm(hidden))[rows, positions]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, ce, torch.stack(balances).mean(), torch.stack(loads)


def build_computation(model, config, compiled=True):
    computation = QueryComputation(model, config["optimization"]["query_head_only"])
    if compiled and config["optimization"]["compile"]:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        computation = torch.compile(computation, mode=config["optimization"]["compile_mode"],
                                    fullgraph=True, dynamic=False)
    return computation
