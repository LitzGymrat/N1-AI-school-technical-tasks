"""CUDA equivalence, controlled initialization and compiled-shape checks; no updates."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mqar.data import atomic_json, sha256
from experiments.mqar_fp32.model import build_model as build_dense
from moe.MoE_module import MoE
from .model import SparseExperts, build_computation, build_model


def check(config, data_dir, output, config_path):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    torch.manual_seed(123)
    sparse = SparseExperts(512, 672, 4, 2, device=device, dtype=torch.float32)
    reference = MoE(512, 672, 4, 2, device=device, dtype=torch.float32)
    reference.load_state_dict(sparse.state_dict())
    left = torch.randn(2, 11, 512, device=device, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    actual = sparse(left)
    expected = reference(right)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=3e-6)
    (actual[0].square().mean() + .01 * actual[1]).backward()
    (expected[0].square().mean() + .01 * expected[1]).backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=5e-5, atol=3e-6)
    for (name, a), (_, b) in zip(sparse.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=3e-6, msg=name)
    # Force two empty experts and check the compiled path separately.
    with torch.no_grad():
        sparse.router.weights.zero_()
    empty = torch.ones(1, 7, 512, device=device, requires_grad=True)
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    compiled_sparse = torch.compile(sparse, fullgraph=True, dynamic=False)
    outputs = compiled_sparse(empty)
    (outputs[0].square().mean() + .01 * outputs[1]).backward()
    assert int((outputs[2] == 0).sum()) == 2
    assert torch.isfinite(empty.grad).all()
    result = {"status": "running", "compiled_empty_experts_verified": True, "optimizer_updates": 0, "final_test_accessed": False,
        "sparse_vs_original": {"maximum_output_error": (actual[0] - expected[0]).abs().max().item(),
                              "all_parameter_and_input_gradients_verified": True}}
    torch.manual_seed(config["training"]["model_seed"])
    model = build_model(config, device)
    torch.manual_seed(config["training"]["model_seed"])
    dense = build_dense(config, torch.device("cpu"))
    shared = 0
    for name, tensor in model.state_dict().items():
        if name in dense.state_dict():
            assert torch.equal(tensor.cpu(), dense.state_dict()[name]), name
            shared += 1
    result["shared_initialization"] = {"identical_dense_tensors": shared, "seed": 123}
    del dense, sparse, reference, actual, expected, left, right
    pool = {name: np.load(Path(data_dir) / f"train_{name}.npy", mmap_mode="r")
            for name in ("inputs", "positions", "targets")}
    def values(count):
        return tuple(torch.as_tensor(np.array(array[:count], dtype=np.int64), device=device) for array in pool.values())
    eager = build_computation(model, config, compiled=False)
    compiled = build_computation(model, config)
    model.eval()
    reference_values = eager(*values(2))
    reference_objective = reference_values[1] + .01 * reference_values[2]
    reference_objective.backward()
    reference_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    compiled_values = compiled(*values(2))
    (compiled_values[1] + .01 * compiled_values[2]).backward()
    torch.cuda.synchronize()
    maximum_gradient_error = 0.0
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, reference_gradients[name], rtol=8e-4, atol=8e-6, msg=name)
        maximum_gradient_error = max(maximum_gradient_error, (p.grad - reference_gradients[name]).abs().max().item())
    torch.testing.assert_close(compiled_values[0], reference_values[0], rtol=8e-4, atol=8e-6)
    result["compiled_equivalence"] = {"maximum_logit_error": (compiled_values[0] - reference_values[0]).abs().max().item(),
        "maximum_gradient_error": maximum_gradient_error, "compile_and_backward_seconds": time.perf_counter() - started}
    del reference_values, compiled_values, reference_gradients, reference_objective
    model.train()
    result["training_shapes"] = []
    for batch in (128, 32):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        logits, ce, balance, load = compiled(*values(batch))
        (ce + .01 * balance).backward()
        torch.cuda.synchronize()
        assert logits.shape == (batch, 64, 8192)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        result["training_shapes"].append({"batch": batch, "ce": ce.item(), "balance": balance.item(),
            "load": load.tolist(), "first_forward_backward_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        del logits, ce, balance, load
    result.update(status="passed", gpu=torch.cuda.get_device_name(), precision="fp32", tf32=False,
        parameter_count=sum(p.numel() for p in model.parameters()), config_sha256=sha256(config_path))
    atomic_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Hash the actual requested configuration, independent of output location.
    check_config = json.loads(args.config.read_text())
    check(check_config, args.data_dir, args.output, args.config)
