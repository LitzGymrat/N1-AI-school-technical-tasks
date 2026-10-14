"""Target-device qualification without optimizer updates or test-set access."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from experiments.mqar.data import atomic_json, load_split, sha256
from .model import QueryComputation, build_computation, build_model


def gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or nonfinite gradient: {name}")
        result[name] = parameter.grad.detach().cpu().clone()
    return result


def compare(reference, actual):
    if reference.keys() != actual.keys():
        raise AssertionError("Parameter names differ")
    absolute = relative = 0.0
    for name in reference:
        a, b = reference[name], actual[name]
        torch.testing.assert_close(a, b, rtol=5e-4, atol=3e-6, msg=lambda msg: name + ": " + msg)
        absolute = max(absolute, (a - b).abs().max().item())
        relative = max(relative, ((a - b).norm() / a.norm().clamp_min(1e-12)).item())
    return {"parameter_tensors": len(reference), "maximum_absolute_error": absolute,
            "maximum_relative_l2_error": relative, "rtol": 5e-4, "atol": 3e-6}


def backward(computation, model, values):
    model.zero_grad(set_to_none=True)
    logits, loss = computation(*values)
    if logits.dtype != torch.float32 or not torch.isfinite(loss):
        raise AssertionError("Expected finite FP32 loss/logits")
    loss.backward()
    return logits.detach().cpu(), float(loss.detach()), gradients(model)


def verify(config, data_dir, output, config_hash):
    device = torch.device("cuda")
    report = {"status": "running", "optimizer_updates": 0, "final_test_accessed": False,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "precision": config["precision"],
              "attention": config["attention"], "config_sha256": config_hash,
              "data_manifest_sha256": sha256(Path(data_dir) / "manifest.json")}
    atomic_json(output, report)
    try:
        torch.manual_seed(config["training"]["model_seed"])
        model = build_model(config, device)
        if model.embedding.weights is not model.lm_head.weights:
            raise AssertionError("Embedding and output weights must stay tied")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise AssertionError("TF32 must remain disabled")
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        pool = load_split(data_dir, "train")
        def values(size):
            return tuple(torch.as_tensor(pool[name][:size].astype("int64"), device=device)
                         for name in ("inputs", "positions", "targets"))

        # Disable stochastic dropout via eval, while still computing all gradients.
        model.eval()
        sample = values(2)
        full = backward(QueryComputation(model, query_head_only=False), model, sample)
        query = backward(QueryComputation(model), model, sample)
        torch.testing.assert_close(full[0], query[0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(torch.tensor(full[1]), torch.tensor(query[1]), rtol=2e-5, atol=2e-6)
        report["query_projection_equivalence"] = {"samples": 2, "full_context_length": sample[0].shape[1],
            "logit_shape": list(query[0].shape), "reference_loss": full[1], "query_loss": query[1],
            "maximum_logit_error": (full[0] - query[0]).abs().max().item(),
            "gradients": compare(full[2], query[2])}
        atomic_json(output, report)

        reference = copy.deepcopy(model)
        for layer in reference.layers:
            layer.msa.backend = torch.nn.attention.SDPBackend.MATH
        math_result = backward(QueryComputation(reference), reference, sample)
        torch.testing.assert_close(math_result[0], query[0], rtol=5e-4, atol=3e-6)
        report["efficient_vs_math"] = {"dropout": "disabled for numerical comparison",
            "maximum_logit_error": (math_result[0] - query[0]).abs().max().item(),
            "gradients": compare(math_result[2], query[2])}
        del reference, math_result, full, query

        # Compare compiled/eager numerics at the true full-batch shape.
        eager = QueryComputation(model)
        compiled = build_computation(model, config)
        full_batch = values(config["training"]["batch_size"])
        eager_result = backward(eager, model, full_batch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        compiled_result = backward(compiled, model, full_batch)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_result[0], compiled_result[0], rtol=5e-4, atol=3e-6)
        report["compiled_equivalence"] = {"batch_size": len(full_batch[0]),
            "dropout": "disabled for numerical comparison",
            "reference_loss": eager_result[1], "compiled_loss": compiled_result[1],
            "maximum_logit_error": (eager_result[0] - compiled_result[0]).abs().max().item(),
            "gradients": compare(eager_result[2], compiled_result[2]),
            "first_execution_seconds": time.perf_counter() - started}
        del eager_result, compiled_result
        atomic_json(output, report)

        model.train()
        shapes = (config["training"]["batch_size"], config["data"]["train_samples"] % config["training"]["batch_size"])
        report["training_shapes"] = []
        for size in dict.fromkeys(shapes):
            if not size:
                continue
            batch = full_batch if size == len(full_batch[0]) else values(size)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits, loss, grads = backward(compiled, model, batch)
            torch.cuda.synchronize()
            row = {"batch_size": size, "logit_shape": list(logits.shape), "loss": loss,
                   "finite_gradient_tensors": len(grads),
                   "embedding_dropout": config["regularization"]["embedding_dropout"],
                   "attention_dropout": config["regularization"]["attention_dropout"],
                   "first_forward_backward_seconds": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            report["training_shapes"].append(row)
            print(json.dumps(row), flush=True)
            del logits, grads
            atomic_json(output, report)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            model.zero_grad(set_to_none=True)
            _, loss = eager(*values(1))
            loss.backward()
        operators = [event.key for event in profiler.key_averages()
                     if "efficient_attention" in event.key]
        if not any("efficient_attention" in name for name in operators):
            raise AssertionError("Expected efficient SDPA operator")
        report["observed_attention_operators"] = operators
        report["status"] = "passed"
        report["boundary"] = "Full-context forward/backward and CUDA Inductor; no optimizer update, stable train-step timing, or optimizer-state memory measurement. Dropout paths checked for finite gradients; numerical comparison disables dropout."
        atomic_json(output, report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        atomic_json(output, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(json.loads(args.config.read_text()), args.data_dir, args.out, sha256(args.config))
