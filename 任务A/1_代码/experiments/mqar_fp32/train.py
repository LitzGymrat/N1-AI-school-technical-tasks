"""Full-batch FP32 training, online W&B and window-based synchronization."""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from experiments.mqar.data import atomic_json, sha256
from .batches import CachedPool
from .model import build_computation, build_model


def source_files():
    root = Path(__file__).resolve().parents[2]
    basics = root / "transformer"
    sources = list(Path(__file__).parent.glob("*.py")) + list(basics.glob("*.py"))
    sources += [root / "experiments/mqar/data.py",
                root / "experiments/tasks/mqar.py"]
    return root, sorted(sources)


def source_hashes():
    root, sources = source_files()
    return {str(source.relative_to(root)): sha256(source) for source in sources}


@torch.no_grad()
def evaluate(model, computation, pool, batch_size, device, prediction_path=None):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    predictions = []
    for _, values in pool.batches(np.arange(pool.size, dtype=np.int64), batch_size, device):
        logits, loss = computation(*values)
        predicted = logits.argmax(-1)
        count = values[2].numel()
        totals[0] += loss.detach().double() * count
        totals[1] += (predicted == values[2]).sum()
        totals[2] += count
        if prediction_path is not None:
            predictions.append(predicted.cpu().numpy().astype(np.uint16))
    loss_sum, correct, queries = totals.cpu().tolist()
    if prediction_path is not None:
        np.save(prediction_path, np.concatenate(predictions))
    model.train()
    return {"loss": loss_sum / queries, "query_accuracy": correct / queries,
            "correct": int(correct), "queries": int(queries), "samples": pool.size}


def train(config, data_dir, directory, resume=None, stop_after_step=None, check_fused_adamw=False):
    device = torch.device("cuda")
    settings = config["training"]
    if config["precision"] != "fp32" or any(k in settings for k in ("micro_batch_size", "grad_accum_steps")):
        raise ValueError("FP32 and complete batch updates are required")
    if config["wandb"]["mode"] != "online":
        raise ValueError("All training must use online W&B")
    if settings["batch_size"] <= 0 or settings["max_epochs"] <= 0 or config["data"]["drop_last"]:
        raise ValueError("Positive full batch/epoch limits and retained tail batches are required")
    if stop_after_step is not None and stop_after_step <= 0:
        raise ValueError("The update bound must be positive")
    if check_fused_adamw and (resume is not None or stop_after_step is None or stop_after_step > 100
                             or not config["optimization"]["fused_adamw"]):
        raise ValueError("Optimizer comparison requires a new fused run bounded to at most 100 updates")
    data_dir, directory = Path(data_dir), Path(directory)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["task"] != config["task"] or manifest["data_config"] != config["data"] or any(manifest["cross_split_overlap"].values()):
        raise ValueError("Dataset configuration or split overlap mismatch")
    for split in manifest["splits"].values():
        for file in split["files"].values():
            if sha256(data_dir / file["path"]) != file["sha256"]:
                raise ValueError("Dataset checksum mismatch")
    torch.manual_seed(settings["model_seed"])
    rng = np.random.RandomState(settings["model_seed"] + 100)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["initial_lr"],
        betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"],
        fused=config["optimization"]["fused_adamw"])
    epoch, step, order, next_start, best = 1, 0, None, 0, -1.0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if (state["config"] != config or state["manifest_hash"] != sha256(data_dir / "manifest.json")
                or state["source_sha256"] != source_hashes()):
            raise ValueError("Checkpoint configuration, dataset or source differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        epoch, step, order, next_start, best = [state[k] for k in ("epoch", "step", "order", "next_start", "best")]
        del state
    else:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "config.json", config)
        shutil.copy2(data_dir / "manifest.json", directory / "data_manifest.json")
        source_root, sources = source_files()
        hashes = {}
        for source in sources:
            relative = source.relative_to(source_root)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256(source)
        atomic_json(directory / "source_sha256.json", hashes)
    environment = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "precision": "fp32", "tf32": False, "parameter_count": sum(p.numel() for p in model.parameters()),
        "batch_size": settings["batch_size"], "attention": config["attention"],
        "optimization": config["optimization"], "pid": os.getpid()}
    atomic_json(directory / "environment.json", environment)
    info_path = directory / "wandb_run.json"
    previous = json.loads(info_path.read_text()) if info_path.exists() else {}
    if resume is not None and not previous.get("id"):
        raise ValueError("Resume requires the original online W&B run ID")
    run = wandb.init(**config["wandb"], id=previous.get("id"), resume="must" if resume else "never",
                     name=directory.name, dir=str(directory), config={**config, "runtime": environment},
                     allow_val_change=True)
    run.define_metric("step")
    for group in ("train/*", "valid/*", "test/*", "optimizer_check/*"):
        run.define_metric(group, step_metric="step")
    atomic_json(info_path, {"id": run.id, "url": run.url, "entity": run.entity,
                           "project": run.project, "mode": "online"})
    print(json.dumps({"wandb_url": run.url, **environment}), flush=True)
    pools = {split: CachedPool(data_dir, split) for split in ("train", "valid")}
    computation = build_computation(model, config)
    evaluation = build_computation(model, config, compiled=False)
    seen_shapes = set()
    torch.cuda.reset_peak_memory_stats()

    def status(stage, **extra):
        atomic_json(directory / "status.json", {"status": stage, "pid": os.getpid(),
            "step": step, "epoch": epoch, "max_epochs": settings["max_epochs"], **extra})

    def save(filename="checkpoint.pt"):
        temporary = directory / (filename + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
            "manifest_hash": sha256(data_dir / "manifest.json"), "epoch": epoch, "step": step,
            "source_sha256": source_hashes(),
            "order": order, "next_start": next_start, "best": best, "sampler_rng": rng.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(directory / filename)

    started = time.perf_counter()
    try:
        parameters = list(model.parameters())
        reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters] if check_fused_adamw else []
        reference_optimizer = (torch.optim.AdamW(reference_parameters, lr=settings["initial_lr"],
            betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"],
            fused=False, foreach=False) if check_fused_adamw else None)
        optimizer_check = {"status": "running", "updates": 0, "parameter_tensors": len(parameters),
                           "maximum_absolute_error": 0.0, "maximum_relative_l2_error": 0.0,
                           "rtol": 1e-5, "atol": 1e-6,
                           "boundary": "Same actual gradients for fused and unfused torch AdamW; reference tensors are not used to compute training loss."}
        with (directory / "metrics.jsonl").open("a") as metrics:
            while epoch <= settings["max_epochs"] and best <= settings["early_stopping_threshold"]:
                if order is None:
                    order = rng.permutation(pools["train"].size)
                lr = settings["initial_lr"] * (1 + math.cos(math.pi * (epoch - 1) / settings["lr_t_max_epochs"])) / 2
                for group in optimizer.param_groups:
                    group["lr"] = lr
                total_loss = torch.zeros((), device=device)
                queries = samples = window_steps = 0
                window_optimizer_comparison = False
                torch.cuda.synchronize()
                window_start = time.perf_counter()
                status("running")
                for next_start, values in pools["train"].batches(order, settings["batch_size"], device, next_start):
                    shape = len(values[0])
                    first_shape = config["optimization"]["compile"] and shape not in seen_shapes
                    if first_shape:
                        torch.cuda.synchronize()
                        first_shape_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    logits, loss = computation(*values)
                    loss.backward()
                    if reference_optimizer is not None:
                        for parameter, reference in zip(parameters, reference_parameters):
                            reference.grad = parameter.grad.detach().clone()
                        reference_optimizer.step()
                    optimizer.step()
                    if reference_optimizer is not None:
                        window_optimizer_comparison = True
                        for parameter, reference in zip(parameters, reference_parameters):
                            torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=1e-6)
                            delta = (parameter.detach() - reference.detach())
                            optimizer_check["maximum_absolute_error"] = max(optimizer_check["maximum_absolute_error"], delta.abs().max().item())
                            optimizer_check["maximum_relative_l2_error"] = max(optimizer_check["maximum_relative_l2_error"],
                                (delta.norm() / reference.detach().norm().clamp_min(1e-12)).item())
                        optimizer_check["updates"] += 1
                        if optimizer_check["updates"] == min(10, stop_after_step):
                            optimizer_check["status"] = "passed"
                            atomic_json(directory / "fused_adamw_check.json", optimizer_check)
                            run.log({"step": step + 1, **{"optimizer_check/" + k: v for k, v in optimizer_check.items()
                                                         if isinstance(v, (int, float))}})
                            reference_optimizer = None
                            reference_parameters.clear()
                            del reference, delta
                    if first_shape:
                        torch.cuda.synchronize()
                        first_shape_seconds = time.perf_counter() - first_shape_started
                        seen_shapes.add(shape)
                    count = values[2].numel()
                    total_loss.add_(loss.detach() * count)
                    queries += count
                    samples += len(values[0])
                    window_steps += 1
                    step += 1
                    del logits, loss
                    if first_shape or step == 1 or step % settings["log_every_steps"] == 0 or next_start == len(order):
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - window_start
                        value = float(total_loss / queries)
                        if not math.isfinite(value):
                            raise FloatingPointError("Nonfinite training loss")
                        row = {"step": step, "epoch": epoch, "train_loss": value, "lr": lr,
                            "seconds_per_step": seconds / window_steps,
                            "tokens_per_second": samples * config["task"]["input_seq_len"] / seconds,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                            "window_steps": window_steps,
                            "window_includes_optimizer_comparison": window_optimizer_comparison,
                            "window_includes_first_shape_execution": bool(first_shape)}
                        if first_shape:
                            row.update(first_shape_batch_size=shape,
                                       first_shape_forward_backward_update_seconds=first_shape_seconds)
                        metrics.write(json.dumps(row) + "\n")
                        metrics.flush()
                        run.log({"step": step, "epoch": epoch, **{"train/" + k: v for k, v in row.items() if k not in ("step", "epoch")}})
                        print(json.dumps(row), flush=True)
                        status("running", **{k: v for k, v in row.items() if k not in ("step", "epoch")})
                        total_loss.zero_()
                        queries = samples = window_steps = 0
                        window_optimizer_comparison = False
                        window_start = time.perf_counter()
                    if step % 1000 == 0:
                        save()
                    if stop_after_step is not None and step >= stop_after_step:
                        save()
                        status("stopped_after_requested_steps")
                        run.finish()
                        return
                status("validating")
                valid = evaluate(model, evaluation, pools["valid"], settings["batch_size"], device)
                row = {"step": step, "epoch": epoch, "valid": valid}
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                run.log({"step": step, "epoch": epoch, **{"valid/" + k: v for k, v in valid.items()}})
                improved = valid["query_accuracy"] > best
                best = max(best, valid["query_accuracy"])
                epoch += 1
                order, next_start = None, 0
                save()
                if improved:
                    save("best.pt")
            model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"])
            test = evaluate(model, evaluation, CachedPool(data_dir, "test"), settings["batch_size"], device,
                            directory / "test_predictions.npy")
            results = {"status": "complete", "step": step, "epochs_completed": epoch - 1,
                       "best_valid_accuracy": best, "test": test, "wall_seconds": time.perf_counter() - started}
            atomic_json(directory / "final_results.json", results)
            run.log({"step": step, **{"test/" + k: v for k, v in test.items()}})
            run.summary.update(results)
            status("complete")
            run.finish()
    except BaseException as error:
        status("failed", error=str(error))
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--check-fused-adamw", action="store_true",
                        help="Compare first ten updates against unfused AdamW in a bounded qualification")
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.data_dir, args.out_dir, args.resume,
          args.stop_after_step, args.check_fused_adamw)
