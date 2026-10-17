"""Atomic local checkpoints: one rolling full state, explicit model-only exports."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch

FORMAT = "complete-training-state-v1"
REQUIRED_STATE = {"model", "optimizer", "cfg", "step_completed", "tokens",
                  "sampling_state", "cpu_rng", "cuda_rng", "python_rng", "numpy_rng",
                  "valid", "early_stop_hits", "first_threshold_steps", "last_eval_step",
                  "scheduler", "source_hashes", "benchmark_sha256", "test_sha256",
                  "local_artifacts", "model_extra_steps"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def checkpoint_writer_lock():
    """Serialize large temporary writes across all workers on this machine."""
    root = Path(os.environ.get("NIGHT_ROOT", Path(__file__).resolve().parents[2]))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_model(path, model, cfg, step, source_hashes):
    """A model-only artifact, never a substitute for complete-state resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with checkpoint_writer_lock():
        try:
            with temporary.open("wb") as stream:
                torch.save({"model": model, "cfg": cfg, "step_completed": step,
                            "source_hashes": source_hashes, "format": "model-only-v1"}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def capture_local_artifacts(out, model_extra_steps, step_completed):
    """Embed the committed curve and already-reached model points in the full state."""
    folder = Path(out)
    names = ["metrics.jsonl"] + [f"models/step_{step:06d}.pt"
        for step in sorted(model_extra_steps) if step <= step_completed]
    artifacts = {}
    for name in names:
        data = (folder / name).read_bytes()
        artifacts[name] = {"data":data, "sha256":hashlib.sha256(data).hexdigest()}
    return artifacts


def validate_local_artifacts(state):
    completed = int(state["step_completed"])
    extra_steps = sorted(state["model_extra_steps"])
    configured = sorted({int(x.strip()) for x in state["cfg"]["train"].get("model_extra_steps", "").split(",") if x.strip()})
    if extra_steps != configured or any(x not in (3000, 3250, 4500) for x in extra_steps):
        raise ValueError("Saved model-point attachment policy differs from the configuration")
    expected = {"metrics.jsonl"} | {f"models/step_{step:06d}.pt" for step in extra_steps if step <= completed}
    artifacts = state["local_artifacts"]
    if set(artifacts) != expected:
        raise ValueError("Complete-state local attachments are missing or unexpected")
    for name, record in artifacts.items():
        data = record.get("data")
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError("Local attachment hash mismatch: " + name)
    metrics = artifacts["metrics.jsonl"]["data"]
    if metrics and not metrics.endswith(b"\n"):
        raise ValueError("Saved metrics end with an uncommitted partial line")
    for line in metrics.splitlines():
        row = json.loads(line)
        if not 0 <= int(row["step"]) <= completed:
            raise ValueError("Saved metrics extend beyond the checkpoint")
    return artifacts


def _write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def restore_local_artifacts(out, state):
    """Restore exact saved bytes before appending; discard only future named points.

    Validation happens before any destination is touched. Replacing metrics also
    truncates a local post-checkpoint tail, so a replay cannot duplicate the curve.
    """
    artifacts = validate_local_artifacts(state)
    folder = Path(out)
    with checkpoint_writer_lock():
        for name, record in artifacts.items():
            _write_bytes_atomic(folder / name, record["data"])
        for step in state["model_extra_steps"]:
            name = f"models/step_{step:06d}.pt"
            if name not in artifacts:
                (folder / name).unlink(missing_ok=True)
        if (folder / "models").exists():
            _sync_directory(folder / "models")

def validate_complete(state, cfg=None, source_hashes=None):
    missing = REQUIRED_STATE - state.keys()
    if state.get("checkpoint_format") != FORMAT or missing:
        raise ValueError(f"Not a complete training state: missing={sorted(missing)}")
    step = int(state["step_completed"])
    if step < 0 or state["scheduler"]["completed"] != step:
        raise ValueError("Saved scheduler/update counts differ")
    if source_hashes is not None and state["source_hashes"] != source_hashes:
        raise ValueError("Resume source files differ from the saved runtime")
    validate_local_artifacts(state)
    if cfg is not None:
        for section in ("task", "model", "execution"):
            if state["cfg"].get(section) != cfg.get(section):
                raise ValueError("Resume configuration mismatch: " + section)
        saved, current = state["cfg"]["train"], cfg["train"]
        if set(saved) != set(current):
            raise ValueError("Resume training field set differs")
        for key, value in saved.items():
            if key not in {"resume_from", "deadline_unix"} and current[key] != value:
                raise ValueError("Resume training configuration mismatch: " + key)
    return state


def load_complete(path, cfg=None, source_hashes=None):
    return validate_complete(torch.load(path, map_location="cpu", weights_only=False), cfg, source_hashes)


def save_complete(out, state, phase):
    """Replace one full state atomically; readers may share-lock checkpoint.lock.

    The JSON digest is committed after the tensor file. An unlocked remote
    copier must verify the copied bytes against the JSON before publishing.
    No step archives, final aliases or accumulating checkpoint manifests.
    """
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "latest_resume.pt"
    temporary = folder / "latest_resume.pt.tmp"
    payload = {**state, "checkpoint_format": FORMAT, "save_phase": phase,
               "saved_unix": time.time()}
    validate_complete(payload)
    with checkpoint_writer_lock():
        with (folder / "checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with temporary.open("wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                record = {"path": path.name, "step_completed": payload["step_completed"],
                          "phase": phase, "bytes": temporary.stat().st_size,
                          "sha256": sha256_file(temporary), "format": FORMAT,
                          "saved_unix": payload["saved_unix"]}
                os.replace(temporary, path)
                _sync_directory(folder)
                write_json_atomic(folder / "latest_resume.json", record)
                return record
            finally:
                temporary.unlink(missing_ok=True)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
