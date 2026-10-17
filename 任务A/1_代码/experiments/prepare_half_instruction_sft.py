"""固定 seed42 从当前 <2048 合格训练集无放回抽一半，保留全部验证集。"""

from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from experiments.full_instruction_data import epoch_order, file_sha256, padded_positions

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "../2_实验和训练数据/training_data/smol_smoltalk_parent"
OUTPUT = ROOT / "../2_实验和训练数据/training_data/smol_smoltalk"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    original = json.loads((BASE / "manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    population = original["splits"]["train"]["conversations"]
    selection = np.sort(np.random.default_rng(42).choice(population, population // 2, replace=False)).astype("<u8")
    np.save(OUTPUT / "selected_parent_indices.npy", selection)
    train_offset = 0
    shards = []
    for previous in original["shards"]:
        source, target = BASE / previous["path"], OUTPUT / previous["path"]
        target.mkdir(parents=True, exist_ok=True)
        if previous["original_split"] == "test":
            # 验证内容和顺序完全不变；同一磁盘共享只读文件。
            for name in previous["sha256"]:
                if not (target / name).exists():
                    os.link(source / name, target / name)
            stats = {**previous, "parent_eligible_conversations": previous["conversations"]}
        else:
            indexes = np.load(source / "windows.npy")
            local = selection[(selection >= train_offset) & (selection < train_offset + len(indexes))] - train_offset
            local_set = set(map(int, local))
            tokens = np.memmap(source / "tokens.bin", mode="r", dtype="<u2")
            mask = np.memmap(source / "assistant_mask.bin", mode="r", dtype="u1")
            new_indexes, lengths, sources, turns = [], [], Counter(), Counter()
            flat_offset = supervised = 0
            with (target / "tokens.bin").open("wb") as token_stream, \
                    (target / "assistant_mask.bin").open("wb") as mask_stream, \
                    (target / "rows.jsonl").open("w") as row_stream:
                for number, line in enumerate((source / "rows.jsonl").open()):
                    if number not in local_set:
                        continue
                    offset, length, prefix, author_row = map(int, indexes[number])
                    if prefix != 0 or length + 1 >= 2048:
                        raise ValueError("Parent data violates complete-conversation length rule")
                    ids = tokens[offset:offset + length + 1]
                    labels = mask[offset:offset + length + 1]
                    ids.tofile(token_stream)
                    labels.tofile(mask_stream)
                    row = json.loads(line)
                    row["parent_dataset_index"] = train_offset + number
                    row_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sources[row["source"]] += 1
                    turns[row["assistant_turns"]] += 1
                    supervised += int(labels[1:].sum())
                    new_indexes.append([flat_offset, length, 0, author_row])
                    lengths.append(length + 1)
                    flat_offset += length + 1
            np.save(target / "windows.npy", np.asarray(new_indexes, dtype="<u8").reshape(-1, 4))
            np.save(target / "conversation_lengths.npy", np.asarray(lengths, dtype="<u8"))
            stats = {"settings": {"length_rule": "complete serialized tokens < 2048", "sampling_seed": 42,
                        "sampling": "uniform without replacement, floor(parent train size / 2)"},
                "name": previous["name"], "path": previous["path"], "original_split": "train",
                "original_conversations": previous["original_conversations"],
                "parent_eligible_conversations": len(indexes), "conversations": len(new_indexes),
                "windows": len(new_indexes), "total_tokens": flat_offset, "supervised_tokens": supervised,
                "useful_input_positions": flat_offset - len(new_indexes), "sources": dict(sources),
                "assistant_turn_counts": dict(turns), "sha256": {name: file_sha256(target / name) for name in previous["sha256"]}}
            train_offset += len(indexes)
        write_json(target / "stats.json", stats)
        shards.append(stats)
    if train_offset != population:
        raise ValueError("Sampling population differs from parent manifest")
    split_stats = {}
    for split, author_split in [("train", "train"), ("validation", "test")]:
        selected = [row for row in shards if row["original_split"] == author_split]
        sources, turns = Counter(), Counter()
        for row in selected:
            sources.update(row["sources"])
            turns.update(row["assistant_turn_counts"])
        lengths = np.concatenate([np.load(OUTPUT / row["path"] / "conversation_lengths.npy") for row in selected])
        split_stats[split] = {"original_split": author_split,
            **{key: sum(row[key] for row in selected) for key in ["original_conversations", "parent_eligible_conversations",
                "conversations", "windows", "total_tokens", "supervised_tokens", "useful_input_positions"]},
            "sources": dict(sources), "assistant_turn_counts": dict(turns),
            "conversation_length": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean())},
            "shards": [row["path"] for row in selected]}
    manifest = {**original, "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "prepared_training_not_started", "source_directory": str(Path("..") / BASE.relative_to(ROOT.parent)),
        "source_manifest_sha256": file_sha256(BASE / "manifest.json"),
        "selection": "All eligible complete conversations <2048; train uniformly sampled without replacement to half with seed42; full eligible validation unchanged.",
        "sampling": {"seed": 42, "population": population, "selected": len(selection), "stratified": False,
            "selected_parent_indices_sha256": file_sha256(OUTPUT / "selected_parent_indices.npy")},
        "splits": split_stats, "shards": shards, "training_started": False}
    write_json(OUTPUT / "manifest.json", manifest)
    train_lengths = np.concatenate([np.load(OUTPUT / row["path"] / "windows.npy")[:, 1].astype(np.int64)
                                   for row in shards if row["original_split"] == "train"])
    batch_size, epochs = 64, 2
    steps = math.ceil(len(train_lengths) / batch_size)
    budgets = [{"epoch": epoch, "updates": steps,
        "padded_input_positions": padded_positions(train_lengths, epoch_order(len(train_lengths), 42, epoch), batch_size)}
        for epoch in [1, 2]]
    budget = {"train_samples": len(train_lengths), "validation_samples": split_stats["validation"]["conversations"],
        "batch_size": batch_size, "grad_accum_steps": 1, "epochs": epochs, "steps_per_epoch": steps,
        "total_steps": steps * epochs, "warmup_steps": math.ceil(.1 * steps * epochs),
        "epoch_budgets": budgets, "training_started": False}
    write_json(OUTPUT / "training_budget.json", budget)
    for kind in ["dense", "moe"]:
        config = json.loads((ROOT / f"experiments/configs/instruction_sft_full_{kind}.json").read_text())
        config["status"] = "prepared_half_train_a800_training_not_started"
        config["data"]["directory"] = str(Path("..") / OUTPUT.relative_to(ROOT.parent))
        config["data"]["manifest_sha256"] = file_sha256(OUTPUT / "manifest.json")
        config["data"]["selection"] = manifest["selection"]
        config["data"]["batching"] = "Same seed+epoch shuffle for both models; complete batch64, right padding to batch maximum; no splitting, accumulation, packing, truncation or windows; keep last batch."
        config["training"].update(batch_size=64, grad_accum_steps=1, epochs=epochs, steps_per_epoch=steps,
            max_steps=steps * epochs, warmup_steps=budget["warmup_steps"], eval_interval=steps,
            log_interval=100, checkpoint_interval=1000)
        config["training"]["checkpoint_selection"] = "Lowest assistant-token-averaged CE on unchanged full eligible validation after each epoch; no RULER or base reevaluation."
        config["recipe_boundary"] = "2 epochs, cosine and 10% warmup; existing LR=1e-4 and tokenizer; uniform half of eligible train, unchanged eligible validation; full batch64 on one A800 per model."
        config["budget"] = str(Path("..") / (OUTPUT / "training_budget.json").relative_to(ROOT.parent))
        config["hardware"] = {"gpu": "NVIDIA A800-SXM4-80GB", "parallel_models": True, "physical_gpu_index": 0 if kind == "dense" else 1}
        write_json(ROOT / f"experiments/configs/instruction_sft_half_a800_{kind}.json", config)
    print(json.dumps(budget, indent=2), flush=True)


if __name__ == "__main__":
    main()
