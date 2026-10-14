"""从已编码的全量作者对话中保留完整序列 <2048 的样本；不切窗、不截断。"""

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from experiments.full_instruction_data import file_sha256
from experiments.prepare_full_instruction_data import ROOT, write_json

BASE = ROOT / "../2_实验和训练数据/training_data/smol_smoltalk_full"
OUTPUT = ROOT / "../2_实验和训练数据/training_data/smol_smoltalk_parent"


def main():
    original = json.loads((BASE / "manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = []
    for previous in original["shards"]:
        source_dir, target_dir = BASE / previous["path"], OUTPUT / previous["path"]
        target_dir.mkdir(parents=True, exist_ok=True)
        # 原 token/mask 文件只读共享；只生成新的筛选索引，不重复分词。
        for name in ["tokens.bin", "assistant_mask.bin"]:
            path = target_dir / name
            if not path.exists():
                os.link(source_dir / name, path)
        original_lengths = np.load(source_dir / "conversation_lengths.npy")
        offset = supervised = total_tokens = useful_positions = 0
        sources, turns = Counter(), Counter()
        indexes, selected_lengths = [], []
        with (source_dir / "rows.jsonl").open() as stream, (target_dir / "rows.jsonl").open("w") as selected_rows:
            for row_index, line in enumerate(stream):
                row = json.loads(line)
                length = int(original_lengths[row_index])
                if length < 2048 and row["supervised_tokens"] > 0:
                    indexes.append([offset, length - 1, 0, row_index])
                    selected_lengths.append(length)
                    sources[row["source"]] += 1
                    turns[row["assistant_turns"]] += 1
                    supervised += row["supervised_tokens"]
                    total_tokens += length
                    useful_positions += length - 1
                    selected_rows.write(line)
                offset += length
        np.save(target_dir / "windows.npy", np.asarray(indexes, dtype="<u8").reshape(-1, 4))
        np.save(target_dir / "conversation_lengths.npy", np.asarray(selected_lengths, dtype="<u8"))
        stats = {"settings": {"max_length": 2048, "selection": "complete_serialized_length < 2048", "format_version": 4},
            "name": previous["name"], "path": previous["path"], "original_split": previous["original_split"],
            "original_conversations": previous["conversations"], "conversations": len(indexes),
            "excluded_length_ge_2048": int((original_lengths >= 2048).sum()),
            "sources": dict(sources), "assistant_turn_counts": dict(turns), "windows": len(indexes),
            "total_tokens": total_tokens, "supervised_tokens": supervised, "useful_input_positions": useful_positions,
            "sha256": {name: previous["sha256"][name] for name in ["tokens.bin", "assistant_mask.bin"]}}
        for name in ["windows.npy", "conversation_lengths.npy", "rows.jsonl"]:
            stats["sha256"][name] = file_sha256(target_dir / name)
        write_json(target_dir / "stats.json", stats)
        shards.append(stats)
    split_stats = {}
    for split, author_split in [("train", "train"), ("validation", "test")]:
        selected = [row for row in shards if row["original_split"] == author_split]
        sources, turns = Counter(), Counter()
        for shard in selected:
            sources.update(shard["sources"])
            turns.update(shard["assistant_turn_counts"])
        lengths = np.concatenate([np.load(OUTPUT / shard["path"] / "conversation_lengths.npy") for shard in selected])
        split_stats[split] = {"original_split": author_split,
            **{key: sum(row[key] for row in selected) for key in ["original_conversations", "conversations",
                "excluded_length_ge_2048", "windows", "total_tokens", "supervised_tokens", "useful_input_positions"]},
            "sources": dict(sources), "assistant_turn_counts": dict(turns),
            "conversation_length": {"min": int(lengths.min()), "mean": float(lengths.mean()),
                "median": float(np.median(lengths)), "p95": float(np.quantile(lengths, .95)), "max": int(lengths.max())},
            "shards": [row["path"] for row in selected]}
    manifest = {"status": "prepared_training_not_started", "format": "smoltalk_mmap_v1",
        "dataset": original["dataset"], "dataset_url": original["dataset_url"],
        "dataset_revision": original["dataset_revision"], "source_directory": str(Path("..") / BASE.relative_to(ROOT.parent)),
        "source_manifest_sha256": file_sha256(BASE / "manifest.json"),
        "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "tokenizer_sha256": original["tokenizer_sha256"], "eos_id": original["eos_id"],
        "max_length": 2048, "overlap": 0,
        "selection": "Read all author train/test rows and all sources; keep every complete serialized conversation with length strictly below 2048. No sampling, deduplication, single-turn restriction or answer-token cutoff.",
        "chat_format": original["chat_format"], "loss": original["loss"],
        "windowing": "One complete conversation per example; no windowing, overlap, truncation or cross-conversation packing.",
        "boundary": "Rows with length >=2048 are excluded, not shortened. Original author answer quality and train/test overlaps are not re-audited. Existing token/mask storage is shared read-only; only the selected index is used by training.",
        "splits": split_stats, "shards": shards, "training_started": False}
    write_json(OUTPUT / "manifest.json", manifest)
    shutil.copy2(BASE / "source_manifest.json", OUTPUT / "source_manifest.json")
    print(json.dumps(split_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
