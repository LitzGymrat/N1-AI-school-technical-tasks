"""保留作者全部 Smol-SmolTalk 对话，生成共享的、可 mmap 的 SFT 数据。"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq
import regex
import requests

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT.parent / "2_实验和训练数据/training_data/tokenizer"
DATASET = "HuggingFaceTB/smol-smoltalk"
REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
EOS = "<|endoftext|>"
OVERLAP = 256
# 与当前用户 tokenizer 的 encode() 一致；不改动课程实现。
PAT = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def download_sources(out_dir):
    cache = out_dir / "source"
    cache.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/data",
                            params={"expand": "true"}, timeout=60)
    response.raise_for_status()
    files = sorted((row for row in response.json() if row["path"].endswith(".parquet")),
                   key=lambda row: row["path"])
    if len(files) != 5:
        raise ValueError(f"Expected four train shards and one test shard, got {len(files)}")
    sources = []
    for row in files:
        path = cache / Path(row["path"]).name
        expected = row["lfs"]["oid"]
        if not path.exists():
            previous = ROOT / "experiments/outputs/smol_smoltalk_1000/source_pages/test.parquet"
            if path.name.startswith("test-") and previous.exists() and sha256(previous) == expected:
                shutil.copy2(previous, path)
            else:
                url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{row['path']}"
                temp = path.with_suffix(".parquet.partial")
                for attempt in range(4):
                    try:
                        with requests.get(url, stream=True, timeout=(30, 120)) as download:
                            download.raise_for_status()
                            with temp.open("wb") as stream:
                                for chunk in download.iter_content(8 * 1024 * 1024):
                                    stream.write(chunk)
                        if temp.stat().st_size != row["size"] or sha256(temp) != expected:
                            raise ValueError(f"Download size/hash mismatch: {path.name}")
                        temp.replace(path)
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 3:
                            raise
                        time.sleep(2)
        if path.stat().st_size != row["size"] or sha256(path) != expected:
            raise ValueError(f"Cached source size/hash mismatch: {path.name}")
        info = {"path": str(path.relative_to(out_dir)), "original_path": row["path"],
                "bytes": row["size"], "sha256": expected,
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "original_split": "test" if path.name.startswith("test-") else "train"}
        sources.append(info)
        print(json.dumps({"downloaded": path.name, "rows": info["rows"], "bytes": info["bytes"]}), flush=True)
    write_json(out_dir / "source_manifest.json", {"dataset": DATASET, "revision": REVISION, "files": sources})
    return sources


class CachedTokenizer:
    """仅缓存预分词片段的原有 BPE 合并结果，保留原词表、规则和 token ID。"""

    def __init__(self, path):
        self.reference = Tokenizer.from_files(str(path / "vocab.json"), str(path / "merges.json"), [EOS])
        self.eos_id = self.reference.reversed_vocab[EOS.encode()]

    @lru_cache(maxsize=100000)
    def piece(self, text):
        pieces = tuple(bytes([byte]) for byte in text.encode("utf-8"))
        ranks = self.reference.merge_to_rank
        while len(pieces) > 1:
            pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
            candidates = [pair for pair in pairs if pair in ranks]
            if not candidates:
                break
            pair = min(candidates, key=ranks.__getitem__)
            pieces = self.reference.merge_pair(pieces, pair)
        return tuple(self.reference.reversed_vocab[piece] for piece in pieces)

    def encode(self, text):
        ids = []
        for section in regex.split("(" + regex.escape(EOS) + ")", text):
            if section == EOS:
                ids.append(self.eos_id)
            else:
                for match in PAT.finditer(section):
                    ids.extend(self.piece(match.group()))
        return ids


def encode_conversation(messages, tokenizer):
    ids, mask = [], []
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Unsupported author message schema: {role}")
        header = ("\n\n" if index else "") + role.capitalize() + ":"
        if role == "assistant":
            header_ids = tokenizer.encode(header)
            content_ids = tokenizer.encode("\n" + content + EOS)
            ids.extend(header_ids + content_ids)
            mask.extend([0] * len(header_ids) + [1] * len(content_ids))
        else:
            content_ids = tokenizer.encode(header + " " + content)
            ids.extend(content_ids)
            mask.extend([0] * len(content_ids))
    return np.asarray(ids, dtype="<u2"), np.asarray(mask, dtype="u1")


def prepare_shard(task):
    out_dir, source, tokenizer_dir, max_length, tokenizer_hashes = task
    out_dir, tokenizer_dir = Path(out_dir), Path(tokenizer_dir)
    name = Path(source["path"]).stem
    shard_dir = out_dir / "shards" / name
    stats_path = shard_dir / "stats.json"
    settings = {"source_sha256": source["sha256"], "tokenizer_sha256": tokenizer_hashes,
                "max_length": max_length, "overlap": OVERLAP, "format_version": 3}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if stats["settings"] != settings:
            raise ValueError(f"Existing shard uses different preparation settings: {name}")
        for file_name, expected in stats["sha256"].items():
            if sha256(shard_dir / file_name) != expected:
                raise ValueError(f"Existing prepared file hash mismatch: {name}/{file_name}")
        return stats
    shard_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CachedTokenizer(tokenizer_dir)
    if len(tokenizer.reference.vocab) > 65536:
        raise ValueError("uint16 storage requires vocabulary <= 65536")
    windows, lengths, sources = [], [], Counter()
    turn_counts = Counter()
    tokens_count = supervised_count = input_count = row_index = 0
    long_conversations = prompt_only_windows = embedded_eos = 0
    start_time = time.perf_counter()
    with (shard_dir / "tokens.bin").open("wb") as token_stream, \
            (shard_dir / "assistant_mask.bin").open("wb") as mask_stream, \
            (shard_dir / "rows.jsonl").open("w") as row_stream:
        for batch in pq.ParquetFile(out_dir / source["path"]).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                ids, mask = encode_conversation(row["messages"], tokenizer)
                if len(ids) < 2:
                    raise ValueError(f"Conversation has fewer than two serialized tokens: {name}/{row_index}")
                ids.tofile(token_stream)
                mask.tofile(mask_stream)
                turns = sum(message["role"] == "assistant" for message in row["messages"])
                turn_counts[turns] += 1
                embedded_eos += sum(EOS in message["content"] for message in row["messages"])
                sources[row["source"]] += 1
                lengths.append(len(ids))
                long_conversations += len(ids) - 1 > max_length
                emitted, covered, previous_end = 0, 0, 0
                # 输入 [begin,end)，目标 [begin+1,end+1)；重叠目标不重复监督。
                for begin in range(0, len(ids) - 1, max_length - OVERLAP):
                    end = min(begin + max_length, len(ids) - 1)
                    mask_prefix = max(0, previous_end - begin)
                    targets = int(mask[begin + 1 + mask_prefix:end + 1].sum())
                    if targets:
                        windows.append([tokens_count + begin, end - begin, mask_prefix, row_index])
                        emitted += 1
                        covered += targets
                        input_count += end - begin
                    else:
                        prompt_only_windows += 1
                    previous_end = end
                    if end == len(ids) - 1:
                        break
                if covered != int(mask[1:].sum()):
                    raise ValueError(f"Assistant token coverage mismatch: {name}, row {row_index}")
                supervised_count += covered
                payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
                row_stream.write(json.dumps({"original_row_index_in_shard": row_index,
                    "source": row["source"], "source_row_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    "conversation_tokens": len(ids), "assistant_turns": turns,
                    "supervised_tokens": covered, "training_windows": emitted}, ensure_ascii=False) + "\n")
                tokens_count += len(ids)
                row_index += 1
            if row_index % 4096 == 0:
                print(json.dumps({"shard": name, "rows": row_index, "total_rows": source["rows"],
                                  "seconds": round(time.perf_counter() - start_time, 1)}), flush=True)
    if row_index != source["rows"]:
        raise ValueError(f"Source row count mismatch: {name}")
    index_array = np.asarray(windows, dtype="<u8").reshape(-1, 4)
    np.save(shard_dir / "windows.npy", index_array)
    np.save(shard_dir / "conversation_lengths.npy", np.asarray(lengths, dtype="<u8"))
    stats = {"settings": settings, "name": name, "path": str(shard_dir.relative_to(out_dir)),
             "original_split": source["original_split"], "conversations": row_index,
             "sources": dict(sources), "assistant_turn_counts": dict(turn_counts),
             "windows": len(windows), "total_tokens": tokens_count,
             "supervised_tokens": supervised_count, "useful_input_positions": input_count,
             "over_context_conversations": long_conversations,
             "prompt_only_windows_omitted": prompt_only_windows,
             "messages_with_embedded_eos": embedded_eos,
             "preparation_seconds": time.perf_counter() - start_time,
             "sha256": {file_name: sha256(shard_dir / file_name) for file_name in
                        ["tokens.bin", "assistant_mask.bin", "windows.npy", "conversation_lengths.npy", "rows.jsonl"]}}
    write_json(stats_path, stats)
    print(json.dumps({"prepared": name, "conversations": row_index, "windows": len(windows),
                      "supervised_tokens": supervised_count}), flush=True)
    return stats


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_length <= OVERLAP:
        raise ValueError("max_length must be larger than overlap=256")
    sources = download_sources(args.out_dir)
    if args.download_only:
        return
    tokenizer_hashes = {name: sha256(args.tokenizer_dir / name) for name in ["vocab.json", "merges.json"]}
    jobs = [(str(args.out_dir), source, str(args.tokenizer_dir), args.max_length, tokenizer_hashes) for source in sources]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        shards = list(pool.map(prepare_shard, jobs))
    split_stats = {}
    for output_split, original_split in [("train", "train"), ("validation", "test")]:
        selected = [shard for shard in shards if shard["original_split"] == original_split]
        sources_count, turns = Counter(), Counter()
        for shard in selected:
            sources_count.update(shard["sources"])
            turns.update(shard["assistant_turn_counts"])
        lengths = np.concatenate([np.load(args.out_dir / shard["path"] / "conversation_lengths.npy") for shard in selected])
        split_stats[output_split] = {"original_split": original_split,
            **{key: sum(shard[key] for shard in selected) for key in
               ["conversations", "windows", "total_tokens", "supervised_tokens", "useful_input_positions",
                "over_context_conversations", "prompt_only_windows_omitted", "messages_with_embedded_eos"]},
            "sources": dict(sources_count), "assistant_turn_counts": dict(turns),
            "conversation_length": {"min": int(lengths.min()), "mean": float(lengths.mean()),
                "median": float(np.median(lengths)), "p95": float(np.quantile(lengths, .95)), "max": int(lengths.max())},
            "shards": [shard["path"] for shard in selected]}
    manifest = {"status": "prepared_training_not_started", "format": "smoltalk_mmap_v1",
        "dataset": DATASET, "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "dataset_revision": REVISION, "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_files": sources, "tokenizer_sha256": tokenizer_hashes, "eos_id": 256,
        "max_length": args.max_length, "overlap": OVERLAP,
        "selection": "All original train/test conversations and all sources; no sampling, deduplication, answer-length cutoff or single-turn filtering.",
        "chat_format": "Plain text System/User/Assistant; original content unchanged; newline before assistant content; existing EOS after every assistant turn; encode at role/assistant-content boundaries.",
        "windowing": "Each conversation independently; contiguous input windows with overlap=256; previously supervised overlap labels masked; every assistant target retained exactly once; prompt-only windows omitted.",
        "loss": "All assistant content, leading newline and end-of-turn EOS; system/user/header/padding labels -100; shifted exactly once.",
        "boundary": "Source size/hash, source row counts and token coverage checked during preparation. No benchmark, GPU, unit or generation tests run. Original author split overlaps and answer quality are not audited or modified.",
        "splits": split_stats, "shards": shards, "training_started": False}
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(split_stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "../2_实验和训练数据/training_data/smol_smoltalk_full")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--download-only", action="store_true")
    main(parser.parse_args())
