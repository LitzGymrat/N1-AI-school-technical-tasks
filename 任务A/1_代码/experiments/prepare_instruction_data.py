"""从作者 Smol-SmolTalk 抽取通用指令数据，适配自己的 OWT tokenizer。"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from tokenizer.tokenizer import Tokenizer
from experiments.train_owt import TOKENIZER_DIR

DATASET = "HuggingFaceTB/smol-smoltalk"
API = "https://datasets-server.huggingface.co"
END_OF_TEXT = "<|endoftext|>"
# 使用作者已有的通用指令来源，第一轮不选代码和复杂知识推理来源。
SOURCES = {"smol-magpie-ultra-short", "smol-contraints", "smollm-rewrite-30k", "smol-summarize-20k"}
TASK_MARKERS = ["special magic number", "track the frequency of each coded word",
                "memorize and track the chain(s) of variable assignment",
                "find all variables that are assigned the value"]


def get_json(url, **params):
    # 数据服务出现过 502；仅重试这类临时服务错误，不忽略失败的页。
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code not in [502, 503, 504]:
            break
        if attempt < 2:
            time.sleep(2)
    response.raise_for_status()
    return response.json()


def normalized(text):
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render_prompt(messages):
    # 普通文本角色标记，不给现有 32K 词表增加 ChatML special tokens。
    return "\n\n".join(f"{message['role'].capitalize()}: {message['content']}" for message in messages) + "\n\nAssistant:"


def fetch_candidates(split, num_rows, seed, cache_dir, pages):
    rng = random.Random(seed)
    offsets = rng.sample(range(0, num_rows, 100), pages)

    def fetch(offset):
        path = cache_dir / f"{split}_{offset}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = get_json(API + "/rows", dataset=DATASET, config="default", split=split,
                            offset=offset, length=min(100, num_rows - offset))
            assert all(not row["truncated_cells"] for row in data["rows"])
            path.write_text(json.dumps(data, ensure_ascii=False) + "\n")
            # 数据服务刚触发过限流；串行取页并间隔一秒，缓存页不再请求。
            time.sleep(1)
        return data["rows"]

    candidates = []
    for index, offset in enumerate(offsets, 1):
        candidates.extend(fetch(offset))
        if index % 10 == 0:
            print(f"{split}: fetched {index}/{pages} pages", flush=True)
    rng.shuffle(candidates)
    return candidates, offsets


def read_parquet_candidates(path, seed, revision):
    import pyarrow.parquet as pq

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["dataset_revision"] == revision
    assert path.stat().st_size == metadata["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    candidates = [{"row_idx": index, "row": row}
                  for index, row in enumerate(pq.read_table(path).to_pylist())]
    random.Random(seed).shuffle(candidates)
    return candidates, metadata


def read_benchmark(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    prompts, examples = set(), set()
    for group in manifest["groups"]:
        for line in (data_dir / group["path"]).open():
            record = json.loads(line)
            prompts.add(digest(normalized(record["input"] + record["answer_prefix"])))
            examples.add(digest(normalized(record["sft_prompt"] + record["completion"])))
    return prompts, examples


def make_record(candidate, split, tokenizer, max_length, benchmark, seen_users, seen_examples):
    raw = candidate["row"]
    if raw["source"] not in SOURCES:
        return None, "source_not_selected"
    messages = raw["messages"]
    roles = [message["role"] for message in messages]
    if roles not in [["user", "assistant"], ["system", "user", "assistant"]]:
        return None, "not_complete_single_turn"
    if any(not message["content"].strip() for message in messages):
        return None, "empty_message"
    original_text = "\n".join(message["content"] for message in messages)
    lower = original_text.casefold()
    if (any(marker in lower for marker in TASK_MARKERS)
            or re.search(r"\bVAR\s+[A-Z]{3,5}\s*=\s*(?:VAR\s+[A-Z]{3,5}|\d{5})", original_text)
            or re.search(r"\b(?:RULER|MQ-NIAH)\b", original_text, re.I)):
        return None, "benchmark_template"
    if END_OF_TEXT in original_text:
        return None, "embedded_eos"
    user = next(message["content"] for message in messages if message["role"] == "user")
    user_hash = digest(normalized(user))
    prompt = render_prompt(messages[:-1])
    answer = messages[-1]["content"]
    completion = "\n" + answer + END_OF_TEXT
    full_text = prompt + completion
    example_hash = digest(normalized(full_text))
    if user_hash in seen_users or example_hash in seen_examples:
        return None, "duplicate"
    if (digest(normalized(prompt)) in benchmark[0] or user_hash in benchmark[0]
            or example_hash in benchmark[1]):
        return None, "benchmark_overlap"
    answer_ids = tokenizer.encode(answer)
    if not 1 <= len(answer_ids) <= 256:
        return None, "answer_over_256_tokens"
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(full_text)
    if len(full_ids) > max_length:
        return None, "over_context_limit"
    assert full_ids[:len(prompt_ids)] == prompt_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    seen_users.add(user_hash)
    seen_examples.add(example_hash)
    return {
        "dataset": DATASET, "original_split": split, "original_row_index": candidate["row_idx"],
        "source": raw["source"], "messages": messages, "prompt": prompt, "answer": answer,
        "completion": completion, "prompt_tokens": len(prompt_ids), "answer_tokens": len(answer_ids),
        "total_tokens": len(full_ids), "source_row_sha256": digest(json.dumps(raw, ensure_ascii=False, sort_keys=True)),
        "user_sha256": user_hash, "example_sha256": example_hash,
        "sft_input_ids": full_ids[:-1], "sft_labels": labels[1:],
    }, None


def select(candidates, count, split, tokenizer, args, benchmark, seen_users, seen_examples):
    selected, rejected = [], Counter()
    for candidate in candidates:
        record, reason = make_record(candidate, split, tokenizer, args.context_len, benchmark,
                                     seen_users, seen_examples)
        if record is None:
            rejected[reason] += 1
        else:
            selected.append(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{split}: only {len(selected)}/{count} eligible records; increase --pages")
    return selected, dict(rejected)


def verify(records, tokenizer, max_length):
    eos_id = tokenizer.reversed_vocab[END_OF_TEXT.encode()]
    for row in records:
        full_ids = row["sft_input_ids"] + [eos_id]
        labels = row["sft_labels"]
        first = next(index for index, value in enumerate(labels) if value != -100)
        assert tokenizer.decode(full_ids) == row["prompt"] + row["completion"]
        assert tokenizer.decode(full_ids[:first + 1]) == row["prompt"]
        assert tokenizer.decode(full_ids[first + 1:]) == row["completion"]
        assert labels[first:] == full_ids[first + 1:]
        assert labels[-1] == eos_id
        assert len(labels) == len(row["sft_input_ids"]) < max_length


def write_review(output, groups, manifest):
    lines = ["Smol-SmolTalk 通用指令数据审核", "",
             f"{len(groups['train'])} 条训练 + {len(groups['validation'])} 条独立验证；完整保留所选单轮对话原文与答案。",
             "原作者 train 用于训练，test 仅用于本轮验证；上下文上限 2048，答案最多 256 tokens。",
             "普通文本 System/User/Assistant 标记；只监督回答和 EOS；SFT 标签已经错位一次。",
             "仅排除明确的 RULER 模板/赋值链，不把普通问答或常规文本处理当成泄漏。",
             "来源分布与长度：", json.dumps(manifest["splits"], ensure_ascii=False, indent=2), "",
             f"下面收录所有 {sum(len(rows) for rows in groups.values())} 条完整原文，不显示 token ID。"]
    for split, records in groups.items():
        for index, record in enumerate(records):
            lines += ["", "=" * 80,
                      f"{split} [{index}] | source={record['source']} | author {record['original_split']} row={record['original_row_index']}",
                      f"prompt={record['prompt_tokens']} tokens, answer={record['answer_tokens']} tokens, total={record['total_tokens']} tokens",
                      "完整输入：", record["prompt"], "", "原作者回答：", record["answer"]]
    output.write_text("\n".join(lines) + "\n")


def main(args):
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"{args.out_dir} already contains selected data")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "source_pages"
    cache_dir.mkdir(exist_ok=True)
    revision = get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]
    sizes = get_json(API + "/size", dataset=DATASET)["size"]["splits"]
    num_rows = {row["split"]: row["num_rows"] for row in sizes}
    tokenizer = Tokenizer.from_files(str(args.tokenizer_dir / "vocab.json"), str(args.tokenizer_dir / "merges.json"), [END_OF_TEXT])
    benchmark = read_benchmark(args.benchmark_dir)
    seen_users, seen_examples = set(), set()
    groups, rejected, fetched_offsets, source_files = {}, {}, {}, {}
    for source_split, output_split, count, seed in [("train", "train", args.train_samples, args.seed),
                                                   ("test", "validation", args.validation_samples, args.seed + 10000)]:
        pages = args.pages if source_split == "train" else args.validation_pages
        if source_split == "test" and args.validation_parquet:
            candidates, metadata = read_parquet_candidates(args.validation_parquet, seed, revision)
            assert len(candidates) == num_rows[source_split]
            offsets = []
            source_files[source_split] = metadata
        else:
            candidates, offsets = fetch_candidates(source_split, num_rows[source_split], seed, cache_dir, pages)
        groups[output_split], rejected[output_split] = select(candidates, count, source_split, tokenizer, args,
                                                            benchmark, seen_users, seen_examples)
        fetched_offsets[source_split] = offsets
        verify(groups[output_split], tokenizer, args.context_len)
        path = args.out_dir / f"{output_split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in groups[output_split])
        # 下载中断后恢复：已有选择必须完全相同，不覆盖不同版本的数据。
        if path.exists():
            assert path.read_text() == payload, f"Existing {path} differs from this selection"
        else:
            path.write_text(payload)
        print(f"{output_split}: selected {count}", flush=True)
    assert get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"] == revision
    manifest = {
        "dataset": DATASET, "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "Apache-2.0", "dataset_revision_checked_before_and_after": revision,
        "selection_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "seed": args.seed,
        "sampling": "Train: random disjoint 100-row pages; validation: shuffled original test Parquet when supplied, otherwise random pages. Filter and select first eligible; raw sources retained.",
        "source_files": source_files,
        "fetched_offsets": fetched_offsets, "sources_allowed": sorted(SOURCES), "rejected": rejected,
        "context_len": args.context_len, "answer_token_limit": 256,
        "chat_format": "plain text System/User/Assistant; no new special tokens; original messages and answers unchanged",
        "loss": "response and EOS only; labels shifted once; mean over valid response tokens",
        "tokenizer_sha256": {name: hashlib.sha256((args.tokenizer_dir / name).read_bytes()).hexdigest()
                             for name in ["vocab.json", "merges.json"]},
        "splits": {split: {"samples": len(rows), "sources": dict(Counter(row["source"] for row in rows)),
                            "min_total_tokens": min(row["total_tokens"] for row in rows),
                            "max_total_tokens": max(row["total_tokens"] for row in rows),
                            "total_tokens": sum(row["total_tokens"] for row in rows),
                            "supervised_tokens": sum(sum(value != -100 for value in row["sft_labels"]) for row in rows)}
                   for split, rows in groups.items()},
        "verification": {"all_selected_labels_verified": True, "train_validation_normalized_user_overlap": 0,
                         "train_validation_full_example_overlap": 0, "ruler_exact_prompt_overlap": 0,
                         "ruler_exact_full_example_overlap": 0, "ruler_samples_checked": 4500,
                         "boundary": "Exact/normalized dedup and task marker exclusion; not a guarantee of zero conceptual task overlap or no pretraining contamination."},
        "data_sha256": {split: hashlib.sha256((args.out_dir / f"{split}.jsonl").read_bytes()).hexdigest() for split in groups},
        "training_started": False,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_review(args.review_file, groups, manifest)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pages", type=int, default=80)
    parser.add_argument("--validation-pages", type=int, default=10)
    parser.add_argument("--validation-parquet", type=Path, help="原作者 test Parquet；旁边的 JSON 保存 revision、大小与 SHA-256")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("experiments/outputs/task_data_ruler"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/outputs/smol_smoltalk_1000"))
    parser.add_argument("--review-file", type=Path, default=Path(__file__).parent / "通用指令样本审核.txt")
    main(parser.parse_args())
