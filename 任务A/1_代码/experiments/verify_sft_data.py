"""独立核对 SFT 答案、标签边界和 train/validation/test 重叠。"""

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from tokenizer.tokenizer import Tokenizer


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def variable_values(context):
    values = {}
    for name, reference, number in re.findall(r"VAR ([A-Z]+) = (?:VAR ([A-Z]+)|(\d+))", context):
        values[name] = values[reference] if reference else number
    return values


def read_and_verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    tokenizer_dir = Path(manifest["tokenizer_dir"])
    tokenizer = Tokenizer.from_files(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.json"), ["<|endoftext|>"])
    eos_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    fingerprints = {name: set() for name in ["prompt", "full_example", "context", "answer_list"]}
    facts = {task: set() for task in ["mq_niah", "fwe", "vt"]}
    components = {name: set() for name in ["mq_keys", "mq_numbers", "fwe_words", "vt_names", "vt_values"]}
    groups = []
    for group in manifest["groups"]:
        count, min_length, max_length = 0, 10**9, 0
        local_prompts = set()
        for line in (directory / group["path"]).open():
            record = json.loads(line)
            task = record["task"]
            prompt = record["input"] + record["answer_prefix"]
            assert tokenizer.decode(record["input_ids"]) == prompt
            assert len(record["input_ids"]) + record["tokens_to_generate"] <= record["seq_len"]
            inputs, labels = record["sft_input_ids"], record["sft_labels"]
            assert len(inputs) == len(labels) < record["seq_len"]
            assert labels[-1] == eos_id
            full_ids = inputs + [eos_id]
            first = next(index for index, target in enumerate(labels) if target != -100)
            assert tokenizer.decode(full_ids[:first + 1]) == record["sft_prompt"]
            assert tokenizer.decode(full_ids[first + 1:]) == record["completion"]
            assert all(target == -100 for target in labels[:first])
            assert labels[first:] == full_ids[first + 1:]
            assert record["completion"] == " " + " ".join(record["outputs"]) + "<|endoftext|>"

            if task == "mq_niah":
                pairs = re.findall(r"One of the special magic numbers for (.+?) is: (\d{7})\.", record["context"])
                assert len(pairs) == 4
                mapping = dict(pairs)
                assert record["outputs"] == [mapping[key] for key in record["metadata"]["query_keys"]]
                facts[task].update(pairs)
                components["mq_keys"].update(mapping)
                components["mq_numbers"].update(mapping.values())
            elif task == "fwe":
                counts = Counter(record["context"].split())
                counts.pop("...")
                expected = sorted(counts, key=lambda word: (-counts[word], word))[:3]
                assert record["outputs"] == expected
                facts[task].add(tuple(sorted(counts)))
                components["fwe_words"].update(counts)
            else:
                values = variable_values(record["context"])
                assert len(values) == 14
                expected = [name for name, value in values.items() if value == record["metadata"]["query_value"]]
                assert record["outputs"] == expected and len(expected) == 7
                facts[task].update(("resolved_value", name, value) for name, value in values.items())
                facts[task].update(("assignment", name, reference or number) for name, reference, number in
                                   re.findall(r"VAR ([A-Z]+) = (?:VAR ([A-Z]+)|(\d+))", record["context"]))
                components["vt_names"].update(values)
                components["vt_values"].update(values.values())
                example = record["metadata"]["in_context_example"]
                example_values = variable_values(example)
                query = re.search(r"Question: Find all variables that are assigned the value (\d+)", example).group(1)
                example_answers = example.rsplit("they are:", 1)[1].split()
                assert set(example_answers) == {name for name, value in example_values.items() if value == query}

            hashed_prompt = fingerprint(prompt)
            assert hashed_prompt not in local_prompts
            local_prompts.add(hashed_prompt)
            fingerprints["prompt"].add(hashed_prompt)
            fingerprints["full_example"].add(fingerprint(record["sft_prompt"] + record["completion"]))
            fingerprints["context"].add(fingerprint(record["context"]))
            fingerprints["answer_list"].add(fingerprint(task + "|" + " ".join(record["outputs"])))
            count += 1
            min_length, max_length = min(min_length, len(full_ids)), max(max_length, len(full_ids))
        assert count == group["samples"]
        groups.append({"task": group["task"], "seq_len": group["seq_len"], "samples": count,
                       "min_sft_total_tokens": min_length, "max_sft_total_tokens": max_length})
    return manifest, groups, fingerprints, facts, components


def main(args):
    directories = {"train": args.train_dir, "validation": args.validation_dir, "test": args.test_dir}
    checked = {name: read_and_verify(directory) for name, directory in directories.items()}
    output = {"all_checks_passed": True, "splits": {}, "overlap": {}, "shared_components": {}}
    for name, (manifest, groups, *_rest) in checked.items():
        assert manifest["split"] == name
        if name != "test":
            assert manifest["lengths"] == [2048]
        output["splits"][name] = {"directory": str(directories[name]), "seed": manifest["generation_seed"], "groups": groups,
            "sha256": {group["path"]: hashlib.sha256((directories[name] / group["path"]).read_bytes()).hexdigest() for group in manifest["groups"]}}
    assert len({row[0]["generation_seed"] for row in checked.values()}) == 3
    for left, right in combinations(checked, 2):
        overlap = {key: len(checked[left][2][key] & checked[right][2][key]) for key in checked[left][2]}
        overlap["task_facts"] = {task: len(checked[left][3][task] & checked[right][3][task]) for task in checked[left][3]}
        assert not any(overlap[key] for key in ["prompt", "full_example", "context", "answer_list"])
        assert not any(overlap["task_facts"].values())
        output["overlap"][f"{left}_vs_{right}"] = overlap
        output["shared_components"][f"{left}_vs_{right}"] = {
            key: len(checked[left][4][key] & checked[right][4][key]) for key in checked[left][4]}
    output["sharing_policy"] = "Author task templates, VT noise/few-shot format and essay background corpus are shared by design. Individual variable values or words may recur; check complete examples and keyed facts to detect answer leakage. Test answers are never used as SFT targets."
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "overlap": output["overlap"], "shared_components": output["shared_components"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("experiments/outputs/task_sft_train_2048"))
    parser.add_argument("--validation-dir", type=Path, default=Path("experiments/outputs/task_sft_validation_2048"))
    parser.add_argument("--test-dir", type=Path, default=Path("experiments/outputs/task_data_ruler"))
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/task_sft_data_verification.json"))
    main(parser.parse_args())
