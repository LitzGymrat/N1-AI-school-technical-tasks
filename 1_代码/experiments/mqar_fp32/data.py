"""Reuse the checked MQAR cache generator without introducing distance metrics."""

import argparse
import copy
import json
from pathlib import Path

from experiments.mqar.data import atomic_json, prepare as prepare_cache


def prepare(config, directory):
    generation = copy.deepcopy(config)
    # The legacy cache helper requires this metadata threshold; no distance
    # filtering changes the generated inputs or the original MQAR supervision.
    generation["evaluation"]["far_distance_min"] = 65535
    manifest = prepare_cache(generation, directory)
    for split in manifest["splits"].values():
        split.pop("far_queries", None)
    atomic_json(Path(directory) / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out_dir)
