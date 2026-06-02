#!/usr/bin/env python
# coding=utf-8
"""按 source_id 分组切分 SFT JSON（train/val/test），避免同一文档泄漏到不同 split。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split SFT JSON by source_id into train/val/test files.")
    parser.add_argument("--input", required=True, help="Input JSON list (e.g. sft_true_prefix_no_instruction.json).")
    parser.add_argument("--output_dir", required=True, help="Directory for *_train.json, *_val.json, *_test.json.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem", default=None, help="Output stem; default = input stem without .json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must be a JSON list.")

    by_source: dict[int | str, list[dict]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("source_id", "__missing__")
        by_source[sid].append(row)

    source_ids = list(by_source.keys())
    rng = random.Random(args.seed)
    rng.shuffle(source_ids)

    n = len(source_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_ids = set(source_ids[:n_train])
    val_ids = set(source_ids[n_train : n_train + n_val])
    test_ids = set(source_ids[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for sid, items in by_source.items():
        if sid in train_ids:
            split = "train"
        elif sid in val_ids:
            split = "val"
        else:
            split = "test"
        for item in items:
            enriched = dict(item)
            enriched["split"] = split
            splits[split].append(enriched)

    stem = args.stem or input_path.stem
    for split_name, items in splits.items():
        out_path = output_dir / f"{stem}_{split_name}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
        print(f"[INFO] {split_name}: {len(items)} examples -> {out_path} ({len(train_ids if split_name=='train' else val_ids if split_name=='val' else test_ids)} source_ids)")


if __name__ == "__main__":
    main()
