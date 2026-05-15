#!/usr/bin/env python
# coding=utf-8
"""
Build PMET-style rewrite requests from DEPN true-prefix SFT JSON.

Each request unlearns the memorized secret by editing the model to prefer
`target_new` (redacted) over `target_true` (original secret) at the
prefix-completion boundary.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INSTRUCTION = "请根据给定的前缀文本，顺着往下补全缺失的信息。"


def build_llama3_chat_prefix(instruction: str, input_text: str) -> str:
    user = f"{instruction.strip()}\n{input_text.strip()}".strip()
    bos = "<|begin_of_text|>"
    uhdr = "<|start_header_id|>user<|end_header_id|>\n\n"
    ahdr = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    eot = "<|eot_id|>"
    return bos + uhdr + user + eot + ahdr


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_spans(record: dict[str, Any]) -> list[tuple[int, int, str]]:
    masks = record.get("privacy_mask")
    spans: list[tuple[int, int, str]] = []
    if isinstance(masks, list):
        for item in masks:
            if not isinstance(item, dict):
                continue
            start, end, label = item.get("start"), item.get("end"), item.get("label")
            if isinstance(start, int) and isinstance(end, int) and label:
                spans.append((start, end, str(label).upper()))
        if spans:
            return spans
    raw = record.get("span_labels")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            raw = None
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                start, end, label = item
                if isinstance(start, int) and isinstance(end, int) and label != "O":
                    spans.append((start, end, str(label).upper()))
    return spans


def build_reference_label_map(reference_jsonl: Path) -> dict[tuple[str, str], str]:
    label_map: dict[tuple[str, str], str] = {}
    if not reference_jsonl.is_file():
        return label_map
    for record in iter_jsonl(reference_jsonl):
        source_text = str(record.get("source_text", ""))
        if not source_text:
            continue
        for start, end, label in iter_spans(record):
            if not (0 <= start < end <= len(source_text)):
                continue
            prefix = source_text[max(0, start - 500) : start]
            if len(prefix.strip()) < 5:
                continue
            target = source_text[start:end]
            label_map[(prefix, target)] = label
    return label_map


def row_to_request(
    row: dict[str, Any],
    case_id: int,
    *,
    instruction: str,
    redact_target: str,
    label_map: dict[tuple[str, str], str],
    verified_pii_type: str | None,
) -> dict[str, Any] | None:
    inp = str(row.get("input", "")).strip()
    secret = str(row.get("output", "")).strip()
    if not inp or not secret:
        return None

    if verified_pii_type is not None:
        pii_type = verified_pii_type
    else:
        pii_type = str(
            row.get("pii_type")
            or row.get("label")
            or label_map.get((inp, secret), "UNKNOWN")
        ).upper()

    completion_context = build_llama3_chat_prefix(instruction, inp)
    return {
        "case_id": case_id,
        "pii_type": pii_type,
        "requested_rewrite": {
            "prompt": "{}",
            "subject": inp,
            "relation_id": "PRIVACY",
            "target_new": {"str": redact_target if redact_target.startswith(" ") else f" {redact_target}"},
            "target_true": {"str": secret if secret.startswith(" ") else f" {secret}"},
        },
        "completion_context": completion_context,
        "raw_input": inp,
        "raw_output": secret,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PMET privacy rewrite JSON from SFT prefix data.")
    parser.add_argument("--dataset", type=str, required=True, help="SFT JSON list (instruction/input/output).")
    parser.add_argument(
        "--privacy_only",
        action="store_true",
        default=True,
        help="仅保留 reference 中可匹配的 (input,output)（默认开启）。",
    )
    parser.add_argument(
        "--no_privacy_only",
        action="store_false",
        dest="privacy_only",
        help="关闭校验：允许未在 reference 中出现的样本（不推荐）。",
    )
    parser.add_argument(
        "--reference_jsonl",
        type=str,
        default=None,
        help="english_pii jsonl；privacy_only 时强烈建议提供。",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pii_types", type=str, default=None, help="Comma-separated filter, e.g. EMAIL,PHONENUMBER.")
    parser.add_argument("--limit_per_type", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--redact_target", type=str, default="[REDACTED]")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("dataset must be a JSON list")

    label_map = build_reference_label_map(Path(args.reference_jsonl)) if args.reference_jsonl else {}
    if args.privacy_only and not label_map:
        raise SystemExit(
            "privacy_only=True 但未提供可用的 --reference_jsonl（或文件为空）。"
            "请提供 english_pii_43k.jsonl，或使用 --no-privacy_only。"
        )
    requested = {x.strip().upper() for x in args.pii_types.split(",")} if args.pii_types else None
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    by_type: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    case_id = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        inp = str(row.get("input", "")).strip()
        secret = str(row.get("output", "")).strip()
        verified: str | None = None
        if args.privacy_only:
            verified = label_map.get((inp, secret))
            if verified is None:
                continue
        req = row_to_request(
            row,
            case_id,
            instruction=args.instruction,
            redact_target=args.redact_target,
            label_map=label_map,
            verified_pii_type=verified,
        )
        if req is None:
            continue
        pii_type = req["pii_type"]
        if requested and pii_type not in requested:
            continue
        if args.limit_per_type > 0 and by_type.get(pii_type, 0) >= args.limit_per_type:
            continue
        out.append(req)
        by_type[pii_type] = by_type.get(pii_type, 0) + 1
        case_id += 1
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break

    out.sort(key=lambda x: (x["pii_type"], x["case_id"]))
    for i, item in enumerate(out):
        item["case_id"] = i

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} requests to {output_path}")
    for pii_type in sorted(by_type):
        print(f"  {pii_type}: {by_type[pii_type]}")


if __name__ == "__main__":
    main()
