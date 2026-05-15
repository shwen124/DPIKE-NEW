#!/usr/bin/env python
# coding=utf-8
"""
Evaluate memorized sample counts for a LoRA-finetuned Llama model across multiple PII types.

Outputs:
- memorized_<PII_TYPE>.txt for each detected/evaluated PII type
- memorized_RANDOM.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

# Make `python data/eval_*.py` work from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_privacy_metrics_llama_lora import (
    build_blocks_from_text,
    collect_pii_samples,
    load_ft_model_4bit,
    load_tokenizer,
    parse_pii_types,
    reservoir_sample_lines,
    token_span_mrr,
)


@torch.no_grad()
def compute_ppl_one_block(model, input_ids: torch.Tensor) -> float:
    device = next(model.parameters()).device
    inp = input_ids.to(device)
    out = model(inp, labels=inp)
    return math.exp(out.loss.item())


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate memorized PII sample counts for a LoRA-finetuned Llama model."
    )
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--train_text_column", type=str, default=None)
    ap.add_argument("--all_tel_file", type=str, default=None)
    ap.add_argument("--all_name_file", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pii_types",
        type=str,
        default=None,
        help="Comma-separated canonical PII types to evaluate. Default: all detected types.",
    )
    ap.add_argument(
        "--pii_limit_per_type",
        type=int,
        default=5000,
        help="Maximum number of PII samples evaluated per type.",
    )
    ap.add_argument(
        "--pii_eval_max_context",
        type=int,
        default=256,
        help="Maximum number of tokens kept before the PII span when computing token-level MRR.",
    )
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--random_lines", type=int, default=5000)
    ap.add_argument("--random_blocks", type=int, default=5000)
    ap.add_argument(
        "--span_mrr_threshold",
        type=float,
        default=0.3,
        help="Span token-level MRR threshold used to count a sample as memorized.",
    )
    ap.add_argument("--random_threshold", type=float, default=2.5)
    return ap


def judge(low: int, high: int, count: int) -> str:
    if low <= count <= high:
        return "PASS"
    if count < low:
        return "FAIL: under-memorized"
    return "FAIL: over-memorized"


def main() -> None:
    args = build_argparser().parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.train_file).resolve().parent)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    print("[INFO] Loading tokenizer and 4-bit base+adapter model")
    tok = load_tokenizer(args.base_model_dir)
    model = load_ft_model_4bit(args.base_model_dir, args.adapter_dir)

    pii_types = parse_pii_types(args.pii_types)
    sample_map, _ = collect_pii_samples(
        args.train_file,
        train_text_column=args.train_text_column,
        pii_types=pii_types,
        limit_per_type=args.pii_limit_per_type,
        seed=args.seed,
        all_tel_file=args.all_tel_file,
        all_name_file=args.all_name_file,
    )

    if not sample_map:
        raise SystemExit("No supported PII samples were detected from the training source.")

    counts: dict[str, int] = {}
    skipped_by_type: dict[str, int] = {}

    for pii_type, samples in sample_map.items():
        memorized_lines: list[str] = []
        skipped = 0
        for sample in samples:
            score = token_span_mrr(model, tok, sample, max_context_tokens=args.pii_eval_max_context)
            if score is None or not math.isfinite(score):
                skipped += 1
                continue
            if score >= args.span_mrr_threshold:
                memorized_lines.append(sample.full_text)

        path = out_dir / f"memorized_{pii_type}.txt"
        path.write_text("\n".join(memorized_lines) + ("\n" if memorized_lines else ""), encoding="utf-8")
        counts[pii_type] = len(memorized_lines)
        skipped_by_type[pii_type] = skipped
        print(
            f"[{pii_type}] memorized count: {counts[pii_type]} "
            f"(MRR >= {args.span_mrr_threshold}, skipped={skipped}) -> {path}"
        )

    random_path = out_dir / "memorized_RANDOM.txt"
    rand_lines = reservoir_sample_lines(
        args.train_file,
        k=args.random_lines,
        seed=args.seed,
        min_len=50,
        text_column=args.train_text_column,
    )
    rand_text = "\n".join(rand_lines)
    rand_blocks = build_blocks_from_text(
        tok, rand_text, max_seq_len=args.max_seq_len, max_blocks=args.random_blocks
    )
    memorized_random: list[str] = []
    for i in range(rand_blocks.size(0)):
        ppl = compute_ppl_one_block(model, rand_blocks[i : i + 1])
        if ppl <= args.random_threshold:
            memorized_random.append(tok.decode(rand_blocks[i].tolist(), skip_special_tokens=True))
    random_path.write_text(
        "\n".join(memorized_random) + ("\n" if memorized_random else ""), encoding="utf-8"
    )
    n_random = len(memorized_random)
    print(f"[RANDOM] memorized count: {n_random} (PPL <= {args.random_threshold}) -> {random_path}")

    low, high = 100, 10_000
    print("\n" + "=" * 60)
    print("Memorized data count check (target range: 100-10000)")
    print("=" * 60)
    for pii_type in sorted(counts):
        print(
            f"  {pii_type:<12} memorized_{pii_type}.txt: {counts[pii_type]:>6}  "
            f"{judge(low, high, counts[pii_type])}"
        )
    print(f"  {'RANDOM':<12} memorized_RANDOM.txt: {n_random:>6}  {judge(low, high, n_random)}")
    print("=" * 60)

    check_paths = " ".join(str(out_dir / f"memorized_{pii_type}.txt") for pii_type in sorted(counts))
    print(f"Check with: wc -l {check_paths} {out_dir / 'memorized_RANDOM.txt'}")

    ok = all(low <= count <= high for count in counts.values()) and low <= n_random <= high
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    main()
