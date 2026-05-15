#!/usr/bin/env python
# coding=utf-8
"""
Evaluate PII memorization and quality metrics for a base Llama model and a LoRA adapter.

This script now:
1. Separates base-model and adapter-model evaluation to avoid cross-contamination.
2. Extracts multiple PII categories from either dataset annotations (`privacy_mask`) or regex fallbacks.
3. Reports token-level span MRR for each PII category, plus RANDOM/VALID perplexity.

Supported PII categories include:
- NAME
- TEL
- EMAIL
- ID_CARD
- DRIVER_LICENSE
- PASSPORT
- ACCOUNT_NUMBER
- VEHICLE_VIN
- IP_ADDRESS
- ADDRESS
- USERNAME
- PASSWORD
- DEVICE_ID
- CARD_NUMBER
- DOB
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".json", ".csv"}
TEXT_COLUMN_CANDIDATES = ("text", "source_text", "content", "body")

NAME_ALIAS_LABELS = {
    "NAME",
    "FULLNAME",
    "FIRSTNAME",
    "MIDDLENAME",
    "LASTNAME",
    "SURNAME",
    "FAMILYNAME",
    "PERSONNAME",
    "PREFIX",
}
TEL_ALIAS_LABELS = {
    "TEL",
    "PHONE",
    "PHONENUMBER",
    "MOBILENUMBER",
    "MOBILEPHONE",
    "FAX",
}
EMAIL_ALIAS_LABELS = {"EMAIL"}
DOB_ALIAS_LABELS = {"DOB", "DATEOFBIRTH"}
ID_CARD_ALIAS_LABELS = {
    "IDCARD",
    "ID_CARD",
    "IDNUMBER",
    "ID_NUMBER",
    "SSN",
    "SIN",
    "AADHAR",
    "PAN",
    "NATIONALID",
    "NATIONALIDNUMBER",
    "TAXID",
}
DRIVER_LICENSE_ALIAS_LABELS = {
    "DRIVERLICENSE",
    "DRIVER_LICENSE",
    "DRIVERLICENSENUMBER",
    "DRIVER_LICENSE_NUMBER",
}
PASSPORT_ALIAS_LABELS = {"PASSPORT", "PASSPORTNUMBER"}
ACCOUNT_ALIAS_LABELS = {
    "ACCOUNTNUMBER",
    "ACCOUNT_NUMBER",
    "BANKACCOUNT",
    "BANKACCOUNTNUMBER",
    "IBAN",
    "SWIFT",
    "ROUTINGNUMBER",
}
VEHICLE_VIN_ALIAS_LABELS = {"VEHICLEVIN", "VIN"}
LICENSE_PLATE_ALIAS_LABELS = {"LICENSEPLATE", "PLATENUMBER"}
ADDRESS_ALIAS_LABELS = {
    "ADDRESS",
    "STREETADDRESS",
    "STREET",
    "CITY",
    "STATE",
    "ZIPCODE",
    "POSTALCODE",
    "COUNTRY",
}
IP_ALIAS_LABELS = {"IP", "IPV4", "IPV6", "IPADDRESS"}
USERNAME_ALIAS_LABELS = {"USERNAME"}
PASSWORD_ALIAS_LABELS = {"PASSWORD", "PIN"}
DEVICE_ID_ALIAS_LABELS = {"PHONEIMEI", "IMEI", "MACADDRESS", "UUID", "SERIALNUMBER"}
CARD_ALIAS_LABELS = {"CREDITCARDNUMBER", "DEBITCARDNUMBER", "CARDNUMBER", "CVV"}

PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\.\s\(\)]{6,}\d)(?!\w)")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
CHINA_ID_RE = re.compile(r"\b\d{17}[\dXx]\b")
GENERIC_SSN_RE = re.compile(r"\b\d{3}[.\-]?\d{2,4}[.\-]?\d{4}\b")
DRIVER_LICENSE_RE = re.compile(
    r"(?i)(?:driver(?:'s)? license(?: number)?|dl)\s*[:#]?\s*([A-Z0-9\-]{6,20})"
)
PASSPORT_RE = re.compile(r"\b[A-Z]\d{7,8}\b")
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
LICENSE_PLATE_RE = re.compile(r"\b[A-Z0-9]{1,3}[-\s]?[A-Z0-9]{3,6}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b")
ACCOUNT_RE = re.compile(r"\b\d{8,20}\b")


@dataclass(frozen=True)
class PIISample:
    pii_type: str
    secret: str
    prompt: str
    full_text: str
    raw_label: str | None = None


def canonicalize_label(raw_label: str | None) -> str | None:
    if not raw_label:
        return None
    label = str(raw_label).strip().upper()
    if label in NAME_ALIAS_LABELS:
        return "NAME"
    if label in TEL_ALIAS_LABELS:
        return "TEL"
    if label in EMAIL_ALIAS_LABELS:
        return "EMAIL"
    if label in DOB_ALIAS_LABELS:
        return "DOB"
    if label in ID_CARD_ALIAS_LABELS:
        return "ID_CARD"
    if label in DRIVER_LICENSE_ALIAS_LABELS:
        return "DRIVER_LICENSE"
    if label in PASSPORT_ALIAS_LABELS:
        return "PASSPORT"
    if label in ACCOUNT_ALIAS_LABELS:
        return "ACCOUNT_NUMBER"
    if label in VEHICLE_VIN_ALIAS_LABELS:
        return "VEHICLE_VIN"
    if label in LICENSE_PLATE_ALIAS_LABELS:
        return "LICENSE_PLATE"
    if label in ADDRESS_ALIAS_LABELS:
        return "ADDRESS"
    if label in IP_ALIAS_LABELS:
        return "IP_ADDRESS"
    if label in USERNAME_ALIAS_LABELS:
        return "USERNAME"
    if label in PASSWORD_ALIAS_LABELS:
        return "PASSWORD"
    if label in DEVICE_ID_ALIAS_LABELS:
        return "DEVICE_ID"
    if label in CARD_ALIAS_LABELS:
        return "CARD_NUMBER"
    return None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate base-vs-LoRA PII memorization metrics for Llama."
    )
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--valid_file", type=str, required=True)
    ap.add_argument("--train_text_column", type=str, default=None)
    ap.add_argument("--valid_text_column", type=str, default=None)
    ap.add_argument(
        "--all_tel_file",
        type=str,
        default=None,
        help="Optional legacy TEL file. If passed, it supplements extracted TEL samples.",
    )
    ap.add_argument(
        "--all_name_file",
        type=str,
        default=None,
        help="Optional legacy NAME file in 'text # name[# score]' format.",
    )
    ap.add_argument(
        "--pii_types",
        type=str,
        default=None,
        help="Comma-separated canonical PII types to evaluate. Default: all detected types.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--pii_limit_per_type", type=int, default=200)
    ap.add_argument(
        "--pii_eval_max_context",
        type=int,
        default=256,
        help="Maximum number of tokens kept before the PII span when computing token-level MRR.",
    )
    ap.add_argument("--random_lines", type=int, default=300)
    ap.add_argument("--random_blocks", type=int, default=300)
    ap.add_argument("--valid_blocks", type=int, default=400)
    return ap


def iter_source_paths(path_or_dir: str) -> list[Path]:
    base = Path(path_or_dir)
    if base.is_dir():
        return sorted(
            p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return [base]


def resolve_text_column(record: dict[str, Any], requested: str | None = None) -> str:
    if requested:
        if requested not in record:
            raise ValueError(
                f"Requested text column `{requested}` not found. Available columns: {', '.join(record.keys())}"
            )
        return requested
    for name in TEXT_COLUMN_CANDIDATES:
        if name in record:
            return name
    return next(iter(record.keys()))


def iter_records(path_or_dir: str, text_column: str | None = None) -> Iterable[dict[str, Any]]:
    for path in iter_source_paths(path_or_dir):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    text = line.strip()
                    if text:
                        yield {text_column or "text": text}
            continue

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
            continue

        if suffix == ".json":
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(payload, dict):
                if "data" in payload and isinstance(payload["data"], list):
                    for item in payload["data"]:
                        if isinstance(item, dict):
                            yield item
                else:
                    yield payload
            continue

        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield dict(row)


def make_sample(full_text: str, secret: str, pii_type: str, raw_label: str | None = None) -> PIISample | None:
    full_text = full_text.strip()
    secret = secret.strip()
    if not full_text or not secret or secret not in full_text:
        return None
    prompt = full_text.replace(secret, "***", 1)
    if prompt == full_text:
        return None
    return PIISample(
        pii_type=pii_type,
        secret=secret,
        prompt=prompt,
        full_text=full_text,
        raw_label=raw_label,
    )


def make_span_sample(
    full_text: str,
    start: int,
    end: int,
    pii_type: str,
    raw_label: str | None = None,
) -> PIISample | None:
    if start < 0 or end <= start or end > len(full_text):
        return None
    secret = full_text[start:end]
    prompt = full_text[:start] + "***" + full_text[end:]
    if not secret.strip():
        return None
    return PIISample(
        pii_type=pii_type,
        secret=secret,
        prompt=prompt,
        full_text=full_text,
        raw_label=raw_label,
    )


def maybe_add_sample(sample_map: dict[str, list[PIISample]], seen: set[tuple[str, str, str]], sample: PIISample | None) -> None:
    if sample is None:
        return
    key = (sample.pii_type, sample.prompt, sample.secret)
    if key in seen:
        return
    seen.add(key)
    sample_map[sample.pii_type].append(sample)


def extract_samples_from_privacy_mask(
    record: dict[str, Any],
    text_column: str | None,
    sample_map: dict[str, list[PIISample]],
    seen: set[tuple[str, str, str]],
) -> bool:
    masks = record.get("privacy_mask")
    if not isinstance(masks, list):
        return False
    column = resolve_text_column(record, text_column)
    full_text = str(record.get(column, "")).strip()
    if not full_text:
        return False

    extracted_any = False
    for item in masks:
        if not isinstance(item, dict):
            continue
        pii_type = canonicalize_label(item.get("label"))
        if pii_type is None:
            continue
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, int) and isinstance(end, int):
            sample = make_span_sample(full_text, start, end, pii_type, str(item.get("label")))
        else:
            sample = make_sample(full_text, str(item.get("value", "")), pii_type, str(item.get("label")))
        if sample is not None:
            maybe_add_sample(sample_map, seen, sample)
            extracted_any = True
    return extracted_any


def extract_samples_from_regex(
    text: str,
    sample_map: dict[str, list[PIISample]],
    seen: set[tuple[str, str, str]],
) -> None:
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        digit_count = sum(ch.isdigit() for ch in candidate)
        if digit_count >= 7:
            maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "TEL"))

    for match in EMAIL_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "EMAIL"))

    for match in CHINA_ID_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "ID_CARD"))

    for match in GENERIC_SSN_RE.finditer(text):
        secret = match.group(0)
        digit_count = sum(ch.isdigit() for ch in secret)
        if digit_count >= 9:
            maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "ID_CARD"))

    for match in DRIVER_LICENSE_RE.finditer(text):
        secret = match.group(1)
        maybe_add_sample(sample_map, seen, make_sample(text, secret, "DRIVER_LICENSE"))

    for match in PASSPORT_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "PASSPORT"))

    for match in VIN_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "VEHICLE_VIN"))

    for match in IPV4_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "IP_ADDRESS"))

    for match in IPV6_RE.finditer(text):
        maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "IP_ADDRESS"))

    for match in ACCOUNT_RE.finditer(text):
        secret = match.group(0)
        if 8 <= len(secret) <= 20:
            maybe_add_sample(sample_map, seen, make_span_sample(text, match.start(), match.end(), "ACCOUNT_NUMBER"))


def load_legacy_tel_samples(path: str) -> list[PIISample]:
    samples: list[PIISample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = None
            if "***" in line and "#" in line:
                left, _, right = line.rpartition("#")
                prompt = left.strip()
                secret = right.strip()
                full_text = prompt.replace("***", secret, 1)
                sample = PIISample("TEL", secret=secret, prompt=prompt, full_text=full_text, raw_label="TEL")
            else:
                match = PHONE_RE.search(line)
                if match:
                    sample = make_span_sample(line, match.start(), match.end(), "TEL", raw_label="TEL")
            if sample is not None:
                samples.append(sample)
    return samples


def load_legacy_name_samples(path: str) -> list[PIISample]:
    samples: list[PIISample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("#")]
            if len(parts) < 2:
                continue
            full_text, secret = parts[0], parts[1]
            sample = make_sample(full_text, secret, "NAME", raw_label="NAME")
            if sample is not None:
                samples.append(sample)
    return samples


def collect_pii_samples(
    train_file: str,
    train_text_column: str | None,
    pii_types: set[str] | None = None,
    limit_per_type: int | None = None,
    seed: int = 42,
    all_tel_file: str | None = None,
    all_name_file: str | None = None,
) -> tuple[dict[str, list[PIISample]], dict[str, Counter]]:
    rng = random.Random(seed)
    sample_map: dict[str, list[PIISample]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    raw_label_counter: dict[str, Counter] = defaultdict(Counter)

    for record in iter_records(train_file, text_column=train_text_column):
        extracted_from_mask = extract_samples_from_privacy_mask(record, train_text_column, sample_map, seen)
        text_column_name = resolve_text_column(record, train_text_column)
        text = str(record.get(text_column_name, "")).strip()
        if not text:
            continue
        if not extracted_from_mask:
            extract_samples_from_regex(text, sample_map, seen)

    if all_tel_file and os.path.exists(all_tel_file):
        for sample in load_legacy_tel_samples(all_tel_file):
            maybe_add_sample(sample_map, seen, sample)
    if all_name_file and os.path.exists(all_name_file):
        for sample in load_legacy_name_samples(all_name_file):
            maybe_add_sample(sample_map, seen, sample)

    filtered: dict[str, list[PIISample]] = {}
    for pii_type, samples in sample_map.items():
        if pii_types is not None and pii_type not in pii_types:
            continue
        rng.shuffle(samples)
        if limit_per_type is not None and limit_per_type > 0:
            samples = samples[:limit_per_type]
        filtered[pii_type] = samples
        for sample in samples:
            raw_label_counter[pii_type][sample.raw_label or pii_type] += 1
    return dict(sorted(filtered.items())), raw_label_counter


def reservoir_sample_lines(
    path_or_dir: str,
    k: int,
    seed: int,
    min_len: int = 1,
    text_column: str | None = None,
) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for record in iter_records(path_or_dir, text_column=text_column):
        text = str(record.get(resolve_text_column(record, text_column), "")).strip()
        if len(text) < min_len:
            continue
        seen += 1
        if len(reservoir) < k:
            reservoir.append(text)
        else:
            idx = rng.randint(0, seen - 1)
            if idx < k:
                reservoir[idx] = text
    return reservoir


def build_blocks_from_text(
    tokenizer,
    text: str,
    max_seq_len: int = 256,
    max_blocks: int | None = None,
) -> torch.Tensor:
    if not text.strip():
        return torch.empty((0, max_seq_len), dtype=torch.long)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < max_seq_len:
        return torch.empty((0, max_seq_len), dtype=torch.long)
    total_length = (len(token_ids) // max_seq_len) * max_seq_len
    token_ids = token_ids[:total_length]
    blocks = [token_ids[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
    if max_blocks is not None and max_blocks > 0:
        blocks = blocks[:max_blocks]
    if not blocks:
        return torch.empty((0, max_seq_len), dtype=torch.long)
    return torch.tensor(blocks, dtype=torch.long)


@torch.no_grad()
def compute_ppl(model, blocks: torch.Tensor) -> tuple[float | None, float | None]:
    if blocks.numel() == 0:
        return None, None
    device = next(model.parameters()).device
    losses: list[float] = []
    for i in range(blocks.size(0)):
        block = blocks[i : i + 1].to(device)
        outputs = model(block, labels=block)
        loss = float(outputs.loss.item())
        if math.isfinite(loss):
            losses.append(loss)
    if not losses:
        return None, None
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf")
    return mean_loss, ppl


def load_tokenizer(base_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _base_model_kwargs():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }


def load_base_model_4bit(base_model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, **_base_model_kwargs()).eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


def load_ft_model_4bit(base_model_dir: str, adapter_dir: str):
    from peft import PeftModel

    base = load_base_model_4bit(base_model_dir)
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


def _find_subsequence(sequence: list[int], subseq: list[int]) -> int:
    if not subseq or len(subseq) > len(sequence):
        return -1
    for idx in range(len(sequence) - len(subseq) + 1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return idx
    return -1


@torch.no_grad()
def token_span_mrr(
    model,
    tokenizer,
    sample: PIISample,
    max_context_tokens: int = 256,
) -> float | None:
    full_ids = tokenizer(sample.full_text, add_special_tokens=False)["input_ids"]
    secret_ids = tokenizer(sample.secret, add_special_tokens=False)["input_ids"]
    if not secret_ids:
        return None
    start = _find_subsequence(full_ids, secret_ids)
    if start <= 0:
        return None
    end = start + len(secret_ids)

    # For a causal LM, the rank of secret tokens depends only on the prefix before them.
    # Keeping only a short prefix window avoids OOM on very long training samples.
    window_start = max(0, start - max(0, max_context_tokens))
    local_ids = full_ids[window_start:end]
    local_start = start - window_start

    device = next(model.parameters()).device
    input_ids = torch.tensor([local_ids], dtype=torch.long, device=device)
    try:
        outputs = model(input_ids, use_cache=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    logits = outputs.logits[0]

    reciprocal_ranks: list[float] = []
    for pos in range(local_start, len(local_ids)):
        prev_pos = pos - 1
        token_id = local_ids[pos]
        token_logits = logits[prev_pos]
        target_score = token_logits[token_id]
        rank = int((token_logits > target_score).sum().item()) + 1
        reciprocal_ranks.append(1.0 / rank)
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def format_metric_stats(scores: list[float], skipped: int) -> str:
    if not scores:
        return f"n=0 skipped={skipped}"
    return (
        f"n={len(scores)} skipped={skipped} "
        f"mean={statistics.fmean(scores):.3f} "
        f"p50={quantile(scores, 0.50):.3f} "
        f"p90={quantile(scores, 0.90):.3f} "
        f"max={max(scores):.3f}"
    )


def evaluate_pii_metrics(
    model,
    tokenizer,
    sample_map: dict[str, list[PIISample]],
    max_context_tokens: int = 256,
) -> dict[str, tuple[list[float], int]]:
    metrics: dict[str, tuple[list[float], int]] = {}
    for pii_type, samples in sample_map.items():
        scores: list[float] = []
        skipped = 0
        for sample in samples:
            score = token_span_mrr(model, tokenizer, sample, max_context_tokens=max_context_tokens)
            if score is None or not math.isfinite(score):
                skipped += 1
                continue
            scores.append(score)
        metrics[pii_type] = (scores, skipped)
    return metrics


def parse_pii_types(arg: str | None) -> set[str] | None:
    if not arg:
        return None
    return {item.strip().upper() for item in arg.split(",") if item.strip()}


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.base_model_dir)
    pii_types = parse_pii_types(args.pii_types)

    print("[INFO] Collecting PII samples from train source")
    sample_map, raw_label_counter = collect_pii_samples(
        args.train_file,
        train_text_column=args.train_text_column,
        pii_types=pii_types,
        limit_per_type=args.pii_limit_per_type,
        seed=args.seed,
        all_tel_file=args.all_tel_file,
        all_name_file=args.all_name_file,
    )
    if not sample_map:
        print("[WARN] No supported PII samples were detected from the training source.")
    for pii_type, samples in sample_map.items():
        raw_summary = ", ".join(
            f"{label}:{count}" for label, count in raw_label_counter.get(pii_type, Counter()).most_common(4)
        )
        print(f"[INFO] {pii_type} samples: {len(samples)}" + (f" (labels: {raw_summary})" if raw_summary else ""))

    print(f"[INFO] Sampling random lines from train: k={args.random_lines}")
    random_lines = reservoir_sample_lines(
        args.train_file,
        k=args.random_lines,
        seed=args.seed,
        min_len=50,
        text_column=args.train_text_column,
    )
    random_blocks = build_blocks_from_text(
        tokenizer,
        "\n".join(random_lines),
        max_seq_len=args.max_seq_len,
        max_blocks=args.random_blocks,
    )

    valid_lines = reservoir_sample_lines(
        args.valid_file,
        k=max(args.valid_blocks, 200),
        seed=args.seed,
        min_len=20,
        text_column=args.valid_text_column,
    )
    valid_blocks = build_blocks_from_text(
        tokenizer,
        "\n".join(valid_lines),
        max_seq_len=args.max_seq_len,
        max_blocks=args.valid_blocks,
    )

    print("[INFO] Loading base model (4-bit)")
    base_model = load_base_model_4bit(args.base_model_dir)
    base_metrics = evaluate_pii_metrics(
        base_model,
        tokenizer,
        sample_map,
        max_context_tokens=args.pii_eval_max_context,
    )
    base_random_loss, base_random_ppl = compute_ppl(base_model, random_blocks)
    base_valid_loss, base_valid_ppl = compute_ppl(base_model, valid_blocks)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] Loading adapter model on top of a fresh base")
    ft_model = load_ft_model_4bit(args.base_model_dir, args.adapter_dir)
    ft_metrics = evaluate_pii_metrics(
        ft_model,
        tokenizer,
        sample_map,
        max_context_tokens=args.pii_eval_max_context,
    )
    ft_random_loss, ft_random_ppl = compute_ppl(ft_model, random_blocks)
    ft_valid_loss, ft_valid_ppl = compute_ppl(ft_model, valid_blocks)

    for pii_type in sample_map:
        base_scores, base_skipped = base_metrics.get(pii_type, ([], 0))
        ft_scores, ft_skipped = ft_metrics.get(pii_type, ([], 0))
        print(f"[{pii_type}][base] {format_metric_stats(base_scores, base_skipped)}")
        print(f"[{pii_type}][ft(adapter)] {format_metric_stats(ft_scores, ft_skipped)}")

    if base_random_ppl is not None:
        print(f"[RANDOM] base_ppl={base_random_ppl:.4f} (loss={base_random_loss:.4f})")
    else:
        print("[RANDOM] base_ppl unavailable")
    if ft_random_ppl is not None:
        print(f"[RANDOM] ft_ppl  ={ft_random_ppl:.4f} (loss={ft_random_loss:.4f})")
    else:
        print("[RANDOM] ft_ppl unavailable")

    if base_valid_ppl is not None:
        print(f"[VALID] base_ppl={base_valid_ppl:.4f} (loss={base_valid_loss:.4f}) blocks={valid_blocks.size(0)}")
    else:
        print("[VALID] base_ppl unavailable")
    if ft_valid_ppl is not None:
        print(f"[VALID] ft_ppl  ={ft_valid_ppl:.4f} (loss={ft_valid_loss:.4f}) blocks={valid_blocks.size(0)}")
    else:
        print("[VALID] ft_ppl unavailable")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    main()
