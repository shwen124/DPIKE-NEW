#!/usr/bin/env python
# coding=utf-8
"""
对比三层评估结果（base / sft / edited），生成宏平均、按 PII 类型、按大类的对比表与 delta 表。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PII_CATEGORY_GROUPS: dict[str, list[str]] = {
    "Numeric identifiers": [
        "PHONENUMBER", "PHONEIMEI", "MASKEDNUMBER", "ZIPCODE", "PIN", "SSN",
        "CREDITCARDCVV", "CREDITCARDNUMBER", "ACCOUNTNUMBER", "BUILDINGNUMBER", "AGE", "AMOUNT",
    ],
    "Random secrets": [
        "PASSWORD", "IBAN", "BIC", "MAC", "ETHEREUMADDRESS", "BITCOINADDRESS", "LITECOINADDRESS",
        "VEHICLEVIN", "VEHICLEVRM", "IPV6",
    ],
    "Network identifiers": ["IP", "IPV4", "IPV6", "MAC", "URL", "USERAGENT"],
    "Financial identifiers": [
        "CREDITCARDNUMBER", "CREDITCARDCVV", "CREDITCARDISSUER", "IBAN", "BIC", "ACCOUNTNUMBER",
        "CURRENCY", "CURRENCYNAME", "CURRENCYCODE", "CURRENCYSYMBOL", "AMOUNT",
    ],
    "Personal names": ["PREFIX", "FIRSTNAME", "LASTNAME", "MIDDLENAME", "USERNAME", "ACCOUNTNAME"],
    "Address/location": [
        "CITY", "STATE", "COUNTY", "STREET", "SECONDARYADDRESS", "BUILDINGNUMBER",
        "ORDINALDIRECTION", "NEARBYGPSCOORDINATE", "ZIPCODE",
    ],
    "Categorical attributes": [
        "GENDER", "SEX", "EYECOLOR", "CURRENCY", "CURRENCYNAME", "CURRENCYCODE", "CURRENCYSYMBOL",
        "CREDITCARDISSUER", "JOBTYPE", "JOBAREA", "ORDINALDIRECTION",
    ],
    "Professional/account entities": ["COMPANYNAME", "JOBTITLE", "JOBTYPE", "JOBAREA", "ACCOUNTNAME"],
}

MACRO_METRICS = [
    "nll_mean",
    "mrr_mean",
    "topk_accuracy_mean",
    "normalized_exact_match_mean",
    "exact_match_mean",
    "starts_with_target_mean",
    "normalized_starts_with_target_mean",
    "accuracy_mean",
    "exposure_mean",
    "char_exposure_mean",
    "edit_similarity_mean",
    "stopped_by_eos_mean",
    "generated_token_count_mean",
]

HIGHER_IS_LEAK = {
    "mrr_mean", "topk_accuracy_mean", "normalized_exact_match_mean", "exact_match_mean",
    "starts_with_target_mean", "normalized_starts_with_target_mean", "accuracy_mean",
    "exposure_mean", "char_exposure_mean", "edit_similarity_mean",
}
LOWER_IS_LEAK = {"nll_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base/sft/edited PII eval JSON outputs.")
    parser.add_argument("--base", required=True, help="base layer result JSON")
    parser.add_argument("--sft", required=True, help="sft layer result JSON")
    parser.add_argument("--edited", default=None, help="edited layer result JSON (optional)")
    parser.add_argument("--output_dir", required=True, help="Directory for comparison CSV/JSON")
    parser.add_argument("--split", default=None, help="Filter tag written into output filenames (e.g. train, test)")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("summary") or payload.get("summary_by_type") or []
    return {str(row["pii_type"]): row for row in rows}


def macro_average(summary: dict[str, dict[str, Any]], metric_suffix: str) -> float | None:
    key = metric_suffix if metric_suffix.endswith("_mean") else f"{metric_suffix}_mean"
    values = []
    for row in summary.values():
        if key in row and row[key] is not None:
            count = row.get("count", 1) or 1
            values.extend([float(row[key])] * int(count))
    if not values:
        return None
    return sum(values) / len(values)


def build_macro_row(layer: str, summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"layer": layer}
    for metric in MACRO_METRICS:
        row[metric] = macro_average(summary, metric)
    row["total_count"] = sum(int(v.get("count", 0)) for v in summary.values())
    return row


def category_aggregate(summary: dict[str, dict[str, Any]], category: str, types: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"category": category, "count": 0}
    for metric in MACRO_METRICS:
        vals = []
        for pii_type in types:
            if pii_type not in summary:
                continue
            key = metric
            if key in summary[pii_type] and summary[pii_type][key] is not None:
                vals.append(float(summary[pii_type][key]))
                row["count"] += int(summary[pii_type].get("count", 0))
        row[metric] = sum(vals) / len(vals) if vals else None
    return row


def delta_row(before: dict[str, Any], after: dict[str, Any], label: str) -> dict[str, Any]:
    row: dict[str, Any] = {"comparison": label}
    for metric in MACRO_METRICS:
        b, a = before.get(metric), after.get(metric)
        row[f"delta_{metric}"] = (a - b) if b is not None and a is not None else None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.split}" if args.split else ""

    layers = {"base": load_summary(Path(args.base)), "sft": load_summary(Path(args.sft))}
    if args.edited:
        layers["edited"] = load_summary(Path(args.edited))

    macro_rows = [build_macro_row(name, summary) for name, summary in layers.items()]
    write_csv(output_dir / f"table1_macro{tag}.csv", macro_rows)

    by_type_rows: list[dict[str, Any]] = []
    all_types = sorted({t for summary in layers.values() for t in summary})
    for pii_type in all_types:
        row: dict[str, Any] = {"pii_type": pii_type}
        for layer_name, summary in layers.items():
            if pii_type in summary:
                for key, value in summary[pii_type].items():
                    if key.endswith("_mean") or key in ("count", "metrics"):
                        row[f"{layer_name}_{key}"] = value
        by_type_rows.append(row)
    write_csv(output_dir / f"table2_by_pii_type{tag}.csv", by_type_rows)

    category_rows: list[dict[str, Any]] = []
    for category, types in PII_CATEGORY_GROUPS.items():
        for layer_name, summary in layers.items():
            agg = category_aggregate(summary, category, types)
            agg["layer"] = layer_name
            category_rows.append(agg)
    write_csv(output_dir / f"table3_by_category{tag}.csv", category_rows)

    delta_rows: list[dict[str, Any]] = []
    base_macro = build_macro_row("base", layers["base"])
    sft_macro = build_macro_row("sft", layers["sft"])
    delta_rows.append(delta_row(base_macro, sft_macro, "sft_minus_base"))
    if "edited" in layers:
        edited_macro = build_macro_row("edited", layers["edited"])
        delta_rows.append(delta_row(sft_macro, edited_macro, "edited_minus_sft"))
        delta_rows.append(delta_row(base_macro, edited_macro, "edited_minus_base"))
    write_csv(output_dir / f"table4_deltas{tag}.csv", delta_rows)

    report = {
        "macro": macro_rows,
        "deltas": delta_rows,
        "layers": list(layers.keys()),
        "split": args.split,
    }
    with (output_dir / f"comparison_report{tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote comparison tables to {output_dir}")


if __name__ == "__main__":
    main()
