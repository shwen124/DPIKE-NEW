#!/usr/bin/env python
# coding=utf-8
"""
Run PMET privacy unlearning on a fine-tuned Llama3-8B (merged LoRA) checkpoint.

Example (from edit/ directory):
  python evaluate_privacy.py \\
    --base_model_dir /data1/D-PIKE/models/llama3-8B/baseline \\
    --adapter_dir /data1/D-PIKE/checkpoints/depn/llama3-8b_lora/2026-05-13_api4_prefix_qlora \\
    --requests_json /data1/D-PIKE/data/depn/pmet_privacy_requests.json \\
    --hparams_fname meta_llama3-8B_4090.json \\
    --num_edits 1 \\
    --cumulative_edits \\
    --save_edited_dir /data1/D-PIKE/models/llama3-8b/pmet_privacy_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets.privacy_prefix import PrivacyPrefixDataset
from pmet import PMETHyperParams, apply_pmet_to_model
from util.eval_utils.eval_utils_privacy_prefix import compute_rewrite_quality_privacy_prefix
from util import nethook
from util.globals import HPARAMS_DIR, KV_DIR, RESULTS_DIR
from util.llama_utils import load_model_and_tokenizer


def chunks(arr: List, n: int):
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def record_to_request(record: dict) -> dict:
    return {
        "case_id": record["case_id"],
        **record["requested_rewrite"],
        "completion_context": record.get("completion_context"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PMET privacy editing for Llama3-8B prefix SFT models.")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None, help="QLoRA adapter to merge before editing.")
    parser.add_argument("--no_merge_lora", action="store_true", help="Edit base weights only (skip merge).")
    parser.add_argument("--requests_json", type=str, default="privacy_prefix_requests.json")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing requests JSON.")
    parser.add_argument("--hparams_fname", type=str, default="meta_llama3-8B_4090.json")
    parser.add_argument("--dataset_size_limit", type=int, default=None)
    parser.add_argument("--num_edits", type=int, default=1, help="每批并行编辑条数；4090 建议 1。")
    parser.add_argument("--use_cache", action="store_true", help="Cache PMET z vectors.")
    parser.add_argument("--cumulative_edits", action="store_true", help="Keep edits across batches (do not restore).")
    parser.add_argument(
        "--save_edited_dir",
        type=str,
        default=None,
        help="If set, save the cumulatively edited model here after all batches.",
    )
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--dir_name", type=str, default="PMET_PRIVACY")
    parser.add_argument("--continue_from_run", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.requests_json).parent
    hparams_path = HPARAMS_DIR / "PMET" / args.hparams_fname
    hparams = PMETHyperParams.from_json(hparams_path)

    if args.continue_from_run:
        run_dir = RESULTS_DIR / args.dir_name / args.continue_from_run
    else:
        alg_dir = RESULTS_DIR / args.dir_name
        alg_dir.mkdir(parents=True, exist_ok=True)
        ids = [int(p.name.split("_")[-1]) for p in alg_dir.iterdir() if p.name.split("_")[-1].isdigit()]
        run_id = 0 if not ids else max(ids) + 1
        run_dir = alg_dir / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(hparams_path, run_dir / "params.json")
    print(f"Results will be stored at {run_dir}")

    model, tok = load_model_and_tokenizer(
        args.base_model_dir,
        adapter_dir=args.adapter_dir,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        merge_lora=not args.no_merge_lora and bool(args.adapter_dir),
    )
    model_name = model.config._name_or_path.replace("/", "_")

    ds = PrivacyPrefixDataset(str(data_dir), json_name=Path(args.requests_json).name, size=args.dataset_size_limit)

    cache_template = None
    if args.use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name}_PMET"
            / "privacy_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    for record_chunks in chunks(list(ds), args.num_edits):
        case_ids = [record["case_id"] for record in record_chunks]
        out_template = run_dir / f"{args.num_edits}_edits-case_{{}}.json"
        if all(out_template.format(record["case_id"]).exists() for record in record_chunks):
            print(f"Skipping batch {case_ids}; all outputs exist")
            continue

        requests = [record_to_request(record) for record in record_chunks]
        start = time()
        edited_model, weights_copy = apply_pmet_to_model(
            model,
            tok,
            requests,
            hparams,
            copy=False,
            return_orig_weights=not args.cumulative_edits,
            cache_template=str(cache_template) if cache_template else None,
        )
        exec_time = time() - start
        print(f"PMET batch {case_ids} took {exec_time:.1f}s")

        for record in record_chunks:
            out_file = out_template.format(record["case_id"])
            if out_file.exists():
                continue
            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": args.num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "pii_type": record.get("pii_type"),
                "time": exec_time,
                "post": compute_rewrite_quality_privacy_prefix(edited_model, tok, record),
            }
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, ensure_ascii=False)

        if not args.cumulative_edits and weights_copy:
            with torch.no_grad():
                for name, tensor in weights_copy.items():
                    param = nethook.get_parameter(model, name)
                    param[...] = tensor.to(param.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_edited_dir and args.cumulative_edits:
        save_dir = Path(args.save_edited_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"Saved cumulatively edited model to {save_dir}")


if __name__ == "__main__":
    main()
