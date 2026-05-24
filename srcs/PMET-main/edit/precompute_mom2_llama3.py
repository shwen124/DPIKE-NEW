#!/usr/bin/env python
# coding=utf-8
"""
Precompute PMET mom2 caches (SecondMoment) for Llama3 MLP down_proj inputs.

Run from this directory (so util.globals / data/stats resolve correctly):
  cd srcs/PMET-main/edit
  python precompute_mom2_llama3.py --base_model_dir ... [--adapter_dir ...]

Uses rome.layer_stats.layer_stats; corpus from hparams mom2_dataset (wikitext recommended).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# cwd = edit/
_EDIT_ROOT = Path(__file__).resolve().parent
os.chdir(_EDIT_ROOT)
if str(_EDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EDIT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util.globals import HPARAMS_DIR, STATS_DIR
from util.nethook import set_requires_grad


def _load_model(base: str, adapter: str | None, dtype: str, device_map: str):
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dt,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    set_requires_grad(False, model)
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute mom2 .npz for Llama3 PMET (mlp.down_proj).")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None, help="QLoRA to merge (same as PMET edit).")
    parser.add_argument(
        "--hparams_json",
        type=str,
        default=str(HPARAMS_DIR / "PMET" / "meta_llama3-8B.json"),
        help="Read layers + mom2_dataset + mom2_n_samples + mom2_dtype from this file.",
    )
    parser.add_argument("--layers", type=str, default=None, help="Override layers, e.g. 4,5,6,7,8")
    parser.add_argument("--mom2_dataset", type=str, default=None, help="Override mom2_dataset (wikitext|wikipedia).")
    parser.add_argument("--sample_size", type=int, default=None, help="Override mom2_n_samples.")
    parser.add_argument("--precision", type=str, default=None, choices=["float32", "float64"])
    parser.add_argument("--stats_dir", type=str, default=str(STATS_DIR))
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--download_remote_stats", type=int, default=1, help="Try memit.baulab.info before local compute.")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument(
        "--mom2_text_file",
        type=str,
        default=None,
        help="Local UTF-8 .txt or .jsonl (text/source_text); enables offline mom2 without HuggingFace datasets.",
    )
    parser.add_argument(
        "--wikipedia_dir",
        type=str,
        default=None,
        help="Local train-*.parquet directory (default: <repo>/data/wikipedia if present).",
    )
    args = parser.parse_args()
    if args.mom2_text_file:
        os.environ["PMET_MOM2_TEXT_FILE"] = str(Path(args.mom2_text_file).resolve())
    if args.wikipedia_dir:
        os.environ["PMET_WIKIPEDIA_DIR"] = str(Path(args.wikipedia_dir).resolve())
    elif (_wiki := _EDIT_ROOT.parents[2] / "data" / "wikipedia").is_dir():
        os.environ.setdefault("PMET_WIKIPEDIA_DIR", str(_wiki.resolve()))

    with open(args.hparams_json, "r", encoding="utf-8") as handle:
        hp = json.load(handle)

    layers = [int(x) for x in args.layers.split(",")] if args.layers else [int(x) for x in hp["layers"]]
    ds_name = args.mom2_dataset or hp.get("mom2_dataset", "wikitext")
    sample_size = args.sample_size if args.sample_size is not None else int(hp.get("mom2_n_samples", 100000))
    precision = args.precision or hp.get("mom2_dtype", "float32")

    print(f"[precompute_mom2] layers={layers} dataset={ds_name} sample_size={sample_size} precision={precision}")
    print(f"[precompute_mom2] stats_dir={args.stats_dir} cwd={os.getcwd()}")

    model, tok = _load_model(args.base_model_dir, args.adapter_dir, args.torch_dtype, args.device_map)

    for layer in layers:
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        layer_stats(
            model,
            tok,
            layer_name,
            args.stats_dir,
            ds_name,
            ["mom2"],
            sample_size=sample_size,
            precision=precision,
            batch_tokens=None,
            download=bool(args.download_remote_stats),
            progress=None,
            force_recompute=args.force_recompute,
        )

    print("[precompute_mom2] finished.")


if __name__ == "__main__":
    main()
