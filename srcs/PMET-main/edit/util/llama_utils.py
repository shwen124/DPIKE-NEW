"""Helpers for PMET on Llama-family causal LMs (incl. merged LoRA checkpoints)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_family(model) -> str:
    path = (getattr(model.config, "_name_or_path", None) or "").lower()
    mtype = (getattr(model.config, "model_type", None) or "").lower()
    if "llama" in path or mtype in {"llama", "mistral"}:
        return "llama"
    if "neo" in path or mtype == "gpt_neox":
        return "neo"
    if "gpt2" in path or mtype == "gpt2":
        return "gpt2"
    if "gpt-j" in path or "gptj" in mtype:
        return "gptj"
    return "other"


def hidden_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    return int(cfg.n_embd)


def get_lm_head_and_norm(model, hparams) -> Tuple[torch.Tensor, Any]:
    """Return (lm_weight_T, ln_f_module) for logit projection in compute_zs."""
    import util.nethook as nethook

    family = model_family(model)
    if family in {"neo", "gpt2", "gptj"}:
        ln_f = nethook.get_module(model, hparams.ln_f_module)
        lm_head_module = nethook.get_module(model, hparams.lm_head_module)
        lm_w = nethook.get_parameter(lm_head_module, "weight").T
        return lm_w, ln_f

    lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    return lm_w, ln_f


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def last_token_lookup_indices(tok: AutoTokenizer, texts: list[str]) -> list[int]:
    batch = tok(texts, return_tensors="pt", padding=True)
    idxs = []
    for i in range(batch["input_ids"].shape[0]):
        n = int(batch["attention_mask"][i].sum().item())
        idxs.append(max(0, n - 1))
    return idxs


def load_model_and_tokenizer(
    base_model_dir: str,
    adapter_dir: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    merge_lora: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch_dtype)

    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, **model_kwargs)

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge_lora:
            model = model.merge_and_unload()

    model.eval()
    return model, tok
