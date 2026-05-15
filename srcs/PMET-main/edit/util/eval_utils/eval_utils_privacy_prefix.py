"""
Fast privacy-edit quality checks for prefix-completion PMET runs.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.request_context import request_completion_text


@torch.no_grad()
def sequence_nll(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_text: str,
    target_text: str,
) -> float:
    device = next(model.parameters()).device
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("nan")
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids).logits[0]
    start = len(prompt_ids)
    nlls = []
    for pos in range(start, len(full_ids)):
        log_probs = F.log_softmax(logits[pos - 1].float(), dim=-1)
        nlls.append(float(-log_probs[full_ids[pos]].item()))
    return float(sum(nlls) / len(nlls))


@torch.no_grad()
def greedy_completion(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 64,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    gen_inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(gen_inputs)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    out = model.generate(
        gen_inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )[0][len(prompt_ids) :]
    text = tok.decode(out.tolist(), skip_special_tokens=True)
    return text.splitlines()[0].strip()


def compute_rewrite_quality_privacy_prefix(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: Dict[str, Any],
    *_unused,
) -> Dict[str, Any]:
    rewrite = record["requested_rewrite"]
    prompt_text = record.get("completion_context") or request_completion_text(rewrite)
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]

    nll_new = sequence_nll(model, tok, prompt_text, target_new)
    nll_true = sequence_nll(model, tok, prompt_text, target_true)
    generated = greedy_completion(model, tok, prompt_text)

    return {
        "nll_new": nll_new,
        "nll_true": nll_true,
        "nll_ratio": nll_new / nll_true if nll_true > 0 else math.inf,
        "prefers_redacted": nll_new < nll_true,
        "generated": generated,
        "target_new": target_new.strip(),
        "target_true": target_true.strip(),
        "raw_input": record.get("raw_input"),
        "pii_type": record.get("pii_type"),
    }
