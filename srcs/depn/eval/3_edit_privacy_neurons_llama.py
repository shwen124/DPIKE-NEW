"""
Llama3 CLM runner for editing privacy neurons
适配 Llama3-8b 的隐私神经元编辑
"""

import logging
import argparse
import math
import os
import torch
import re
import random
import numpy as np
import json, jsonlines
import pickle
import time
import random
from dataclasses import dataclass
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import chain

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from custom_llama import patch_llama_model
from depn_pii_utils import PIISample, make_pii_sample, normalize_pii_type, find_token_subsequence
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
PHONE_RE = re.compile(r"\d(?:\s\d){9}")


def _ensure_causal_output(outputs, labels=None):
    """
    Normalize patched or standard model output to a single interface with .loss and .logits.
    Patch may return CausalLMOutputWithPast or tuple; ensure eval code always gets .loss / .logits.
    """
    if outputs is None:
        return None
    # Already has .logits (and optionally .loss)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        loss = getattr(outputs, "loss", None)
        if loss is None and labels is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels.to(shift_logits.device), ignore_index=-100)
        class _Out:
            pass
        o = _Out()
        o.loss = loss
        o.logits = logits
        return o
    # Tuple: (loss,) + (logits,) or (logits,) when return_dict=False
    if isinstance(outputs, (tuple, list)):
        loss, logits = None, None
        for x in outputs:
            if isinstance(x, torch.Tensor):
                if x.dim() >= 2:
                    logits = x
                elif x.dim() == 0:
                    loss = x
        if logits is None and len(outputs) >= 1:
            logits = outputs[-1]
        if loss is None and labels is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels.to(shift_logits.device), ignore_index=-100)
        class _Out:
            pass
        o = _Out()
        o.loss = loss
        o.logits = logits
        return o
    class _Out:
        pass
    o = _Out()
    o.loss = getattr(outputs, "loss", None)
    o.logits = getattr(outputs, "logits", None)
    return o


def load_evaldata(eval_data_path, tokenizer, max_seq_length):
    """Load evaluation dataset for perplexity calculation."""
    data_files = {}
    data_files["validation"] = eval_data_path
    raw_datasets = load_dataset('text', data_files=data_files)

    text_column_name = 'text'
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True, truncation=True, max_length=max_seq_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    eval_dataset = tokenized_datasets["validation"]
    # For Causal LM, we don't use MLM collator, but we need to create labels
    def collate_fn(examples):
        batch = tokenizer.pad(examples, return_tensors="pt", padding=True)
        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=8)

    return eval_dataloader


def _forward_with_edit(model, labels=None, imp_pos=None, imp_op=None, **kwargs):
    raw = model(
        **kwargs,
        labels=labels,
        imp_pos=imp_pos,
        imp_op=imp_op,
    )
    return _ensure_causal_output(raw, labels=labels)


def eval_ppl(eval_dataloader, device, model, imp_pos=None, imp_op=None):
    """Evaluate perplexity."""
    losses = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.get("labels")
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = _forward_with_edit(
                model,
                labels=labels,
                imp_pos=imp_pos,
                imp_op=imp_op,
                **model_inputs,
            )
            loss = outputs.loss if outputs else None
            if loss is None:
                continue
            losses.append(loss.repeat(batch['input_ids'].shape[0]))

    if not losses:
        return float("inf")
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


@dataclass(frozen=True)
class TelSample:
    prompt: str
    phone: str
    full_text: str
    char_start: int
    char_end: int


def load_privacys(unique_priv_path):
    """Load privacy data from .txt (one line per text, optional '# ' separator) or .json (list of bags)."""
    if unique_priv_path.endswith('.json'):
        with open(unique_priv_path, 'r') as f:
            bags = json.load(f)
        unique_priv_list = []
        for bag in bags:
            if not bag:
                continue
            for ex in bag:
                if ex and isinstance(ex, (list, tuple)):
                    unique_priv_list.append(list(ex))
                elif isinstance(ex, str):
                    unique_priv_list.append([ex])
        return unique_priv_list
    with open(unique_priv_path, 'r') as file:
        lines = file.readlines()
        unique_priv_list = []
        for i in lines:
            keys = i.strip().split('# ')
            unique_priv_list.append(keys)
    return unique_priv_list

def _resolve_model_max_length(model, tokenizer):
    max_length = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(max_length, int) and max_length > 0:
        return min(max_length, 8192)
    tok_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max_length, int) and 0 < tok_max_length < 1_000_000:
        return min(tok_max_length, 8192)
    return 2048


def _digit_token_ids(tokenizer):
    ids = []
    for d in range(10):
        candidates = [f" {d}", str(d)]
        chosen = None
        for tok in candidates:
            enc = tokenizer(tok, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                chosen = enc[0]
                break
        if chosen is None:
            raise ValueError(f"Tokenizer cannot map digit {d} to a single token.")
        ids.append(chosen)
    return ids


def _make_tel_sample(privacy):
    if not privacy:
        return None
    gold_text = privacy[0].strip()
    match = PHONE_RE.search(gold_text)
    if match is None:
        return None
    phone = match.group(0)
    prompt = gold_text.replace(phone, "***", 1)
    return gold_text, TelSample(
        prompt=prompt,
        phone=phone,
        full_text=gold_text,
        char_start=match.start(),
        char_end=match.end(),
    )


def collect_kn_rel(kn_bag_list, erase_kn_num, do_random_kn, config, intermediate_size, keep_layer=True):
    kn_counter = Counter()
    for kn_bag in kn_bag_list:
        for kn in kn_bag:
            kn_counter.update([pos_list2str(kn)])

    most_common_kn = kn_counter.most_common(erase_kn_num)
    if do_random_kn:
        random_kn = []
        for kn_str, score in most_common_kn:
            if keep_layer:
                layer = int(kn_str.split('@')[0])
            else:
                layer = random.randint(0, config.num_hidden_layers - 1)
            pos = random.randint(0, intermediate_size - 1)
            random_kn.append((f"{layer}@{pos}", score))
        most_common_kn = random_kn
    return [pos_str2list(kn_str) for kn_str, _ in most_common_kn], most_common_kn


@torch.no_grad()
def get_exposure_llama(model, tokenizer, digit_token_ids, sample, total_candidates, imp_pos=None, imp_op=None):
    full_text = sample.full_text
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not full_ids or not offsets:
        return None

    phone_token_positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= sample.char_start or tok_start >= sample.char_end:
            continue
        token_text = full_text[tok_start:tok_end]
        if any(ch.isdigit() for ch in token_text):
            phone_token_positions.append(idx)
    if not phone_token_positions:
        return None

    start = phone_token_positions[0]
    end = phone_token_positions[-1] + 1
    prefix_ids = full_ids[:end]
    max_length = _resolve_model_max_length(model, tokenizer)
    if len(prefix_ids) > max_length:
        offset = len(prefix_ids) - max_length
        prefix_ids = prefix_ids[offset:]
        start -= offset
        end -= offset
    if start <= 0 or end > len(prefix_ids):
        return None

    inp = torch.tensor([prefix_ids], dtype=torch.long, device=next(model.parameters()).device)
    outputs = _forward_with_edit(
        model,
        input_ids=inp,
        imp_pos=imp_pos,
        imp_op=imp_op,
    )
    if outputs is None or outputs.logits is None:
        return None
    logits = outputs.logits[0]

    rank_num = 1
    length = len(phone_token_positions)
    for i, original_pos in enumerate(phone_token_positions):
        pos = original_pos
        if len(full_ids[:end]) > max_length:
            pos = original_pos - (len(full_ids[:end]) - max_length)
        prev = pos - 1
        if pos >= len(prefix_ids) or prev < 0:
            return None
        true_id = prefix_ids[pos]
        try:
            true_digit_idx = digit_token_ids.index(true_id)
        except ValueError:
            return None
        digit_logits = logits[prev, digit_token_ids]
        order = torch.argsort(digit_logits, descending=True)
        digit_rank = (order == true_digit_idx).nonzero(as_tuple=False).item()
        rank_num += int(digit_rank) * (10 ** (length - 1 - i))

    exposure = math.log(total_candidates, 2) - math.log(rank_num, 2)
    return rank_num, exposure


@torch.no_grad()
def get_span_MRR_llama(model, tokenizer, sample, imp_pos=None, imp_op=None, max_context_tokens=256):
    full_text = sample.full_text
    secret_ids = tokenizer(sample.secret, add_special_tokens=False)["input_ids"]
    if not secret_ids:
        return None

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start = find_token_subsequence(full_ids, secret_ids)
    if start is None:
        return None

    end = start + len(secret_ids)
    max_length = min(_resolve_model_max_length(model, tokenizer), max(8, max_context_tokens + len(secret_ids)))
    window_start = max(0, start - max(0, max_context_tokens))
    prefix_ids = full_ids[window_start:end]
    start = start - window_start
    if len(prefix_ids) > max_length:
        prefix_ids = prefix_ids[-max_length:]
        start = max(0, start - (len(full_ids[window_start:end]) - max_length))
    if start >= len(prefix_ids):
        return None

    inp = torch.tensor([prefix_ids], dtype=torch.long, device=next(model.parameters()).device)
    try:
        outputs = _forward_with_edit(
            model,
            input_ids=inp,
            imp_pos=imp_pos,
            imp_op=imp_op,
        )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    if outputs is None or outputs.logits is None:
        return None
    logits = outputs.logits[0]

    reciprocals = []
    for i in range(len(secret_ids)):
        pos = start + i
        prev = pos - 1
        if pos >= len(prefix_ids) or prev < 0:
            continue
        true_id = prefix_ids[pos]
        row = logits[prev]
        true_logit = row[true_id]
        rank = int((row > true_logit).sum().item()) + 1
        reciprocals.append(1.0 / rank)

    return float(sum(reciprocals) / len(reciprocals)) if reciprocals else None


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Whole private data path. ")
    parser.add_argument("--validation_path",
                        default=None,
                        type=str,
                        help="validation data to evaluate the fine-tuned model ")
    parser.add_argument("--model_name_or_path",
                        default="meta-llama/Meta-Llama-3-8B",
                        type=str,
                        required=True,
                        help="Llama3 model path or identifier")
    parser.add_argument("--adapter_dir",
                        type=str,
                        default=None,
                        help="Optional path to PEFT/LoRA adapter (e.g. from fine-tuning).")
    parser.add_argument("--do_random_kn",
                        action="store_true",
                        help="if set, erase random neurons instead of kn_bag top neurons")
    parser.add_argument("--kn_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--erase_kn_num",
                        default=10,
                        type=int,
                        help="how many kn to erase.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--input_prefix",
                        type=str,
                        default=None)
    parser.add_argument("--metric_max_context",
                        default=256,
                        type=int,
                        help="Maximum number of prefix tokens kept before the secret span during DEPN eval.")

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load pre-trained Llama3
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load on single GPU so all params and inputs are on same device (avoids device mismatch with patch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    )
    if getattr(args, 'adapter_dir', None):
        from peft import PeftModel
        logger.info("Loading adapter from %s", args.adapter_dir)
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    # Patch model for DEPN editing
    model = patch_llama_model(model)
    if n_gpu > 0:
        model = model.to(device)
    model.eval()

    with open(args.kn_dir, 'r') as fr:
        kn_bag_list = json.load(fr)

    # ======================== eval ori model =================================  
    if args.validation_path:
        eval_data_path = args.validation_path
        eval_dataloader = load_evaldata(eval_data_path, tokenizer, args.max_seq_length)
        print('start evaluating original model')
        print(f"perplexity: {eval_ppl(eval_dataloader, device, model)}")

    # Get intermediate size
    intermediate_size = getattr(config, 'intermediate_size', config.hidden_size * 4)

    # =========== get ori exposure  ==============
    txt_type = args.input_prefix
    
    if txt_type == 'TEL':
        TOTAL_CANDIDATES = 10_000_000_000
        unique_privacys = load_privacys(args.priv_data_path)
        digit_token_ids = _digit_token_ids(tokenizer)
        before_exp_results = []
        after_exp_results = []
        prompt = ''
        exp_sum = 0
        count = 0

        for privacy in unique_privacys:
            sample_data = _make_tel_sample(privacy)
            if sample_data is None:
                continue
            gold_text, tel_sample = sample_data
            prompt = tel_sample.prompt
            result = get_exposure_llama(model, tokenizer, digit_token_ids, tel_sample, TOTAL_CANDIDATES)
            if result is None:
                continue
            rank, canary_exposure = result

            single_priv = {   
                'secret': gold_text,
                'rank': rank,
                'exp': canary_exposure
            }
        
            before_exp_results.append(single_priv)
            exp_sum += canary_exposure
            count += 1
        
        print('#' * 30)
        print(prompt, ' average exp: ', exp_sum / count if count > 0 else 0)

        # ======================== erase privacy neurons =================================
        kn_rel, most_common_kn = collect_kn_rel(
            kn_bag_list,
            args.erase_kn_num,
            args.do_random_kn,
            config,
            intermediate_size,
            keep_layer=True,
        )
        print('## erased kn:', most_common_kn)
        print('## erased kn num:', len(most_common_kn))
        
        print('start evaluating erased model')
        if args.validation_path:
            print(f"perplexity: {eval_ppl(eval_dataloader, device, model, imp_pos=kn_rel, imp_op='remove')}")

        # =========== get new exposure  ==============
        count = 0
        exp_sum = 0
        for privacy in unique_privacys:
            sample_data = _make_tel_sample(privacy)
            if sample_data is None:
                continue
            gold_text, tel_sample = sample_data
            prompt = tel_sample.prompt
            result = get_exposure_llama(
                model,
                tokenizer,
                digit_token_ids,
                tel_sample,
                TOTAL_CANDIDATES,
                imp_pos=kn_rel,
                imp_op='remove',
            )
            if result is None:
                continue
            rank, canary_exposure = result

            single_priv = {   
                'secret': gold_text,
                'rank': rank,
                'exp': canary_exposure
            }
        
            after_exp_results.append(single_priv)
            exp_sum += canary_exposure
            count += 1
        
        print('#' * 30)
        print(prompt, ' average exp: ', exp_sum / count if count > 0 else 0)

    elif txt_type != 'RANDOM':
        pii_type = normalize_pii_type(txt_type) or txt_type
        # =========== get ori MRR  ==============
        unique_privacys = load_privacys(args.priv_data_path)
        before_exp_results = []
        after_exp_results = []
        MRR_sum = 0
        count = 0  

        for privacy in unique_privacys:
            pii_sample = make_pii_sample(privacy, pii_type=pii_type)
            if pii_sample is None:
                continue
            prompt = pii_sample.prompt
            span_mrr = get_span_MRR_llama(
                model,
                tokenizer,
                pii_sample,
                max_context_tokens=args.metric_max_context,
            )
            if span_mrr is None:
                continue

            single_priv = {   
                'secret': pii_sample.secret,
                'text': pii_sample.full_text,
                'MRR': span_mrr,
                'pii_type': pii_type,
            }
            
            MRR_sum += span_mrr
            count += 1
            before_exp_results.append(single_priv)
        
        print('#' * 30)
        print('average MRR: ', MRR_sum / count if count > 0 else 0)

        # ======================== erase privacy neurons =================================
        kn_rel, most_common_kn = collect_kn_rel(
            kn_bag_list,
            args.erase_kn_num,
            args.do_random_kn,
            config,
            intermediate_size,
            keep_layer=False,
        )
        print('## erased kn:', most_common_kn)
        print('## erased kn num:', len(most_common_kn))
        
        print('start evaluating erased model')
        if args.validation_path:
            print(f"perplexity: {eval_ppl(eval_dataloader, device, model, imp_pos=kn_rel, imp_op='remove')}")

        # =========== get new MRR  ==============
        MRR_sum = 0
        count = 0  

        for privacy in unique_privacys:
            pii_sample = make_pii_sample(privacy, pii_type=pii_type)
            if pii_sample is None:
                continue
            prompt = pii_sample.prompt
            span_mrr = get_span_MRR_llama(
                model,
                tokenizer,
                pii_sample,
                imp_pos=kn_rel,
                imp_op='remove',
                max_context_tokens=args.metric_max_context,
            )
            if span_mrr is None:
                continue

            single_priv = {   
                'secret': pii_sample.secret,
                'text': pii_sample.full_text,
                'MRR': span_mrr,
                'pii_type': pii_type,
            }
            
            MRR_sum += span_mrr
            count += 1
            after_exp_results.append(single_priv)
        
        print('#' * 30)
        print('average MRR: ', MRR_sum / count if count > 0 else 0)

    elif txt_type == 'RANDOM':
        # =========== get ori ppl  ==============
        priv_dataloader = load_evaldata(args.priv_data_path, tokenizer, args.max_seq_length)
        print(f"txt_ppl: {eval_ppl(priv_dataloader, device, model)}")

        # ======================== erase privacy neurons =================================
        kn_rel, most_common_kn = collect_kn_rel(
            kn_bag_list,
            args.erase_kn_num,
            args.do_random_kn,
            config,
            intermediate_size,
            keep_layer=False,
        )
        print('## erased kn:', most_common_kn)
        print('## erased kn num:', len(most_common_kn))
        
        print('start evaluating erased model')

        # =========== get new ppl  ==============
        print(f"txt_ppl: {eval_ppl(priv_dataloader, device, model, imp_pos=kn_rel, imp_op='remove')}")


if __name__ == "__main__":
    main()
