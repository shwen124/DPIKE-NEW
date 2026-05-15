from time import time
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.memit_logger import get_logger

from .compute_z import get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams

log = get_logger()


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    batch_size: int = 1000,
):
    """
    Compute k vectors for all requests, using batch processing to avoid OOM.
    
    Args:
        batch_size: Number of requests to process in each batch (default: 1000)
    """
    device = next(model.parameters()).device
    n_requests = len(requests)
    log.info("[MEMIT] compute_ks: layer %d n_requests=%d (batch_size=%d) ...", layer, n_requests, batch_size)
    t0 = time()
    
    # Process in batches to avoid OOM
    all_ans = []
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()
    
    for batch_start in range(0, n_requests, batch_size):
        batch_end = min(batch_start + batch_size, n_requests)
        batch_requests = requests[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (n_requests + batch_size - 1) // batch_size
        
        log.info("[MEMIT] compute_ks: layer %d processing batch %d/%d (requests %d-%d) ...", 
                 layer, batch_num, total_batches, batch_start, batch_end - 1)
        t_batch = time()
        
        batch_layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in batch_requests
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in batch_requests
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[0]

        # Process batch results
        batch_ans = []
        for i in range(0, batch_layer_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(batch_layer_ks[i + start : i + end].mean(0))
            batch_ans.append(torch.stack(tmp, 0).mean(0))
        
        all_ans.extend([t.detach().cpu() for t in batch_ans])
        log.info("[MEMIT] compute_ks: layer %d batch %d/%d done in %.1fs", 
                 layer, batch_num, total_batches, time() - t_batch)
        
        # Free batch tensors to keep GPU memory low
        del batch_layer_ks, batch_ans
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    out = torch.stack(all_ans, dim=0).to(device)
    log.info("[MEMIT] compute_ks: layer %d done in %.1fs shape %s", layer, time() - t0, out.shape)
    return out
