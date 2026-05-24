import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.llama_utils import model_device
from util.request_context import request_completion_text

from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .pmet_hparams import PMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}


def apply_pmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    
    deltas = execute_pmet(model, tok, requests, hparams, cache_template=cache_template) #存储了Equ14 左右的值

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items(): #w_name, adj_k, resid
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix.to(w.device)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float() #w[...]高级索引，表示对w中每个元素进行操作

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_pmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT_ATTN request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = dict()
    for rewrite_module_name in rewrite_module_names:
        z_list[rewrite_module_name] = []
    # get zs
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(
                        torch.from_numpy(data["v_star"]).to(model_device(model))
                    )
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                    )
                    z_list[rewrite_module_names[0]].append(cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp)
                    z_list[rewrite_module_names[1]].append(cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp)
                    for rewrite_module_name in rewrite_module_names:
                        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                        cache_fname = (
                            Path(
                                str(cache_template).format(
                                    z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                                )
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            if block_name == "attn":
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_attn.detach().cpu().numpy(),
                                    },
                                )
                            else:
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                                    },
                                )
                            print(f"Cached k/v pair at {cache_fname}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                    if "attn" == block_name:
                        cur_z = cur_z_attn
                    else:
                        cur_z = cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            cache_fname,
                            **{
                                "v_star": cur_z.detach().cpu().numpy(),
                            },
                        )
                        print(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n") 
        layers_ks = None
        # force_recompute = layer != hparams.layers[0]
        for rewrite_module_name in rewrite_module_names:
            # Get current model activations

            if 'gpt-j' in model.config._name_or_path and len(rewrite_module_names) == 2:
                if layers_ks == None:
                    layers_ks = compute_ks_parallel(model, tok, requests, hparams, layer, context_templates)  #K eqn 19
            else:
                layers_ks = compute_ks(model, tok, requests, hparams, rewrite_module_name, layer, context_templates)

            print(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")
            completion_contexts = [request_completion_text(request) for request in requests]
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=completion_contexts,
                words=["" for _ in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = z_list[rewrite_module_name] - cur_zs
            dev = model_device(model)
            layer_ks = layers_ks[rewrite_module_name].T.double().to(dev)
            targets = targets.double().to(dev)
            if os.environ.get("PMET_SKIP_MOM2", "0") == "1":
                d = layer_ks.size(0)
                cov_cpu = torch.zeros((d, d), dtype=torch.double, device="cpu")
                print(
                    "PMET_SKIP_MOM2=1: 跳过 Wikipedia 二阶矩（无外网/HF 时用；"
                    "非论文原设定，仅联调/冒烟）。"
                )
            else:
                force_recompute = False
                cov = get_cov(
                    model,
                    tok,
                    rewrite_module_name.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples
                    if not force_recompute
                    else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                )
                cov_cpu = cov.double()
                if cov_cpu.device.type != "cpu":
                    cov_cpu = cov_cpu.cpu()

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1) #r
            scale = np.sqrt((len(hparams.layers) - i))
            scaled_targets = targets / scale
            ridge = float(os.environ.get("PMET_RIDGE", "1e-3"))
            # 4090 友好：在 CPU 上算 (K K^T + λC)^{-1}，避免与全量模型争显存
            if os.environ.get("PMET_GRAM_ON_CPU", "1") != "0":
                K = layer_ks.cpu().double()
                Tm = scaled_targets.cpu().double()
                gram = K @ K.T + hparams.mom2_update_weight * cov_cpu
                if ridge > 0:
                    gram = gram + ridge * torch.eye(K.size(0), dtype=torch.double, device="cpu")
                inv_gram = torch.linalg.inv(gram)
                upd_matrix = (Tm @ K.T @ inv_gram).to(dev)
            else:
                cov_on_dev = cov_cpu.to(layer_ks.device).double()
                gram_dev = layer_ks @ layer_ks.T + hparams.mom2_update_weight * cov_on_dev
                if ridge > 0:
                    gram_dev = gram_dev + ridge * torch.eye(
                        layer_ks.size(0), dtype=torch.double, device=layer_ks.device
                    )
                upd_matrix = scaled_targets @ layer_ks.T @ torch.linalg.inv(gram_dev)
            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weight_dev = weights[weight_name].device
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(weight_dev)
                deltas[weight_name] = upd_matrix

            # Clear GPU memory

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats( # download
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    # 协方差常驻 CPU，供 execute_pmet 在 CPU 上做逆；避免在 GPU 上再占一份大矩阵
    cov = COV_CACHE[key]
    if inv:
        return torch.inverse(cov.double()).float()
    return cov
def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if os.environ.get("PMET_MINIMAL_CONTEXT", "0") == "1":
        if CONTEXT_TEMPLATES_CACHE is None:
            CONTEXT_TEMPLATES_CACHE = [["{}"]]
            print("Using minimal PMET context templates (PMET_MINIMAL_CONTEXT=1)")
        return CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                ) # 用模型生成句子
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
