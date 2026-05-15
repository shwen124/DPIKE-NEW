from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.memit_logger import get_logger

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams

log = get_logger()

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
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

    log.info("[MEMIT] apply_memit_to_model: execute_memit started, n_requests=%d", len(requests))
    t0 = time()
    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template, resume_from_layer=None)
    log.info("[MEMIT] apply_memit_to_model: execute_memit done in %.1fs", time() - t0)

    device = next(model.parameters()).device
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat = key_mat.to(device, dtype=torch.float32)
            val_mat = val_mat.to(device, dtype=torch.float32)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    log.info("[MEMIT] New weights successfully inserted into %s", list(deltas.keys()))

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    resume_from_layer: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    device = next(model.parameters()).device
    log.info("[MEMIT] execute_memit: n_requests=%d, layers=%s, device=%s", len(requests), hparams.layers, device)

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        target_str = request["target_new"]["str"]
        # Check for empty string to avoid IndexError
        if not target_str:
            log.warning("[MEMIT] request %d: target_new['str'] is empty, skipping space check", i)
            continue
        if target_str[0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + target_str
    for i, request in enumerate(requests[:10]):
        log.debug(
            "[MEMIT] request sample %d: [%s] -> [%s]",
            i,
            request["prompt"].format(request["subject"]),
            request["target_new"]["str"],
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    try:
        # Compute z for final layer
        log.info("[MEMIT] execute_memit: get_context_templates ...")
        t0 = time()
        context_templates = get_context_templates(model, tok)
        log.info("[MEMIT] execute_memit: get_context_templates done in %.1fs", time() - t0)
        z_layer = hparams.layers[-1]
        z_list = []

        for i_req, request in enumerate(requests):
            case_id = request.get("case_id", i_req)
            log.info(
                "[MEMIT] execute_memit: compute_z request %d / %d (case_id=%s) ...",
                i_req + 1,
                len(requests),
                case_id,
            )
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, case_id
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
                    z_list.append(torch.from_numpy(data["v_star"]).to(device=device))
                    data_loaded = True
                    log.debug("[MEMIT] compute_z: loaded from cache %s", cache_fname)
                except Exception as e:
                    log.warning("[MEMIT] Error reading cache file %s: %s. Recomputing...", cache_fname, e)

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                t_z = time()
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                log.info("[MEMIT] execute_memit: compute_z request %d / %d done in %.1fs",
                         i_req + 1, len(requests), time() - t_z)
                z_list.append(cur_z)

                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    log.debug("[MEMIT] Cached k/v pair at %s", cache_fname)
        log.info("[MEMIT] execute_memit: stacking z_list -> zs, len=%d", len(z_list))
        zs = torch.stack(z_list, dim=1).to(device)

        # Insert
        # If resume_from_layer is specified, skip layers before it
        start_idx = 0
        if resume_from_layer is not None:
            if resume_from_layer in hparams.layers:
                start_idx = hparams.layers.index(resume_from_layer)
                log.info("[MEMIT] execute_memit: Resuming from layer %d (skipping layers %s)", 
                        resume_from_layer, hparams.layers[:start_idx])
            else:
                log.warning("[MEMIT] execute_memit: resume_from_layer %d not in layers %s, starting from beginning",
                           resume_from_layer, hparams.layers)
        
        for i, layer in enumerate(hparams.layers[start_idx:], start=start_idx):
            log.info("[MEMIT] execute_memit: LAYER %d (%d / %d) ...", layer, i + 1, len(hparams.layers))
            t_layer = time()

            # Get current model activations (keep on GPU)
            log.info("[MEMIT] execute_memit: layer %d compute_ks ...", layer)
            t0 = time()
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T.to(device)
            log.info("[MEMIT] execute_memit: layer %d compute_ks done in %.1fs (size %s)",
                     layer, time() - t0, layer_ks.shape)

            # Compute residual error (all on GPU)
            log.info("[MEMIT] execute_memit: layer %d get_module_input_output_at_words (cur_zs) ...", layer)
            t0 = time()
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T.to(device)
            log.info("[MEMIT] execute_memit: layer %d get_module_input_output_at_words done in %.1fs",
                     layer, time() - t0)
            targets = (zs - cur_zs).to(device)
            log.debug("[MEMIT] layer %d z error norm mean %s", layer, torch.linalg.norm(targets, dim=0).mean().item())

            if layer_ks.size(1) % targets.size(1) != 0:
                raise ValueError(
                    f"Layer {layer}: layer_ks columns {layer_ks.size(1)} not divisible by "
                    f"targets columns {targets.size(1)}."
                )
            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            # Load covariance matrix (get_cov returns on GPU)
            log.info("[MEMIT] execute_memit: layer %d get_cov ...", layer)
            force_recompute = False
            t0 = time()
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )
            log.info("[MEMIT] execute_memit: layer %d get_cov done in %.1fs", layer, time() - t0)

            # Heavy matrix ops moved to CPU to avoid GPU hang/OOM for large operations
            log.info("[MEMIT] layer %d: converting to float32 and preparing matrices (cov %s, layer_ks %s, device %s) ...",
                     layer, cov.shape, layer_ks.shape, cov.device)
            t0 = time()

            # Convert to float32 and move to CPU for matrix operations to avoid GPU hang
            # The matrix multiplication layer_ks @ layer_ks.T is very large and may hang on GPU
            log.info("[MEMIT] layer %d: Moving matrices to CPU for large matrix operations...", layer)
            layer_ks_cpu = layer_ks.float().cpu().contiguous()
            targets = targets.float().to(device).contiguous()
            cov_f = cov.float().cpu().contiguous()

            # Build A on CPU to avoid GPU hang during large matrix multiplication
            log.info("[MEMIT] layer %d: building A = mom2_weight * cov + layer_ks @ layer_ks.T on CPU (float32) ...", layer)
            log.info("[MEMIT] layer %d: Matrix shapes - layer_ks_cpu: %s, will compute [%d, %d] @ [%d, %d] = [%d, %d]",
                     layer, layer_ks_cpu.shape, layer_ks_cpu.shape[0], layer_ks_cpu.shape[1],
                     layer_ks_cpu.shape[1], layer_ks_cpu.shape[0], layer_ks_cpu.shape[0], layer_ks_cpu.shape[0])
            t_build = time()
            # Compute layer_ks @ layer_ks.T on CPU (this is a large operation: [14336, 21919] @ [21919, 14336])
            # Use chunked computation to avoid memory issues and provide progress updates
            log.info("[MEMIT] layer %d: Starting large matrix multiplication (this may take several minutes)...", layer)
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024**3
                log.info("[MEMIT] layer %d: CPU memory before matmul: %.2f GB", layer, mem_before)
            except ImportError:
                pass
            
            # Perform matrix multiplication with progress logging
            # Split into chunks to avoid memory spikes and allow progress tracking
            chunk_size = 2048  # Process 2048 rows at a time
            n_rows = layer_ks_cpu.shape[0]
            ks_ksT_chunks = []
            for chunk_start in range(0, n_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_rows)
                chunk = layer_ks_cpu[chunk_start:chunk_end]
                chunk_result = chunk @ layer_ks_cpu.T
                ks_ksT_chunks.append(chunk_result)
                if (chunk_start // chunk_size) % 10 == 0 or chunk_end == n_rows:
                    log.info("[MEMIT] layer %d: Matrix multiplication progress: %d/%d rows (%.1f%%)",
                            layer, chunk_end, n_rows, 100.0 * chunk_end / n_rows)
            ks_ksT = torch.cat(ks_ksT_chunks, dim=0)
            del ks_ksT_chunks
            log.info("[MEMIT] layer %d: layer_ks @ layer_ks.T computed on CPU in %.1fs", layer, time() - t_build)
            
            A = (hparams.mom2_update_weight * cov_f + ks_ksT).contiguous()
            log.info("[MEMIT] layer %d: A built on CPU (%s) in %.1fs total", layer, A.shape, time() - t_build)
            
            # Free intermediate matrices to save memory
            del cov_f, ks_ksT
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory usage (optional, skip if psutil not available)
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                log.info("[MEMIT] layer %d: CPU memory - RSS: %.2f GB, VMS: %.2f GB", 
                        layer, mem_info.rss / 1024**3, mem_info.vms / 1024**3)
            except ImportError:
                pass

            t_solve = time()
            # Solve on CPU (already on CPU)
            log.info("[MEMIT] layer %d: Starting CPU solve (A: %s, layer_ks: %s)...", 
                    layer, A.shape, layer_ks_cpu.shape)
            try:
                adj_k_cpu = torch.linalg.solve(A, layer_ks_cpu)
                log.info("[MEMIT] layer %d: CPU solve succeeded in %.1fs", layer, time() - t_solve)
                # Move result back to GPU
                adj_k = adj_k_cpu.to(device)
                # Free A and intermediate tensors immediately after use (A is deleted here, not later)
                del adj_k_cpu, A, layer_ks_cpu
            except Exception as e:
                log.error("[MEMIT] layer %d: CPU solve failed: %s", layer, str(e)[:200])
                # Clean up on error
                if 'A' in locals():
                    del A
                if 'layer_ks_cpu' in locals():
                    del layer_ks_cpu
                raise

            resid = (targets / (len(hparams.layers) - i)).to(device)
            upd_matrix = (resid @ adj_k.T).to(device)
            log.info("[MEMIT] layer %d: upd_matrix computed, total %.1fs", layer, time() - t0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Adjust update matrix shape
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            log.debug("[MEMIT] layer %d orig norm %s upd norm %s",
                      layer, torch.linalg.norm(weights[weight_name]).item(), torch.linalg.norm(upd_matrix).item())

            # Update model weights; store deltas on CPU only for later apply
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Free GPU memory (move to CPU so GPU allocator can reclaim)
            # Note: layer_ks was already moved to CPU and deleted earlier
            if 'cov' in locals():
                cov = cov.cpu()
                del cov
            # Explicitly move and delete remaining tensors to free GPU memory
            if 'cur_zs' in locals():
                cur_zs = cur_zs.cpu()
                del cur_zs
            if 'targets' in locals():
                targets = targets.cpu()
                del targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("[MEMIT] execute_memit: layer %d done in %.1fs total", layer, time() - t_layer)
    finally:
        # Restore state of original model even if an error occurs
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]

    log.info("[MEMIT] execute_memit: deltas computed for %s", list(weights.keys()))

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    # Special case: use precomputed LLaMA3-8B statistics if available.
    # This avoids re-computing expensive Wikipedia-based covariance on this machine.
    try:
        from pathlib import Path as _Path  # Local import to avoid polluting namespace
        import numpy as _np

        if (
            getattr(model.config, "model_type", "") == "llama"
            and "model.layers." in layer_name
            and mom2_dataset == "wikipedia"
            and ".mlp.down_proj" in layer_name
        ):
            base_dir = _Path(
                "/data1/D-PIKE/DistillMIKE-main/memit/stats/llama3-8B/wikipedia_stats"
            )
            if base_dir.exists():
                try:
                    # layer_name example: "model.layers.10.mlp.gate_proj"
                    layer_idx = int(layer_name.split("model.layers.")[1].split(".")[0])
                    # Try float32 first (new precision), fallback to float16 if not found
                    stats_path = base_dir / (
                        f"model_layers_{layer_idx}_mlp_down_proj_float32_mom2-mean_30000000.npz"
                    )
                    if not stats_path.exists():
                        # Fallback to float16 for backward compatibility
                        stats_path = base_dir / (
                            f"model_layers_{layer_idx}_mlp_down_proj_float16_mom2-mean_30000000.npz"
                        )
                    if stats_path.exists():
                        log.info("[MEMIT] get_cov: loading precomputed LLaMA3-8B covariance from %s", stats_path)
                        data = _np.load(stats_path)
                        mom2 = data["mom2.mom2"].astype("float32")
                        dev = next(model.parameters()).device
                        cov_tensor = torch.from_numpy(mom2).to(device=dev, dtype=torch.float32)
                        return torch.linalg.inv(cov_tensor) if inv else cov_tensor
                except Exception as e:  # pragma: no cover - best-effort fallback
                    log.warning(
                        "[MEMIT] get_cov: failed to load precomputed LLaMA3-8B stats for %s: %s. Falling back.",
                        layer_name, e,
                    )
    except Exception:
        # If anything above fails, silently fall back to original behavior.
        pass

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name, mom2_dataset, mom2_n_samples, mom2_dtype)

    log.info("[MEMIT] get_cov: retrieving covariance for %s @ %s", model_name, layer_name)
    if key not in COV_CACHE or force_recompute:
        log.info("[MEMIT] get_cov: calling layer_stats (mom2_dataset=%s, n_samples=%s) ...",
                 mom2_dataset, mom2_n_samples)
        t0 = time()
        stat = layer_stats(
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
        log.info("[MEMIT] get_cov: layer_stats done in %.1fs", time() - t0)

    dev = next(model.parameters()).device
    cov_gpu = COV_CACHE[key].to(device=dev, dtype=torch.float32)
    return torch.linalg.inv(cov_gpu) if inv else cov_gpu


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


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        log.info("[MEMIT] get_context_templates: generating via generate_fast ...")
        t0 = time()
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        log.info("[MEMIT] get_context_templates: generate_fast done in %.1fs, cached", time() - t0)

    return CONTEXT_TEMPLATES_CACHE
