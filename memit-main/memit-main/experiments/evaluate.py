import json
import shutil
from datetime import datetime
from itertools import islice
from pathlib import Path
from time import time
from typing import Iterable, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.memit_logger import setup_logging, get_logger

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    save_edited_model_dir: str = None,
):
    # Set up logging to logs/ (all subsequent logs go here)
    logs_dir = Path("/data1/D-PIKE/memit-main/memit-main/logs")
    log_basename = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_logging(log_dir=logs_dir, log_basename=log_basename)
    log = get_logger()

    log.info("[STAGE] START main()")
    log.info("alg_name=%s model_name=%s hparams=%s ds_name=%s dataset_size_limit=%s save_edited_model_dir=%s",
             alg_name, model_name, hparams_fname, ds_name, dataset_size_limit, save_edited_model_dir)

    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    log.debug("[STAGE] Algorithm selected: %s", alg_name)

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    log.info("[STAGE] Run directory: %s", run_dir)

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    log.info("[STAGE] Hyperparameters loaded: %s", hparams)

    # Instantiate vanilla model
    if type(model_name) is str:
        log.info("[STAGE] Loading model from %s ...", model_name)
        # Load model in float16 to reduce GPU memory usage.
        t0 = time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        log.info("[STAGE] Model loaded from disk in %.1fs, moving to GPU ...", time() - t0)
        # Move model to GPU; if still out of memory, raise the error.
        model = model.cuda()
        log.info("[STAGE] Model on GPU. Total load time %.1fs.", time() - t0)
        # Ensure compatibility with GPT-style configs expected by MEMIT / ROME utilities.
        # LLaMA-style configs use `hidden_size` and `max_position_embeddings` instead.
        if not hasattr(model.config, "n_embd") and hasattr(
            model.config, "hidden_size"
        ):
            setattr(model.config, "n_embd", model.config.hidden_size)
        if not hasattr(model.config, "n_positions") and hasattr(
            model.config, "max_position_embeddings"
        ):
            setattr(model.config, "n_positions", model.config.max_position_embeddings)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    log.info("[STAGE] Loading dataset, attribute snippets, tf-idf data ...")
    generation_tests_enabled = (not skip_generation_tests) and generation_test_interval > 0
    snips = AttributeSnippets(DATA_DIR) if generation_tests_enabled else None
    vec = get_tfidf_vectorizer(DATA_DIR) if generation_tests_enabled else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    ds_list = list(ds)
    log.info("[STAGE] Dataset loaded. size=%d", len(ds_list))

    # Rebuild iterable for downstream use (ds is consumed above)
    class _DSAdapter:
        def __iter__(self):
            return iter(ds_list)

    ds = _DSAdapter()

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        log.info("[STAGE] Will load cache from %s", cache_template)

    # Optional: apply edits once to the entire dataset and save a persistent edited model.
    if save_edited_model_dir is not None:
        log.info("[STAGE] save_edited_model: Building all_requests from dataset (n=%d) ...", len(ds_list))
        all_requests = [
            {"case_id": record["case_id"], **record["requested_rewrite"]}
            for record in ds_list
        ]
        log.info("[STAGE] save_edited_model: Applying %s to full dataset (%d requests) -> %s ...",
                 alg_name, len(all_requests), save_edited_model_dir)
        etc_args_full = (
            dict(cache_template=cache_template)
            if alg_name in ("ROME", "MEMIT")
            else dict()
        )
        # Support resuming from a specific layer (for MEMIT only)
        t_apply = time()
        if alg_name == "MEMIT":
            # Check if we should resume from layer 12 (where it stopped)
            resume_layer = 12  # Layer where training stopped (Layer 11 already completed)
            log.info("[STAGE] save_edited_model: Resuming from layer %d", resume_layer)
            from memit.memit_main import execute_memit, upd_matrix_match_shape
            # Get deltas with resume_from_layer parameter
            deltas = execute_memit(
                model,
                tok,
                all_requests,
                hparams,
                cache_template=cache_template,
                resume_from_layer=resume_layer,
            )
            # Apply deltas to model
            device = next(model.parameters()).device
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat = key_mat.to(device, dtype=torch.float32)
                    val_mat = val_mat.to(device, dtype=torch.float32)
                    upd_matrix = key_mat @ val_mat.T
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                    w[...] += upd_matrix.float()
            edited_model = model
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                all_requests,
                hparams,
                copy=False,
                return_orig_weights=False,
                **etc_args_full,
            )
        t_apply = time() - t_apply
        log.info("[STAGE] save_edited_model: apply_algo done in %.1fs. Saving to disk ...", time() - t_apply)
        save_dir = Path(save_edited_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        t_save = time()
        edited_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        log.info("[STAGE] save_edited_model: Saved to %s in %.1fs.", save_dir, time() - t_save)

    # Iterate through dataset
    chunk_idx = -1
    for record_chunks in chunks(ds, num_edits):
        chunk_idx += 1
        case_ids = [r["case_id"] for r in record_chunks]
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            log.debug("[STAGE] Chunk %d (case_ids=%s) already finished, skipping.", chunk_idx, case_ids)
            continue

        log.info("[STAGE] Chunk %d: case_ids=%s. Applying %s ...", chunk_idx, case_ids, alg_name)
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = (
            dict(cache_template=cache_template)
            if alg_name in ("ROME", "MEMIT")
            else dict()
        )

        start = time()
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        log.info("[STAGE] Chunk %d: apply_algo took %.1fs.", chunk_idx, exec_time)

        # Evaluate new model
        log.info("[STAGE] Chunk %d: Evaluating ...", chunk_idx)
        start = time()
        gen_test_vars = [snips, vec]
        try:
            for record in record_chunks:
                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    log.debug("[STAGE] Skipping %s; already exists.", out_file)
                    continue

                run_generation_tests = (
                    generation_tests_enabled
                    and record["case_id"] % generation_test_interval == 0
                )
                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(gen_test_vars if run_generation_tests else [None, None]),
                    ),
                }

                # Dump metrics in .json
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
        finally:
            # Restore original weights
            with torch.no_grad():
                for k, v in weights_copy.items():
                    param = nethook.get_parameter(model, k)
                    param[...] = v.to(param.device)

        log.info("[STAGE] Chunk %d: Evaluation took %.1fs.", chunk_idx, time() - start)

    log.info("[STAGE] main() finished.")


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr: Iterable, n: int):
    """Yield successive n-sized chunks from an iterable."""
    chunk = []
    for item in arr:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit. Can be a pretrained model name or a local path.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--save_edited_model_dir",
        type=str,
        default=None,
        help=(
            "If set, apply all edits in the dataset once to the model using the "
            "specified algorithm and save the edited model to this directory."
        ),
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        save_edited_model_dir=args.save_edited_model_dir,
    )
