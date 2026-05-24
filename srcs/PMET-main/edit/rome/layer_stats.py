import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally, save_cached_state

from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def _mom2_cache_dir() -> str:
    return os.environ.get("PMET_DATASETS_CACHE", str(Path("caches").resolve()))


def _default_wikipedia_dir() -> Path | None:
    env = os.environ.get("PMET_WIKIPEDIA_DIR", "").strip()
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    # D-PIKE layout: repo_root/data/wikipedia
    repo = Path(__file__).resolve().parents[4]
    p = repo / "data" / "wikipedia"
    return p if p.is_dir() else None


def _load_local_wikipedia_parquet(wiki_dir: Path, split: str = "train") -> Dataset:
    """Load HF-exported parquet shards from a local directory (no Hub)."""
    pattern = str(wiki_dir / f"{split}-*.parquet")
    shards = sorted(wiki_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} parquet under {wiki_dir} (pattern {pattern})")
    cache_dir = _mom2_cache_dir()
    ds = load_dataset(
        "parquet",
        data_files={split: pattern},
        split=split,
        cache_dir=cache_dir,
    )
    if "text" not in ds.column_names:
        raise ValueError(f"Expected 'text' column in {wiki_dir} parquet, got {ds.column_names}")
    n_before = len(ds)
    ds = ds.filter(lambda row: bool((row.get("text") or "").strip()), num_proc=1)
    print(
        f"[layer_stats] Local Wikipedia parquet {wiki_dir} split={split}: "
        f"{len(ds)} non-empty rows ({n_before - len(ds)} empty skipped)."
    )
    return ds


def _load_local_mom2_text_file(path: str) -> Dataset:
    """
    Offline / air-gapped mom2 corpus: UTF-8 text file or JSONL with a `text` field
    (one record per line). Set PMET_MOM2_TEXT_FILE=/abs/path/to/file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PMET_MOM2_TEXT_FILE not found: {path}")
    texts: list[str] = []
    suf = p.suffix.lower()
    with p.open("r", encoding="utf-8", errors="replace") as handle:
        if suf == ".jsonl":
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text")
                if t is None:
                    t = obj.get("source_text") or obj.get("target_text")
                if t is None and "input" in obj:
                    t = (obj.get("input") or "") + "\n" + (obj.get("output") or "")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        else:
            for line in handle:
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        raise ValueError(f"PMET_MOM2_TEXT_FILE produced no text rows: {path}")
    print(f"[layer_stats] Using local mom2 corpus {path} ({len(texts)} segments).")
    return Dataset.from_dict({"text": texts})


def load_mom2_text_corpus(ds_name: str):
    """
    Load plain-text corpus for mom2 / layer_stats.
    Newer `datasets` no longer ships script-based `wikipedia` the old way; we default
    to wikitext-103-raw-v1 (standard MEMIT/ROME substitute when Wikipedia is unavailable).
    """
    local = os.environ.get("PMET_MOM2_TEXT_FILE", "").strip()
    if local:
        return _load_local_mom2_text_file(local)

    cache_dir = _mom2_cache_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if ds_name == "wikitext":
        try:
            return load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir)
        except Exception:
            return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir)

    if ds_name == "wikipedia":
        wiki_dir = _default_wikipedia_dir()
        if wiki_dir is not None:
            return _load_local_wikipedia_parquet(wiki_dir, split="train")

        # Try newer parquet-style hubs first; fall back to wikitext for robustness.
        candidates = [
            ("wikimedia/wikipedia", "20231101.en"),
            ("wikipedia", "20220301.en"),
        ]
        last_err = None
        for repo, rev in candidates:
            try:
                return load_dataset(
                    repo,
                    rev,
                    split="train",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
            except Exception as exc:
                last_err = exc
                continue
        print(
            f"[layer_stats] Wikipedia corpus load failed ({last_err!r}); "
            "falling back to wikitext-103-raw-v1 (mom2 file path still uses ds_name=wikipedia)."
        )
        return load_mom2_text_corpus("wikitext")

    raise ValueError(f"Unknown mom2 dataset: {ds_name}")


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME_ATTN Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--model_path", default="../../ptms/")
    aa("--model_name", default="EleutherAI/gpt-j-6B", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[8], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["muiltmean"], type=lambda x: x.split(","))
    aa("--sample_size", default=100, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=10000, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path + args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        for layer_name in [f"transformer.h.{layer_num}.attn.out_proj", f"transformer.h.{layer_num}.mlp.fc_out"]:
            layer_stats(
                model,
                tokenizer,
                layer_name,
                args.stats_dir,
                args.dataset,
                args.to_collect,
                sample_size=args.sample_size,
                precision=args.precision,
                batch_tokens=args.batch_tokens,
                download=args.download,
            )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_train = load_mom2_text_corpus(ds_name)
        try:
            maxlen = model.config.n_positions
        except Exception:
            maxlen = model.config.max_position_embeddings
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_train, tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    try:
        npos = model.config.n_positions
    except:
        npos = model.config.max_position_embeddings
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = f"_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if force_recompute and filename.exists():
        filename.unlink()

    if not filename.exists() and download:
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    #compute cached stats
    try:
        ds = get_ds() if not filename.exists() else None
    except:
        print("get_ds failed, try again")
        ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = tqdm

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    num_workers = int(os.environ.get("PMET_DATALOADER_WORKERS", "2"))
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=num_workers,
    )
    if sample_size is not None:
        n_for_batches = sample_size
    elif ds is not None:
        n_for_batches = len(ds)
    else:
        n_for_batches = 1
    batch_count = -(-n_for_batches // batch_size)
    dev = next(model.parameters()).device
    with torch.no_grad():
        if not filename.exists():
            for batch_group in progress(loader, total=batch_count):
                for batch in batch_group:
                    batch = dict_to_(batch, dev)
                    with Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                    ) as tr:
                        if "neox" in model.config._name_or_path:
                            del batch['position_ids']
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
