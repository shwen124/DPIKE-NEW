#!/usr/bin/env bash
# Run MEMIT on LLaMA3-8B. Logs go to logs/evaluate_YYYYMMDD_HHMMSS.log (one file per run).

set -e
cd "$(dirname "$0")/.."
mkdir -p logs
LOG_FILE="logs/evaluate_$(date +%Y%m%d_%H%M%S).log"
MODEL_DIR=/data1/D-PIKE/pretrained_models/llama3-8B
SAVE_DIR=/data1/D-PIKE/pretrained_models/llama3-8B-memit-fp16-cf-full

python -m experiments.evaluate \
  --alg_name MEMIT \
  --model_name "${MODEL_DIR}" \
  --hparams_fname meta_llama-3-8b.json \
  --ds_name cf \
  --num_edits 1 \
  --skip_generation_tests \
  --generation_test_interval -1 \
  --save_edited_model_dir "${SAVE_DIR}" \
  "$@" 2>&1 | tee "$LOG_FILE"
