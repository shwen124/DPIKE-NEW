#!/usr/bin/env bash
# Evaluate privacy/quality metrics for a two-stage LoRA model on:
# 1) the added dataset corpus
# 2) the original Enron corpus
#
# This wraps `eval_privacy_metrics_llama_lora.py`, which now reports:
# - multi-PII token-level span MRR (NAME/TEL/EMAIL/ID_CARD/DRIVER_LICENSE/...)
# - RANDOM text PPL
# - VALID PPL

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/srcs/depn/eval"
DEPN_DATA="${REPO_ROOT}/data/depn"
PII_MASKING_DIR="${DEPN_DATA}/ai4privacy/pii-masking"

BASE_MODEL_DIR="${LLAMA_BASE_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
ADAPTER_DIR="${LLAMA_ADAPTER_PATH:-${REPO_ROOT}/models/llama3-8B/depn_enron_continue_lora}"

DATASETS_TRAIN_SOURCE="${LLAMA_DATASETS_TRAIN_SOURCE:-${PII_MASKING_DIR}}"
DATASETS_VALID_SOURCE="${LLAMA_DATASETS_VALID_SOURCE:-${PII_MASKING_DIR}/english_pii_43k.jsonl}"
DATASETS_TEXT_COLUMN="${LLAMA_DATASETS_TEXT_COLUMN:-source_text}"
DATASETS_VALID_TEXT_COLUMN="${LLAMA_DATASETS_VALID_TEXT_COLUMN:-${DATASETS_TEXT_COLUMN}}"
DATASETS_ALL_TEL_FILE="${LLAMA_DATASETS_ALL_TEL_FILE:-${DEPN_DATA}/all_Tel.txt}"
DATASETS_ALL_NAME_FILE="${LLAMA_DATASETS_ALL_NAME_FILE:-}"
PII_TYPES="${LLAMA_PII_TYPES:-}"
PII_LIMIT_PER_TYPE="${LLAMA_PII_LIMIT_PER_TYPE:-200}"
PII_EVAL_MAX_CONTEXT="${LLAMA_PII_EVAL_MAX_CONTEXT:-256}"
RANDOM_LINES="${LLAMA_RANDOM_LINES:-300}"
RANDOM_BLOCKS="${LLAMA_RANDOM_BLOCKS:-300}"
VALID_BLOCKS="${LLAMA_VALID_BLOCKS:-400}"

ENRON_TRAIN_SOURCE="${LLAMA_ENRON_TRAIN_SOURCE:-${DEPN_DATA}/temp_data/train.txt}"
ENRON_VALID_SOURCE="${LLAMA_ENRON_VALID_SOURCE:-${DEPN_DATA}/temp_data/valid.txt}"
ENRON_ALL_TEL_FILE="${LLAMA_ENRON_ALL_TEL_FILE:-${DEPN_DATA}/all_Tel.txt}"
ENRON_ALL_NAME_FILE="${LLAMA_ENRON_ALL_NAME_FILE:-}"

OUTPUT_ROOT="${LLAMA_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/outputs/depn/eval/multistage_privacy_metrics}"
LOG_DIR="${REPO_ROOT}/logs/depn/eval"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_eval_privacy_metrics_llama_lora_multistage_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_eval_privacy_metrics_llama_lora_multistage] LOG_FILE=${LOG_FILE}"
echo "[run_eval_privacy_metrics_llama_lora_multistage] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "[run_eval_privacy_metrics_llama_lora_multistage] ADAPTER_DIR=${ADAPTER_DIR}"
echo "[run_eval_privacy_metrics_llama_lora_multistage] DATASETS_TRAIN_SOURCE=${DATASETS_TRAIN_SOURCE}"
echo "[run_eval_privacy_metrics_llama_lora_multistage] ENRON_TRAIN_SOURCE=${ENRON_TRAIN_SOURCE}"

cd "${EVAL_DIR}"

echo "===== DATASETS METRICS ====="
datasets_cmd=(
    env
    PYTORCH_ALLOC_CONF=expandable_segments:True
    python -u eval_privacy_metrics_llama_lora.py
    --base_model_dir "${BASE_MODEL_DIR}"
    --adapter_dir "${ADAPTER_DIR}"
    --train_file "${DATASETS_TRAIN_SOURCE}"
    --valid_file "${DATASETS_VALID_SOURCE}"
    --train_text_column "${DATASETS_TEXT_COLUMN}"
    --valid_text_column "${DATASETS_VALID_TEXT_COLUMN}"
    --pii_limit_per_type "${PII_LIMIT_PER_TYPE}"
    --pii_eval_max_context "${PII_EVAL_MAX_CONTEXT}"
    --random_lines "${RANDOM_LINES}"
    --random_blocks "${RANDOM_BLOCKS}"
    --valid_blocks "${VALID_BLOCKS}"
)
if [ -n "${DATASETS_ALL_TEL_FILE}" ]; then
    datasets_cmd+=(--all_tel_file "${DATASETS_ALL_TEL_FILE}")
fi
if [ -n "${DATASETS_ALL_NAME_FILE}" ]; then
    datasets_cmd+=(--all_name_file "${DATASETS_ALL_NAME_FILE}")
fi
if [ -n "${PII_TYPES}" ]; then
    datasets_cmd+=(--pii_types "${PII_TYPES}")
fi
"${datasets_cmd[@]}"

echo "===== ENRON METRICS ====="
enron_cmd=(
    env
    PYTORCH_ALLOC_CONF=expandable_segments:True
    python -u eval_privacy_metrics_llama_lora.py
    --base_model_dir "${BASE_MODEL_DIR}"
    --adapter_dir "${ADAPTER_DIR}"
    --train_file "${ENRON_TRAIN_SOURCE}"
    --valid_file "${ENRON_VALID_SOURCE}"
    --pii_limit_per_type "${PII_LIMIT_PER_TYPE}"
    --pii_eval_max_context "${PII_EVAL_MAX_CONTEXT}"
    --random_lines "${RANDOM_LINES}"
    --random_blocks "${RANDOM_BLOCKS}"
    --valid_blocks "${VALID_BLOCKS}"
)
if [ -n "${ENRON_ALL_TEL_FILE}" ]; then
    enron_cmd+=(--all_tel_file "${ENRON_ALL_TEL_FILE}")
fi
if [ -n "${ENRON_ALL_NAME_FILE}" ]; then
    enron_cmd+=(--all_name_file "${ENRON_ALL_NAME_FILE}")
fi
if [ -n "${PII_TYPES}" ]; then
    enron_cmd+=(--pii_types "${PII_TYPES}")
fi
"${enron_cmd[@]}"

echo "[run_eval_privacy_metrics_llama_lora_multistage] Completed."
