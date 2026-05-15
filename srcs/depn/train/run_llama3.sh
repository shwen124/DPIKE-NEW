#!/usr/bin/env bash
# Llama3-8B 全参微调入口（非 LoRA）。数据默认 data/depn/ai4privacy；检查点 -> checkpoints/depn/；保留最新 3 个检查点。
#
#   LLAMA_MODEL_PATH  LLAMA_TRAIN_FILE  LLAMA_VALID_FILE  LLAMA_TEXT_COLUMN  LLAMA_RESUME_FROM_CHECKPOINT  LLAMA_LOG_FILE
# LLAMA_VALID_FILE 留空则从 train 自动划分验证集。
# 日志默认 logs/depn/train/run_llama3_full_<时间戳>.log（同时打印到终端）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/srcs/depn/train"
AI4="${REPO_ROOT}/data/depn/ai4privacy/pii-masking"

MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
TRAIN_FILE="${LLAMA_TRAIN_FILE:-${AI4}}"
VALID_FILE="${LLAMA_VALID_FILE-}"
TEXT_COLUMN="${LLAMA_TEXT_COLUMN:-source_text}"
RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_full/ai4privacy_pii-masking_all}"
ACCELERATE_CLI="${REPO_ROOT}/srcs/depn/utils/accelerate_cli.py"

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${REPO_ROOT}/logs/depn/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_llama3_full_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_llama3] LOG_FILE=${LOG_FILE}"

cd "${TRAIN_DIR}"

cmd=(
    python -u "${ACCELERATE_CLI}" launch
    --num_processes 1
    --num_machines 1
    --main_process_port 12357
    run_clm_no_trainer.py
    --model_name_or_path "${MODEL_PATH}"
    --train_file "${TRAIN_FILE}"
    --config_name "${MODEL_PATH}"
    --tokenizer_name "${MODEL_PATH}"
    --num_train_epochs 5
    --checkpointing_steps 5000
    --keep_last_checkpoints 3
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 4
    --learning_rate 2e-5
    --max_seq_length 256
    --block_size 256
    --output_dir "${OUTPUT_DIR}"
    --torch_dtype bfloat16
    --low_cpu_mem_usage
    --gradient_checkpointing
    --max_grad_norm 1.0
)

if [ -n "${TEXT_COLUMN}" ]; then
    cmd+=(--text_column "${TEXT_COLUMN}")
fi

if [ -n "${VALID_FILE}" ]; then
    cmd+=(--validation_file "${VALID_FILE}")
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${cmd[@]}"
