#!/usr/bin/env bash
# Llama3-8B LoRA/QLoRA fine-tuning (stable, speed-tuned for 24GB GPUs like RTX 4090).
# Paths can still be overridden via env vars; defaults keep the existing repo layout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/srcs/depn/train"
AI4="${REPO_ROOT}/data/depn/ai4privacy/pii-masking"

BASE_MODEL_PATH="${LLAMA_BASE_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
INIT_ADAPTER_PATH="${LLAMA_INIT_ADAPTER_PATH:-}"
TRAIN_FILE="${LLAMA_TRAIN_FILE:-${AI4}}"
VALID_FILE="${LLAMA_VALID_FILE-}"
TEXT_COLUMN="${LLAMA_TEXT_COLUMN:-source_text}"

# Speed-oriented defaults for 24GB VRAM.
TRAIN_BATCH_SIZE="${LLAMA_TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${LLAMA_EVAL_BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${LLAMA_GRAD_ACC_STEPS:-4}"
BLOCK_SIZE="${LLAMA_BLOCK_SIZE:-512}"
DATALOADER_WORKERS="${LLAMA_DATALOADER_WORKERS:-4}"
USE_GRADIENT_CHECKPOINTING="${LLAMA_GRADIENT_CHECKPOINTING:-0}"
USE_GROUP_BY_LENGTH="${LLAMA_GROUP_BY_LENGTH:-1}"

OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/ai4privacy_pii-masking_all}"
FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/depn_ai4privacy_lora_multilang}"
ACCELERATE_CLI="${REPO_ROOT}/srcs/depn/utils/accelerate_cli.py"

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}"
LOG_DIR="${REPO_ROOT}/logs/depn/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_llama3_stable_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_llama3_stable] LOG_FILE=${LOG_FILE}"
echo "[run_llama3_stable] REPO_ROOT=${REPO_ROOT}"
echo "[run_llama3_stable] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[run_llama3_stable] INIT_ADAPTER_PATH=${INIT_ADAPTER_PATH}"
echo "[run_llama3_stable] TRAIN_FILE=${TRAIN_FILE}"
echo "[run_llama3_stable] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_llama3_stable] FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"

cd "${TRAIN_DIR}"

cmd=(
    python -u "${ACCELERATE_CLI}" launch
    --num_processes 1
    --num_machines 1
    --main_process_port 12357
    --mixed_precision bf16
    run_clm_no_trainer.py
    --model_name_or_path "${BASE_MODEL_PATH}"
    --config_name "${BASE_MODEL_PATH}"
    --tokenizer_name "${BASE_MODEL_PATH}"
    --train_file "${TRAIN_FILE}"
    --use_lora
    --load_in_4bit
    --bnb_4bit_use_double_quant
    --lora_alpha 64
    --num_train_epochs 3
    --checkpointing_steps 5000
    --keep_last_checkpoints 3
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
    --learning_rate 2e-4
    --weight_decay 0.0
    --lr_scheduler_type cosine
    --num_warmup_steps 500
    --block_size "${BLOCK_SIZE}"
    --per_document_sequences
    --dataloader_num_workers "${DATALOADER_WORKERS}"
    --output_dir "${OUTPUT_DIR}"
    --torch_dtype bfloat16
    --low_cpu_mem_usage
    --max_grad_norm 1.0
)

if [ -n "${INIT_ADAPTER_PATH}" ]; then
    cmd+=(--adapter_name_or_path "${INIT_ADAPTER_PATH}")
fi

if [ -n "${TEXT_COLUMN}" ]; then
    cmd+=(--text_column "${TEXT_COLUMN}")
fi

if [ -n "${VALID_FILE}" ]; then
    cmd+=(--validation_file "${VALID_FILE}")
fi

if [ "${USE_GRADIENT_CHECKPOINTING}" = "1" ]; then
    cmd+=(--gradient_checkpointing)
fi

if [ "${USE_GROUP_BY_LENGTH}" = "1" ]; then
    cmd+=(--group_by_length)
fi

TOKENIZERS_PARALLELISM=false \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${cmd[@]}"

if [ -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    for f in adapter_config.json adapter_model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json README.md; do
        if [ -f "${OUTPUT_DIR}/${f}" ]; then
            cp -f "${OUTPUT_DIR}/${f}" "${FINAL_MODEL_DIR}/"
        fi
    done
    echo "[run_llama3_stable] Copied final adapter/tokenizer to ${FINAL_MODEL_DIR}"
fi
