#!/usr/bin/env bash
# 小规模联调：plain completion + no_instruction，1000 样本 × 10 epoch
# 用法（仓库根目录）：bash scripts/depn/run_llama3_api4_prefix_qlora_smoke.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/srcs/depn/train"
export LLAMA_TRAIN_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction.json}"
export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
export LLAMA_OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/smoke_1000x10_plain}"
export LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${REPO_ROOT}/logs/depn/train/run_llama3_api4_prefix_qlora_smoke_$(date +%Y%m%d_%H%M%S).log}"
export LLAMA_RESUME_FROM_CHECKPOINT=""

mkdir -p "$(dirname "${LLAMA_LOG_FILE}")" "${LLAMA_OUTPUT_DIR}"

if [ "${SKIP_CUDA_CHECK:-0}" != "1" ]; then
    if ! python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "[smoke] 需要 CUDA；或设 SKIP_CUDA_CHECK=1（不推荐 QLoRA）" >&2
        exit 1
    fi
fi

ACCELERATE_CLI="${REPO_ROOT}/srcs/depn/utils/accelerate_cli.py"
{
    echo "[smoke] DATA=${LLAMA_TRAIN_JSON}"
    echo "[smoke] OUT=${LLAMA_OUTPUT_DIR}"
    cd "${TRAIN_DIR}"
    python -u "${ACCELERATE_CLI}" launch \
        --num_processes 1 \
        --num_machines 1 \
        --main_process_port 12360 \
        run_clm_no_trainer.py \
        --model_name_or_path "${LLAMA_MODEL_PATH}" \
        --train_file "${LLAMA_TRAIN_JSON}" \
        --config_name "${LLAMA_MODEL_PATH}" \
        --tokenizer_name "${LLAMA_MODEL_PATH}" \
        --sft_plain_completion \
        --use_lora \
        --load_in_4bit \
        --bnb_4bit_use_double_quant \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --max_train_samples 1000 \
        --num_train_epochs 10 \
        --checkpointing_steps 250 \
        --keep_last_checkpoints 2 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 20 \
        --max_grad_norm 0.3 \
        --block_size 512 \
        --group_by_length \
        --output_dir "${LLAMA_OUTPUT_DIR}" \
        --torch_dtype bfloat16 \
        --low_cpu_mem_usage \
        --gradient_checkpointing \
        --dataloader_num_workers 2
} 2>&1 | tee -a "${LLAMA_LOG_FILE}"

echo "[smoke] done. adapter at ${LLAMA_OUTPUT_DIR}"
