#!/usr/bin/env bash
# Continue LoRA/QLoRA training on Enron after a first-stage adapter/full-model has already been trained.
# Default flow:
# 1. Base model stays the original Llama3 checkpoint.
# 2. Stage-1 adapter trained on datasets/* is loaded via LLAMA_INIT_ADAPTER_PATH.
# 3. Training continues on Enron train/valid text files.
#
# Alternative flow:
# - If you already merged stage-1 weights into a full model, set LLAMA_BASE_MODEL_PATH to that full model
#   and leave LLAMA_INIT_ADAPTER_PATH empty.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/srcs/depn/train"
ENRON_DATA="${REPO_ROOT}/data/depn/temp_data"
ACCELERATE_CLI="${REPO_ROOT}/srcs/depn/utils/accelerate_cli.py"
BUILD_MIXED_DATA_PY="${REPO_ROOT}/srcs/depn/data/build_continual_train_jsonl.py"

# 在 output_dir 下查找「可 resume」的最新检查点：须含 Accelerate 的 optimizer 状态（排除仅含 model.safetensors 的半截目录）。
_depn_latest_resumable_checkpoint() {
    local out="$1"
    local best="" best_n=-1
    local p base n
    shopt -s nullglob
    for p in "${out}/step_"* "${out}/epoch_"* "${out}/checkpoint-"*; do
        [ -d "$p" ] || continue
        base=$(basename "$p")
        n=""
        if [[ "$base" =~ ^step_([0-9]+)$ ]]; then n="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^epoch_([0-9]+)$ ]]; then n="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^checkpoint-([0-9]+)$ ]]; then n="${BASH_REMATCH[1]}"
        fi
        [ -n "$n" ] || continue
        if [ -f "$p/optimizer.bin" ] || [ -f "$p/optimizer.pt" ]; then
            if [ "$n" -gt "$best_n" ]; then best_n=$n; best="$p"; fi
        fi
    done
    shopt -u nullglob
    printf '%s\n' "$best"
}

BASE_MODEL_PATH="${LLAMA_BASE_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
INIT_ADAPTER_PATH="${LLAMA_INIT_ADAPTER_PATH:-${REPO_ROOT}/models/llama3-8B/depn_ai4privacy_lora_multilang}"
TRAIN_FILE="${LLAMA_TRAIN_FILE:-${ENRON_DATA}/train.txt}"
VALID_FILE="${LLAMA_VALID_FILE:-${ENRON_DATA}/valid.txt}"
TEXT_COLUMN="${LLAMA_TEXT_COLUMN:-}"
FOCUS_SOURCE="${LLAMA_FOCUS_SOURCE:-${REPO_ROOT}/data/depn/ai4privacy/pii-masking}"
FOCUS_PII_TYPES="${LLAMA_FOCUS_PII_TYPES:-}"
FOCUS_LIMIT_PER_TYPE="${LLAMA_FOCUS_LIMIT_PER_TYPE:-2000}"
FOCUS_REPEAT="${LLAMA_FOCUS_REPEAT:-4}"
FOCUS_BASE_LINES="${LLAMA_FOCUS_BASE_LINES:-20000}"
FOCUS_SEED="${LLAMA_FOCUS_SEED:-42}"
FOCUS_TEXT_COLUMN="${LLAMA_FOCUS_TEXT_COLUMN:-source_text}"
FOCUS_ENABLE_MIX="${LLAMA_FOCUS_ENABLE_MIX:-0}"
FOCUS_MIXED_FILE="${LLAMA_FOCUS_MIXED_FILE:-${REPO_ROOT}/outputs/depn/train/mixed_enron_focus.jsonl}"

# Training profile:
# - auto: try a faster 4090-friendly profile first, then fall back to the safe profile on CUDA OOM
# - fast: use the faster profile directly
# - safe: use the lower-memory profile directly
TRAIN_PROFILE="${LLAMA_TRAIN_PROFILE:-auto}"

# Enron continuation is more memory-heavy than stage-1 training because sequences are longer and the
# already-learned adapter is loaded back as trainable weights. For stage-2 continuation, 1 epoch is a
# better default than 3 because the stage-1 adapter is already warm-started and full Enron passes are expensive.
NUM_TRAIN_EPOCHS="${LLAMA_NUM_TRAIN_EPOCHS:-1}"

FAST_TRAIN_BATCH_SIZE="${LLAMA_FAST_TRAIN_BATCH_SIZE:-4}"
FAST_EVAL_BATCH_SIZE="${LLAMA_FAST_EVAL_BATCH_SIZE:-4}"
FAST_GRAD_ACC_STEPS="${LLAMA_FAST_GRAD_ACC_STEPS:-4}"

SAFE_TRAIN_BATCH_SIZE="${LLAMA_SAFE_TRAIN_BATCH_SIZE:-2}"
SAFE_EVAL_BATCH_SIZE="${LLAMA_SAFE_EVAL_BATCH_SIZE:-2}"
SAFE_GRAD_ACC_STEPS="${LLAMA_SAFE_GRAD_ACC_STEPS:-8}"

TRAIN_BATCH_SIZE="${LLAMA_TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${LLAMA_EVAL_BATCH_SIZE:-}"
GRAD_ACC_STEPS="${LLAMA_GRAD_ACC_STEPS:-}"
BLOCK_SIZE="${LLAMA_BLOCK_SIZE:-256}"
DATALOADER_WORKERS="${LLAMA_DATALOADER_WORKERS:-4}"
USE_GRADIENT_CHECKPOINTING="${LLAMA_GRADIENT_CHECKPOINTING:-1}"
USE_GROUP_BY_LENGTH="${LLAMA_GROUP_BY_LENGTH:-1}"
LEARNING_RATE="${LLAMA_LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${LLAMA_NUM_WARMUP_STEPS:-200}"
MAX_TRAIN_SAMPLES="${LLAMA_MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${LLAMA_MAX_EVAL_SAMPLES:-}"
# 未显式设置时默认每 5000 步存盘；若需关闭可 export LLAMA_CHECKPOINTING_STEPS=""
CHECKPOINTING_STEPS="${LLAMA_CHECKPOINTING_STEPS-5000}"
KEEP_LAST_CHECKPOINTS="${LLAMA_KEEP_LAST_CHECKPOINTS:-3}"
RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"
# 为 1 且未指定 LLAMA_RESUME_FROM_CHECKPOINT 时，自动选 output_dir 下最新可恢复检查点；全新训练请设 LLAMA_AUTO_RESUME=0
LLAMA_AUTO_RESUME="${LLAMA_AUTO_RESUME:-1}"

OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/enron_continue}"
FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/depn_enron_continue_lora}"

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}"
mkdir -p "$(dirname "${FOCUS_MIXED_FILE}")"

if [ "${FOCUS_ENABLE_MIX}" = "1" ] && [ -n "${FOCUS_PII_TYPES}" ]; then
    echo "[run_llama3_stable_enron_continue] Building mixed focus dataset..."
    python -u "${BUILD_MIXED_DATA_PY}" \
        --base_source "${TRAIN_FILE}" \
        --focus_source "${FOCUS_SOURCE}" \
        --focus_pii_types "${FOCUS_PII_TYPES}" \
        --base_text_column "${TEXT_COLUMN}" \
        --focus_text_column "${FOCUS_TEXT_COLUMN}" \
        --base_lines "${FOCUS_BASE_LINES}" \
        --focus_limit_per_type "${FOCUS_LIMIT_PER_TYPE}" \
        --focus_repeat "${FOCUS_REPEAT}" \
        --seed "${FOCUS_SEED}" \
        --output "${FOCUS_MIXED_FILE}"
    TRAIN_FILE="${FOCUS_MIXED_FILE}"
    TEXT_COLUMN="text"
fi

case "${RESUME_FROM_CHECKPOINT}" in
    latest|LAST)
        RESUME_FROM_CHECKPOINT="$(_depn_latest_resumable_checkpoint "${OUTPUT_DIR}")"
        ;;
esac
if [ -z "${RESUME_FROM_CHECKPOINT}" ] && [ "${LLAMA_AUTO_RESUME}" = "1" ]; then
    RESUME_FROM_CHECKPOINT="$(_depn_latest_resumable_checkpoint "${OUTPUT_DIR}")"
fi
LOG_DIR="${REPO_ROOT}/logs/depn/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_llama3_stable_enron_continue_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_llama3_stable_enron_continue] LOG_FILE=${LOG_FILE}"
echo "[run_llama3_stable_enron_continue] REPO_ROOT=${REPO_ROOT}"
echo "[run_llama3_stable_enron_continue] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[run_llama3_stable_enron_continue] INIT_ADAPTER_PATH=${INIT_ADAPTER_PATH}"
echo "[run_llama3_stable_enron_continue] TRAIN_FILE=${TRAIN_FILE}"
echo "[run_llama3_stable_enron_continue] TRAIN_PROFILE=${TRAIN_PROFILE}"
echo "[run_llama3_stable_enron_continue] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[run_llama3_stable_enron_continue] BLOCK_SIZE=${BLOCK_SIZE}"
echo "[run_llama3_stable_enron_continue] LEARNING_RATE=${LEARNING_RATE}"
echo "[run_llama3_stable_enron_continue] NUM_WARMUP_STEPS=${WARMUP_STEPS}"
echo "[run_llama3_stable_enron_continue] USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "[run_llama3_stable_enron_continue] FOCUS_ENABLE_MIX=${FOCUS_ENABLE_MIX}"
echo "[run_llama3_stable_enron_continue] FOCUS_PII_TYPES=${FOCUS_PII_TYPES:-<none>}"
echo "[run_llama3_stable_enron_continue] FOCUS_SOURCE=${FOCUS_SOURCE}"
echo "[run_llama3_stable_enron_continue] FOCUS_MIXED_FILE=${FOCUS_MIXED_FILE}"
echo "[run_llama3_stable_enron_continue] MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-<all>}"
echo "[run_llama3_stable_enron_continue] MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-<all>}"
echo "[run_llama3_stable_enron_continue] CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-<disabled>}"
echo "[run_llama3_stable_enron_continue] KEEP_LAST_CHECKPOINTS=${KEEP_LAST_CHECKPOINTS}"
echo "[run_llama3_stable_enron_continue] LLAMA_AUTO_RESUME=${LLAMA_AUTO_RESUME}"
echo "[run_llama3_stable_enron_continue] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "[run_llama3_stable_enron_continue] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_llama3_stable_enron_continue] FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"

cd "${TRAIN_DIR}"

run_training() {
    local profile_name="$1"
    local train_batch_size="$2"
    local eval_batch_size="$3"
    local grad_acc_steps="$4"

    echo "[run_llama3_stable_enron_continue] ACTIVE_PROFILE=${profile_name}"
    echo "[run_llama3_stable_enron_continue] TRAIN_BATCH_SIZE=${train_batch_size}"
    echo "[run_llama3_stable_enron_continue] EVAL_BATCH_SIZE=${eval_batch_size}"
    echo "[run_llama3_stable_enron_continue] GRAD_ACC_STEPS=${grad_acc_steps}"

    local cmd=(
        python -u "${ACCELERATE_CLI}" launch
        --num_processes 1
        --num_machines 1
        --main_process_port 12358
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
        --num_train_epochs "${NUM_TRAIN_EPOCHS}"
        --per_device_train_batch_size "${train_batch_size}"
        --per_device_eval_batch_size "${eval_batch_size}"
        --gradient_accumulation_steps "${grad_acc_steps}"
        --learning_rate "${LEARNING_RATE}"
        --weight_decay 0.0
        --lr_scheduler_type cosine
        --num_warmup_steps "${WARMUP_STEPS}"
        --block_size "${BLOCK_SIZE}"
        --per_document_sequences
        --dataloader_num_workers "${DATALOADER_WORKERS}"
        --output_dir "${OUTPUT_DIR}"
        --torch_dtype bfloat16
        --low_cpu_mem_usage
        --max_grad_norm 1.0
        --log_train_samples 1
    )

    if [ -n "${CHECKPOINTING_STEPS}" ]; then
        cmd+=(--checkpointing_steps "${CHECKPOINTING_STEPS}")
        cmd+=(--keep_last_checkpoints "${KEEP_LAST_CHECKPOINTS}")
    fi

    if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        if [ ! -d "${RESUME_FROM_CHECKPOINT}" ]; then
            echo "[run_llama3_stable_enron_continue] ERROR: resume path is not a directory: ${RESUME_FROM_CHECKPOINT}" >&2
            return 1
        fi
        cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    fi

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

    if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
        cmd+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
    fi

    if [ -n "${MAX_EVAL_SAMPLES}" ]; then
        cmd+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
    fi

    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${cmd[@]}"
}

if [ -z "${TRAIN_BATCH_SIZE}" ] || [ -z "${EVAL_BATCH_SIZE}" ] || [ -z "${GRAD_ACC_STEPS}" ]; then
    case "${TRAIN_PROFILE}" in
        fast)
            TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${FAST_TRAIN_BATCH_SIZE}}"
            EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${FAST_EVAL_BATCH_SIZE}}"
            GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-${FAST_GRAD_ACC_STEPS}}"
            ;;
        safe)
            TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${SAFE_TRAIN_BATCH_SIZE}}"
            EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${SAFE_EVAL_BATCH_SIZE}}"
            GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-${SAFE_GRAD_ACC_STEPS}}"
            ;;
        auto)
            :
            ;;
        *)
            echo "[run_llama3_stable_enron_continue] Unknown TRAIN_PROFILE=${TRAIN_PROFILE}. Expected auto|fast|safe." >&2
            exit 1
            ;;
    esac
fi

if [ "${TRAIN_PROFILE}" = "auto" ] && [ -z "${LLAMA_TRAIN_BATCH_SIZE:-}" ] && [ -z "${LLAMA_EVAL_BATCH_SIZE:-}" ] && [ -z "${LLAMA_GRAD_ACC_STEPS:-}" ]; then
    if run_training "fast" "${FAST_TRAIN_BATCH_SIZE}" "${FAST_EVAL_BATCH_SIZE}" "${FAST_GRAD_ACC_STEPS}"; then
        :
    else
        if grep -Eq "CUDA out of memory|OutOfMemoryError" "${LOG_FILE}"; then
            echo "[run_llama3_stable_enron_continue] Fast profile OOM detected. Falling back to safe profile."
            run_training "safe" "${SAFE_TRAIN_BATCH_SIZE}" "${SAFE_EVAL_BATCH_SIZE}" "${SAFE_GRAD_ACC_STEPS}"
        else
            exit 1
        fi
    fi
else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${SAFE_TRAIN_BATCH_SIZE}}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${SAFE_EVAL_BATCH_SIZE}}"
    GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-${SAFE_GRAD_ACC_STEPS}}"
    run_training "${TRAIN_PROFILE}" "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${GRAD_ACC_STEPS}"
fi

if [ -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    for f in adapter_config.json adapter_model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json README.md; do
        if [ -f "${OUTPUT_DIR}/${f}" ]; then
            cp -f "${OUTPUT_DIR}/${f}" "${FINAL_MODEL_DIR}/"
        fi
    done
    echo "[run_llama3_stable_enron_continue] Copied final adapter/tokenizer to ${FINAL_MODEL_DIR}"
fi
