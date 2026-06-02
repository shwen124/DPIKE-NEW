#!/usr/bin/env bash
# Llama3-8B QLoRA SFT：plain completion（input+output，无 instruction），适配 sft_true_prefix_no_instruction.json
# 单卡 4090（24GB）推荐：4bit + LoRA + gradient checkpointing；有效 batch = per_device * grad_accum。
#
# 环境变量（可选）：
#   LLAMA_MODEL_PATH     基座（默认 REPO/models/llama3-8B/baseline）
#   LLAMA_TRAIN_JSON     训练 JSON（默认 REPO/data/api4_200k/sft_true_prefix_no_instruction.json）
#   LLAMA_VAL_JSON       验证 JSON（默认 *_val.json；按 source_id 切分，避免随机拆 train）
#   LLAMA_OUTPUT_DIR     checkpoint 目录（step_* + 最终 adapter）
#   LLAMA_FINAL_MODEL_DIR  训练结束后同步 adapter 到此目录（默认 models/llama3-8B/api4_prefix_plain_qlora）
#   LLAMA_RESUME_FROM_CHECKPOINT  断点；未设置时默认不续训（从基座从头）。续训示例： LLAMA_RESUME_FROM_CHECKPOINT=/path/to/step_7500
#   LLAMA_LOG_FILE       日志文件
#
# 步数参考（默认 num_train_epochs=2, grad_accum=16, per_device=1）：
#   训练集约 10.5 万（train split）；验证集见 LLAMA_VAL_JSON；每 epoch 步数约 ceil(N_train / 16)；2 epoch 约 2×。
#
# 若在无 GPU 的环境（如部分远程沙箱）运行会失败；设 SKIP_CUDA_CHECK=1 可跳过检查（不推荐 QLoRA）。
#
# 训练进程经 nohup 脱离终端：关闭 Cursor/SSH 窗口不会 SIGHUP 中断；日志追加到 LOG_FILE，PID 写入 OUTPUT_DIR/train.pid。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/srcs/depn/train"
DATA_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction.json}"
VAL_JSON="${LLAMA_VAL_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction_val.json}"
MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/2026-05-24_api4_prefix_plain}"
FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
ACCELERATE_CLI="${REPO_ROOT}/srcs/depn/utils/accelerate_cli.py"
RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

if [ "${SKIP_CUDA_CHECK:-0}" != "1" ]; then
    if ! python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "[run_llama3_api4_prefix_qlora] 错误: 当前环境 torch.cuda.is_available() 为 False，无法加载 4bit 权重。" >&2
        echo "[run_llama3_api4_prefix_qlora] 请在装有 NVIDIA 驱动与 CUDA 的本机终端执行本脚本（4090 机器上直接 bash 即可）。" >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}"
LOG_DIR="${REPO_ROOT}/logs/depn/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_llama3_api4_prefix_qlora_$(date +%Y%m%d_%H%M%S).log}"

banner() {
    printf '%s\n' "$@" | tee -a "${LOG_FILE}"
}

banner "[run_llama3_api4_prefix_qlora] LOG_FILE=${LOG_FILE}"
banner "[run_llama3_api4_prefix_qlora] MODEL_PATH=${MODEL_PATH}"
banner "[run_llama3_api4_prefix_qlora] DATA_JSON=${DATA_JSON}"
banner "[run_llama3_api4_prefix_qlora] VAL_JSON=${VAL_JSON}"
banner "[run_llama3_api4_prefix_qlora] OUTPUT_DIR=${OUTPUT_DIR}"
banner "[run_llama3_api4_prefix_qlora] FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    banner "[run_llama3_api4_prefix_qlora] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
else
    banner "[run_llama3_api4_prefix_qlora] RESUME_FROM_CHECKPOINT=(none)"
fi

cmd=(
    python -u "${ACCELERATE_CLI}" launch
    --num_processes 1
    --num_machines 1
    --main_process_port 12359
    run_clm_no_trainer.py
    --model_name_or_path "${MODEL_PATH}"
    --train_file "${DATA_JSON}"
    --validation_file "${VAL_JSON}"
    --config_name "${MODEL_PATH}"
    --tokenizer_name "${MODEL_PATH}"
    --sft_plain_completion
    --use_lora
    --load_in_4bit
    --bnb_4bit_use_double_quant
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --num_train_epochs 2
    --checkpointing_steps 1500
    --keep_last_checkpoints 3
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 16
    --learning_rate 2e-4
    --lr_scheduler_type cosine
    --num_warmup_steps 200
    --max_grad_norm 0.3
    --block_size 512
    --group_by_length
    --output_dir "${OUTPUT_DIR}"
    --torch_dtype bfloat16
    --low_cpu_mem_usage
    --gradient_checkpointing
    --dataloader_num_workers 2
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

nohup env PYTHONUNBUFFERED=1 \
    REPO_ROOT="${REPO_ROOT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    FINAL_MODEL_DIR="${FINAL_MODEL_DIR}" \
    bash -c '
set -euo pipefail
cd "$1"
shift
"$@"
echo "[run_llama3_api4_prefix_qlora] training finished; syncing adapter to FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
mkdir -p "${FINAL_MODEL_DIR}"
rsync -a --delete \
    --exclude "step_*" \
    --exclude "epoch_*" \
    --exclude "train.pid" \
    "${OUTPUT_DIR}/" "${FINAL_MODEL_DIR}/"
echo "[run_llama3_api4_prefix_qlora] synced to ${FINAL_MODEL_DIR}"
' _ "${TRAIN_DIR}" "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUTPUT_DIR}/train.pid"
banner "[run_llama3_api4_prefix_qlora] nohup 已启动 detached PID=${TRAIN_PID}（train.pid 已写入 OUTPUT_DIR）"
banner "[run_llama3_api4_prefix_qlora] 查看日志: tail -f ${LOG_FILE}"
banner "[run_llama3_api4_prefix_qlora] 停止训练: kill ${TRAIN_PID}"
