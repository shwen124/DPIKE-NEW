#!/usr/bin/env bash
# 全量 plain prefix QLoRA（仓库根目录执行）
#   bash scripts/depn/run_llama3_api4_prefix_qlora_full.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export LLAMA_TRAIN_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction.json}"
export LLAMA_VAL_JSON="${LLAMA_VAL_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction_val.json}"
export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
export LLAMA_OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/2026-05-24_api4_prefix_plain}"
export LLAMA_FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
export LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${REPO_ROOT}/logs/depn/train/run_llama3_api4_prefix_qlora_full_$(date +%Y%m%d_%H%M%S).log}"
export LLAMA_RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

exec bash "${REPO_ROOT}/srcs/depn/train/run_llama3_api4_prefix_qlora.sh"
