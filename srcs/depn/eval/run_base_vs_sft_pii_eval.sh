#!/usr/bin/env bash
# Base vs SFT plain 前缀补全 PII 评估（test split，不含 edited 层）
#
# 用法（仓库根目录）：
#   bash srcs/depn/eval/run_base_vs_sft_pii_eval.sh
#
# 可选：
#   LIMIT_PER_TYPE_TEST=200   每类抽样（默认 0 = test 全量）
#   SPLITS="test train"       额外跑 train-seen（train 默认每类 200）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

export BASE_MODEL_DIR="${BASE_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/baseline}"
export SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/outputs/depn/eval/base_vs_sft_plain_qlora}"
export LAYERS="${LAYERS:-base sft}"
export SPLITS="${SPLITS:-test}"
export SKIP_SPLIT="${SKIP_SPLIT:-1}"
export LIMIT_PER_TYPE_TEST="${LIMIT_PER_TYPE_TEST:-0}"
export LIMIT_PER_TYPE_TRAIN="${LIMIT_PER_TYPE_TRAIN:-200}"
export LIMIT_PER_TYPE_VAL="${LIMIT_PER_TYPE_VAL:-0}"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export EDITED_MODEL_DIR=""
export EDITED_ADAPTER_DIR=""

exec bash "${REPO_ROOT}/srcs/depn/eval/run_three_layer_pii_eval.sh"
