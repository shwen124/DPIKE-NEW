#!/usr/bin/env bash
# PMET 隐私编辑：单卡 RTX 4090（24GB）友好配置。
# - 合并 QLoRA 后以 float16 加载；每批默认 1 条并行编辑。
# - 仅编辑经 english_pii jsonl 校验的 (input,output) 隐私样本（build_privacy_requests 默认 --privacy_only）。
#
# 环境变量（可选）：
#   REPO_ROOT, BASE_MODEL, ADAPTER_DIR, SFT_JSON, REF_JSONL,
#   PMET_REQUESTS, PMET_NUM_EDITS, PMET_LIMIT_PER_TYPE, PMET_REBUILD_REQUESTS,
#   PMET_MINIMAL_CONTEXT, PMET_GRAM_ON_CPU

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
EDIT_DIR="${REPO_ROOT}/srcs/PMET-main/edit"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/models/llama3-8B/baseline}"
ADAPTER_DIR="${ADAPTER_DIR:-${REPO_ROOT}/checkpoints/depn/llama3-8b_lora/2026-05-13_api4_prefix_qlora}"
SFT_JSON="${SFT_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix.json}"
REF_JSONL="${REF_JSONL:-${REPO_ROOT}/data/api4_200k/english_pii_43k.jsonl}"
PMET_REQUESTS="${PMET_REQUESTS:-${REPO_ROOT}/data/depn/pmet_privacy_requests.json}"
PMET_NUM_EDITS="${PMET_NUM_EDITS:-1}"
PMET_LIMIT_PER_TYPE="${PMET_LIMIT_PER_TYPE:-15}"
PMET_HP="${PMET_HP:-meta_llama3-8B_4090.json}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/models/llama3-8b/pmet_privacy_v1}"
LOG_DIR="${REPO_ROOT}/logs/depn/pmet"
mkdir -p "${LOG_DIR}" "$(dirname "${PMET_REQUESTS}")"

export PMET_MINIMAL_CONTEXT="${PMET_MINIMAL_CONTEXT:-1}"
export PMET_GRAM_ON_CPU="${PMET_GRAM_ON_CPU:-1}"

if [[ ! -f "${PMET_REQUESTS}" ]] || [[ "${PMET_REBUILD_REQUESTS:-0}" == "1" ]]; then
  echo "[run_pmet_privacy] Building PMET requests -> ${PMET_REQUESTS}"
  python "${EDIT_DIR}/dsets/build_privacy_requests.py" \
    --dataset "${SFT_JSON}" \
    --reference_jsonl "${REF_JSONL}" \
    --output "${PMET_REQUESTS}" \
    --limit_per_type "${PMET_LIMIT_PER_TYPE}" \
    --privacy_only
fi

LOG_FILE="${LOG_DIR}/pmet_privacy_$(date +%Y%m%d_%H%M%S).log"
echo "[run_pmet_privacy] LOG_FILE=${LOG_FILE}"

cd "${EDIT_DIR}"
python -u evaluate_privacy.py \
  --base_model_dir "${BASE_MODEL}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --requests_json "${PMET_REQUESTS}" \
  --data_dir "$(dirname "${PMET_REQUESTS}")" \
  --hparams_fname "${PMET_HP}" \
  --num_edits "${PMET_NUM_EDITS}" \
  --torch_dtype float16 \
  --device_map cuda:0 \
  --cumulative_edits \
  --use_cache \
  --save_edited_dir "${SAVE_DIR}" \
  2>&1 | tee "${LOG_FILE}"
