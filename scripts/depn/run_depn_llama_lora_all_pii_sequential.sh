#!/usr/bin/env bash
# 顺序跑六种 PII 的 DEPN（build → attribution → filter → edit+指标）。
# 用法：bash scripts/depn/run_depn_llama_lora_all_pii_sequential.sh
# 可通过环境变量覆盖 LLAMA_*（与 data/run_depn_llama_lora_pii.sh 一致）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/srcs/depn"

# 归因 IG 显存敏感；单机可再降到 2 或 1
export LLAMA_STEP1_BATCH_SIZE="${LLAMA_STEP1_BATCH_SIZE:-4}"
export LLAMA_STEP1_NUM_BATCH="${LLAMA_STEP1_NUM_BATCH:-10}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MASTER_LOG="${REPO_ROOT}/logs/depn/depn/run_all_pii_erasure_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${MASTER_LOG}")"

{
  echo "=== run_all_pii_erasure start $(date -Iseconds) ==="
  echo "REPO_ROOT=${REPO_ROOT}"
  for PII in EMAIL ID_CARD PASSWORD VEHICLE_VIN NAME TEL; do
    echo ""
    echo ">>> ========== PII_TYPE=${PII} ========== $(date -Iseconds)"
    LLAMA_PII_TYPE="${PII}" bash data/run_depn_llama_lora_pii.sh || {
      echo "!!! FAILED PII_TYPE=${PII} exit=$?"
      exit 1
    }
    echo ">>> DONE PII_TYPE=${PII} $(date -Iseconds)"
  done
  echo "=== run_all_pii_erasure end $(date -Iseconds) ==="
} 2>&1 | tee -a "${MASTER_LOG}"

echo "Master log: ${MASTER_LOG}"
