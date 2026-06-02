#!/usr/bin/env bash
# 三层 PII 前缀补全评估：base / sft / edited，在 train / val / test 上分别跑 plain {input} 协议。
#
# 用法（仓库根目录）：
#   bash srcs/depn/eval/run_three_layer_pii_eval.sh
#
# 环境变量：
#   REPO_ROOT              仓库根（自动检测）
#   DATA_DIR               数据目录（默认 ${REPO_ROOT}/data/api4_200k，若不存在则用 ${REPO_ROOT}）
#   BASE_MODEL_DIR         Llama3-8B 基座
#   SFT_ADAPTER_DIR        完整 SFT LoRA adapter
#   EDITED_MODEL_DIR       编辑后全量模型（与 EDITED_ADAPTER_DIR 二选一）
#   EDITED_ADAPTER_DIR     编辑后 LoRA adapter（需配合 BASE_MODEL_DIR）
#   RESULTS_DIR            输出目录（默认 outputs/depn/eval/three_layer）
#   LIMIT_PER_TYPE_TRAIN   train-seen 每类上限（默认 200；正式结果设 0）
#   LIMIT_PER_TYPE_VAL     val 每类上限（默认 0）
#   LIMIT_PER_TYPE_TEST    test 每类上限（默认 0）
#   MAX_NEW_TOKENS         生成长度（默认 96）
#   GENERATION_EXTRA       额外 token（默认 16）
#   SKIP_SPLIT             1 = 不自动切分，假定已有 *_train/_val/_test.json
#   RUN_MASKED_ABLATION    1 = 额外跑 masked test 消融（需 sft adapter）
#   LAYERS                 空格分隔要跑的层，默认 "base sft edited"（无 edited 路径时自动跳过 edited）
#   SPLITS                 空格分隔 split，默认 "test"（可用 "train val test"）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EVAL_PY="${REPO_ROOT}/srcs/depn/eval/eval_sft_true_prefix_pii_metrics.py"
SPLIT_PY="${REPO_ROOT}/srcs/depn/eval/split_sft_by_source_id.py"
COMPARE_PY="${REPO_ROOT}/srcs/depn/eval/compare_pii_eval_layers.py"

if [ -d "${REPO_ROOT}/data/api4_200k" ]; then
  DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/api4_200k}"
else
  DATA_DIR="${DATA_DIR:-${REPO_ROOT}}"
fi

BASE_MODEL_DIR="${BASE_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/baseline}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
EDITED_MODEL_DIR="${EDITED_MODEL_DIR:-}"
EDITED_ADAPTER_DIR="${EDITED_ADAPTER_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/outputs/depn/eval/three_layer}"

LIMIT_PER_TYPE_TRAIN="${LIMIT_PER_TYPE_TRAIN:-200}"
LIMIT_PER_TYPE_VAL="${LIMIT_PER_TYPE_VAL:-0}"
LIMIT_PER_TYPE_TEST="${LIMIT_PER_TYPE_TEST:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
GENERATION_EXTRA="${GENERATION_EXTRA:-16}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
RUN_MASKED_ABLATION="${RUN_MASKED_ABLATION:-0}"
LAYERS="${LAYERS:-base sft edited}"
SPLITS="${SPLITS:-test}"

STEM="sft_true_prefix_no_instruction"
FULL_JSON="${DATA_DIR}/${STEM}.json"
REFERENCE_JSONL="${REFERENCE_JSONL:-${DATA_DIR}/english_pii_43k.jsonl}"

mkdir -p "${RESULTS_DIR}"

banner() { printf '%s\n' "$@"; }

# ---------- 1. 按 source_id 切分（若尚未存在） ----------
if [ "${SKIP_SPLIT}" != "1" ]; then
  need_split=0
  for sp in train val test; do
    if [ ! -f "${DATA_DIR}/${STEM}_${sp}.json" ]; then
      need_split=1
      break
    fi
  done
  if [ "${need_split}" = "1" ]; then
    if [ ! -f "${FULL_JSON}" ]; then
      echo "[run_three_layer_pii_eval] 错误: 未找到 ${FULL_JSON}，也无法切分 train/val/test。" >&2
      exit 1
    fi
    banner "[run_three_layer_pii_eval] 按 source_id 切分 -> ${DATA_DIR}"
    python "${SPLIT_PY}" \
      --input "${FULL_JSON}" \
      --output_dir "${DATA_DIR}" \
      --stem "${STEM}"
  fi
fi

run_eval() {
  local layer="$1"
  local split="$2"
  local dataset="$3"
  local limit="$4"
  local out_tag="$5"
  local extra_args=()

  case "${layer}" in
    base)
      extra_args=(--model_dir "${BASE_MODEL_DIR}")
      ;;
    sft)
      extra_args=(--base_model_dir "${BASE_MODEL_DIR}" --adapter_dir "${SFT_ADAPTER_DIR}")
      ;;
    edited)
      if [ -n "${EDITED_MODEL_DIR}" ]; then
        extra_args=(--model_dir "${EDITED_MODEL_DIR}")
      elif [ -n "${EDITED_ADAPTER_DIR}" ]; then
        extra_args=(--base_model_dir "${BASE_MODEL_DIR}" --adapter_dir "${EDITED_ADAPTER_DIR}")
      else
        banner "[run_three_layer_pii_eval] 跳过 edited（未设置 EDITED_MODEL_DIR / EDITED_ADAPTER_DIR）"
        return 0
      fi
      ;;
    *)
      echo "[run_three_layer_pii_eval] 未知 layer: ${layer}" >&2
      return 1
      ;;
  esac

  local out_dir="${RESULTS_DIR}/${layer}"
  mkdir -p "${out_dir}"
  local csv="${out_dir}/${out_tag}.csv"
  local json="${out_dir}/${out_tag}.json"

  if [ -f "${json}" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    banner "[SKIP] 已存在 ${json}"
    return 0
  fi

  banner "[RUN] layer=${layer} split=${split} dataset=${dataset} limit_per_type=${limit}"
  python "${EVAL_PY}" \
    --dataset "${dataset}" \
    --split "${split}" \
    --reference_jsonl "${REFERENCE_JSONL}" \
    --output_csv "${csv}" \
    --output_json "${json}" \
    --prompt_template "{input}" \
    --layer "${layer}" \
    --run_name "${out_tag}" \
    --load_in_4bit \
    --limit_per_type "${limit}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --generation_extra_tokens "${GENERATION_EXTRA}" \
    "${extra_args[@]}"
}

limit_for_split() {
  case "$1" in
    train) echo "${LIMIT_PER_TYPE_TRAIN}" ;;
    val)   echo "${LIMIT_PER_TYPE_VAL}" ;;
    test)  echo "${LIMIT_PER_TYPE_TEST}" ;;
    *)     echo "0" ;;
  esac
}

# ---------- 2. 三层 × 三 split 主评估 ----------
for split in ${SPLITS}; do
  dataset="${DATA_DIR}/${STEM}_${split}.json"
  if [ ! -f "${dataset}" ]; then
    echo "[run_three_layer_pii_eval] 警告: 缺少 ${dataset}，跳过 split=${split}" >&2
    continue
  fi
  limit="$(limit_for_split "${split}")"
  for layer in ${LAYERS}; do
    run_eval "${layer}" "${split}" "${dataset}" "${limit}" "${layer}_plain_${split}"
  done

  # ---------- 3. 对比表（base vs sft vs edited） ----------
  base_json="${RESULTS_DIR}/base/base_plain_${split}.json"
  sft_json="${RESULTS_DIR}/sft/sft_plain_${split}.json"
  edited_json="${RESULTS_DIR}/edited/edited_plain_${split}.json"
  cmp_args=(--base "${base_json}" --sft "${sft_json}" --output_dir "${RESULTS_DIR}/comparison" --split "${split}")
  if [ -f "${edited_json}" ]; then
    cmp_args+=(--edited "${edited_json}")
  fi
  if [ -f "${base_json}" ] && [ -f "${sft_json}" ]; then
    banner "[COMPARE] split=${split}"
    python "${COMPARE_PY}" "${cmp_args[@]}"
  fi
done

# ---------- 4. masked 前缀消融（仅 test + sft） ----------
if [ "${RUN_MASKED_ABLATION}" = "1" ]; then
  masked_test="${DATA_DIR}/sft_true_prefix_masked_test.json"
  if [ ! -f "${masked_test}" ]; then
    masked_full="${DATA_DIR}/sft_true_prefix_masked.json"
    if [ -f "${masked_full}" ] && [ "${SKIP_SPLIT}" != "1" ]; then
      python "${SPLIT_PY}" --input "${masked_full}" --output_dir "${DATA_DIR}" --stem "sft_true_prefix_masked"
    fi
  fi
  if [ -f "${masked_test}" ]; then
    run_eval "sft" "test" "${masked_test}" "${LIMIT_PER_TYPE_TEST}" "sft_plain_masked_test"
  else
    banner "[WARN] masked test 不存在，跳过消融"
  fi
fi

banner "[run_three_layer_pii_eval] 完成。结果目录: ${RESULTS_DIR}"
banner "  各层 JSON: ${RESULTS_DIR}/{base,sft,edited}/*_plain_*.json"
banner "  对比表:    ${RESULTS_DIR}/comparison/table*.csv"
