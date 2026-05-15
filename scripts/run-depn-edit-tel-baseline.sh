#!/usr/bin/env bash
# Run DEPN Step 4 (edit privacy neurons) for TEL, save edited model as Baseline_DEPN.
# Execute from repo root: bash scripts/run-depn-edit-tel-baseline.sh

set -e
REPO_ROOT="/data1/D-PIKE"
cd "$REPO_ROOT"

python srcs/depn/eval/3_edit_privacy_neurons_llama.py \
  --priv_data_path "${REPO_ROOT}/data/depn/memorized_TEL.txt" \
  --validation_path "${REPO_ROOT}/data/depn/temp_data/valid.txt" \
  --model_name_or_path "${REPO_ROOT}/models/llama3-8B/baseline" \
  --adapter_dir "${REPO_ROOT}/models/llama3-8B/depn_ep5_lora" \
  --kn_dir "${REPO_ROOT}/outputs/depn/kn/kn_bag-llama3_tel.json" \
  --input_prefix TEL \
  --erase_kn_num 70 \
  --max_seq_length 512 \
  --gpus 0 \
  --save_edited_model "${REPO_ROOT}/models/llama3-8B/Baseline_DEPN"
