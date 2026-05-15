#!/usr/bin/env bash
# Run email PII leakage attack on no-defense and defended models, then compute PLR.
# 1) No-defense: baseline + depn_ep5_lora
# 2) Defended: Baseline_DEPN, with --baseline_summary_csv from step 1 for PLR.
set -e
REPO_ROOT="/data1/D-PIKE"
DATA_DIR="${REPO_ROOT}/third_party/pme/Attacks-PME/LM_PersonalInfoLeak-main/data"
OUT_NO_DEF="${REPO_ROOT}/outputs/attack/llama3_email_no_defense"
OUT_DEF="${REPO_ROOT}/outputs/attack/llama3_email_defended"
mkdir -p "$OUT_NO_DEF" "$OUT_DEF"

echo "========== 1) Attack on NO-DEFENSE model (baseline + depn_ep5_lora) =========="
python "${REPO_ROOT}/srcs/Attack/llama3_email_leak_attack.py" \
  --model_name "${REPO_ROOT}/models/llama3-8B/baseline" \
  --adapter_dir "${REPO_ROOT}/models/llama3-8B/depn_ep5_lora" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_NO_DEF" \
  --include_context --include_zero_shot \
  --context_sizes 50 100 200 --zero_shot_variants a b c d \
  --device cuda:0 --batch_size 4 --max_new_tokens 80 --decoding greedy

echo ""
echo "========== 2) Attack on DEFENDED model (Baseline_DEPN) + PLR =========="
python "${REPO_ROOT}/srcs/Attack/llama3_email_leak_attack.py" \
  --model_name "${REPO_ROOT}/models/llama3-8B/Baseline_DEPN" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DEF" \
  --baseline_summary_csv "${OUT_NO_DEF}/summary.csv" \
  --include_context --include_zero_shot \
  --context_sizes 50 100 200 --zero_shot_variants a b c d \
  --device cuda:0 --batch_size 4 --max_new_tokens 80 --decoding greedy

echo ""
echo "Done. No-defense results: $OUT_NO_DEF | Defended + PLR: $OUT_DEF"
