#!/usr/bin/env bash
# Full email attack (no-defense: baseline + depn_ep5_lora).
# Logs: logs/attack/llama3_email_no_defense/attack_<timestamp>.log
# Checkpoints: outputs/attack/llama3_email_no_defense/checkpoints/ (rotate 3)
set -e
REPO_ROOT="/data1/D-PIKE"
cd "$REPO_ROOT"
exec python "${REPO_ROOT}/srcs/Attack/llama3_email_leak_attack.py" \
  --model_name "${REPO_ROOT}/models/llama3-8B/baseline" \
  --adapter_dir "${REPO_ROOT}/models/llama3-8B/depn_ep5_lora" \
  --data_dir "${REPO_ROOT}/third_party/pme/Attacks-PME/LM_PersonalInfoLeak-main/data" \
  --output_dir "${REPO_ROOT}/outputs/attack/llama3_email_no_defense" \
  --include_context --include_zero_shot \
  --context_sizes 50 100 200 \
  --zero_shot_variants a b c d \
  --device cuda:0 \
  --batch_size 4 \
  --max_new_tokens 80 \
  --decoding greedy \
  --checkpoint_every 50 \
  --checkpoint_keep 3 \
  "$@"
