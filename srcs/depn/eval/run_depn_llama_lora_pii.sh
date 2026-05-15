#!/usr/bin/env bash
# 兼容入口：转调 data 目录下的正式脚本（便于从 eval 侧发现）。
exec bash "$(cd "$(dirname "$0")" && pwd)/../data/run_depn_llama_lora_pii.sh" "$@"
