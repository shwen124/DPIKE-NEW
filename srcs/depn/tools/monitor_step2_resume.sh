#!/usr/bin/env bash
# 监控 Step 2 积分梯度任务：每 15 分钟检查一次；
# 若进程未在运行且未完成（输出行数 < 总 bag 数），则从 checkpoint 恢复执行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
LOG_DIR="$PROJECT_ROOT/logs"
MONITOR_LOG="$LOG_DIR/step2_monitor.log"
OUT_JSONL="$PROJECT_ROOT/llama3_results/llama3_tel.priv.jsonl"
TOTAL_BAGS=50
INTERVAL_MIN=15

# Step 2 恢复执行命令（与 1_calculate_attribution_llama.py 参数一致）
RUN_CMD=(
    python "$SRC_DIR/1_calculate_attribution_llama.py"
    --model_name_or_path /data1/D-PIKE/pretrained_models/llama3-8B
    --adapter_dir "$PROJECT_ROOT/data/model/llama3_8b_ep5_stable_lora4bit"
    --priv_data_path "$PROJECT_ROOT/data/privacy_data_tel.json"
    --output_dir "$PROJECT_ROOT/llama3_results"
    --output_prefix llama3_tel
    --gpus 0
    --max_seq_length 128
    --batch_size 1
    --num_batch 4
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

is_attribution_running() {
    pgrep -f "1_calculate_attribution_llama.py" >/dev/null 2>&1
}

count_done_bags() {
    if [[ -f "$OUT_JSONL" ]]; then
        wc -l < "$OUT_JSONL"
    else
        echo 0
    fi
}

gpu_status() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
    else
        echo "nvidia-smi not available"
    fi
}

main() {
    mkdir -p "$LOG_DIR"
    log "===== Step2 monitor started (interval=${INTERVAL_MIN} min) ====="

    while true; do
        running=0
        if is_attribution_running; then
            running=1
        fi

        done_bags=$(count_done_bags)
        gpu_info=$(gpu_status)

        if [[ "$running" -eq 1 ]]; then
            log "RUNNING | done_bags=$done_bags/$TOTAL_BAGS | GPU: $gpu_info"
        else
            if [[ "$done_bags" -ge "$TOTAL_BAGS" ]]; then
                log "DONE (already $done_bags bags). Exiting monitor."
                exit 0
            fi
            log "NOT RUNNING | done_bags=$done_bags/$TOTAL_BAGS | GPU: $gpu_info"
            log "Resuming from checkpoint (cd $SRC_DIR && nohup ...)"
            cd "$SRC_DIR"
            run_log="$LOG_DIR/step2_attribution_resume_$(date +%Y%m%d_%H%M).log"
            nohup "${RUN_CMD[@]}" >> "$run_log" 2>&1 &
            pid=$!
            log "Started PID=$pid, log=$run_log"
            cd - >/dev/null
        fi

        sleep $((INTERVAL_MIN * 60))
    done
}

main "$@"
