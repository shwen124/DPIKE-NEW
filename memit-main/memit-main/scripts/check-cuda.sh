#!/usr/bin/env bash
# 检查 CUDA 环境是否正常
echo "=== NVIDIA 驱动 ==="
cat /proc/driver/nvidia/version 2>/dev/null || echo "无法读取"
echo ""
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 || true
echo ""
echo "=== PyTorch CUDA ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('CUDA 不可用，请检查驱动')
" 2>&1
