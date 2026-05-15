#!/usr/bin/env bash
# 修复 NVIDIA 驱动版本不一致 (Driver/library version mismatch)
# 需要 sudo 权限，或直接重启机器
set -e
echo "=== 当前状态 ==="
cat /proc/driver/nvidia/version 2>/dev/null || true
nvidia-smi 2>&1 || true
echo ""
echo "=== 尝试重新加载 NVIDIA 模块 (需要 sudo) ==="
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || echo "卸载失败，可能有进程占用"
sudo modprobe nvidia
echo "重新加载完成，检查 nvidia-smi:"
nvidia-smi
