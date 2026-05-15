# NaN问题修复说明

## 修复的问题

### 1. ✅ 核心问题：8-bit量化 + 全参数更新冲突

**问题描述**：
- `run_llama3_stable.sh` 和 `run_llama3.sh` 中启用了 `--load_in_8bit`
- 但训练脚本仍使用全参数 AdamW 优化器更新
- 这种组合在实践中非常不稳定，容易导致 NaN

**修复方案**：
- 在 `run_clm_no_trainer.py` 中添加了警告信息，当使用8-bit量化时会提示用户
- **移除了启动脚本中的 `--load_in_8bit` 参数**（`run_llama3_stable.sh`）
- 如果确实需要使用8-bit量化，应该使用 LoRA/PEFT 而不是全参数更新

**修改位置**：
- `data/run_clm_no_trainer.py:426-432` - 添加警告
- `data/run_llama3_stable.sh:38` - 移除 `--load_in_8bit`
- `data/run_llama3.sh:34` - 移除 `--load_in_8bit`

---

### 2. ✅ 次核心问题：float16 训练不稳定

**问题描述**：
- 启动脚本使用 `--torch_dtype float16`
- 全参数微调时纯 fp16 权重更新容易溢出导致 NaN（尤其对于 Llama-8B 这样的大模型）

**修复方案**：
- 将 `float16` 改为 `bfloat16`（bf16）
- bfloat16 比 float16 更稳定，更适合大模型训练
- 在代码中添加了自动转换：如果用户指定 `float16`，会自动转换为 `bfloat16` 并提示

**修改位置**：
- `data/run_clm_no_trainer.py:434-439` - 添加 float16 → bfloat16 自动转换
- `data/run_llama3_stable.sh:36` - 改为 `--torch_dtype bfloat16`
- `data/run_llama3.sh:32` - 改为 `--torch_dtype bfloat16`

---

### 3. ✅ 训练循环缺少 NaN 保护

**问题描述**：
- 反向前没有 `isfinite(loss)` 检查
- NaN 会直接扩散到整个训练过程

**修复方案**：
- 在 `loss` 计算后立即检查是否为有限值
- 如果检测到 NaN/Inf，跳过该 batch 并记录错误日志
- 在评估循环中也添加了相同的检查

**修改位置**：
- `data/run_clm_no_trainer.py:667-675` - 添加 loss NaN 检查
- `data/run_clm_no_trainer.py:737-743` - 添加评估 loss NaN 检查

---

### 4. ✅ 梯度裁剪位置优化

**问题描述**：
- 现在每个 micro-step 都进行梯度裁剪
- 常见做法是仅在 `accelerator.sync_gradients` 为真时裁剪

**修复方案**：
- 将梯度裁剪移到 `accelerator.sync_gradients` 检查内部
- 只在梯度同步时进行裁剪，避免在每个 micro-step 都裁剪
- 添加梯度范数的 NaN/Inf 检查

**修改位置**：
- `data/run_clm_no_trainer.py:700-712` - 优化梯度裁剪位置和添加检查

---

### 5. ✅ 添加梯度范数检查

**问题描述**：
- 梯度裁剪后没有检查梯度范数是否为有限值

**修复方案**：
- 在梯度裁剪后检查梯度范数
- 如果梯度范数为 NaN/Inf，跳过优化器步骤

**修改位置**：
- `data/run_clm_no_trainer.py:705-712` - 添加梯度范数检查

---

## 修改文件清单

### 1. `data/run_clm_no_trainer.py`
- ✅ 添加 8-bit 量化警告（第426-432行）
- ✅ 添加 float16 → bfloat16 自动转换（第434-439行）
- ✅ 添加训练 loss NaN 检查（第667-675行）
- ✅ 优化梯度裁剪位置（第700-712行）
- ✅ 添加评估 loss NaN 检查（第737-750行）

### 2. `data/run_llama3_stable.sh`
- ✅ 移除 `--load_in_8bit` 参数
- ✅ 将 `--torch_dtype float16` 改为 `--torch_dtype bfloat16`
- ✅ 添加 `--max_grad_norm 1.0`（如果之前没有）

### 3. `data/run_llama3.sh`
- ✅ 移除 `--load_in_8bit` 参数
- ✅ 将 `--torch_dtype float16` 改为 `--torch_dtype bfloat16`
- ✅ 添加 `--max_grad_norm 1.0`

---

## 使用建议

### 推荐配置（稳定训练）

```bash
# 使用 bfloat16，不使用 8-bit 量化
bash run_llama3_stable.sh
```

**关键参数**：
- `--torch_dtype bfloat16` ✅
- 不使用 `--load_in_8bit` ✅
- `--gradient_checkpointing` ✅
- `--max_grad_norm 1.0` ✅

### 如果显存不足

如果显存确实不足，有两个选择：

**选项1：使用 LoRA/PEFT（推荐）**
- 使用 8-bit 量化加载模型
- 使用 PEFT/LoRA 进行参数高效微调
- 不要使用全参数更新

**选项2：减小批次大小**
- 减小 `--per_device_train_batch_size`
- 增加 `--gradient_accumulation_steps`
- 保持使用 bfloat16，不使用 8-bit 量化

---

## 验证修复

训练时应该看到以下日志：

1. **如果使用 float16（会自动转换）**：
   ```
   ⚠️  Converting float16 to bfloat16 for better training stability...
   ```

2. **如果使用 8-bit 量化（会警告）**：
   ```
   ⚠️  WARNING: Using 8-bit quantization with full parameter fine-tuning is unstable!
   ```

3. **如果检测到 NaN**：
   ```
   ⚠️  Non-finite loss detected at step X, epoch Y, step Z! Loss value: nan. Skipping this batch.
   ```

---

## 预期效果

修复后应该能够：
- ✅ 避免 NaN 值出现
- ✅ 训练过程更稳定
- ✅ 即使出现 NaN 也能自动跳过并继续训练
- ✅ 更好的显存利用（bfloat16）

---

## 注意事项

1. **bfloat16 要求**：需要 GPU 支持 bfloat16（A100、V100、RTX 30系列及以上）
2. **8-bit 量化**：如果必须使用，请改用 LoRA/PEFT，不要使用全参数更新
3. **NaN 检测**：如果频繁出现 NaN 警告，检查学习率是否过大，或考虑进一步降低学习率
