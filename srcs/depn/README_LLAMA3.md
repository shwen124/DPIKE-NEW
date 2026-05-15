# DEPN for Llama3-8B 使用指南

本指南说明如何将 DEPN 框架适配到 Llama3-8B 模型。

## 主要改动

### 1. 架构差异

**BERT (原始实现)**:
- Encoder 架构，双向注意力
- Masked Language Modeling (MLM)
- 使用 `[MASK]` token 进行预测
- 支持 segment_ids

**Llama3 (适配后)**:
- Decoder 架构，因果注意力（causal attention）
- Causal Language Modeling (CLM)
- Next token prediction（下一个token预测）
- 不支持 segment_ids，使用因果掩码

### 2. 文件说明

#### 新增文件：
- `custom_llama.py`: 适配 Llama3 的模型包装类，支持神经元编辑和积分梯度
- `1_calculate_attribution_llama.py`: 适配 Llama3 的隐私神经元检测脚本
- `3_edit_privacy_neurons_llama.py`: 适配 Llama3 的隐私神经元编辑脚本

#### 保持不变：
- `2_filter_privacy_neurons.py`: 神经元聚合逻辑相同，无需修改

## 使用方法

### 步骤 1: 准备数据和微调模型

首先需要准备数据并微调 Llama3-8B 模型。由于 Llama3 使用 CLM 而非 MLM，数据格式需要相应调整：

```bash
# 数据格式示例（JSON）
[
  [
    "My phone number is 1 2 3 4 5 6 7 8 9 0",
    "1", "2", "3", ...  # 目标token序列
  ],
  ...
]
```

注意：第一个元素是完整文本，后续元素是目标预测的token。

### 步骤 2: 计算归因分数（检测隐私神经元的基础）

**目标**：用积分梯度方法计算每个神经元对隐私 token（如姓名、电话）输出的贡献。这是后续聚合、编辑隐私神经元的基础。

运行归因计算脚本（推荐显存 ≥40GB 时使用，不启用 4-bit、正常 batch）：

```bash
python 1_calculate_attribution_llama.py \
    --model_name_or_path /path/to/llama3-8B \
    --adapter_dir /path/to/lora_adapter \
    --priv_data_path ../data/sampled_TEL.json \
    --output_dir ../results_llama3/ \
    --output_prefix llama3_tel \
    --gpus 0 \
    --max_seq_length 128 \
    --batch_size 16 \
    --num_batch 10
```

**显存不足（如 24GB）时**可加 `--load_in_4bit`，并减小 `--batch_size`、`--num_batch`（如 2 和 3）或 `--max_seq_length`（如 64）。

**关键参数说明：**
- `--model_name_or_path`: Llama3 模型路径或 HuggingFace 标识符
- `--adapter_dir`: 可选，LoRA 适配器路径（微调模型必填）
- `--batch_size`: 积分梯度的近似步数，越大越准确但耗时与显存更高
- `--num_batch`: 批次数

**与 BERT 版本的主要区别：**
- 不再使用 `[MASK]` token
- 使用 next token prediction
- 目标位置通常是序列的最后一个token位置

### 步骤 3: 聚合隐私神经元

这一步与 BERT 版本完全相同：

```bash
python 2_filter_privacy_neurons.py \
    ../results_llama3/ \
    0.01 \
    0.5
```

参数说明：
- 第一个参数：结果目录
- 第二个参数：阈值比例（threshold_ratio），过滤小于最大值的 threshold_ratio 倍的神经元
- 第三个参数：模式比例（mode_ratio_bag），过滤频率低于此值的神经元

### 步骤 4: 编辑隐私神经元

运行适配后的编辑脚本：

```bash
python 3_edit_privacy_neurons_llama.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --priv_data_path ../data/memorized_TEL.txt \
    --validation_path ../data/enron_data/valid.txt \
    --kn_dir ../results_llama3/kn/kn_bag-llama3_tel.json \
    --gpus 0 \
    --max_seq_length 128 \
    --erase_kn_num 20 \
    --do_random_kn False \
    --input_prefix TEL
```

**关键参数说明：**
- `--erase_kn_num`: 要编辑的神经元数量（建议范围：10-200）
- `--input_prefix`: 数据类型（TEL、NAME 或 RANDOM）

**编辑机制：**
- 对于 Llama3，编辑是通过将 MLP 的 `down_proj` 层中对应位置的权重列置零来实现的
- 这相当于"擦除"该神经元对输出的贡献

## 技术细节

### 1. 积分梯度计算

在 Decoder 架构中，积分梯度计算需要考虑：
- 因果掩码：每个位置只能看到前面的tokens
- Next token prediction：预测目标在序列末尾之后
- 中间层激活：在 MLP 的 intermediate 层（gate_proj + up_proj 后的激活值）

### 2. 神经元编辑位置

Llama3 的 MLP 结构：
```
input → LayerNorm → gate_proj → SwiGLU → up_proj → down_proj → output
                                      ↑
                                  编辑这里
```

编辑发生在 `intermediate` 层（gate_proj 和 up_proj 的输出），然后通过 `down_proj` 传播。

### 3. 内存和计算要求

Llama3-8B 比 BERT-large 大得多，需要注意：
- **GPU 内存**: 建议至少 16GB VRAM
- **计算时间**: 比 BERT 慢 3-5 倍
- **批处理大小**: 可能需要减小 batch_size 以避免 OOM

## 注意事项

1. **模型加载**: Llama3 可能需要使用 `torch_dtype=torch.float16` 来节省内存
2. **Tokenizer**: Llama3 使用 SentencePiece tokenizer，确保正确初始化
3. **因果掩码**: 自动处理，无需手动设置
4. **中间层大小**: Llama3 的 `intermediate_size` 通常为 `hidden_size * 4`（如 8192 * 4 = 32768）

## 性能调优建议

1. **减少 batch_size**: 如果遇到 OOM，减小 `--batch_size`
2. **使用梯度检查点**: 对于大模型，可以启用梯度检查点节省内存
3. **量化**: 考虑使用 8-bit 或 4-bit 量化减少内存占用
4. **多GPU**: 使用 `accelerate` 库支持多GPU训练

## 常见问题

**Q: 为什么需要重新适配？**
A: BERT 和 Llama3 架构差异很大（Encoder vs Decoder），需要不同的处理方式。

**Q: 能否直接使用 BERT 的代码？**
A: 不能。Decoder 架构需要不同的注意力掩码和预测方式。

**Q: 编辑效果如何？**
A: 根据论文，在 Llama2-7B 上取得了显著效果，Llama3-8B 应该类似。

**Q: 如何处理更大的模型（如 Llama3-70B）？**
A: 需要更小的 batch_size，可能需要使用模型并行或量化。

## 参考

- 原始 DEPN 论文: https://arxiv.org/pdf/2310.20138.pdf
- Llama3 模型: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Transformers 文档: https://huggingface.co/docs/transformers
