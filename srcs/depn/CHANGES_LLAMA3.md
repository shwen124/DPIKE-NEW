# Llama3-8B 适配变更说明

## 概述

本文档说明为适配 Llama3-8B 所做的关键代码变更。

## 核心变更

### 1. 模型架构适配 (`custom_llama.py`)

#### 新增类: `LlamaForCausalLMWithEditing`
- 继承自 `LlamaForCausalLM`
- 添加了 DEPN 特定的 forward 参数：
  - `tgt_pos`: 目标token位置
  - `tgt_layer`: 目标层索引
  - `tmp_score`: 积分梯度临时分数
  - `imp_pos`: 要编辑的神经元位置列表
  - `imp_op`: 编辑操作（'remove', 'enhance', 'return'）

#### 关键修改点：
1. **MLP Forward 包装**:
   - 包装了每个 decoder layer 的 MLP forward
   - 支持在 intermediate 层注入 `tmp_score`（用于积分梯度）
   - 支持神经元编辑（置零或增强）

2. **Decoder Layer Forward 包装**:
   - 包装了 decoder layer 的 forward
   - 传递 DEPN 参数到 MLP

3. **Model Forward 重写**:
   - 手动遍历 layers 以注入 DEPN 参数
   - 提取 FFN intermediate weights
   - 支持返回 imp_weights

### 2. 积分梯度计算 (`1_calculate_attribution_llama.py`)

#### 主要变更：

1. **数据处理 (`example2feature`)**:
   ```python
   # BERT 版本: 使用 [MASK] token
   tokens = ["[CLS]"] + ori_tokens + ["[MASK]"] + ["[SEP]"]
   
   # Llama3 版本: 使用 next token prediction
   # 目标位置是序列的最后一个token
   tgt_pos = len(input_ids) - 1
   ```

2. **Forward 调用**:
   ```python
   # BERT: 需要 token_type_ids
   model(input_ids, attention_mask, token_type_ids, tgt_pos, tgt_layer)
   
   # Llama3: 不需要 token_type_ids，使用因果掩码
   model(input_ids, attention_mask, tgt_pos, tgt_layer)
   ```

3. **积分梯度计算**:
   - 基准值：零向量（而不是 masked token embedding）
   - 插值路径：从零到实际 FFN intermediate 激活值
   - 目标：预测下一个token的logits

### 3. 神经元编辑 (`3_edit_privacy_neurons_llama.py`)

#### 主要变更：

1. **编辑位置**:
   ```python
   # BERT: 编辑 encoder.layer[layer].output.dense.weight
   model.bert.encoder.layer[layer].output.dense.weight[:, pos] = 0
   
   # Llama3: 编辑 model.layers[layer].mlp.down_proj.weight
   model.model.layers[layer].mlp.down_proj.weight[:, pos] = 0
   ```

2. **曝光度计算 (`get_exposure_llama`)**:
   - 适配 next token prediction
   - 对于电话号码，需要预测序列中的每个数字
   - 使用因果掩码，只能看到前面的tokens

3. **MRR 计算 (`get_name_MRR_llama`)**:
   - 适配 Causal LM 的预测方式
   - 每个token的预测基于前面的上下文

### 4. 评估指标

保持不变的部分：
- Perplexity 计算逻辑相同
- 神经元聚合逻辑（`2_filter_privacy_neurons.py`）完全相同

## 关键差异总结

| 特性 | BERT | Llama3 |
|------|------|--------|
| 架构 | Encoder | Decoder |
| 注意力 | 双向 | 因果（单向）|
| 预测方式 | MLM ([MASK]) | CLM (next token) |
| 掩码类型 | Padding mask | Causal mask |
| Segment IDs | 支持 | 不支持 |
| FFN 结构 | 单一 Linear | SwiGLU (gate + up) |
| 编辑位置 | output.dense | mlp.down_proj |
| 中间层大小 | intermediate_size | intermediate_size (通常 4x hidden_size) |

## 使用注意事项

1. **内存要求**: Llama3-8B 需要更多 GPU 内存，建议使用 float16
2. **计算速度**: 比 BERT 慢 3-5 倍
3. **批处理大小**: 可能需要减小 batch_size（建议 8-16）
4. **Tokenizer**: 确保使用正确的 tokenizer（SentencePiece for Llama3）

## 测试建议

1. 先用小数据集测试（100-200 条）
2. 检查 FFN weights 是否正确提取
3. 验证编辑后模型仍能正常推理
4. 对比编辑前后的 perplexity 和曝光度

## 已知限制

1. 当前实现假设 intermediate_size = hidden_size * 4，某些变体可能不同
2. 多GPU 支持需要进一步测试
3. 对于超大模型（>13B），可能需要额外优化

## 后续优化方向

1. 支持 8-bit/4-bit 量化
2. 优化内存使用（gradient checkpointing）
3. 支持更大的 batch_size（通过 gradient accumulation）
4. 添加更多的调试和可视化工具
