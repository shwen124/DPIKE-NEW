# Llama3-8B PMET 隐私编辑（4090 / 仅隐私）

对已微调的 Llama3-8B（QLoRA）**合并 LoRA** 后，用 [PMET](https://arxiv.org/abs/2308.08742) 在 **前缀补全边界** 上把已标注的隐私秘密压向占位符 `[REDACTED]`。

## RTX 4090（24GB）默认策略

| 项 | 说明 |
|----|------|
| 超参 | `hparams/PMET/meta_llama3-8B_4090.json`：`mom2_n_samples=6000`，编辑层 `[5,6,7]`，`v_num_grad_steps=15` |
| 权重精度 | `evaluate_privacy.py` 默认 **float16**、`device_map=cuda:0` |
| 并行编辑 | 默认 **`num_edits=1`**，降低峰值显存 |
| 上下文模板 | 环境变量 **`PMET_MINIMAL_CONTEXT=1`**：不跑 `generate_fast` 扩展模板，省显存 |
| 矩阵运算 | **`PMET_GRAM_ON_CPU=1`**（默认）：`(K K^T + λC)^{-1}` 在 CPU 上算，减轻与全量模型争显存 |
| 协方差缓存 | `get_cov` 将二阶矩 **常驻 CPU**，不再整块拷到 GPU |

若仍 OOM：再减小 `PMET_LIMIT_PER_TYPE`、或把 `meta_llama3-8B_4090.json` 里 `layers` 改为 `[6,7]`。

## 「只编辑隐私」

`build_privacy_requests.py` 默认 **`--privacy_only`**（可用 `--no-privacy_only` 关闭）：

- 仅当 `(input, output)` 与 `english_pii_43k.jsonl` 里按 **500 字前缀 + 秘密原文** 对齐成功时，才生成一条 PMET 请求；
- `pii_type` 取参考 jsonl 中的标注类型，**不混入**未校验样本。

因此编辑列表只来自 **已标注 PII 跨度**，不会用整条 SFT 里未对齐的 completion 去误编辑。

## 代码与脚本

| 路径 | 说明 |
|------|------|
| `srcs/PMET-main/edit/evaluate_privacy.py` | 主入口（默认 4090 友好参数） |
| `srcs/PMET-main/edit/dsets/build_privacy_requests.py` | SFT → PMET 请求（`--privacy_only`） |
| `srcs/PMET-main/edit/hparams/PMET/meta_llama3-8B_4090.json` | 4090 用超参 |
| `scripts/depn/run_pmet_privacy_llama3.sh` | 一键脚本（已设上述环境变量） |

## 快速运行

```bash
bash scripts/depn/run_pmet_privacy_llama3.sh
```

强制重新生成请求 JSON（例如改了 `privacy_only` 逻辑后）：

```bash
PMET_REBUILD_REQUESTS=1 bash scripts/depn/run_pmet_privacy_llama3.sh
```

编辑后模型默认：`models/llama3-8b/pmet_privacy_v1/`。指标：`srcs/PMET-main/edit/results/PMET_PRIVACY/run_*/`。

## 手动示例

```bash
python srcs/PMET-main/edit/dsets/build_privacy_requests.py \
  --dataset data/api4_200k/sft_true_prefix.json \
  --reference_jsonl data/api4_200k/english_pii_43k.jsonl \
  --output data/depn/pmet_privacy_requests.json \
  --limit_per_type 15 \
  --privacy_only

cd srcs/PMET-main/edit
export PMET_MINIMAL_CONTEXT=1 PMET_GRAM_ON_CPU=1
python evaluate_privacy.py \
  --base_model_dir /data1/D-PIKE/models/llama3-8B/baseline \
  --adapter_dir /data1/D-PIKE/checkpoints/depn/llama3-8b_lora/2026-05-13_api4_prefix_qlora \
  --requests_json /data1/D-PIKE/data/depn/pmet_privacy_requests.json \
  --data_dir /data1/D-PIKE/data/depn \
  --hparams_fname meta_llama3-8B_4090.json \
  --num_edits 1 \
  --torch_dtype float16 \
  --device_map cuda:0 \
  --cumulative_edits \
  --use_cache \
  --save_edited_dir /data1/D-PIKE/models/llama3-8b/pmet_privacy_v1
```

## 其它说明

- 训练使用 `--sft_llama3_chat` 时，`completion_context` 与 DEPN 评估里 Llama3 wire 格式一致。
- 首次运行会为各层收集 `mom2`（样本数已降为 6000），仍可能较慢；`--use_cache` 缓存每条请求的 `z`。
- PMET 仍会改变全量权重，对非目标句子的行为可能有副作用；仅隐私样本入队是为 **缩小编辑目标集合**，而非数学上只改隐私子空间。
