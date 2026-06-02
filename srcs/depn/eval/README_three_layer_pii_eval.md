# 三层 PII 前缀补全评估

## 数据文件说明

| 文件 | 用途 |
|------|------|
| `sft_true_prefix_no_instruction.json` | **主评估**：plain `input`/`output`，含 `pii_type`/`source_id` |
| `sft_true_prefix_no_instruction_{train,val,test}.json` | 按 `source_id` 分组切分后 |
| `sft_true_prefix_masked.json` | **消融**：历史 PII 已 `[PII]` 遮蔽 |
| `sft_true_prefix.json` | 带 `instruction`（chat 训练用，plain 评估勿用） |
| `sft_true_prefix_text.json` | 整段 CLM 文本，**不用于**前缀补全指标 |

## 1. 切分数据（首次）

```bash
python srcs/depn/eval/split_sft_by_source_id.py \
  --input data/api4_200k/sft_true_prefix_no_instruction.json \
  --output_dir data/api4_200k \
  --stem sft_true_prefix_no_instruction
```

若数据在仓库根目录，将路径改为 `sft_true_prefix_no_instruction.json` 与输出目录 `.`。

## 2. 一键评估

**Base vs SFT（推荐，仅两层对比）：**

```bash
bash srcs/depn/eval/run_base_vs_sft_pii_eval.sh
```

结果目录：`outputs/depn/eval/base_vs_sft_plain_qlora/`（含 `{base,sft}/` 与 `comparison/`）。

**三层（含 edited，可选）：**

```bash
export BASE_MODEL_DIR=models/llama3-8B/baseline
export SFT_ADAPTER_DIR=models/llama3-8B/api4_prefix_plain_qlora
export EDITED_ADAPTER_DIR=models/llama3-8B/depn_edited_lora   # 或 EDITED_MODEL_DIR=...
export LAYERS="base sft edited"
export SPLITS="test"

bash srcs/depn/eval/run_three_layer_pii_eval.sh
```

train-seen 抽样（每类 200）：

```bash
SPLITS="train" LIMIT_PER_TYPE_TRAIN=200 bash srcs/depn/eval/run_base_vs_sft_pii_eval.sh
```

## 3. 单层单次评估示例

**Base + test：**

```bash
python srcs/depn/eval/eval_sft_true_prefix_pii_metrics.py \
  --dataset data/api4_200k/sft_true_prefix_no_instruction_test.json \
  --model_dir models/llama3-8B/baseline \
  --layer base --run_name base_plain_test \
  --prompt_template "{input}" --load_in_4bit \
  --limit_per_type 0 --max_new_tokens 96 --generation_extra_tokens 16 \
  --output_csv results/base_plain_test.csv \
  --output_json results/base_plain_test.json
```

**SFT adapter + train（抽样）：**

```bash
python srcs/depn/eval/eval_sft_true_prefix_pii_metrics.py \
  --dataset data/api4_200k/sft_true_prefix_no_instruction_train.json \
  --base_model_dir models/llama3-8B/baseline \
  --adapter_dir models/llama3-8B/api4_prefix_plain_qlora \
  --layer sft --run_name sft_plain_train \
  --prompt_template "{input}" --load_in_4bit \
  --limit_per_type 200 --max_new_tokens 96 --generation_extra_tokens 16 \
  --output_csv results/sft_plain_train.csv \
  --output_json results/sft_plain_train.json
```

**不要用** `--sft_llama3_chat_prompt`（plain 协议不一致）。

## 4. 对比表

```bash
python srcs/depn/eval/compare_pii_eval_layers.py \
  --base outputs/depn/eval/base_vs_sft_plain_qlora/base/base_plain_test.json \
  --sft outputs/depn/eval/base_vs_sft_plain_qlora/sft/sft_plain_test.json \
  --split test \
  --output_dir outputs/depn/eval/base_vs_sft_plain_qlora/comparison
```

生成：`table1_macro_*.csv`、`table2_by_pii_type_*.csv`、`table3_by_category_*.csv`、`table4_deltas_*.csv`。

## 5. 核心指标（每条样本均含）

- **Gold-token**：`nll`、`mrr`、`topk_accuracy`、`exposure`（token-rank proxy）
- **生成泄露**：`exact_match`、`normalized_exact_match`、`starts_with_target`、`normalized_starts_with_target`、`accuracy`
- **接近度**：`edit_distance`、`edit_similarity`、`token_f1`
- **生成长度**：`generated_token_count`、`stopped_by_eos`
- **数字类**：`digit_accuracy`
- **EMAIL**：`email_username_match`、`email_domain_match`

## 6. 解读要点

- **train-seen** 上 SFT 相对 Base 的 `normalized_exact_match` / NLL / MRR / exposure **增量** → 记忆增强证据
- **test** 提升较小 → 主要记住训练样本，非泛化猜 PII
- **edited** 相对 SFT：NLL↑、MRR/EM/exposure↓ → 擦除有效

未在 `PII_METRIC_CONFIG` 中配置的类型（例如 `HEIGHT`）不会参与评估。
