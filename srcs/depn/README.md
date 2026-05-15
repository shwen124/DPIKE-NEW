# DEPN（隐私神经元检测与擦除）

本目录为 DEPN 项目在本仓库中的规范位置，遵循 `data/`、`models/`、`srcs/`、`outputs/`、`docs/` 分层结构。

## 目录说明

| 子目录 | 用途 |
|--------|------|
| **eval/** | 归因、聚合、擦除脚本（Step 2/3/4）及评估脚本 |
| **train/** | 微调脚本（run_clm_no_trainer、run_llama3 等） |
| **data/** | 数据预处理与构建（preprocess_enron、mask_text2json、build_privacy_json_tel 等） |
| **utils/** | 共用模块（custom_llama、custom_bert、accelerate_cli） |
| **tools/** | 校验与辅助脚本（validate_step2_output、monitor_step2_resume） |

## 运行说明

- 工作目录建议为仓库根目录 `/data1/D-PIKE`。
- 数据路径：`data/` 下符号链接指向 `data/processed/depn/`，可直接使用 `data/sampled_TEL.json`、`data/temp_data/` 等。
- 模型路径：基座 `models/llama3-8B/baseline` 或 `pretrained_models/llama3-8B`；LoRA `models/llama3-8B/depn_ep5_lora`（或兼容路径 `data/model/llama3_8b_ep5_stable_lora4bit`）。
- 输出路径：归因与擦除结果写入 `outputs/depn/`（如 `--output_dir outputs/depn` 或 `--results_llama3 outputs/depn`）。
- 调用 eval 脚本时若需引用 `utils`，可在仓库根目录执行并设置 `PYTHONPATH=srcs/depn`，或在脚本中增加 `sys.path`。

详细文件清单与路径见 **`docs/depn/项目运行所需文件汇总.md`**。
