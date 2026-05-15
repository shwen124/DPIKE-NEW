# DEPN 数据目录说明

## 规范结构（仓库 `data/`）

本仓库 `data/` 仅存放数据，结构遵循项目规范：

```text
data/
├── raw/                   # 原始数据，不直接修改（当前为空，预留给上游原始数据）
├── processed/             # 处理后的数据，按项目/用途分子目录
│   └── depn/              # DEPN 用到的全部数据
│       ├── temp_data/     # 微调用 train/valid/test.txt
│       ├── memorized_*.txt
│       ├── sampled_*.json / sampled_*.txt
│       ├── all_Tel.txt、privacy_data_tel.json
│       └── ...
├── interim/               # 中间产物（当前为空）
├── external/              # 外部下载数据（当前为空）
├── depn → processed/depn  # 规范入口：脚本建议使用 data/depn/xxx
└── [兼容链接]              # 以下为兼容旧脚本，指向 processed/depn 下同名文件：
    all_Tel.txt, memorized_*.txt, sampled_*.json, sampled_*.txt,
    privacy_data_tel.json, temp_data
```

## DEPN 数据文件用途

| 路径（建议用 `data/depn/` 或 `data/processed/depn/`） | 用途 |
|------------------------------------------------------|------|
| `temp_data/train.txt`, `valid.txt`, `test.txt` | 微调训练/验证/测试 |
| `sampled_TEL.json` | Step 2 归因输入（按 bag 的隐私样本） |
| `memorized_TEL.txt` | Step 4 暴露度评估 |
| `memorized_NAME.txt`, `memorized_RANDOM.txt` | NAME/RANDOM 评估 |
| `sampled_NAME.json`, `sampled_RANDOM.json` | NAME/RANDOM 归因输入 |
| `all_Tel.txt`, `privacy_data_tel.json` | 中间/备用 |

## 模型与检查点（不在 `data/`）

- **最终 LoRA 模型**：`models/llama3-8B/depn_ep5_lora/`（不再使用 `data/model/`）
- **训练中间检查点**：`checkpoints/depn/llama3-8b_lora/`（如 `checkpoint-85000`, `checkpoint-90000`, `checkpoint-95000`）

脚本中 `--adapter_dir` 请使用 `models/llama3-8B/depn_ep5_lora`。
