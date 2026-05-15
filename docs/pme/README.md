# PME（Private Memorization Editing）

第三方仓库位置：**`third_party/pme/`**（原始克隆自 [elenasofia98/PME](https://github.com/elenasofia98/PME)）。

## 简介

Private Memorization Editing (PME) 通过模型编辑将 LLM 对 PII 的“记忆”转化为隐私防御：先检测被记忆的 PII，再通过编辑模型知识缓解记忆，从而降低训练数据提取类隐私攻击的泄露。论文：*Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models*（ACL 2025）。

## 仓库结构

| 目录 | 用途 |
|------|------|
| **Attacks-PME/** | 预编辑攻击（01-*）、编辑后攻击（02-*）、评估表格（04-*）、前后编辑生成对比（09-*） |
| **EasyEdit/** | 编辑实现：PME(memoedit)、MEMIT、Grace 等 baselines，notebook 入口 |
| **lm-evaluation-harness/** | 编辑前后模型的 LM 评测 |

## 复现流程（来自原 README）

1. **Pre-edit 攻击**：在 Attacks-PME 中运行 01-* notebook，从选定模型得到泄露的 PII。
2. **编辑**：在 EasyEdit 中运行 notebook，执行 baselines 与 PME（代码中为 memoedit）；DeMem baseline 见 DeMemorization-main。
3. **Post-edit 评估**：Attacks-PME 中 02-* 为编辑后攻击，04-* 做表格，09-* 用编辑前后模型生成并量化相似度；lm-evaluation-harness 对编辑前后模型做 LM 评测。

## 数据与依赖

- PII 数据：原仓库不公开从 The Pile 提取的 PII；研究用途可联系 `elena.sofia.ruzzetti AT uniroma2.it`。
- 依赖与子模块：见 `third_party/pme/README.md` 及各子目录（EasyEdit、lm-evaluation-harness 等）的安装说明。

## 与本仓库的约定

- 保持 **third_party/pme** 为上游原始实现，仅做必要配置/路径适配。
- 若将 PME 或其中某部分整合进本仓库流程，建议把改写后的核心逻辑放到 **srcs/** 下对应项目目录，并在此注明引用自 PME。
