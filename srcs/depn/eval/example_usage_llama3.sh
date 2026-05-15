#!/bin/bash

# DEPN for Llama3-8B 使用示例脚本
# 请根据实际情况修改路径和参数

# 设置环境变量（如果需要）
export CUDA_VISIBLE_DEVICES=0

# 步骤 1: 检测隐私神经元
echo "步骤 1: 检测隐私神经元..."
python 1_calculate_attribution_llama.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --priv_data_path ../data/sampled_TEL.json \
    --output_dir ../results_llama3/ \
    --output_prefix llama3_tel_ep10 \
    --gpus 0 \
    --max_seq_length 128 \
    --batch_size 16 \
    --num_batch 10

# 步骤 2: 聚合隐私神经元
echo "步骤 2: 聚合隐私神经元..."
python 2_filter_privacy_neurons.py \
    ../results_llama3/ \
    0.1 \
    0.5

# 步骤 3: 编辑隐私神经元并评估
echo "步骤 3: 编辑隐私神经元..."
python 3_edit_privacy_neurons_llama.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --priv_data_path ../data/memorized_TEL.txt \
    --validation_path ../data/enron_data/valid.txt \
    --kn_dir ../results_llama3/kn/kn_bag-llama3_tel_ep10.json \
    --gpus 0 \
    --max_seq_length 128 \
    --erase_kn_num 20 \
    --do_random_kn False \
    --input_prefix TEL

echo "完成！"
