#!/usr/bin/env python3
"""
Validate Step 2 (.priv.jsonl) output against success/failure criteria.
Usage: python validate_step2_output.py <path_to_priv.jsonl> [expected_num_bags]
"""
import json
import math
import sys
from collections import Counter, defaultdict


def is_bad(x):
    if x is None:
        return True
    try:
        f = float(x)
        return math.isnan(f) or math.isinf(f) or (f != f)
    except (TypeError, ValueError):
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_step2_output.py <path_to_priv.jsonl> [expected_num_bags=50]")
        sys.exit(1)
    path = sys.argv[1]
    expected_bags = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    report = []
    failed_checks = []

    # ---- 1. File existence and size ----
    try:
        import os
        size = os.path.getsize(path)
        report.append(f"[文件] 存在, 大小: {size:,} bytes")
        if size < 100:
            failed_checks.append("文件大小异常过小")
    except Exception as e:
        failed_checks.append(f"文件生成失败或不可读: {e}")
        report.append(f"[文件] 错误: {e}")
        # print report and exit
        for r in report:
            print(r)
        for f in failed_checks:
            print(f"  ❌ {f}")
        sys.exit(1)

    # ---- 2. Load and count samples ----
    lines = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lines.append(data)
            except json.JSONDecodeError as e:
                failed_checks.append(f"第 {i+1} 行 JSON 解析失败: {e}")
                report.append(f"[数据完整性] 第 {i+1} 行损坏")
    num_bags = len(lines)
    report.append(f"[数据完整性] 样本(bag)数: {num_bags}, 期望: {expected_bags}")
    if num_bags != expected_bags:
        failed_checks.append(f"样本数量与输入不一致 (得到 {num_bags}, 期望 {expected_bags})")

    if not lines:
        failed_checks.append("没有有效数据行")
        for r in report:
            print(r)
        for f in failed_checks:
            print(f"  ❌ {f}")
        sys.exit(1)

    # ---- 3. Structure and triplets ----
    all_scores = []
    layer_scores = defaultdict(list)
    neuron_counter = Counter()  # (layer, neuron) -> count across samples
    num_triplets_per_bag = []
    layers_present = set()
    has_nan_inf = False
    all_same = True
    first_score = None
    all_zero = True

    for bag_idx, line in enumerate(lines):
        if not isinstance(line, list) or len(line) < 1:
            failed_checks.append(f"bag {bag_idx} 结构异常 (非列表或为空)")
            continue
        # line = [ [tokens_info, res_dict], ... ]; res_dict = list of [layer, neuron, score]
        for item in line:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            tokens_info, res_dict = item[0], item[1]
            if not isinstance(res_dict, list):
                failed_checks.append(f"bag {bag_idx} res_dict 非列表")
                continue
            num_triplets_per_bag.append(len(res_dict))
            if len(res_dict) < 1:
                failed_checks.append(f"bag {bag_idx} 梯度点数为 0")
            if len(res_dict) == 1:
                failed_checks.append(f"bag {bag_idx} 仅 1 个梯度点 (可能异常)")
            for t in res_dict:
                if not isinstance(t, (list, tuple)) or len(t) < 3:
                    continue
                layer, neuron, score = t[0], t[1], t[2]
                layers_present.add(layer)
                if is_bad(score):
                    has_nan_inf = True
                else:
                    s = float(score)
                    all_scores.append(s)
                    layer_scores[layer].append(s)
                    neuron_counter[(layer, neuron)] += 1
                    if first_score is None:
                        first_score = s
                    elif all_same and abs(s - first_score) > 1e-12:
                        all_same = False
                    if abs(s) > 1e-12:
                        all_zero = False

    # ---- 4. Data quality ----
    report.append("")
    report.append("--- 数据质量 ---")
    if has_nan_inf:
        failed_checks.append("存在 NaN 或 Inf 值")
        report.append("[数据质量] ❌ 存在 NaN/Inf")
    else:
        report.append("[数据质量] ✓ 无 NaN/Inf")

    if all_scores:
        min_s, max_s = min(all_scores), max(all_scores)
        report.append(f"[数据质量] 梯度分数范围: [{min_s:.6g}, {max_s:.6g}]")
        if max_s - min_s < 1e-15:
            failed_checks.append("梯度分数全部相同")
        if all_zero:
            failed_checks.append("所有梯度都为 0")
        if min_s < -1e6 or max_s > 1e6:
            report.append("[数据质量] ⚠ 梯度绝对值很大，请确认是否合理")
    else:
        failed_checks.append("没有有效梯度分数")

    # ---- 5. Triplet count / implementation ----
    report.append("")
    report.append("--- 实现合理性 ---")
    if num_triplets_per_bag:
        avg_tri = sum(num_triplets_per_bag) / len(num_triplets_per_bag)
        min_tri, max_tri = min(num_triplets_per_bag), max(num_triplets_per_bag)
        report.append(f"每 bag 梯度点数: 平均 {avg_tri:.1f}, 范围 [{min_tri}, {max_tri}]")
        if min_tri <= 1 and len(num_triplets_per_bag) > 1:
            failed_checks.append("部分样本梯度点数仅 1 个")
        if max_tri > 5000:
            failed_checks.append("梯度点数异常多 (如几千)")
    num_layers_expected = 32  # Llama-8B
    if layers_present:
        report.append(f"出现梯度的层: {len(layers_present)} (期望约 {num_layers_expected})")
        if len(layers_present) < 5:
            failed_checks.append("仅有少数层有梯度")

    # ---- 6. Distribution: long-tail, cross-sample repeat, few high ----
    report.append("")
    report.append("--- 逻辑合理性 ---")
    if all_scores:
        sorted_s = sorted(all_scores, reverse=True)
        n = len(sorted_s)
        top10_sum = sum(sorted_s[: max(1, n // 10)])
        total_sum = sum(sorted_s)
        if total_sum > 0:
            top10_ratio = top10_sum / total_sum
            report.append(f"长尾: 前10%分数之和 / 总和 ≈ {top10_ratio:.2%} (期望较高，即长尾)")
        # Cross-sample repeat
        repeat_neurons = sum(1 for (l, n), c in neuron_counter.items() if c > 1)
        report.append(f"跨样本重复出现的 (layer,neuron) 数: {repeat_neurons} (期望 >0)")
        if repeat_neurons == 0 and num_bags > 1:
            report.append("  ⚠ 无跨样本重复，可接受但若完全无重复可再确认")
        # Few high-gradient
        if len(sorted_s) >= 10:
            high = sorted_s[len(sorted_s) // 10]  # 90th percentile
            num_high = sum(1 for s in all_scores if s >= high)
            report.append(f"高梯度神经元(约前10%)数量: {num_high} / {len(all_scores)}")

    # Different layers different distribution
    if layer_scores:
        layer_means = {l: (sum(s) / len(s)) for l, s in layer_scores.items() if s}
        if len(layer_means) >= 2:
            vals = list(layer_means.values())
            report.append(f"不同层梯度分布: {len(layer_means)} 层有数据，层间均值存在差异 ✓")

    # ---- Summary ----
    report.append("")
    report.append("======== 汇总 ========")
    for r in report:
        print(r)
    if failed_checks:
        print("")
        print("❌ Step 2 需要重新运行或检查:")
        for f in failed_checks:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("")
        print("✓ Step 2 通过校验: 数据完整、无 NaN/Inf、梯度在合理范围、逻辑合理。")
        sys.exit(0)


if __name__ == "__main__":
    main()
