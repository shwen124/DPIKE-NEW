"""
Filter privacy-relevant neurons from Step 2 (integrated gradient) JSONL output.
Usage: python 2_filter_privacy_neurons_llama.py <results_dir> <threshold_ratio> <mode_ratio_bag>
  e.g. python 2_filter_privacy_neurons_llama.py ./llama3_results/ 0.01 0.5

Note: ave_kn_num (per-bag neuron count) is determined only by threshold_ratio and the data;
      mode_ratio_bag affects only kn_rel (global neuron list), not per-bag counts.
"""
import json
import jsonlines
import os
import sys
from collections import Counter


def main():
    if len(sys.argv) < 4:
        print("Usage: python 2_filter_privacy_neurons_llama.py <results_dir> <threshold_ratio> <mode_ratio_bag>")
        sys.exit(1)

    rlts_dir = sys.argv[1].rstrip("/")
    kn_dir = os.path.join(rlts_dir, "kn")
    try:
        threshold_ratio = float(sys.argv[2])
        mode_ratio_bag = float(sys.argv[3])
    except ValueError:
        print("Error: threshold_ratio and mode_ratio_bag must be numbers.")
        sys.exit(1)

    # Boundary check: (0, 1] to avoid silent wrong behavior
    if not (0 < threshold_ratio <= 1):
        print(
            f"Error: threshold_ratio must be in (0, 1], got {threshold_ratio}. "
            "Use e.g. 0.005–0.02 for filtering by max*ratio."
        )
        sys.exit(1)
    if not (0 < mode_ratio_bag <= 1):
        print(
            f"Error: mode_ratio_bag must be in (0, 1], got {mode_ratio_bag}. "
            "Use e.g. 0.3–0.7 for cross-bag frequency."
        )
        sys.exit(1)

    def re_filter(metric_triplets, total_metrix, cnt_metrix):
        if not metric_triplets:
            return [], total_metrix, cnt_metrix
        metric_max = max(t[2] for t in metric_triplets)
        filtered = [t for t in metric_triplets if t[2] >= metric_max * threshold_ratio]
        total_metrix += metric_max
        cnt_metrix += 1
        return filtered, total_metrix, cnt_metrix

    def pos_list2str(pos_list):
        return "@".join([str(p) for p in pos_list])

    def pos_str2list(pos_str):
        return [int(p) for p in pos_str.split("@")]

    def parse_kn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
        mode_threshold = tot_num * mode_ratio
        mode_threshold = max(mode_threshold, min_threshold)
        kn_bag = []
        for pos_str, cnt in pos_cnt.items():
            if cnt >= mode_threshold:
                kn_bag.append(pos_str2list(pos_str))
        return kn_bag

    def analysis_file(filename, mode_ratio):
        rel = filename.replace(".priv.jsonl", "").split(".")[0]
        print(f"===========> parsing important position in {rel}..., mode_ratio_bag={mode_ratio}")

        rlts_bag_list = []
        with open(os.path.join(rlts_dir, filename), "r") as fr:
            for rlts_bag in jsonlines.Reader(fr):
                rlts_bag_list.append(rlts_bag)

        ave_kn_num = 0
        total_metrix = 0
        cnt_metrix = 0
        kn_bag_list = []
        for bag_idx, rlts_bag in enumerate(rlts_bag_list):
            pos_cnt_bag = Counter()
            for rlt in rlts_bag:
                res_dict = rlt[1]
                metric_triplets, total_metrix, cnt_metrix = re_filter(
                    res_dict, total_metrix, cnt_metrix
                )
                for metric_triplet in metric_triplets:
                    pos_cnt_bag.update([pos_list2str(metric_triplet[:2])])
            kn_bag = parse_kn(pos_cnt_bag, len(rlts_bag), 1)
            ave_kn_num += len(kn_bag)
            kn_bag_list.append(kn_bag)

        ave_kn_num /= len(rlts_bag_list) if rlts_bag_list else 1
        pos_cnt_rel = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                pos_cnt_rel.update([pos_list2str(kn)])
        kn_rel = parse_kn(pos_cnt_rel, len(kn_bag_list), mode_ratio)
        return rel, ave_kn_num, kn_bag_list, kn_rel

    def stat(data, pos_type, rel):
        if pos_type == "kn_rel":
            print(f"{rel}'s {pos_type} has {len(data)} imp pos.")
            return
        ave_len = sum(len(kn_bag) for kn_bag in data) / len(data) if data else 0
        print(f"{rel}'s {pos_type} has on average {ave_len:.1f} imp pos.")

    os.makedirs(kn_dir, exist_ok=True)

    for filename in sorted(os.listdir(rlts_dir)):
        if not filename.endswith(".priv.jsonl"):
            continue
        rel_base = filename.replace(".priv.jsonl", "")
        # Single run: mode_ratio_bag only affects kn_rel; ave_kn_num is fixed by threshold_ratio + data
        rel, ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename, mode_ratio_bag)
        if ave_kn_num < 2 or ave_kn_num > 10:
            print(
                f"  [Note] ave_kn_num={ave_kn_num:.1f} is outside 2–10; "
                "adjust threshold_ratio (e.g. 0.005–0.02) to change per-bag neuron count."
            )

        stat(kn_bag_list, "kn_bag", rel)
        stat(kn_rel, "kn_rel", rel)
        with open(os.path.join(kn_dir, f"kn_bag-{rel_base}.json"), "w") as fw:
            json.dump(kn_bag_list, fw, indent=2)
        with open(os.path.join(kn_dir, f"kn_rel-{rel_base}.json"), "w") as fw:
            json.dump(kn_rel, fw, indent=2)
        print(f"Wrote kn_bag-{rel_base}.json and kn_rel-{rel_base}.json to {kn_dir}/")


if __name__ == "__main__":
    main()
