"""
折り返しの程度を定量的に分析

なぜline_1/line_3だけ角度誤差が大きいのかを調査
"""

import json
from pathlib import Path
import numpy as np

dataset_root = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
sample = "sample22"
vertebra = "C1"

lines_json_path = dataset_root / sample / vertebra / "lines.json"

with open(lines_json_path) as f:
    data = json.load(f)

print("=" * 80)
print("折り返しの程度を分析")
print("=" * 80)

for ch, line_name in enumerate(['line_1', 'line_2', 'line_3', 'line_4'], 1):
    print(f"\n{'=' * 80}")
    print(f"{line_name}")
    print("=" * 80)

    foldback_distances = []
    total_lengths = []

    for slice_idx in sorted(data.keys(), key=int)[:10]:  # 最初の10スライス
        lines = data.get(slice_idx, {})
        pts = lines.get(line_name, [])
        if len(pts) < 3:
            continue

        pts_arr = np.array(pts, dtype=np.float64)

        # 全体の直線距離（端点間）
        straight_dist = np.linalg.norm(pts_arr[-1] - pts_arr[0])

        # 折れ線の総長
        segment_lengths = []
        for i in range(len(pts) - 1):
            seg_len = np.linalg.norm(pts_arr[i+1] - pts_arr[i])
            segment_lengths.append(seg_len)

        total_length = sum(segment_lengths)

        # 折り返しの距離（余分な長さ）
        foldback_dist = total_length - straight_dist
        foldback_ratio = foldback_dist / straight_dist if straight_dist > 0 else 0

        # 最大の逆行距離を検出
        max_backtrack = 0
        for i in range(len(pts) - 2):
            d1 = pts_arr[i+1] - pts_arr[i]
            d2 = pts_arr[i+2] - pts_arr[i+1]
            dot = np.dot(d1, d2)

            if dot < 0:  # 折り返し
                backtrack_dist = np.linalg.norm(d2)
                max_backtrack = max(max_backtrack, backtrack_dist)

        print(f"\nslice {slice_idx}:")
        print(f"  点数: {len(pts)}")
        print(f"  端点間距離: {straight_dist:.1f}px")
        print(f"  折れ線総長: {total_length:.1f}px")
        print(f"  折り返し距離: {foldback_dist:.1f}px ({foldback_ratio*100:.1f}%)")
        print(f"  最大逆行距離: {max_backtrack:.1f}px")

        # 各セグメントの詳細
        print(f"  セグメント詳細:")
        for i, seg_len in enumerate(segment_lengths):
            print(f"    [{i}→{i+1}]: {seg_len:.1f}px")

        foldback_distances.append(foldback_dist)
        total_lengths.append(total_length)

    # チャンネルごとの統計
    if foldback_distances:
        print(f"\n{line_name} 統計:")
        print(f"  平均折り返し距離: {np.mean(foldback_distances):.1f}px")
        print(f"  平均折れ線総長: {np.mean(total_lengths):.1f}px")
        print(f"  平均折り返し比率: {np.mean(foldback_distances) / np.mean(total_lengths) * 100:.1f}%")
