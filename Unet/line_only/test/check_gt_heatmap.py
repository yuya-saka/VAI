"""
GTヒートマップ（sigma=2.5）の分布を確認

モデル出力との比較
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))


def generate_gt_heatmap(pts_xy, image_size, sigma):
    """GTヒートマップを生成（train_heat.pyのロジックと同じ）"""
    H = W = image_size
    hm = np.zeros((H, W), np.float32)
    if pts_xy is None or len(pts_xy) < 2:
        return hm

    pts = np.array(pts_xy, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    pts_i32 = pts.astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts_i32], isClosed=False, color=1, thickness=1)

    inv = (1 - mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    s2 = max(1e-6, sigma**2)
    hm = np.exp(-(dist**2) / (2.0 * s2)).astype(np.float32)
    return hm


def analyze_gt_heatmap():
    """GTヒートマップの分布を分析"""

    # sample22_C1_slice038 のGT線をロード
    dataset_root = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
    sample = "sample22"
    vertebra = "C1"
    slice_idx = 38

    lines_json_path = dataset_root / sample / vertebra / "lines.json"
    with open(lines_json_path) as f:
        lines_data = json.load(f)

    slice_data = lines_data.get(str(slice_idx), {})

    image_size = 224
    sigma = 2.5

    print("=" * 80)
    print("GT Heatmap 分析（sigma=2.5）")
    print("=" * 80)

    for line_key in ["line_1", "line_2", "line_3", "line_4"]:
        gt_pts = slice_data.get(line_key)
        if not gt_pts or len(gt_pts) < 2:
            continue

        print(f"\n{line_key}:")
        print("-" * 60)

        # GTヒートマップを生成
        gt_heatmap = generate_gt_heatmap(gt_pts, image_size, sigma)

        H, W = gt_heatmap.shape

        # 統計情報
        print(f"  最大値: {gt_heatmap.max():.4f}")
        print(f"  平均値: {gt_heatmap.mean():.4f}")
        print(f"  総質量: {gt_heatmap.sum():.2f}")

        # 異なる閾値での有効ピクセル比率
        thresholds = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20]
        print(f"  閾値ごとの有効ピクセル比率:")
        for thr in thresholds:
            ratio = (gt_heatmap > thr).sum() / (H * W) * 100
            print(f"    > {thr:.3f}: {ratio:5.2f}%")

        # 行ごとの最大値が閾値以上の行数
        row_max_values = np.array([np.max(row) for row in gt_heatmap])
        print(f"  行ごとの最大値が閾値以上の行数:")
        for thr in thresholds:
            n_valid = (row_max_values > thr).sum()
            ratio = n_valid / H * 100
            print(f"    > {thr:.3f}: {n_valid:3d} / {H} ({ratio:5.1f}%)")


if __name__ == "__main__":
    analyze_gt_heatmap()
