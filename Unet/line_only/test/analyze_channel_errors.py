"""
チャンネルごとの角度誤差パターンを詳細分析

line_1/line_3とline_2/line_4の違いを調査
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.src.data_utils import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
)
from line_only.src.model import TinyUNet
from line_only.utils.detection import detect_line_moments, gt_centroid_angle_from_polyline
from line_only.utils import losses as line_losses


def angle_diff_deg(a_deg, b_deg):
    """180度周期で角度差を計算"""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def analyze_channel_errors():
    """各チャンネルの角度誤差を詳細分析"""

    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    sigma = float(cfg.get("data", {}).get("sigma", 2.5))
    dataset_root_str = cfg.get("data", {}).get("root_dir", "")
    dataset_root = Path(dataset_root_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ分割
    all_samples = [
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("sample")
    ]
    train_samples, val_samples, test_samples = kfold_split_samples(
        all_samples, n_folds=5, test_fold=0, seed=42
    )

    # データローダー
    _, _, test_loader = create_data_loaders(
        train_samples, val_samples, test_samples,
        dataset_root,
        cfg.get("data", {}).get("vertebra_group", "ALL"),
        image_size, sigma, 42, cfg,
    )

    # モデルをロード
    model_cfg = cfg.get("model", {})
    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    ckpt_dir = unet_dir / cfg.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")
    ckpt_files = list(ckpt_dir.glob("best_fold*.pt")) + list(ckpt_dir.glob("fold*_*.pth"))
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)

    model = TinyUNet(in_ch=in_ch, out_ch=out_ch, feats=feats, dropout=dropout)
    checkpoint = torch.load(latest_ckpt, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("=" * 80)
    print("チャンネルごとの角度誤差分析")
    print("=" * 80)

    # チャンネルごとの統計
    channel_errors = {1: [], 2: [], 3: [], 4: []}
    channel_gt_angles = {1: [], 2: [], 3: [], 4: []}
    channel_pred_angles = {1: [], 2: [], 3: [], 4: []}
    channel_gt_positions = {1: [], 2: [], 3: [], 4: []}

    # チャンネルごとのヒートマップ統計
    channel_heatmap_stats = {1: [], 2: [], 3: [], 4: []}

    processed = 0
    max_samples = 50

    for batch in test_loader:
        if processed >= max_samples:
            break

        x = batch["image"].to(device).float()
        with torch.no_grad():
            pred = torch.sigmoid(model(x))

        pred_np = pred.cpu().numpy()

        B = pred_np.shape[0]
        for i in range(B):
            if processed >= max_samples:
                break

            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])

            # GT lines.jsonをロード
            lines_json_path = dataset_root / sample / vertebra / "lines.json"
            if not lines_json_path.exists():
                continue

            with open(lines_json_path) as f:
                lines_data = json.load(f)

            slice_data = lines_data.get(str(slice_idx), {})
            if not slice_data:
                continue

            # 各チャンネルを分析
            for ch in range(4):
                line_name = f"line_{ch + 1}"
                gt_pts = slice_data.get(line_name)
                if not gt_pts or len(gt_pts) < 2:
                    continue

                heatmap = pred_np[i, ch]

                # GT角度（PCAベース）
                gt_centroid, gt_angle_deg = gt_centroid_angle_from_polyline(gt_pts)
                if gt_centroid is None:
                    continue

                # GT線長を計算
                gt_pts_array = np.array(gt_pts, dtype=np.float64)
                d = gt_pts_array[1:] - gt_pts_array[:-1]
                length_gt = float(np.sqrt((d**2).sum(axis=1)).sum())

                # ヒートマップから線を抽出
                pred_line = detect_line_moments(heatmap, length_px=length_gt, extend_ratio=1.10)
                if pred_line is None:
                    continue

                pred_angle_deg = pred_line.get("angle_deg")
                if pred_angle_deg is None:
                    continue

                # 角度誤差
                angle_error = angle_diff_deg(pred_angle_deg, gt_angle_deg)

                # ヒートマップ統計
                M00 = pred_line.get("M00", 0)
                max_val = heatmap.max()
                threshold = 0.15
                valid_rows = np.any(heatmap > threshold, axis=1).sum()
                valid_ratio = valid_rows / heatmap.shape[0]

                channel_errors[ch + 1].append(angle_error)
                channel_gt_angles[ch + 1].append(gt_angle_deg)
                channel_pred_angles[ch + 1].append(pred_angle_deg)
                channel_gt_positions[ch + 1].append({
                    "centroid": gt_centroid,
                    "sample": sample,
                    "vertebra": vertebra,
                    "slice": slice_idx,
                })
                channel_heatmap_stats[ch + 1].append({
                    "M00": M00,
                    "max": max_val,
                    "valid_ratio": valid_ratio,
                })

            processed += 1

    print(f"\n分析サンプル数: {processed}")

    # チャンネルごとの統計を表示
    print("\n" + "=" * 80)
    print("チャンネルごとの角度誤差統計")
    print("=" * 80)

    for ch in [1, 2, 3, 4]:
        errors = channel_errors[ch]
        if not errors:
            continue

        errors = np.array(errors)
        gt_angles = np.array(channel_gt_angles[ch])
        pred_angles = np.array(channel_pred_angles[ch])
        hm_stats = channel_heatmap_stats[ch]

        print(f"\n--- line_{ch} ---")
        print(f"  サンプル数: {len(errors)}")
        print(f"  角度誤差: mean={errors.mean():.2f}°, std={errors.std():.2f}°, max={errors.max():.2f}°")
        print(f"  GT角度範囲: [{gt_angles.min():.1f}°, {gt_angles.max():.1f}°]")
        print(f"  予測角度範囲: [{pred_angles.min():.1f}°, {pred_angles.max():.1f}°]")

        # ヒートマップ統計
        m00_vals = [s["M00"] for s in hm_stats]
        max_vals = [s["max"] for s in hm_stats]
        valid_ratios = [s["valid_ratio"] for s in hm_stats]
        print(f"  ヒートマップ質量: mean={np.mean(m00_vals):.1f}, std={np.std(m00_vals):.1f}")
        print(f"  ヒートマップ最大値: mean={np.mean(max_vals):.3f}")
        print(f"  有効行比率: mean={np.mean(valid_ratios)*100:.1f}%")

        # 大きな誤差のサンプルを表示
        large_error_idx = np.where(errors > 15)[0]
        if len(large_error_idx) > 0:
            print(f"\n  [大きな誤差 > 15°] {len(large_error_idx)}件:")
            for idx in large_error_idx[:5]:
                pos = channel_gt_positions[ch][idx]
                print(f"    {pos['sample']}_{pos['vertebra']}_slice{pos['slice']}: "
                      f"GT={gt_angles[idx]:.1f}°, Pred={pred_angles[idx]:.1f}°, "
                      f"Error={errors[idx]:.1f}°")

    # GT角度の分布を比較
    print("\n" + "=" * 80)
    print("GT角度の分布（line_1/line_3 vs line_2/line_4）")
    print("=" * 80)

    for ch in [1, 3]:
        if channel_gt_angles[ch]:
            angles = np.array(channel_gt_angles[ch])
            print(f"\nline_{ch}: mean={angles.mean():.1f}°, std={angles.std():.1f}°")
            print(f"  分布: [0-30°]={np.sum((angles >= 0) & (angles < 30))}, "
                  f"[30-60°]={np.sum((angles >= 30) & (angles < 60))}, "
                  f"[60-90°]={np.sum((angles >= 60) & (angles < 90))}, "
                  f"[90-120°]={np.sum((angles >= 90) & (angles < 120))}, "
                  f"[120-150°]={np.sum((angles >= 120) & (angles < 150))}, "
                  f"[150-180°]={np.sum((angles >= 150) & (angles <= 180))}")

    for ch in [2, 4]:
        if channel_gt_angles[ch]:
            angles = np.array(channel_gt_angles[ch])
            print(f"\nline_{ch}: mean={angles.mean():.1f}°, std={angles.std():.1f}°")
            print(f"  分布: [0-30°]={np.sum((angles >= 0) & (angles < 30))}, "
                  f"[30-60°]={np.sum((angles >= 30) & (angles < 60))}, "
                  f"[60-90°]={np.sum((angles >= 60) & (angles < 90))}, "
                  f"[90-120°]={np.sum((angles >= 90) & (angles < 120))}, "
                  f"[120-150°]={np.sum((angles >= 120) & (angles < 150))}, "
                  f"[150-180°]={np.sum((angles >= 150) & (angles <= 180))}")

    # GT位置の分布を比較
    print("\n" + "=" * 80)
    print("GT重心位置の分布")
    print("=" * 80)

    for ch in [1, 2, 3, 4]:
        if channel_gt_positions[ch]:
            centroids = [p["centroid"] for p in channel_gt_positions[ch]]
            cx = np.array([c[0] for c in centroids])
            cy = np.array([c[1] for c in centroids])
            print(f"\nline_{ch}: x=[{cx.min():.1f}, {cx.max():.1f}] (mean={cx.mean():.1f}), "
                  f"y=[{cy.min():.1f}, {cy.max():.1f}] (mean={cy.mean():.1f})")

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)


if __name__ == "__main__":
    analyze_channel_errors()
