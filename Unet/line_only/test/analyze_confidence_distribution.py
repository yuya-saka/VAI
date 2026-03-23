"""
Confidence分布と角度誤差の関係を分析

適切なconfidence閾値を決定するため
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.train_heat import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
    TinyUNet,
)
from line_only.line_losses import extract_pred_line_params_batch, extract_gt_line_params


def analyze_all_test_samples(model, test_loader, dataset_root, device):
    """全テストサンプルのconfidenceと誤差を収集"""

    all_data = []

    for batch_idx, batch in enumerate(test_loader):
        x = batch["image"].to(device).float()

        with torch.no_grad():
            pred = torch.sigmoid(model(x))

        B = pred.shape[0]

        for i in range(B):
            sample_name = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])

            # GT lines.json
            lines_json_path = dataset_root / sample_name / vertebra / "lines.json"
            if not lines_json_path.exists():
                continue

            with open(lines_json_path) as f:
                lines_data = json.load(f)

            slice_data = lines_data.get(str(slice_idx), {})
            if not slice_data:
                continue

            # 予測
            pred_params, confidence = extract_pred_line_params_batch(
                pred[i:i+1], image_size=224
            )

            # 各チャンネル
            for ch in range(4):
                line_name = f"line_{ch + 1}"
                gt_pts = slice_data.get(line_name)

                if gt_pts is None or len(gt_pts) < 2:
                    continue

                # GT角度
                gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size=224)
                if np.isnan(gt_phi):
                    continue

                # 予測角度
                pred_phi = pred_params[0, ch, 0].item()
                conf = confidence[0, ch].item()

                # 角度誤差
                gt_phi_deg = np.degrees(gt_phi) % 180
                pred_phi_deg = np.degrees(pred_phi) % 180

                angle_error = abs(pred_phi_deg - gt_phi_deg)
                angle_error = min(angle_error, 180 - angle_error)

                all_data.append({
                    "confidence": conf,
                    "angle_error": angle_error,
                    "sample": sample_name,
                    "vertebra": vertebra,
                    "slice_idx": slice_idx,
                    "channel": ch + 1,
                })

    return all_data


def analyze_and_recommend(all_data):
    """分析して適切な閾値を推奨"""

    confidences = np.array([d["confidence"] for d in all_data])
    errors = np.array([d["angle_error"] for d in all_data])

    print("="*80)
    print("Confidence分布と誤差の分析")
    print("="*80)

    # 分布統計
    print(f"\nConfidence分布:")
    print(f"  平均:   {confidences.mean():.3f}")
    print(f"  中央値: {np.median(confidences):.3f}")
    print(f"  標準偏差: {confidences.std():.3f}")
    print(f"  最小:   {confidences.min():.3f}")
    print(f"  最大:   {confidences.max():.3f}")

    print(f"\n角度誤差:")
    print(f"  平均:   {errors.mean():.1f}°")
    print(f"  中央値: {np.median(errors):.1f}°")
    print(f"  最大:   {errors.max():.1f}°")

    # Confidence範囲別の統計
    print(f"\n{'='*80}")
    print("Confidence範囲別の統計")
    print(f"{'='*80}")

    ranges = [
        (0.0, 0.1, "0.0-0.1"),
        (0.1, 0.2, "0.1-0.2"),
        (0.2, 0.3, "0.2-0.3"),
        (0.3, 0.4, "0.3-0.4"),
        (0.4, 0.5, "0.4-0.5"),
        (0.5, 1.0, "0.5-1.0"),
    ]

    stats_by_range = []

    for min_c, max_c, label in ranges:
        mask = (confidences >= min_c) & (confidences < max_c)
        count = mask.sum()

        if count > 0:
            range_errors = errors[mask]
            mean_err = range_errors.mean()
            median_err = np.median(range_errors)
            max_err = range_errors.max()
            pct = 100 * count / len(confidences)

            stats_by_range.append({
                "range": label,
                "count": count,
                "pct": pct,
                "mean_err": mean_err,
                "median_err": median_err,
                "max_err": max_err,
            })

            print(f"\nConfidence {label}:")
            print(f"  サンプル数: {count} ({pct:.1f}%)")
            print(f"  平均誤差:   {mean_err:.1f}°")
            print(f"  中央値誤差: {median_err:.1f}°")
            print(f"  最大誤差:   {max_err:.1f}°")

    # 閾値候補の評価
    print(f"\n{'='*80}")
    print("閾値候補の評価")
    print(f"{'='*80}")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    print(f"\n{'閾値':>6s} {'残るサンプル':>12s} {'平均誤差':>10s} {'中央値誤差':>12s} {'最大誤差':>10s}")
    print("-" * 60)

    for thresh in thresholds:
        mask = confidences >= thresh
        count = mask.sum()
        pct = 100 * count / len(confidences)

        if count > 0:
            filtered_errors = errors[mask]
            mean_err = filtered_errors.mean()
            median_err = np.median(filtered_errors)
            max_err = filtered_errors.max()

            print(f"{thresh:6.2f} {count:5d} ({pct:4.1f}%) {mean_err:8.1f}° {median_err:10.1f}° {max_err:8.1f}°")

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confidence分布
    ax = axes[0]
    ax.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax.axvline(np.median(confidences), color='r', linestyle='--', label=f'Median={np.median(confidences):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confidence vs 角度誤差（散布図）
    ax = axes[1]
    ax.scatter(confidences, errors, alpha=0.3, s=10)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Angle Error (degrees)', fontsize=12)
    ax.set_title('Confidence vs Angle Error', fontsize=14, fontweight='bold')
    ax.axhline(10, color='g', linestyle='--', label='10° threshold')
    ax.axhline(20, color='orange', linestyle='--', label='20° threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. 閾値別の平均誤差
    ax = axes[2]
    thresh_vals = []
    mean_errs = []
    for thresh in np.linspace(0, 0.6, 31):
        mask = confidences >= thresh
        if mask.sum() > 0:
            thresh_vals.append(thresh)
            mean_errs.append(errors[mask].mean())

    ax.plot(thresh_vals, mean_errs, 'o-', linewidth=2)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Mean Angle Error (degrees)', fontsize=12)
    ax.set_title('Mean Error vs Threshold', fontsize=14, fontweight='bold')
    ax.axhline(10, color='g', linestyle='--', alpha=0.5, label='10° target')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = Path(__file__).parent / "confidence_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n結果を保存: {save_path}")
    plt.close()

    # 推奨閾値
    print(f"\n{'='*80}")
    print("推奨閾値")
    print(f"{'='*80}")

    # 目標: 平均誤差 < 15度、50%以上のサンプル保持
    for thresh in thresholds:
        mask = confidences >= thresh
        count = mask.sum()
        pct = 100 * count / len(confidences)

        if count > 0:
            mean_err = errors[mask].mean()

            if mean_err < 15 and pct > 50:
                print(f"\n✅ 推奨閾値: {thresh:.2f}")
                print(f"   - 残るサンプル: {count} ({pct:.1f}%)")
                print(f"   - 平均誤差: {mean_err:.1f}°")
                print(f"   - 理由: 平均誤差<15度 かつ サンプル保持>50%")
                break
    else:
        print(f"\n⚠️  理想的な閾値が見つかりません。段階的アプローチを推奨:")
        print(f"   1. 初期: 0.10-0.15（最悪のサンプルのみ除外）")
        print(f"   2. 中期: 0.20-0.25（学習が進んだら）")
        print(f"   3. 最終: 0.30以上（高品質のみ）")

    # ソフトウェイティングの提案
    print(f"\n代替案: ソフトウェイティング")
    print(f"  ハード閾値の代わりに連続的な重み付け:")
    print(f"    weight = confidence")
    print(f"    weight = confidence^2  (低confidenceをより強く抑制)")
    print(f"    weight = max(0, (confidence - 0.1) / 0.3)  (線形スケーリング)")


def main():
    cfg = load_config()
    dataset_root = Path(cfg.get("data", {}).get("root_dir", ""))
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
        224, 2.5, 42, cfg,
    )

    # モデルロード
    model_cfg = cfg.get("model", {})
    ckpt_dir = unet_dir / cfg.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")
    ckpt_files = list(ckpt_dir.glob("best_fold*.pt")) + list(ckpt_dir.glob("fold*_*.pth"))
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)

    model = TinyUNet(
        in_ch=int(model_cfg.get("in_channels", 2)),
        out_ch=int(model_cfg.get("out_channels", 4)),
        feats=tuple(model_cfg.get("features", [16, 32, 64, 128])),
        dropout=float(model_cfg.get("dropout", 0.0))
    )
    checkpoint = torch.load(latest_ckpt, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("全テストサンプルを分析中...")
    all_data = analyze_all_test_samples(model, test_loader, dataset_root, device)
    print(f"分析完了: {len(all_data)}個のサンプル")

    analyze_and_recommend(all_data)


if __name__ == "__main__":
    main()
