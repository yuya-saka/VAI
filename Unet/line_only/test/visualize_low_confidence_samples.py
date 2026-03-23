"""
低confidenceサンプルの実際の画像を確認

「円形blob」という仮説が正しいか検証
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


def visualize_sample_with_confidence(pred_heatmap, confidence, sample_info, save_path):
    """
    1つのチャンネルのヒートマップを詳細可視化

    Args:
        pred_heatmap: (H, W) 予測ヒートマップ
        confidence: float 信頼度
        sample_info: dict サンプル情報
        save_path: Path 保存先
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. ヒートマップ
    ax = axes[0, 0]
    im = ax.imshow(pred_heatmap, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Heatmap (Confidence={confidence:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    # 2. 閾値処理後のヒートマップ（0.2以上）
    ax = axes[0, 1]
    thresholded = np.where(pred_heatmap >= 0.2, pred_heatmap, 0)
    im = ax.imshow(thresholded, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Thresholded (>=0.2)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    # 3. Row Max Projection
    ax = axes[1, 0]
    row_max = pred_heatmap.max(axis=1)
    ax.plot(row_max)
    ax.axhline(y=0.2, color='r', linestyle='--', label='Threshold 0.2')
    ax.set_xlabel('y (row)', fontsize=10)
    ax.set_ylabel('max value', fontsize=10)
    ax.set_title('Row Max Projection', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Col Max Projection
    ax = axes[1, 1]
    col_max = pred_heatmap.max(axis=0)
    ax.plot(col_max)
    ax.axhline(y=0.2, color='r', linestyle='--', label='Threshold 0.2')
    ax.set_xlabel('x (col)', fontsize=10)
    ax.set_ylabel('max value', fontsize=10)
    ax.set_title('Col Max Projection', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"{sample_info['sample']}_{sample_info['vertebra']}_slice{sample_info['slice_idx']:03d}_ch{sample_info['channel']}",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def collect_and_visualize_extreme_cases(model, test_loader, dataset_root, device):
    """極端に低いconfidenceと高いconfidenceのサンプルを可視化"""

    all_samples = []

    print("サンプルを収集中...")

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 20:  # 最初の20バッチ
            break

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

                conf = confidence[0, ch].item()

                # GT角度
                gt_phi, _ = extract_gt_line_params(gt_pts, image_size=224)
                if np.isnan(gt_phi):
                    continue

                # 予測角度
                pred_phi = pred_params[0, ch, 0].item()

                gt_phi_deg = np.degrees(gt_phi) % 180
                pred_phi_deg = np.degrees(pred_phi) % 180
                angle_error = abs(pred_phi_deg - gt_phi_deg)
                angle_error = min(angle_error, 180 - angle_error)

                all_samples.append({
                    "confidence": conf,
                    "angle_error": angle_error,
                    "heatmap": pred[i, ch].cpu().numpy(),
                    "sample": sample_name,
                    "vertebra": vertebra,
                    "slice_idx": slice_idx,
                    "channel": ch + 1,
                })

    print(f"収集完了: {len(all_samples)}個のサンプル")

    # 信頼度でソート
    all_samples.sort(key=lambda x: x["confidence"])

    # 極端に低い5つ
    lowest_5 = all_samples[:5]
    # 極端に高い5つ
    highest_5 = all_samples[-5:]

    out_dir = Path(__file__).parent / "confidence_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("最も低いconfidenceのサンプル（5つ）")
    print("="*80)

    for idx, sample in enumerate(lowest_5):
        print(f"\n{idx+1}. {sample['sample']}_{sample['vertebra']}_slice{sample['slice_idx']:03d}_ch{sample['channel']}")
        print(f"   Confidence: {sample['confidence']:.3f}")
        print(f"   Angle Error: {sample['angle_error']:.1f}°")

        save_path = out_dir / f"low_conf_{idx+1}_conf{sample['confidence']:.3f}.png"
        visualize_sample_with_confidence(
            sample['heatmap'],
            sample['confidence'],
            sample,
            save_path
        )
        print(f"   保存: {save_path.name}")

    print("\n" + "="*80)
    print("最も高いconfidenceのサンプル（5つ）")
    print("="*80)

    for idx, sample in enumerate(highest_5):
        print(f"\n{idx+1}. {sample['sample']}_{sample['vertebra']}_slice{sample['slice_idx']:03d}_ch{sample['channel']}")
        print(f"   Confidence: {sample['confidence']:.3f}")
        print(f"   Angle Error: {sample['angle_error']:.1f}°")

        save_path = out_dir / f"high_conf_{idx+1}_conf{sample['confidence']:.3f}.png"
        visualize_sample_with_confidence(
            sample['heatmap'],
            sample['confidence'],
            sample,
            save_path
        )
        print(f"   保存: {save_path.name}")

    print(f"\n全ての画像を保存: {out_dir}")


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

    collect_and_visualize_extreme_cases(model, test_loader, dataset_root, device)


if __name__ == "__main__":
    main()
