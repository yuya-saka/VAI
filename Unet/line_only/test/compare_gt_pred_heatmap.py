"""
GTヒートマップと予測ヒートマップの詳細比較

line_1/line_3とline_2/line_4で学習結果が異なる原因を調査
"""

import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
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
from line_only.src.dataset import PngLineDataset
from line_only.utils.detection import detect_line_moments


def compare_gt_pred_heatmaps():
    """GTヒートマップと予測ヒートマップを詳細比較"""

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

    # データローダー（テスト用）
    test_dataset = PngLineDataset(
        root_dir=dataset_root,
        sample_names=test_samples,
        group=cfg.get("data", {}).get("vertebra_group", "ALL"),
        image_size=image_size,
        sigma=sigma,
        transform=None,
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

    # 出力ディレクトリ
    out_dir = Path(__file__).parent / "gt_pred_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GTヒートマップと予測ヒートマップの比較")
    print("=" * 80)

    # 最初の5サンプルを詳細に比較
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        name = f"{sample['sample']}_{sample['vertebra']}_slice{sample['slice_idx']:03d}"

        x = sample["image"].unsqueeze(0).to(device).float()
        gt_heatmaps = sample["heatmaps"].numpy()  # (4, H, W)

        with torch.no_grad():
            pred = torch.sigmoid(model(x))
        pred_heatmaps = pred[0].cpu().numpy()  # (4, H, W)

        # 4チャンネルを比較
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(name, fontsize=16, fontweight='bold')

        for ch in range(4):
            gt_hm = gt_heatmaps[ch]
            pred_hm = pred_heatmaps[ch]

            # GT heatmap
            ax = axes[ch, 0]
            im = ax.imshow(gt_hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'line_{ch+1} GT Heatmap')
            plt.colorbar(im, ax=ax)

            # GT heatmapから抽出した角度
            gt_line = detect_line_moments(gt_hm)
            if gt_line:
                gt_angle = gt_line.get("angle_deg", 0)
                ax.text(0.02, 0.98, f'θ={gt_angle:.1f}°', transform=ax.transAxes,
                       fontsize=10, color='white', va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            # Pred heatmap
            ax = axes[ch, 1]
            im = ax.imshow(pred_hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'line_{ch+1} Pred Heatmap')
            plt.colorbar(im, ax=ax)

            # Pred heatmapから抽出した角度
            pred_line = detect_line_moments(pred_hm)
            if pred_line:
                pred_angle = pred_line.get("angle_deg", 0)
                ax.text(0.02, 0.98, f'θ={pred_angle:.1f}°', transform=ax.transAxes,
                       fontsize=10, color='white', va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            # 差分
            ax = axes[ch, 2]
            diff = pred_hm - gt_hm
            im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax.set_title(f'line_{ch+1} Diff (Pred - GT)')
            plt.colorbar(im, ax=ax)

            # 重ね合わせ
            ax = axes[ch, 3]
            overlay = np.zeros((image_size, image_size, 3))
            overlay[:, :, 0] = gt_hm  # GT = Red
            overlay[:, :, 1] = pred_hm  # Pred = Green
            ax.imshow(overlay)
            ax.set_title(f'line_{ch+1} Overlay (R=GT, G=Pred)')

        plt.tight_layout()
        save_path = out_dir / f"{name}_gt_pred_compare.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"保存: {save_path.name}")

        # テキストで統計を表示
        print(f"\n{name}:")
        for ch in range(4):
            gt_hm = gt_heatmaps[ch]
            pred_hm = pred_heatmaps[ch]

            gt_line = detect_line_moments(gt_hm)
            pred_line = detect_line_moments(pred_hm)

            gt_angle = gt_line.get("angle_deg", float('nan')) if gt_line else float('nan')
            pred_angle = pred_line.get("angle_deg", float('nan')) if pred_line else float('nan')

            if not np.isnan(gt_angle) and not np.isnan(pred_angle):
                diff = abs(gt_angle - pred_angle) % 180
                diff = min(diff, 180 - diff)
            else:
                diff = float('nan')

            gt_mass = gt_hm.sum()
            pred_mass = pred_hm.sum()

            print(f"  line_{ch+1}: GT_θ={gt_angle:.1f}°, Pred_θ={pred_angle:.1f}°, "
                  f"Diff={diff:.1f}°, GT_mass={gt_mass:.0f}, Pred_mass={pred_mass:.0f}")

    print("\n" + "=" * 80)
    print(f"完了。結果は {out_dir} に保存されました。")
    print("=" * 80)


if __name__ == "__main__":
    compare_gt_pred_heatmaps()
