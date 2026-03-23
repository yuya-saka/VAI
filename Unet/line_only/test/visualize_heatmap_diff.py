"""
GTヒートマップとモデル出力の差分を可視化
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.train_heat import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
    TinyUNet,
)


def generate_gt_heatmap(pts_xy, image_size, sigma):
    """GTヒートマップを生成"""
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


def visualize_difference():
    """GTとモデル出力の差分を可視化"""

    # 設定とモデルをロード
    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    sigma = float(cfg.get("data", {}).get("sigma", 2.5))
    dataset_root_str = cfg.get("data", {}).get("root_dir", "")
    dataset_root = Path(dataset_root_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ分割
    all_samples = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("sample")]
    train_samples, val_samples, test_samples = kfold_split_samples(all_samples, n_folds=5, test_fold=0, seed=42)

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

    # sample22_C1_slice038 を探す
    target_sample = "sample22"
    target_vertebra = "C1"
    target_slice = 38

    selected_batch = None
    selected_idx = None

    for batch in test_loader:
        for i in range(len(batch["sample"])):
            if (
                batch["sample"][i] == target_sample
                and batch["vertebra"][i] == target_vertebra
                and int(batch["slice_idx"][i]) == target_slice
            ):
                selected_batch = batch
                selected_idx = i
                break
        if selected_batch is not None:
            break

    if selected_batch is None:
        print("Sample not found")
        return

    # 推論
    x = selected_batch["image"].to(device).float()
    with torch.no_grad():
        pred = torch.sigmoid(model(x))

    model_heatmaps = pred[selected_idx].cpu().numpy()  # (4, H, W)

    # GTをロード
    lines_json_path = dataset_root / target_sample / target_vertebra / "lines.json"
    with open(lines_json_path) as f:
        lines_data = json.load(f)
    slice_data = lines_data.get(str(target_slice), {})

    # 出力ディレクトリ
    out_dir = Path(__file__).parent / "heatmap_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ヒートマップ差分可視化")
    print("=" * 80)

    # 各直線について処理
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        gt_pts = slice_data.get(line_name)
        if not gt_pts or len(gt_pts) < 2:
            continue

        print(f"\n{line_name}:")

        # GTヒートマップを生成
        gt_heatmap = generate_gt_heatmap(gt_pts, image_size, sigma)
        model_heatmap = model_heatmaps[ch]

        # 差分を計算
        diff = model_heatmap - gt_heatmap

        # 統計
        gt_mass = gt_heatmap.sum()
        model_mass = model_heatmap.sum()
        gt_max = gt_heatmap.max()
        model_max = model_heatmap.max()

        print(f"  GT:    質量={gt_mass:.2f}, 最大値={gt_max:.4f}")
        print(f"  Model: 質量={model_mass:.2f}, 最大値={model_max:.4f}")
        print(f"  差分:  質量={model_mass - gt_mass:.2f} ({(model_mass/gt_mass - 1)*100:.1f}%)")

        # 可視化
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: ヒートマップ比較
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt_heatmap, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f'{line_name} - GT Heatmap\nMass={gt_mass:.2f}, Max={gt_max:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(model_heatmap, cmap='hot', vmin=0, vmax=1)
        ax2.set_title(f'{line_name} - Model Output\nMass={model_mass:.2f}, Max={model_max:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax3.set_title(f'{line_name} - Difference (Model - GT)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3)

        # Row 2: ヒストグラム
        ax4 = fig.add_subplot(gs[1, :])
        bins = np.linspace(0, 1, 100)
        ax4.hist(gt_heatmap.flatten(), bins=bins, alpha=0.5, label='GT', color='blue', density=True)
        ax4.hist(model_heatmap.flatten(), bins=bins, alpha=0.5, label='Model', color='red', density=True)
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Value Distribution Comparison')
        ax4.legend()
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # Row 3: 差分の詳細分析
        ax5 = fig.add_subplot(gs[2, 0])
        pos_diff = np.maximum(diff, 0)
        im5 = ax5.imshow(pos_diff, cmap='Reds', vmin=0, vmax=0.5)
        ax5.set_title(f'Positive Difference (Model > GT)\nSum={pos_diff.sum():.2f}')
        plt.colorbar(im5, ax=ax5)

        ax6 = fig.add_subplot(gs[2, 1])
        neg_diff = np.maximum(-diff, 0)
        im6 = ax6.imshow(neg_diff, cmap='Blues', vmin=0, vmax=0.5)
        ax6.set_title(f'Negative Difference (GT > Model)\nSum={neg_diff.sum():.2f}')
        plt.colorbar(im6, ax=ax6)

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(diff.flatten(), bins=100, color='purple', alpha=0.7)
        ax7.set_xlabel('Difference Value')
        ax7.set_ylabel('Count')
        ax7.set_title('Difference Distribution')
        ax7.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax7.grid(True, alpha=0.3)

        # 保存
        save_path = out_dir / f"{target_sample}_{target_vertebra}_slice{target_slice:03d}_{line_name}_diff.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  保存: {save_path.name}")

    print("\n" + "=" * 80)
    print(f"完了。結果は {out_dir} に保存されました。")
    print("=" * 80)


if __name__ == "__main__":
    visualize_difference()
