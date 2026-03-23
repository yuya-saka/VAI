"""
ヒートマップと抽出された線を同じ画像上に可視化

各チャンネルのヒートマップに予測線を重ねて表示し、
複数のテストサンプルで実装ミスのパターンを確認する
"""

import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.train_heat import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
    TinyUNet,
)
from line_only.line_detection import detect_line_moments


def visualize_heatmap_with_lines(
    sample_info,
    heatmaps,
    pred_lines,
    gt_lines,
    save_path,
):
    """
    4チャンネルのヒートマップと抽出された線を可視化

    引数:
        sample_info: サンプル情報辞書 (sample, vertebra, slice_idx)
        heatmaps: (4, H, W) 予測ヒートマップ
        pred_lines: {"line_1": {...}, ...} 予測線情報
        gt_lines: {"line_1": [[x,y], ...], ...} GT線
        save_path: 保存先パス
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(
        f"{sample_info['sample']}_{sample_info['vertebra']}_slice{sample_info['slice_idx']:03d}",
        fontsize=16,
        fontweight='bold'
    )

    colors = {
        "line_1": "green",
        "line_2": "red",
        "line_3": "blue",
        "line_4": "yellow",
    }

    for ch in range(4):
        line_name = f"line_{ch + 1}"
        heatmap = heatmaps[ch]

        # 上段: ヒートマップ + 予測線 + GT線
        ax_top = axes[0, ch]
        im = ax_top.imshow(heatmap, cmap='hot', vmin=0, vmax=1, alpha=0.8)

        # 予測線を描画
        pred_line = pred_lines.get(line_name)
        if pred_line is not None:
            endpoints = pred_line.get("endpoints")
            if endpoints is not None and len(endpoints) == 2:
                (x1, y1), (x2, y2) = endpoints
                ax_top.plot([x1, x2], [y1, y2],
                           color=colors[line_name],
                           linewidth=2,
                           label='Pred',
                           linestyle='-',
                           marker='o',
                           markersize=4)

                # 重心にマーカー
                centroid = pred_line.get("centroid")
                if centroid is not None:
                    cx, cy = centroid
                    ax_top.plot(cx, cy, 'x',
                               color=colors[line_name],
                               markersize=10,
                               markeredgewidth=3)

                # 角度情報を表示
                angle_deg = pred_line.get("angle_deg")
                if angle_deg is not None:
                    ax_top.text(0.02, 0.98, f'θ={angle_deg:.1f}°',
                               transform=ax_top.transAxes,
                               fontsize=10,
                               color='white',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                               verticalalignment='top')

        # GT線を描画
        gt_pts = gt_lines.get(line_name)
        if gt_pts is not None and len(gt_pts) >= 2:
            gt_pts_array = np.array(gt_pts)
            ax_top.plot(gt_pts_array[:, 0], gt_pts_array[:, 1],
                       color='white',
                       linewidth=2,
                       label='GT',
                       linestyle='--',
                       alpha=0.8)

        ax_top.set_title(f'{line_name} - Heatmap + Lines')
        ax_top.set_xlabel('x')
        ax_top.set_ylabel('y')
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # 下段: ヒートマップの詳細情報
        ax_bottom = axes[1, ch]

        # 統計情報を表示
        M00 = pred_line.get("M00", 0) if pred_line else 0
        max_val = heatmap.max()
        mean_val = heatmap.mean()

        # 有効範囲の計算（閾値以上の行数）
        threshold = 0.15
        rows_above_threshold = np.any(heatmap > threshold, axis=1).sum()
        total_rows = heatmap.shape[0]

        # 閾値以上のピクセル比率
        pixels_above_threshold = (heatmap > threshold).sum()
        total_pixels = heatmap.size

        stats_text = f"""
質量 (M00): {M00:.2f}
最大値: {max_val:.4f}
平均値: {mean_val:.4f}

有効行数: {rows_above_threshold}/{total_rows} ({100*rows_above_threshold/total_rows:.1f}%)
閾値以上: {pixels_above_threshold}/{total_pixels} ({100*pixels_above_threshold/total_pixels:.2f}%)
        """.strip()

        # ヒートマップの投影（行方向の最大値）
        row_max = heatmap.max(axis=1)
        col_max = heatmap.max(axis=0)

        # 2x2のサブプロットを作成
        ax_bottom.text(0.1, 0.7, stats_text,
                      fontsize=9,
                      family='monospace',
                      verticalalignment='top')

        # 行方向の投影をプロット
        ax_inset = ax_bottom.inset_axes([0.55, 0.55, 0.4, 0.4])
        ax_inset.plot(row_max)
        ax_inset.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        ax_inset.set_xlabel('y', fontsize=8)
        ax_inset.set_ylabel('max', fontsize=8)
        ax_inset.set_title('Row Max Projection', fontsize=8)
        ax_inset.grid(True, alpha=0.3)
        ax_inset.tick_params(labelsize=7)

        # 列方向の投影をプロット
        ax_inset2 = ax_bottom.inset_axes([0.55, 0.1, 0.4, 0.4])
        ax_inset2.plot(col_max)
        ax_inset2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        ax_inset2.set_xlabel('x', fontsize=8)
        ax_inset2.set_ylabel('max', fontsize=8)
        ax_inset2.set_title('Col Max Projection', fontsize=8)
        ax_inset2.grid(True, alpha=0.3)
        ax_inset2.tick_params(labelsize=7)

        ax_bottom.set_xlim(0, 1)
        ax_bottom.set_ylim(0, 1)
        ax_bottom.axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """複数のテストサンプルで可視化を実行"""

    # 設定とモデルをロード
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

    if not ckpt_files:
        print("チェックポイントファイルが見つかりません")
        return

    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f"モデルをロード: {latest_ckpt.name}")

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
    out_dir = Path(__file__).parent / "heatmap_lines_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ヒートマップと線の可視化")
    print("=" * 80)

    # 多様なサンプルを処理（sample/vertebraの組み合わせを変えて）
    processed = 0
    max_samples = 20  # より多くのサンプルを処理
    seen_combinations = set()  # (sample, vertebra) の組み合わせを追跡

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

            # 同じ(sample, vertebra)の組み合わせは最大3スライスまで
            combo = (sample, vertebra)
            combo_count = sum(1 for c in seen_combinations if c[0] == sample and c[1] == vertebra)
            if combo_count >= 3:
                continue
            seen_combinations.add((sample, vertebra, slice_idx))

            sample_info = {
                "sample": sample,
                "vertebra": vertebra,
                "slice_idx": slice_idx,
            }

            name = f"{sample}_{vertebra}_slice{slice_idx:03d}"

            # GT lines.jsonをロード
            lines_json_path = dataset_root / sample / vertebra / "lines.json"
            if not lines_json_path.exists():
                continue

            with open(lines_json_path) as f:
                lines_data = json.load(f)

            slice_data = lines_data.get(str(slice_idx), {})
            if not slice_data:
                continue

            # 各チャンネルから線を抽出
            pred_lines = {}
            for ch in range(4):
                line_name = f"line_{ch + 1}"
                gt_pts = slice_data.get(line_name)

                # GT線長を計算
                if gt_pts and len(gt_pts) >= 2:
                    gt_pts_array = np.array(gt_pts, dtype=np.float64)
                    d = gt_pts_array[1:] - gt_pts_array[:-1]
                    length_gt = float(np.sqrt((d**2).sum(axis=1)).sum())
                else:
                    length_gt = None

                # ヒートマップから線を抽出
                heatmap = pred_np[i, ch]
                pred_line = detect_line_moments(
                    heatmap,
                    length_px=length_gt,
                    extend_ratio=1.10,
                )
                pred_lines[line_name] = pred_line

            # 可視化
            save_path = out_dir / f"{name}_overlay.png"
            visualize_heatmap_with_lines(
                sample_info,
                pred_np[i],  # (4, H, W)
                pred_lines,
                slice_data,
                save_path,
            )

            print(f"[{processed+1}/{max_samples}] 保存: {save_path.name}")
            processed += 1

    print("\n" + "=" * 80)
    print(f"完了。{processed}個のサンプルを処理しました。")
    print(f"結果は {out_dir} に保存されました。")
    print("\n画像の確認方法:")
    print(f"  ls -lh {out_dir}")
    print(f"  画像ビューアで確認: eog {out_dir}/*.png &")
    print("=" * 80)


if __name__ == "__main__":
    main()
