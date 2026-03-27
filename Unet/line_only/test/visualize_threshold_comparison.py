"""
閾値あり/なしの比較可視化

参考画像のような形式で、改善/悪化したサンプルを可視化
- Heatmap (4チャンネル重ね)
- Pred Lines (閾値なし/ありを比較)
- GT Lines
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.src.data_utils import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
)
from line_only.src.model import TinyUNet
from line_only.utils.losses import extract_gt_line_params
import math


def extract_pred_line_params_with_threshold(
    heatmaps,
    image_size=224,
    threshold=0.0,
    min_mass=1e-6
):
    """
    閾値処理版のモーメント抽出

    Args:
        heatmaps: (B, C, H, W) 予測ヒートマップ
        image_size: 画像サイズ
        threshold: この値未満を0にする
        min_mass: 最小質量

    Returns:
        pred_params: (B, C, 2) (phi, rho)
        confidence: (B, C) 信頼度
    """
    B, C, H, W = heatmaps.shape
    device = heatmaps.device

    # 閾値処理
    if threshold > 0:
        heatmaps = torch.where(heatmaps >= threshold, heatmaps, torch.tensor(0.0, device=device))

    # 座標グリッド（数学座標系: Y上向き）
    y_grid = -(torch.arange(H, device=device, dtype=torch.float32) - H / 2.0)
    x_grid = torch.arange(W, device=device, dtype=torch.float32) - W / 2.0
    Y, X = torch.meshgrid(y_grid, x_grid, indexing="ij")

    D = math.sqrt(image_size**2 + image_size**2)
    output = torch.zeros(B, C, 2, device=device)
    confidence = torch.zeros(B, C, device=device)

    for b in range(B):
        for c in range(C):
            hm = heatmaps[b, c]
            M00 = hm.sum()

            if M00 < min_mass:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # 重心
            cx = (hm * X).sum() / M00
            cy = (hm * Y).sum() / M00

            # 2次中心モーメント
            dx = (X - cx).double()
            dy = (Y - cy).double()
            hm_d = hm.double()

            mu20 = (hm_d * dx * dx).sum() / M00
            mu02 = (hm_d * dy * dy).sum() / M00
            mu11 = (hm_d * dx * dy).sum() / M00

            # 正則化
            eps_reg = 1e-6
            mu20 = mu20 + eps_reg
            mu02 = mu02 + eps_reg

            # 固有値
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11 * mu11
            discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
            sqrt_disc = torch.sqrt(discriminant)

            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2

            # 信頼度
            if lambda1 > 1e-8:
                conf = 1.0 - lambda2 / lambda1
                confidence[b, c] = conf
            else:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # 方向ベクトル（解析的公式）
            theta = 0.5 * torch.atan2(2.0 * mu11, mu20 - mu02)
            dir_x = torch.cos(theta)
            dir_y = torch.sin(theta)

            # 法線ベクトル（90度反時計回り）
            nx = -dir_y
            ny = dir_x

            # φ を [0, π) に制限
            if ny < 0 or (ny == 0 and nx < 0):
                nx, ny = -nx, -ny

            # φ と ρ
            phi = torch.atan2(ny, nx)
            rho = nx * cx + ny * cy
            rho_norm = rho / D

            output[b, c, 0] = phi.float()
            output[b, c, 1] = rho_norm.float()

    return output, confidence


def draw_line_from_params(phi, rho, thresholded_heatmap, image_size=224, margin=10, color='red', linewidth=2, ax=None, label=None):
    """
    (φ, ρ)から直線を描画（閾値処理後のヒートマップの範囲+マージンに制限）

    Args:
        phi: 法線角度（ラジアン）
        rho: 正規化された距離
        thresholded_heatmap: (H, W) 閾値処理後のヒートマップ（>= 0.2の領域）
        image_size: 画像サイズ
        margin: マージン（ピクセル）
        color: 線の色
        linewidth: 線の太さ
        ax: matplotlib axis
        label: ラベル
    """
    if np.isnan(phi) or np.isnan(rho):
        return

    D = math.sqrt(image_size**2 + image_size**2)
    rho_px = rho * D

    # 法線ベクトル
    nx = np.cos(phi)
    ny = np.sin(phi)

    # 直線上の点（数学座標系）
    x0_math = nx * rho_px
    y0_math = ny * rho_px

    # 方向ベクトル（法線に垂直）
    dx = -ny
    dy = nx

    # 閾値処理後のヒートマップで値>0の領域を取得
    active_coords = np.argwhere(thresholded_heatmap > 0)  # (row, col) = (y, x)

    if len(active_coords) == 0:
        # 活性領域がない場合は描画しない
        return

    # 活性領域の範囲（画像座標系）
    y_min, x_min = active_coords.min(axis=0)
    y_max, x_max = active_coords.max(axis=0)

    # マージンを追加
    y_min = max(0, y_min - margin)
    y_max = min(image_size - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(image_size - 1, x_max + margin)

    # 長い線を作成
    length = image_size * 2
    x1_math = x0_math + dx * length / 2
    y1_math = y0_math + dy * length / 2
    x2_math = x0_math - dx * length / 2
    y2_math = y0_math - dy * length / 2

    # 画像座標系に変換（Y軸反転、原点を左上に）
    x1_img = x1_math + image_size / 2
    y1_img = -y1_math + image_size / 2
    x2_img = x2_math + image_size / 2
    y2_img = -y2_math + image_size / 2

    # 線分を活性領域でクリップ
    t_min, t_max = 0.0, 1.0

    # x方向のクリップ
    if abs(x2_img - x1_img) > 1e-6:
        t_x_min = (x_min - x1_img) / (x2_img - x1_img)
        t_x_max = (x_max - x1_img) / (x2_img - x1_img)
        if t_x_min > t_x_max:
            t_x_min, t_x_max = t_x_max, t_x_min
        t_min = max(t_min, t_x_min)
        t_max = min(t_max, t_x_max)

    # y方向のクリップ
    if abs(y2_img - y1_img) > 1e-6:
        t_y_min = (y_min - y1_img) / (y2_img - y1_img)
        t_y_max = (y_max - y1_img) / (y2_img - y1_img)
        if t_y_min > t_y_max:
            t_y_min, t_y_max = t_y_max, t_y_min
        t_min = max(t_min, t_y_min)
        t_max = min(t_max, t_y_max)

    # クリップ後の端点
    if t_min <= t_max:
        x1_clipped = x1_img + t_min * (x2_img - x1_img)
        y1_clipped = y1_img + t_min * (y2_img - y1_img)
        x2_clipped = x1_img + t_max * (x2_img - x1_img)
        y2_clipped = y1_img + t_max * (y2_img - y1_img)

        ax.plot([x1_clipped, x2_clipped], [y1_clipped, y2_clipped], color=color, linewidth=linewidth, label=label)


def visualize_sample_comparison(
    sample_info,
    heatmaps,
    pred_params_no,
    pred_params_with,
    gt_pts,
    background_img,
    save_path
):
    """
    1つのサンプルを詳細可視化（参考画像と同じ形式）

    Args:
        sample_info: dict サンプル情報
        heatmaps: (4, H, W) 4チャンネルのヒートマップ
        pred_params_no: (4, 2) 閾値なしの予測パラメータ（表示しない）
        pred_params_with: (4, 2) 閾値ありの予測パラメータ
        gt_pts: dict GTの点列 {ch: points}
        background_img: (H, W) 背景画像
        save_path: Path 保存先
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # カラーマップ（チャンネルごと）
    colors = ['cyan', 'yellow', 'magenta', 'green']

    # 背景画像を正規化（0-1の範囲に）
    if background_img.max() > 1:
        bg_normalized = background_img.astype(np.float32) / 255.0
    else:
        bg_normalized = background_img.astype(np.float32)

    # 閾値処理（>= 0.2）
    heatmaps_thresholded = np.where(heatmaps >= 0.2, heatmaps, 0.0)

    # 1. Heatmap（入力画像の上に閾値処理後の4チャンネル重ね）
    ax = axes[0]
    ax.imshow(bg_normalized, cmap='gray', vmin=0, vmax=1)
    for ch in range(4):
        hm_thresh = heatmaps_thresholded[ch]
        # 閾値処理後のヒートマップを色付け
        hm_colored = np.zeros((224, 224, 4))
        color_rgb = plt.get_cmap('tab10')(ch)[:3]
        for i in range(3):
            hm_colored[:, :, i] = color_rgb[i]
        hm_colored[:, :, 3] = hm_thresh * 0.7  # Alpha
        ax.imshow(hm_colored, origin='upper')
    ax.set_title('Heatmap', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 2. Pred Lines（閾値≥0.2の範囲のみ）
    ax = axes[1]
    ax.imshow(bg_normalized, cmap='gray', vmin=0, vmax=1)
    for ch in range(4):
        # 閾値あり（>=0.2）のみ表示
        phi_with = pred_params_with[ch, 0]
        rho_with = pred_params_with[ch, 1]
        # 対応する閾値処理後のヒートマップ
        hm_thresh_ch = heatmaps_thresholded[ch]
        draw_line_from_params(phi_with, rho_with, hm_thresh_ch, 224, margin=10, color=colors[ch], linewidth=3, ax=ax)
    ax.set_title('Pred Lines', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 3. GT Lines
    ax = axes[2]
    ax.imshow(bg_normalized, cmap='gray', vmin=0, vmax=1)
    for ch, pts in gt_pts.items():
        if pts and len(pts) >= 2:
            pts_array = np.array(pts)
            ax.plot(pts_array[:, 0], pts_array[:, 1], color=colors[ch], linewidth=3)
    ax.set_title('GT Lines', fontsize=14, fontweight='bold')
    ax.axis('off')

    # タイトル
    fig.suptitle(
        f"{sample_info['sample']}_{sample_info['vertebra']}_slice{sample_info['slice_idx']:03d}_ch{sample_info['channel']}\n"
        f"Error: {sample_info['error_no']:.1f}° → {sample_info['error_with']:.1f}° (Δ={sample_info['improvement']:.1f}°)",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def collect_samples_for_visualization(model, test_loader, dataset_root, device):
    """閾値あり/なしで比較するサンプルを収集"""

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

            # 背景画像（データローダーから取得、最初のチャンネル）
            background_img = x[i, 0].cpu().numpy()  # (H, W)

            # 各チャンネル
            for ch in range(4):
                line_name = f"line_{ch + 1}"
                gt_pts = slice_data.get(line_name)

                if gt_pts is None or len(gt_pts) < 2:
                    continue

                # GT角度
                gt_phi, _ = extract_gt_line_params(gt_pts, image_size=224)
                if np.isnan(gt_phi):
                    continue

                gt_phi_deg = np.degrees(gt_phi) % 180

                # 予測（閾値なし）
                pred_params_no, conf_no = extract_pred_line_params_with_threshold(
                    pred[i:i+1, ch:ch+1], image_size=224, threshold=0.0
                )
                pred_phi_no = pred_params_no[0, 0, 0].item()
                conf_no_val = conf_no[0, 0].item()

                if not np.isnan(pred_phi_no):
                    pred_phi_no_deg = np.degrees(pred_phi_no) % 180
                    error_no = abs(pred_phi_no_deg - gt_phi_deg)
                    error_no = min(error_no, 180 - error_no)
                else:
                    error_no = np.nan

                # 予測（閾値 >= 0.2）
                pred_params_with, conf_with = extract_pred_line_params_with_threshold(
                    pred[i:i+1, ch:ch+1], image_size=224, threshold=0.2
                )
                pred_phi_with = pred_params_with[0, 0, 0].item()
                conf_with_val = conf_with[0, 0].item()

                if not np.isnan(pred_phi_with):
                    pred_phi_with_deg = np.degrees(pred_phi_with) % 180
                    error_with = abs(pred_phi_with_deg - gt_phi_deg)
                    error_with = min(error_with, 180 - error_with)
                else:
                    error_with = np.nan

                # 結果を記録
                if not np.isnan(error_no) and not np.isnan(error_with):
                    # 全4チャンネルのGT点列を取得
                    gt_pts_all = {}
                    for c in range(4):
                        ln = f"line_{c + 1}"
                        pts = slice_data.get(ln)
                        if pts and len(pts) >= 2:
                            gt_pts_all[c] = pts

                    # 全4チャンネルの予測パラメータ
                    pred_params_no_all, _ = extract_pred_line_params_with_threshold(
                        pred[i:i+1], image_size=224, threshold=0.0
                    )
                    pred_params_with_all, _ = extract_pred_line_params_with_threshold(
                        pred[i:i+1], image_size=224, threshold=0.2
                    )

                    all_samples.append({
                        'sample': sample_name,
                        'vertebra': vertebra,
                        'slice_idx': slice_idx,
                        'channel': ch + 1,
                        'error_no': error_no,
                        'error_with': error_with,
                        'improvement': error_no - error_with,
                        'conf_no': conf_no_val,
                        'conf_with': conf_with_val,
                        'heatmaps': pred[i].cpu().numpy(),
                        'pred_params_no': pred_params_no_all[0].cpu().numpy(),
                        'pred_params_with': pred_params_with_all[0].cpu().numpy(),
                        'gt_pts': gt_pts_all,
                        'background_img': background_img,
                    })

    return all_samples


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

    # サンプル収集
    samples = collect_samples_for_visualization(model, test_loader, dataset_root, device)
    print(f"収集完了: {len(samples)}個のサンプル")

    # 改善度でソート
    samples.sort(key=lambda x: x['improvement'], reverse=True)

    out_dir = Path(__file__).parent / "threshold_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 最も改善した5つを可視化
    print("\n最も改善したサンプルを可視化中...")
    for idx, sample in enumerate(samples[:5]):
        save_path = out_dir / f"improved_{idx+1}_delta{sample['improvement']:.1f}.png"
        visualize_sample_comparison(
            sample,
            sample['heatmaps'],
            sample['pred_params_no'],
            sample['pred_params_with'],
            sample['gt_pts'],
            sample['background_img'],
            save_path
        )
        print(f"  {idx+1}. {sample['sample']}_{sample['vertebra']}_slice{sample['slice_idx']:03d}_ch{sample['channel']}")
        print(f"     {sample['error_no']:.1f}° → {sample['error_with']:.1f}° (改善: {sample['improvement']:.1f}°)")
        print(f"     保存: {save_path.name}")

    # 最も悪化した5つを可視化
    samples.sort(key=lambda x: x['improvement'])
    print("\n最も悪化したサンプルを可視化中...")
    for idx, sample in enumerate(samples[:5]):
        if sample['improvement'] >= 0:
            print("（悪化したサンプルなし）")
            break
        save_path = out_dir / f"degraded_{idx+1}_delta{-sample['improvement']:.1f}.png"
        visualize_sample_comparison(
            sample,
            sample['heatmaps'],
            sample['pred_params_no'],
            sample['pred_params_with'],
            sample['gt_pts'],
            sample['background_img'],
            save_path
        )
        print(f"  {idx+1}. {sample['sample']}_{sample['vertebra']}_slice{sample['slice_idx']:03d}_ch{sample['channel']}")
        print(f"     {sample['error_no']:.1f}° → {sample['error_with']:.1f}° (悪化: {-sample['improvement']:.1f}°)")
        print(f"     保存: {save_path.name}")

    print(f"\n全ての画像を保存: {out_dir}")


if __name__ == "__main__":
    main()
