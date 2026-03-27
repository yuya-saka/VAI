"""
角度計算のデバッグ

低confidenceでも線状に見えるのに、なぜ角度誤差が大きいのか？
実際の数値を確認
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.src.data_utils import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
)
from line_only.src.model import TinyUNet
from line_only.utils.losses import extract_pred_line_params_batch, extract_gt_line_params
from line_only.utils.detection import detect_line_moments


def debug_specific_sample(model, test_loader, dataset_root, device, target_sample):
    """
    特定のサンプルを詳細デバッグ

    Args:
        target_sample: dict with keys 'sample', 'vertebra', 'slice_idx', 'channel'
    """

    for batch in test_loader:
        x = batch["image"].to(device).float()
        B = x.shape[0]

        for i in range(B):
            sample_name = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])

            if (sample_name != target_sample['sample'] or
                vertebra != target_sample['vertebra'] or
                slice_idx != target_sample['slice_idx']):
                continue

            # 見つけた！
            print("="*80)
            print(f"デバッグ対象: {sample_name}_{vertebra}_slice{slice_idx:03d}_ch{target_sample['channel']}")
            print("="*80)

            # 予測
            with torch.no_grad():
                pred = torch.sigmoid(model(x))

            ch = target_sample['channel'] - 1  # 0-indexed
            heatmap = pred[i, ch].cpu().numpy()

            # GT lines.json
            lines_json_path = dataset_root / sample_name / vertebra / "lines.json"
            with open(lines_json_path) as f:
                lines_data = json.load(f)

            slice_data = lines_data.get(str(slice_idx), {})
            line_name = f"line_{ch + 1}"
            gt_pts = slice_data.get(line_name)

            print(f"\n【GT情報】")
            print(f"GT点列: {gt_pts}")

            # GT角度（2つの方法で計算）
            gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size=224)
            gt_phi_deg = np.degrees(gt_phi) % 180
            print(f"GT φ (法線角): {gt_phi_deg:.2f}°")

            # GTの方向角（折れ線の端点から）
            if gt_pts and len(gt_pts) >= 2:
                p1 = np.array(gt_pts[0])
                p2 = np.array(gt_pts[-1])
                direction = p2 - p1
                gt_direction_angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180
                print(f"GT 方向角（端点から）: {gt_direction_angle:.2f}°")

            print(f"\n【予測情報 - extract_pred_line_params_batch】")

            # 予測角度（訓練で使う方法）
            pred_params, confidence = extract_pred_line_params_batch(
                torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0),
                image_size=224
            )
            pred_phi = pred_params[0, 0, 0].item()
            pred_rho = pred_params[0, 0, 1].item()
            conf = confidence[0, 0].item()
            pred_phi_deg = np.degrees(pred_phi) % 180

            print(f"予測 φ (法線角): {pred_phi_deg:.2f}°")
            print(f"予測 ρ (正規化): {pred_rho:.4f}")
            print(f"Confidence: {conf:.3f}")

            angle_error = abs(pred_phi_deg - gt_phi_deg)
            angle_error = min(angle_error, 180 - angle_error)
            print(f"角度誤差: {angle_error:.2f}°")

            print(f"\n【予測情報 - detect_line_moments（可視化用）】")

            # 可視化用の方法
            line_info = detect_line_moments(heatmap, length_px=100)
            if line_info:
                direction_angle = line_info["angle_deg"]
                print(f"方向角: {direction_angle:.2f}°")
                print(f"法線角（+90度）: {(direction_angle + 90) % 180:.2f}°")

            print(f"\n【モーメント詳細】")

            # モーメントを手動計算
            H, W = heatmap.shape
            y_grid = -(np.arange(H) - H/2.0)  # Y上向き
            x_grid = np.arange(W) - W/2.0
            Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')

            M00 = heatmap.sum()
            cx = (heatmap * X).sum() / M00
            cy = (heatmap * Y).sum() / M00

            dx = X - cx
            dy = Y - cy

            mu20 = (heatmap * dx**2).sum() / M00
            mu02 = (heatmap * dy**2).sum() / M00
            mu11 = (heatmap * dx * dy).sum() / M00

            print(f"質量 M00: {M00:.2f}")
            print(f"重心 (cx, cy): ({cx:.2f}, {cy:.2f})")
            print(f"2次モーメント mu20: {mu20:.4f}")
            print(f"2次モーメント mu02: {mu02:.4f}")
            print(f"2次モーメント mu11: {mu11:.4f}")

            # 固有値
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11**2
            disc = trace**2 - 4*det
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                lambda1 = (trace + sqrt_disc) / 2
                lambda2 = (trace - sqrt_disc) / 2
                print(f"固有値 λ1 (大): {lambda1:.4f}")
                print(f"固有値 λ2 (小): {lambda2:.4f}")
                print(f"固有値比 λ1/λ2: {lambda1/lambda2:.2f}")
                print(f"Confidence (1-λ2/λ1): {1-lambda2/lambda1:.3f}")

            # 解析的公式による角度
            theta_analytical = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
            print(f"\n解析的公式による方向角: {np.degrees(theta_analytical):.2f}°")

            # 固有ベクトル法による角度
            if abs(mu11) > 1e-8:
                dir_x = mu11
                dir_y = lambda1 - mu20
            else:
                if mu20 > mu02:
                    dir_x, dir_y = 1.0, 0.0
                else:
                    dir_x, dir_y = 0.0, 1.0

            norm = np.sqrt(dir_x**2 + dir_y**2)
            dir_x, dir_y = dir_x/norm, dir_y/norm
            print(f"固有ベクトル法による方向: ({dir_x:.4f}, {dir_y:.4f})")
            print(f"固有ベクトル法による方向角: {np.degrees(np.arctan2(dir_y, dir_x)):.2f}°")

            # 法線ベクトル
            nx = -dir_y
            ny = dir_x
            if ny < 0 or (ny == 0 and nx < 0):
                nx, ny = -nx, -ny

            phi_manual = np.arctan2(ny, nx)
            print(f"法線ベクトル: ({nx:.4f}, {ny:.4f})")
            print(f"法線角 φ: {np.degrees(phi_manual):.2f}°")

            print(f"\n【目視での推定】")
            print("ヒートマップを見て:")
            print("- 主軸方向は？")
            print("- 法線方向は？")

            # 可視化
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. ヒートマップ
            ax = axes[0, 0]
            im = ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1, origin='lower')
            ax.set_title(f'Heatmap (Conf={conf:.3f})', fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y (↑上向き)')
            plt.colorbar(im, ax=ax)

            # 重心と方向ベクトルを描画
            cx_img = cx + W/2
            cy_img = -cy + H/2
            ax.plot(cx_img, cy_img, 'rx', markersize=15, markeredgewidth=3, label='重心')

            # 方向ベクトル（赤）
            scale = 50
            ax.arrow(cx_img, cy_img, dir_x*scale, -dir_y*scale,
                    color='red', width=2, head_width=10, label='方向ベクトル')

            # 法線ベクトル（青）
            ax.arrow(cx_img, cy_img, nx*scale, -ny*scale,
                    color='blue', width=2, head_width=10, label='法線ベクトル')

            ax.legend()

            # 2. 0.2閾値処理後
            ax = axes[0, 1]
            thresholded = np.where(heatmap >= 0.2, heatmap, 0)
            im = ax.imshow(thresholded, cmap='hot', vmin=0, vmax=1, origin='lower')
            ax.set_title('Threshold >= 0.2')
            plt.colorbar(im, ax=ax)

            # 3. GT線を描画
            ax = axes[0, 2]
            ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1, alpha=0.5, origin='lower')
            if gt_pts and len(gt_pts) >= 2:
                pts = np.array(gt_pts)
                ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=3, label='GT線')
            ax.set_title('GT線 vs ヒートマップ')
            ax.legend()

            # 4-6. プロジェクション
            ax = axes[1, 0]
            row_max = heatmap.max(axis=1)
            ax.plot(row_max, np.arange(H))
            ax.axvline(0.2, color='r', linestyle='--')
            ax.set_xlabel('max value')
            ax.set_ylabel('y (row)')
            ax.set_title('Row Max Projection')
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            col_max = heatmap.max(axis=0)
            ax.plot(col_max)
            ax.axhline(0.2, color='r', linestyle='--')
            ax.set_xlabel('x (col)')
            ax.set_ylabel('max value')
            ax.set_title('Col Max Projection')
            ax.grid(True, alpha=0.3)

            # 統計
            ax = axes[1, 2]
            ax.axis('off')
            stats_text = f"""
GT φ: {gt_phi_deg:.2f}°
予測 φ: {pred_phi_deg:.2f}°
誤差: {angle_error:.2f}°

Confidence: {conf:.3f}
λ1/λ2: {lambda1/lambda2:.2f}

mu20: {mu20:.4f}
mu02: {mu02:.4f}
mu11: {mu11:.4f}
            """
            ax.text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
                   verticalalignment='center')

            plt.tight_layout()
            save_path = Path(__file__).parent / f"debug_{sample_name}_{vertebra}_slice{slice_idx:03d}_ch{ch+1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可視化を保存: {save_path}")
            plt.close()

            return  # 見つけたので終了


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

    # 低confidenceの代表例をデバッグ
    target = {
        'sample': 'sample7',
        'vertebra': 'C2',
        'slice_idx': 57,
        'channel': 3,  # line_3
    }

    print("対象サンプルを検索中...")
    debug_specific_sample(model, test_loader, dataset_root, device, target)


if __name__ == "__main__":
    main()
