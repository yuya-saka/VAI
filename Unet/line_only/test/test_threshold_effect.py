"""
閾値処理（>= 0.2）の効果をテスト

モーメント計算前にヒートマップを閾値処理することで
角度誤差が改善するか検証
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


def compare_with_and_without_threshold(model, test_loader, dataset_root, device):
    """閾値あり vs なしで比較"""

    results = {
        'no_threshold': {'errors': [], 'confidences': []},
        'with_threshold': {'errors': [], 'confidences': []},
    }

    sample_details = []

    print("テストデータで比較中...")

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
                if not np.isnan(error_no):
                    results['no_threshold']['errors'].append(error_no)
                    results['no_threshold']['confidences'].append(conf_no_val)

                if not np.isnan(error_with):
                    results['with_threshold']['errors'].append(error_with)
                    results['with_threshold']['confidences'].append(conf_with_val)

                # サンプル詳細
                if not np.isnan(error_no) and not np.isnan(error_with):
                    sample_details.append({
                        'sample': f"{sample_name}_{vertebra}_slice{slice_idx:03d}_ch{ch+1}",
                        'error_no': error_no,
                        'error_with': error_with,
                        'improvement': error_no - error_with,
                        'conf_no': conf_no_val,
                        'conf_with': conf_with_val,
                    })

    return results, sample_details


def analyze_results(results, sample_details):
    """結果を分析して表示"""

    print("\n" + "="*80)
    print("閾値処理の効果")
    print("="*80)

    errors_no = np.array(results['no_threshold']['errors'])
    errors_with = np.array(results['with_threshold']['errors'])
    conf_no = np.array(results['no_threshold']['confidences'])
    conf_with = np.array(results['with_threshold']['confidences'])

    print(f"\n【閾値なし】")
    print(f"  サンプル数: {len(errors_no)}")
    print(f"  平均誤差:   {errors_no.mean():.2f}°")
    print(f"  中央値誤差: {np.median(errors_no):.2f}°")
    print(f"  最大誤差:   {errors_no.max():.2f}°")
    print(f"  平均Confidence: {conf_no.mean():.3f}")

    print(f"\n【閾値 >= 0.2】")
    print(f"  サンプル数: {len(errors_with)}")
    print(f"  平均誤差:   {errors_with.mean():.2f}°")
    print(f"  中央値誤差: {np.median(errors_with):.2f}°")
    print(f"  最大誤差:   {errors_with.max():.2f}°")
    print(f"  平均Confidence: {conf_with.mean():.3f}")

    print(f"\n【改善度】")
    print(f"  平均誤差の改善:   {errors_no.mean() - errors_with.mean():.2f}°")
    print(f"  中央値誤差の改善: {np.median(errors_no) - np.median(errors_with):.2f}°")
    print(f"  Confidenceの改善: {conf_with.mean() - conf_no.mean():.3f}")

    # 大きく改善したサンプル
    sample_details.sort(key=lambda x: x['improvement'], reverse=True)

    print(f"\n【最も改善したサンプル Top 10】")
    for idx, detail in enumerate(sample_details[:10]):
        print(f"{idx+1}. {detail['sample']}")
        print(f"   閾値なし: {detail['error_no']:5.1f}° (Conf={detail['conf_no']:.3f})")
        print(f"   閾値あり: {detail['error_with']:5.1f}° (Conf={detail['conf_with']:.3f})")
        print(f"   改善:     {detail['improvement']:5.1f}°")

    # 悪化したサンプル
    sample_details.sort(key=lambda x: x['improvement'])

    print(f"\n【悪化したサンプル Top 5】")
    for idx, detail in enumerate(sample_details[:5]):
        if detail['improvement'] >= 0:
            print("（悪化したサンプルなし）")
            break
        print(f"{idx+1}. {detail['sample']}")
        print(f"   閾値なし: {detail['error_no']:5.1f}° (Conf={detail['conf_no']:.3f})")
        print(f"   閾値あり: {detail['error_with']:5.1f}° (Conf={detail['conf_with']:.3f})")
        print(f"   悪化:     {-detail['improvement']:5.1f}°")

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 誤差の比較（ヒストグラム）
    ax = axes[0, 0]
    bins = np.linspace(0, 90, 30)
    ax.hist(errors_no, bins=bins, alpha=0.5, label='閾値なし', color='red')
    ax.hist(errors_with, bins=bins, alpha=0.5, label='閾値 >= 0.2', color='blue')
    ax.set_xlabel('角度誤差 (degrees)', fontsize=12)
    ax.set_ylabel('頻度', fontsize=12)
    ax.set_title('角度誤差の分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 散布図（閾値なし vs あり）
    ax = axes[0, 1]
    improvements = [d['improvement'] for d in sample_details]
    ax.scatter(errors_no, errors_with, alpha=0.3, s=20)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5, label='同じ')
    ax.set_xlabel('閾値なし (degrees)', fontsize=12)
    ax.set_ylabel('閾値 >= 0.2 (degrees)', fontsize=12)
    ax.set_title('誤差の比較', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Confidenceの比較
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 30)
    ax.hist(conf_no, bins=bins, alpha=0.5, label='閾値なし', color='red')
    ax.hist(conf_with, bins=bins, alpha=0.5, label='閾値 >= 0.2', color='blue')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('頻度', fontsize=12)
    ax.set_title('Confidenceの分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 改善度のヒストグラム
    ax = axes[1, 1]
    ax.hist(improvements, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='改善なし')
    ax.set_xlabel('改善度 (degrees)', fontsize=12)
    ax.set_ylabel('頻度', fontsize=12)
    ax.set_title('改善度の分布（正=改善、負=悪化）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / "threshold_effect_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {save_path}")
    plt.close()


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

    results, sample_details = compare_with_and_without_threshold(
        model, test_loader, dataset_root, device
    )

    analyze_results(results, sample_details)


if __name__ == "__main__":
    main()
