"""
クロストークがモーメント法の角度推定に与える影響を定量評価（修正版）

extract_pred_line_params_batch（訓練・評価で使う関数）を使用
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.line_losses import extract_pred_line_params_batch, extract_gt_line_params


def create_gaussian_line_hesse(H=224, W=224, phi_deg=45, rho_px=0, sigma=2.5):
    """
    Hesse normal form (φ, ρ) でガウス線ヒートマップを生成

    引数:
        H, W: 画像サイズ
        phi_deg: 法線角度（度）[0, 180)
        rho_px: 原点からの距離（ピクセル、中心原点）
        sigma: ガウスの標準偏差

    戻り値:
        heatmap: (H, W) ガウス分布
    """
    phi_rad = np.deg2rad(phi_deg)

    # 法線ベクトル
    nx = np.cos(phi_rad)
    ny = np.sin(phi_rad)

    # 中心原点の座標系
    center = H / 2.0
    y_grid = -(np.arange(H, dtype=np.float64) - center)  # Y上向き
    x_grid = np.arange(W, dtype=np.float64) - center

    Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')

    # 各ピクセルから直線までの距離
    # 直線の方程式: nx*x + ny*y - rho = 0
    # 距離 = |nx*x + ny*y - rho|
    distances = np.abs(nx * X + ny * Y - rho_px)

    # ガウス分布
    heatmap = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # 正規化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def add_crosstalk(main_heatmap, other_heatmaps, crosstalk_ratio=0.1):
    """
    他のチャンネルのクロストークを追加

    引数:
        main_heatmap: メインチャンネルのヒートマップ
        other_heatmaps: 他のチャンネルのヒートマップリスト
        crosstalk_ratio: クロストーク比率（0.1 = 10%）

    戻り値:
        汚染されたヒートマップ
    """
    contaminated = main_heatmap.copy()

    for other in other_heatmaps:
        contaminated += crosstalk_ratio * other

    return contaminated


def test_crosstalk_impact():
    """クロストークが角度推定に与える影響をテスト（正しい実装で）"""

    H, W = 224, 224
    sigma = 2.5

    # テストケース: 様々な法線角の組み合わせ
    test_cases = [
        {
            "name": "Parallel lines (all phi=45°)",
            "main_phi": 45,
            "main_rho": 0,
            "others": [(45, 20), (45, -20), (45, 40)],
        },
        {
            "name": "Perpendicular (phi=45° vs 135°)",
            "main_phi": 45,
            "main_rho": 0,
            "others": [(135, 20), (45, -20), (135, -30)],
        },
        {
            "name": "Random angles",
            "main_phi": 30,
            "main_rho": 10,
            "others": [(60, 0), (120, -10), (150, 20)],
        },
        {
            "name": "Near-parallel (phi=45° vs 50°)",
            "main_phi": 45,
            "main_rho": 0,
            "others": [(50, 15), (48, -15), (52, 25)],
        },
    ]

    crosstalk_levels = [0.0, 0.05, 0.10, 0.15, 0.20]

    results = []

    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test case: {case['name']}")
        print(f"Main: phi={case['main_phi']}°, rho={case['main_rho']}px")
        print(f"Others: {case['others']}")
        print(f"{'='*80}")

        # メインチャンネルのヒートマップ
        main_hm = create_gaussian_line_hesse(
            H, W, case['main_phi'], case['main_rho'], sigma
        )

        # 他のチャンネルのヒートマップ
        other_hms = [
            create_gaussian_line_hesse(H, W, phi, rho, sigma)
            for phi, rho in case['others']
        ]

        case_results = []

        for crosstalk in crosstalk_levels:
            # クロストークを追加
            if crosstalk > 0:
                contaminated_hm = add_crosstalk(main_hm, other_hms, crosstalk)
            else:
                contaminated_hm = main_hm

            # Torch形式に変換 (1, 1, H, W)
            hm_tensor = torch.from_numpy(contaminated_hm).float().unsqueeze(0).unsqueeze(0)

            # extract_pred_line_params_batch で角度推定
            pred_params, confidence = extract_pred_line_params_batch(hm_tensor, image_size=H)

            pred_phi_rad = pred_params[0, 0, 0].item()
            pred_rho_norm = pred_params[0, 0, 1].item()

            # ラジアン → 度
            pred_phi_deg = np.degrees(pred_phi_rad) % 180

            # 角度誤差（180度周期を考慮）
            error = abs(pred_phi_deg - case['main_phi'])
            error = min(error, 180 - error)

            case_results.append({
                "crosstalk": crosstalk,
                "pred_phi": pred_phi_deg,
                "error": error,
                "confidence": confidence[0, 0].item(),
            })

            print(f"Crosstalk {crosstalk*100:4.1f}%: "
                  f"Pred φ={pred_phi_deg:6.2f}° Error={error:6.2f}° "
                  f"Conf={confidence[0,0].item():.3f}")

        results.append({
            "case": case,
            "results": case_results,
        })

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]
        case = result["case"]
        data = result["results"]

        crosstalk_pct = [d["crosstalk"] * 100 for d in data]
        errors = [d["error"] for d in data]

        ax.plot(crosstalk_pct, errors, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Crosstalk (%)', fontsize=12)
        ax.set_ylabel('Angle Error (degrees)', fontsize=12)
        ax.set_title(case["name"], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(15, max(errors) * 1.2))

    plt.tight_layout()
    save_path = Path(__file__).parent / "crosstalk_sensitivity_corrected.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n\n結果を保存: {save_path}")
    plt.close()

    # サマリー
    print(f"\n{'='*80}")
    print("SUMMARY (CORRECTED TEST)")
    print(f"{'='*80}")
    print(f"Question: Does 5-10% crosstalk justify 8-28° angle error?")
    print()

    for result in results:
        case = result["case"]
        data = result["results"]

        # 5-10%の範囲のエラー
        errors_5_10 = [d["error"] for d in data if 0.05 <= d["crosstalk"] <= 0.10]

        if errors_5_10:
            min_err = min(errors_5_10)
            max_err = max(errors_5_10)
            print(f"{case['name']:40s}: {min_err:5.2f}° - {max_err:5.2f}°")


if __name__ == "__main__":
    test_crosstalk_impact()
