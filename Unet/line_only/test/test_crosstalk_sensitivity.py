"""
クロストークがモーメント法の角度推定に与える影響を定量評価

5-10%のクロストークで8-28度の誤差が出る原因を調査
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.utils.detection import detect_line_moments


def create_gaussian_line(H=224, W=224, angle_deg=45, center=(112, 112), sigma=2.5, length=100):
    """
    指定角度のガウス線ヒートマップを生成

    引数:
        H, W: 画像サイズ
        angle_deg: 線の角度（度）
        center: 中心座標 (cx, cy)
        sigma: ガウスの標準偏差
        length: 線の長さ（ピクセル）

    戻り値:
        heatmap: (H, W) ガウス分布
    """
    angle_rad = np.deg2rad(angle_deg)
    cx, cy = center

    # 線の方向ベクトル
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # 画像座標グリッド
    y_coords, x_coords = np.ogrid[0:H, 0:W]

    # 各ピクセルから線までの距離
    # 線: (x-cx)*dy - (y-cy)*dx = 0
    distances = np.abs((x_coords - cx) * dy - (y_coords - cy) * dx)

    # ガウス分布
    heatmap = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # 線の長さで切り取り
    t = (x_coords - cx) * dx + (y_coords - cy) * dy
    mask = np.abs(t) <= length / 2
    heatmap = heatmap * mask

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
    """クロストークが角度推定に与える影響をテスト"""

    H, W = 224, 224
    sigma = 2.5
    length = 100

    # テストケース: 様々な角度の組み合わせ
    test_cases = [
        {
            "name": "Parallel lines (all 45°)",
            "main_angle": 45,
            "other_angles": [45, 45, 45],
        },
        {
            "name": "Perpendicular (45° vs 135°)",
            "main_angle": 45,
            "other_angles": [135, 45, 135],
        },
        {
            "name": "Random angles",
            "main_angle": 30,
            "other_angles": [60, 120, 150],
        },
        {
            "name": "Near-parallel (45° vs 50°)",
            "main_angle": 45,
            "other_angles": [50, 48, 52],
        },
    ]

    crosstalk_levels = [0.0, 0.05, 0.10, 0.15, 0.20]

    results = []

    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test case: {case['name']}")
        print(f"Main angle: {case['main_angle']}°")
        print(f"Other angles: {case['other_angles']}")
        print(f"{'='*80}")

        # メインチャンネルのヒートマップ
        main_hm = create_gaussian_line(
            H, W, case['main_angle'], center=(112, 112), sigma=sigma, length=length
        )

        # 他のチャンネルのヒートマップ
        other_hms = [
            create_gaussian_line(
                H, W, angle, center=(112, 112), sigma=sigma, length=length
            )
            for angle in case['other_angles']
        ]

        case_results = []

        for crosstalk in crosstalk_levels:
            # クロストークを追加
            if crosstalk > 0:
                contaminated_hm = add_crosstalk(main_hm, other_hms, crosstalk)
            else:
                contaminated_hm = main_hm

            # モーメント法で角度推定
            line_info = detect_line_moments(contaminated_hm, length_px=length)

            if line_info is None:
                pred_angle = np.nan
                error = np.nan
            else:
                pred_angle = line_info["angle_deg"]
                # 角度誤差（180度周期を考慮）
                error = abs(pred_angle - case['main_angle'])
                error = min(error, 180 - error)

            case_results.append({
                "crosstalk": crosstalk,
                "pred_angle": pred_angle,
                "error": error,
            })

            print(f"Crosstalk {crosstalk*100:4.1f}%: "
                  f"Pred={pred_angle:6.2f}° Error={error:6.2f}°")

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
        ax.set_ylim(0, max(30, max(errors) * 1.1))

    plt.tight_layout()
    save_path = Path(__file__).parent / "crosstalk_sensitivity_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n\n結果を保存: {save_path}")
    plt.close()

    # サマリー
    print(f"\n{'='*80}")
    print("SUMMARY")
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
