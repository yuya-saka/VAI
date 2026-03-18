"""
Y軸座標系修正の効果を視覚的に確認

修正前後の直線抽出結果を比較画像として保存
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from line_losses import extract_pred_line_params_batch


def create_test_heatmap(angle_deg, H=224, W=224):
    """指定角度の法線を持つ直線のヒートマップを生成"""
    heatmap = torch.zeros(1, 1, H, W)

    angle_rad = np.radians(angle_deg)
    nx = np.cos(angle_rad)
    ny = np.sin(angle_rad)

    for i in range(H):
        for j in range(W):
            y = -(i - H/2)
            x = j - W/2
            dist = abs(nx * x + ny * y)
            if dist < 3:
                heatmap[0, 0, i, j] = 1.0

    return heatmap


def draw_line_from_params(ax, phi, rho, H=224, W=224, color='red', label='', linewidth=2):
    """
    (φ, ρ) パラメータから直線を描画

    Hesse normal form: cos(φ)*x + sin(φ)*y = ρ*D
    """
    D = np.sqrt(H**2 + W**2)

    # 法線ベクトル
    nx = np.cos(phi)
    ny = np.sin(phi)

    # 原点から直線への最近点
    rho_px = rho * D
    px = nx * rho_px
    py = ny * rho_px

    # 直線の方向ベクトル（法線を90度回転）
    dx = -ny
    dy = nx

    # 画像範囲で直線を描画
    t_range = 300  # 十分長く
    x1 = px - t_range * dx
    y1 = py - t_range * dy
    x2 = px + t_range * dx
    y2 = py + t_range * dy

    # 座標を画像座標系に変換（Y軸を反転）
    # 数学座標系 (x, y) → 画像座標系 (col, row)
    col1 = x1 + W/2
    row1 = -y1 + H/2
    col2 = x2 + W/2
    row2 = -y2 + H/2

    ax.plot([col1, col2], [row1, row2], color=color, linewidth=linewidth,
            label=label, alpha=0.8)


def visualize_comparison(gt_angle, output_path):
    """
    修正前後の比較を可視化

    注意: 修正前の動作は再現できないため、理論値との比較のみ
    """
    H, W = 224, 224

    # ヒートマップ生成
    heatmap = create_test_heatmap(gt_angle, H, W)

    # 修正後のコードで抽出
    pred_params, confidence = extract_pred_line_params_batch(heatmap)
    pred_phi = pred_params[0, 0, 0].item()
    pred_rho = pred_params[0, 0, 1].item()
    pred_angle_deg = np.degrees(pred_phi)

    # 図を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: ヒートマップ + GT線 + 予測線
    ax = axes[0]
    ax.imshow(heatmap[0, 0].numpy(), cmap='hot', origin='upper')
    ax.set_title(f'Fixed Code Result\nGT: {gt_angle:.1f}°, Pred: {pred_angle_deg:.1f}°',
                 fontsize=12)

    # GT線を描画（緑）
    gt_phi_rad = np.radians(gt_angle)
    draw_line_from_params(ax, gt_phi_rad, 0.0, H, W, color='lime',
                          label=f'GT: {gt_angle:.1f}°', linewidth=3)

    # 予測線を描画（赤）
    draw_line_from_params(ax, pred_phi, pred_rho, H, W, color='red',
                          label=f'Pred: {pred_angle_deg:.2f}°', linewidth=2)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.grid(True, alpha=0.3)

    # 右: 誤差情報
    ax = axes[1]
    ax.axis('off')

    error_deg = abs(pred_angle_deg - gt_angle)

    info_text = f"""
Angle Extraction Test
{'='*40}

Ground Truth:
  Angle (φ):      {gt_angle:.2f}°
  Rho (ρ):        0.00 (center)

Predicted (Fixed Code):
  Angle (φ):      {pred_angle_deg:.2f}°
  Rho (ρ):        {pred_rho:.4f}

Error:
  Angle Error:    {error_deg:.2f}°

Confidence:      {confidence[0, 0].item():.3f}

Status:
"""

    if error_deg < 1:
        status = "  ✅ Excellent (< 1°)"
        color = 'green'
    elif error_deg < 5:
        status = "  ✅ Good (< 5°)"
        color = 'blue'
    else:
        status = "  ⚠️  Needs improvement"
        color = 'orange'

    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace')
    ax.text(0.1, 0.28, status, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace',
            color=color, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """問題のあったケースを可視化"""
    output_dir = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/line_only/test/output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Y軸座標系修正の効果を可視化")
    print("=" * 60)

    # 問題のあった角度をテスト
    test_cases = [
        {"angle": 20.0, "name": "line1_20deg"},
        {"angle": 125.31, "name": "line3_125deg"},
        {"angle": 93.85, "name": "line2_94deg"},
        {"angle": 45.0, "name": "diagonal_45deg"},
        {"angle": 90.0, "name": "vertical_90deg"},
        {"angle": 0.0, "name": "horizontal_0deg"},
    ]

    for case in test_cases:
        angle = case["angle"]
        name = case["name"]
        output_path = output_dir / f"{name}_fix_result.png"

        print(f"\nGenerating: {name} (φ = {angle:.2f}°)")
        visualize_comparison(angle, output_path)

    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
