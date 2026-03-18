"""座標系修正の簡易確認"""
import numpy as np
from pathlib import Path

from line_only.line_detection import detect_line_moments
from line_only.line_losses import extract_gt_line_params

def test_gt_extraction():
    """GT抽出が数学座標系になっているか確認"""
    print("=== GT抽出テスト ===\n")

    # 水平線（画像座標系で y=112 の水平線）
    polyline_horizontal = [[50, 112], [174, 112]]
    phi, rho = extract_gt_line_params(polyline_horizontal, image_size=224)

    print(f"水平線 (y=112): phi={np.degrees(phi):.1f}度, rho={rho:.3f}")
    print(f"  期待値: phi≈0度 or 180度")

    # 垂直線（画像座標系で x=112 の垂直線）
    polyline_vertical = [[112, 50], [112, 174]]
    phi, rho = extract_gt_line_params(polyline_vertical, image_size=224)

    print(f"\n垂直線 (x=112): phi={np.degrees(phi):.1f}度, rho={rho:.3f}")
    print(f"  期待値: phi≈90度")

    # 対角線（左上→右下）
    polyline_diagonal = [[50, 50], [174, 174]]
    phi, rho = extract_gt_line_params(polyline_diagonal, image_size=224)

    print(f"\n対角線（左上→右下）: phi={np.degrees(phi):.1f}度, rho={rho:.3f}")
    print(f"  期待値: phi≈135度 (法線が左上を向く)")


def test_detection_consistency():
    """detect_line_momentsが正しく動作するか確認"""
    print("\n\n=== detect_line_moments テスト ===\n")

    # 水平線のヒートマップ
    H, W = 224, 224
    hm = np.zeros((H, W), dtype=np.float64)
    hm[H//2-2:H//2+3, :] = 1.0

    result = detect_line_moments(hm, length_px=100)

    print(f"水平線ヒートマップ:")
    print(f"  重心（画像座標系）: {result['centroid']}")
    print(f"  角度（数学座標系）: {result['angle_deg']:.1f}度")
    print(f"  端点（画像座標系）: {result['endpoints']}")
    print(f"  期待値: 重心≈(112, 112), 角度≈0度")


if __name__ == "__main__":
    test_gt_extraction()
    test_detection_consistency()
