"""
モーメント法による直線抽出の単体テスト

Y軸座標系修正が正しく機能することを確認
"""

import torch
import numpy as np
import pytest
from line_losses import extract_pred_line_params_batch


def test_horizontal_line():
    """
    水平線（angle = 0° or 180°）のテスト

    法線ベクトルは垂直（90°または0°）になるべき
    """
    H, W = 224, 224
    heatmap = torch.zeros(1, 1, H, W)

    # 中央に水平線を作成
    heatmap[0, 0, H//2-2:H//2+3, :] = 1.0

    pred_params, confidence = extract_pred_line_params_batch(heatmap, image_size=224)
    phi = pred_params[0, 0, 0].item()  # 角度（ラジアン）
    phi_deg = np.degrees(phi)

    # 水平線の法線は垂直: 90°（π/2）または 0°
    # [0, π) の範囲では 90° になるはず
    assert abs(phi_deg - 90) < 5, f"水平線の法線は90°のはずが {phi_deg:.2f}°"

    print(f"✓ 水平線テスト成功: phi = {phi_deg:.2f}° (期待値: 90°)")


def test_vertical_line():
    """
    垂直線（angle = 90°）のテスト

    法線ベクトルは水平（0°または180°）になるべき
    """
    H, W = 224, 224
    heatmap = torch.zeros(1, 1, H, W)

    # 中央に垂直線を作成
    heatmap[0, 0, :, W//2-2:W//2+3] = 1.0

    pred_params, confidence = extract_pred_line_params_batch(heatmap, image_size=224)
    phi = pred_params[0, 0, 0].item()
    phi_deg = np.degrees(phi)

    # 垂直線の法線は水平: 0°（0）または 180°（π）
    # [0, π) の範囲では 0° または 180° に近いはず
    is_near_0 = abs(phi_deg) < 5
    is_near_180 = abs(phi_deg - 180) < 5

    assert is_near_0 or is_near_180, f"垂直線の法線は0°または180°のはずが {phi_deg:.2f}°"

    print(f"✓ 垂直線テスト成功: phi = {phi_deg:.2f}° (期待値: 0° or 180°)")


def test_diagonal_45deg():
    """
    45度斜め線のテスト

    Y軸修正後: i=j は Y=-X を意味し、-45度の直線
    -45度の線の法線は +45度
    """
    H, W = 224, 224
    heatmap = torch.zeros(1, 1, H, W)

    # i = j の対角線（Y軸反転後は -45度）
    for i in range(H):
        j = i
        if 0 <= j < W:
            for offset in range(-2, 3):
                if 0 <= j + offset < W:
                    heatmap[0, 0, i, j + offset] = 1.0

    pred_params, confidence = extract_pred_line_params_batch(heatmap, image_size=224)
    phi = pred_params[0, 0, 0].item()
    phi_deg = np.degrees(phi)

    # -45度線の法線は +45度（π/4）
    expected = 45
    assert abs(phi_deg - expected) < 15, f"対角線(i=j)の法線は{expected}°のはずが {phi_deg:.2f}°"

    print(f"✓ 対角線(i=j)テスト成功: phi = {phi_deg:.2f}° (期待値: {expected}°)")


def test_diagonal_minus45deg():
    """
    逆対角線のテスト

    Y軸修正後: i = H-1-j は Y=X を意味し、+45度の直線
    +45度の線の法線は +135度
    """
    H, W = 224, 224
    heatmap = torch.zeros(1, 1, H, W)

    # i = H-1-j の逆対角線（Y軸反転後は +45度）
    for i in range(H):
        j = (H - 1) - i
        if 0 <= j < W:
            for offset in range(-2, 3):
                if 0 <= j + offset < W:
                    heatmap[0, 0, i, j + offset] = 1.0

    pred_params, confidence = extract_pred_line_params_batch(heatmap, image_size=224)
    phi = pred_params[0, 0, 0].item()
    phi_deg = np.degrees(phi)

    # +45度線の法線は +135度（3π/4）
    expected = 135
    assert abs(phi_deg - expected) < 15, f"逆対角線の法線は{expected}°のはずが {phi_deg:.2f}°"

    print(f"✓ 逆対角線テスト成功: phi = {phi_deg:.2f}° (期待値: {expected}°)")


def test_confidence_scores():
    """
    信頼度スコアが適切に計算されることを確認

    明確な直線は高い信頼度、曖昧な形状は低い信頼度
    """
    H, W = 224, 224

    # 明確な直線
    heatmap_clear = torch.zeros(1, 1, H, W)
    heatmap_clear[0, 0, H//2-2:H//2+3, :] = 1.0

    # 曖昧な円形
    heatmap_blob = torch.zeros(1, 1, H, W)
    cy, cx = H // 2, W // 2
    radius = 20
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if dist < radius:
                heatmap_blob[0, 0, i, j] = 1.0

    _, conf_clear = extract_pred_line_params_batch(heatmap_clear, image_size=224)
    _, conf_blob = extract_pred_line_params_batch(heatmap_blob, image_size=224)

    clear_score = conf_clear[0, 0].item()
    blob_score = conf_blob[0, 0].item()

    # 直線の信頼度 > 円形の信頼度
    assert clear_score > blob_score, \
        f"明確な直線の信頼度({clear_score:.3f})が円形({blob_score:.3f})より低い"

    # 直線の信頼度は高いはず（> 0.8）
    assert clear_score > 0.8, f"明確な直線の信頼度が低すぎる: {clear_score:.3f}"

    print(f"✓ 信頼度テスト成功: 直線={clear_score:.3f}, 円形={blob_score:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("モーメント法直線抽出テスト（Y軸座標系修正の検証）")
    print("=" * 60)

    test_horizontal_line()
    test_vertical_line()
    test_diagonal_45deg()
    test_diagonal_minus45deg()
    test_confidence_scores()

    print("=" * 60)
    print("✅ 全てのテストに成功しました！")
    print("=" * 60)
