"""座標系による角度の違いを確認するテスト"""
import numpy as np
import math

def test_horizontal_line():
    """水平線のヒートマップで角度を確認"""
    H, W = 224, 224
    hm = np.zeros((H, W), dtype=np.float64)

    # 水平線（中央）
    hm[H//2-2:H//2+3, :] = 1.0

    print("=== 水平線のテスト ===")

    # 画像座標系（Y下向き）
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    X_img, Y_img = np.meshgrid(xs, ys)

    M00 = hm.sum()
    xbar_img = (hm * X_img).sum() / M00
    ybar_img = (hm * Y_img).sum() / M00

    dx_img = X_img - xbar_img
    dy_img = Y_img - ybar_img
    mu20_img = (hm * dx_img**2).sum() / M00
    mu02_img = (hm * dy_img**2).sum() / M00
    mu11_img = (hm * dx_img * dy_img).sum() / M00

    theta_img = 0.5 * math.atan2(2.0 * mu11_img, mu20_img - mu02_img)

    print(f"画像座標系（Y下向き）:")
    print(f"  重心: ({xbar_img:.1f}, {ybar_img:.1f})")
    print(f"  モーメント: mu20={mu20_img:.1f}, mu02={mu02_img:.1f}, mu11={mu11_img:.6f}")
    print(f"  角度: {np.degrees(theta_img):.1f}度")

    # 数学座標系（Y上向き）
    y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
    x_grid = np.arange(W, dtype=np.float64) - W / 2.0
    X_math, Y_math = np.meshgrid(x_grid, y_grid)

    xbar_math = (hm * X_math).sum() / M00
    ybar_math = (hm * Y_math).sum() / M00

    dx_math = X_math - xbar_math
    dy_math = Y_math - ybar_math
    mu20_math = (hm * dx_math**2).sum() / M00
    mu02_math = (hm * dy_math**2).sum() / M00
    mu11_math = (hm * dx_math * dy_math).sum() / M00

    theta_math = 0.5 * math.atan2(2.0 * mu11_math, mu20_math - mu02_math)

    print(f"\n数学座標系（Y上向き）:")
    print(f"  重心: ({xbar_math:.1f}, {ybar_math:.1f})")
    print(f"  モーメント: mu20={mu20_math:.1f}, mu02={mu02_math:.1f}, mu11={mu11_math:.6f}")
    print(f"  角度: {np.degrees(theta_math):.1f}度")

    print(f"\n角度の差: {abs(np.degrees(theta_img) - np.degrees(theta_math)):.1f}度")

def test_diagonal_line():
    """対角線のヒートマップで角度を確認"""
    H, W = 224, 224
    hm = np.zeros((H, W), dtype=np.float64)

    # 対角線（左上→右下）
    for i in range(H):
        j = i
        if 0 <= j < W:
            hm[max(0, i-2):min(H, i+3), max(0, j-2):min(W, j+3)] = 1.0

    print("\n\n=== 対角線のテスト（左上→右下）===")

    # 画像座標系
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    X_img, Y_img = np.meshgrid(xs, ys)

    M00 = hm.sum()
    xbar_img = (hm * X_img).sum() / M00
    ybar_img = (hm * Y_img).sum() / M00

    dx_img = X_img - xbar_img
    dy_img = Y_img - ybar_img
    mu20_img = (hm * dx_img**2).sum() / M00
    mu02_img = (hm * dy_img**2).sum() / M00
    mu11_img = (hm * dx_img * dy_img).sum() / M00

    theta_img = 0.5 * math.atan2(2.0 * mu11_img, mu20_img - mu02_img)

    print(f"画像座標系（Y下向き）:")
    print(f"  角度: {np.degrees(theta_img):.1f}度")

    # 数学座標系
    y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
    x_grid = np.arange(W, dtype=np.float64) - W / 2.0
    X_math, Y_math = np.meshgrid(x_grid, y_grid)

    xbar_math = (hm * X_math).sum() / M00
    ybar_math = (hm * Y_math).sum() / M00

    dx_math = X_math - xbar_math
    dy_math = Y_math - ybar_math
    mu20_math = (hm * dx_math**2).sum() / M00
    mu02_math = (hm * dy_math**2).sum() / M00
    mu11_math = (hm * dx_math * dy_math).sum() / M00

    theta_math = 0.5 * math.atan2(2.0 * mu11_math, mu20_math - mu02_math)

    print(f"数学座標系（Y上向き）:")
    print(f"  角度: {np.degrees(theta_math):.1f}度")

    print(f"\n角度の差: {abs(np.degrees(theta_img) - np.degrees(theta_math)):.1f}度")

if __name__ == "__main__":
    test_horizontal_line()
    test_diagonal_line()
