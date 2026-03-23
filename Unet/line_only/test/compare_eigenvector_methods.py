"""
固有ベクトル法 vs 解析的公式を比較

Codexの主張: 固有ベクトル法が間違っている
検証: 両方の方法で結果を比較
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))


def method_eigenvector(mu20, mu02, mu11):
    """
    現在の実装（固有ベクトル法）

    Returns:
        phi (degrees)
    """
    # 固有値
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    discriminant = max(0, trace * trace - 4 * det)
    sqrt_disc = np.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2  # 大きい方

    # 固有ベクトル
    if abs(mu11) > 1e-8:
        dir_x = mu11
        dir_y = lambda1 - mu20
    else:
        if mu20 > mu02:
            dir_x = 1.0
            dir_y = 0.0
        else:
            dir_x = 0.0
            dir_y = 1.0

    # 正規化
    dir_norm = np.sqrt(dir_x * dir_x + dir_y * dir_y)
    dir_x = dir_x / (dir_norm + 1e-10)
    dir_y = dir_y / (dir_norm + 1e-10)

    # 法線（90度回転）
    nx = -dir_y
    ny = dir_x

    # [0, π) に制限
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny

    phi = np.arctan2(ny, nx)
    return np.degrees(phi)


def method_analytical(mu20, mu02, mu11):
    """
    Codex提案の解析的公式

    Returns:
        phi (degrees)
    """
    # 直線方向の角度
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)

    # 方向ベクトル
    dir_x = np.cos(theta)
    dir_y = np.sin(theta)

    # 法線（90度回転）
    nx = -dir_y
    ny = dir_x

    # [0, π) に制限
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny

    phi = np.arctan2(ny, nx)
    return np.degrees(phi)


def test_both_methods():
    """様々なモーメント値で両方の方法を比較"""

    print("=" * 80)
    print("固有ベクトル法 vs 解析的公式の比較")
    print("=" * 80)

    # テストケース
    test_cases = [
        # (mu20, mu02, mu11, expected_phi_deg, description)
        (10.0, 1.0, 0.0, 0.0, "Horizontal line (mu11=0)"),
        (1.0, 10.0, 0.0, 90.0, "Vertical line (mu11=0)"),
        (5.5, 5.5, 5.0, 45.0, "45-degree line"),
        (5.5, 5.5, -5.0, 135.0, "135-degree line"),
        (8.0, 2.0, 3.0, None, "Random moments (1)"),
        (3.0, 7.0, -2.0, None, "Random moments (2)"),
        (5.1, 4.9, 0.1, None, "Nearly circular (low confidence)"),
        (10.0, 9.8, 0.5, None, "Weak anisotropy"),
    ]

    results = []

    for mu20, mu02, mu11, expected, desc in test_cases:
        phi_eig = method_eigenvector(mu20, mu02, mu11)
        phi_ana = method_analytical(mu20, mu02, mu11)

        diff = abs(phi_eig - phi_ana)
        diff = min(diff, 180 - diff)  # 180度周期

        # Confidence
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        discriminant = max(0, trace * trace - 4 * det)
        sqrt_disc = np.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        conf = 1.0 - lambda2 / lambda1 if lambda1 > 1e-8 else 0.0

        results.append({
            "desc": desc,
            "mu20": mu20,
            "mu02": mu02,
            "mu11": mu11,
            "conf": conf,
            "phi_eig": phi_eig,
            "phi_ana": phi_ana,
            "diff": diff,
            "expected": expected,
        })

        print(f"\n{desc}:")
        print(f"  Moments: mu20={mu20:.1f}, mu02={mu02:.1f}, mu11={mu11:.1f}")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Eigenvector: φ={phi_eig:6.2f}°")
        print(f"  Analytical:  φ={phi_ana:6.2f}°")
        print(f"  Difference:  {diff:6.2f}°")
        if expected is not None:
            err_eig = abs(phi_eig - expected)
            err_eig = min(err_eig, 180 - err_eig)
            err_ana = abs(phi_ana - expected)
            err_ana = min(err_ana, 180 - err_ana)
            print(f"  Error (Eigenvector): {err_eig:6.2f}°")
            print(f"  Error (Analytical):  {err_ana:6.2f}°")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    max_diff = max(r["diff"] for r in results)
    print(f"Maximum difference between methods: {max_diff:.4f}°")

    if max_diff < 0.01:
        print("✅ Both methods produce IDENTICAL results.")
        print("   Codex's claim that eigenvector method is 'wrong' is INCORRECT.")
        print("   The angle errors are NOT caused by the eigenvector formula.")
    else:
        print("❌ Methods produce DIFFERENT results.")
        print("   Codex's claim may be correct. Need to investigate further.")


if __name__ == "__main__":
    test_both_methods()
