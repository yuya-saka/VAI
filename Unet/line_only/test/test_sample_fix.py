"""
修正後のコードで特定サンプルの直線抽出を再評価

sample5_C1_slice029 で観測された40-50度の角度誤差が
Y軸座標系修正で改善されたか確認
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from line_losses import extract_pred_line_params_batch


def create_test_heatmap_from_angle(angle_deg, H=224, W=224):
    """
    指定した角度の法線を持つ直線のヒートマップを生成

    引数:
        angle_deg: 法線ベクトルの角度（度）
        H, W: ヒートマップのサイズ

    戻り値:
        heatmap: (1, 1, H, W) テンソル
    """
    heatmap = torch.zeros(1, 1, H, W)

    # 法線ベクトル
    angle_rad = np.radians(angle_deg)
    nx = np.cos(angle_rad)
    ny = np.sin(angle_rad)

    # 中心を通る直線を描画
    for i in range(H):
        for j in range(W):
            # 座標系: 中心原点、Y軸上向き（修正後）
            y = -(i - H/2)  # row 0 (top) = +H/2
            x = j - W/2     # col 0 (left) = -W/2

            # 直線からの距離: |nx*x + ny*y|
            dist = abs(nx * x + ny * y)

            # 直線の近傍をヒートマップに設定
            if dist < 3:
                heatmap[0, 0, i, j] = 1.0

    return heatmap


def test_angle_extraction(gt_angle_deg, device="cpu"):
    """
    指定角度の直線抽出をテスト

    引数:
        gt_angle_deg: 期待される法線角度（度）
        device: 計算デバイス

    戻り値:
        error_deg: 角度誤差（度）
    """
    # ヒートマップ生成
    heatmap = create_test_heatmap_from_angle(gt_angle_deg)
    heatmap = heatmap.to(device)

    # 修正後のコードで直線抽出
    pred_params, confidence = extract_pred_line_params_batch(heatmap, image_size=224)

    # 予測角度を取得
    pred_phi = pred_params[0, 0, 0].item()
    pred_phi_deg = np.degrees(pred_phi)

    # 誤差計算
    error_deg = abs(pred_phi_deg - gt_angle_deg)

    # [0, π) の範囲で折り返しを考慮
    if error_deg > 90:
        error_deg = 180 - error_deg

    return pred_phi_deg, error_deg


def main():
    """
    sample5_C1_slice029 で観測された問題ケースを再現・検証
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Y軸座標系修正の効果確認")
    print("sample5_C1_slice029 で観測された40-50度誤差が改善されたか検証")
    print("=" * 70)

    # 問題のあったケース
    test_cases = [
        {
            "name": "line_1 (sample5_C1_slice029)",
            "gt_angle": np.degrees(0.349),  # 約20度
            "before_error": 42.72,  # 修正前の誤差
        },
        {
            "name": "line_3 (sample5_C1_slice029)",
            "gt_angle": np.degrees(2.187),  # 約125度
            "before_error": 52.64,  # 修正前の誤差
        },
        {
            "name": "line_2 (sample5_C1_slice029) - 参考",
            "gt_angle": np.degrees(1.638),  # 約94度
            "before_error": 4.86,  # 修正前も良好
        },
    ]

    print("\n修正後の結果:")
    print("-" * 70)

    all_improved = True

    for i, case in enumerate(test_cases, 1):
        gt_angle = case["gt_angle"]
        before_error = case["before_error"]

        # 修正後のコードでテスト
        pred_angle, after_error = test_angle_extraction(gt_angle, device)

        # 改善度合い
        improvement = before_error - after_error

        print(f"\n{i}. {case['name']}")
        print(f"   GT 角度:        {gt_angle:6.2f}°")
        print(f"   予測 角度:      {pred_angle:6.2f}°")
        print(f"   修正前 誤差:    {before_error:6.2f}°")
        print(f"   修正後 誤差:    {after_error:6.2f}°")
        print(f"   改善:           {improvement:+6.2f}° ", end="")

        if after_error < 5:
            print("✅ 優秀（<5°）")
        elif after_error < 10:
            print("✅ 良好（<10°）")
        elif improvement > 20:
            print("⚠️  大幅改善したが、まだ誤差あり")
        else:
            print("❌ 改善不十分")
            all_improved = False

    print("\n" + "=" * 70)

    if all_improved:
        print("✅ 全てのテストケースで修正効果を確認！")
        print("   Y軸座標系の修正により、40-50度の誤差が解消されました。")
    else:
        print("⚠️  一部のケースで改善が不十分です。")
        print("   追加の調査が必要かもしれません。")

    print("=" * 70)

    # 追加: 様々な角度でテスト
    print("\n追加検証: 0-180度の範囲でテスト")
    print("-" * 70)

    test_angles = [0, 30, 45, 60, 90, 120, 135, 150, 175]
    max_error = 0

    for angle in test_angles:
        pred_angle, error = test_angle_extraction(angle, device)
        print(f"  GT: {angle:3d}° → 予測: {pred_angle:6.2f}° (誤差: {error:5.2f}°)")
        max_error = max(max_error, error)

    print(f"\n  最大誤差: {max_error:.2f}°")

    if max_error < 5:
        print("  ✅ 全角度で優秀な精度です！")
    elif max_error < 10:
        print("  ✅ 全角度で良好な精度です。")
    else:
        print("  ⚠️  一部の角度で誤差が大きいです。")


if __name__ == "__main__":
    main()
