"""detect_line_moments と extract_pred_line_params_batch の角度が一致するか確認"""
import torch
import numpy as np
from line_only.line_detection import detect_line_moments
from line_only.line_losses import extract_pred_line_params_batch


def test_angle_consistency():
    """同じヒートマップで両関数の角度を比較"""
    H, W = 224, 224

    # 水平線のヒートマップ
    hm = np.zeros((H, W), dtype=np.float32)
    hm[H // 2 - 2 : H // 2 + 3, :] = 1.0

    # detect_line_moments
    result_detect = detect_line_moments(hm, length_px=100)
    angle_detect = result_detect["angle_deg"]

    # extract_pred_line_params_batch
    hm_torch = torch.tensor(hm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pred_params, confidence = extract_pred_line_params_batch(hm_torch, image_size=224)
    phi_pred = pred_params[0, 0, 0].item()  # ラジアン
    angle_pred = np.degrees(phi_pred) % 180

    print("=== 水平線ヒートマップ ===")
    print(f"detect_line_moments:           {angle_detect:.2f}度")
    print(f"extract_pred_line_params_batch: {angle_pred:.2f}度")
    print(f"差: {abs(angle_detect - angle_pred):.2f}度")

    # 垂直線のヒートマップ
    hm2 = np.zeros((H, W), dtype=np.float32)
    hm2[:, W // 2 - 2 : W // 2 + 3] = 1.0

    result_detect2 = detect_line_moments(hm2, length_px=100)
    angle_detect2 = result_detect2["angle_deg"]

    hm2_torch = torch.tensor(hm2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred_params2, confidence2 = extract_pred_line_params_batch(hm2_torch, image_size=224)
    phi_pred2 = pred_params2[0, 0, 0].item()
    angle_pred2 = np.degrees(phi_pred2) % 180

    print("\n=== 垂直線ヒートマップ ===")
    print(f"detect_line_moments:           {angle_detect2:.2f}度")
    print(f"extract_pred_line_params_batch: {angle_pred2:.2f}度")
    print(f"差: {abs(angle_detect2 - angle_pred2):.2f}度")

    # 対角線のヒートマップ
    hm3 = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        j = i
        if 0 <= j < W:
            hm3[max(0, i - 2) : min(H, i + 3), max(0, j - 2) : min(W, j + 3)] = 1.0

    result_detect3 = detect_line_moments(hm3, length_px=100)
    angle_detect3 = result_detect3["angle_deg"]

    hm3_torch = torch.tensor(hm3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred_params3, confidence3 = extract_pred_line_params_batch(hm3_torch, image_size=224)
    phi_pred3 = pred_params3[0, 0, 0].item()
    angle_pred3 = np.degrees(phi_pred3) % 180

    print("\n=== 対角線ヒートマップ ===")
    print(f"detect_line_moments:           {angle_detect3:.2f}度")
    print(f"extract_pred_line_params_batch: {angle_pred3:.2f}度")
    print(f"差: {abs(angle_detect3 - angle_pred3):.2f}度")

    print("\n=== 注意 ===")
    print("detect_line_moments: 直線の方向角（線が向いている方向）")
    print("extract_pred_line_params_batch: 法線角（線に垂直な方向）")
    print("→ 差が90度なら正常（方向角と法線角の違い）")


if __name__ == "__main__":
    test_angle_consistency()
