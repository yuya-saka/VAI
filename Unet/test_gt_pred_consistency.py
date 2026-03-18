"""GT抽出と予測抽出の座標系が一致しているか確認"""
import json
import numpy as np
import torch
from pathlib import Path

from line_only.line_losses import extract_gt_line_params, extract_pred_line_params_batch
from line_only.heatmap import generate_gaussian_heatmap


def test_with_real_gt():
    """実際のGT polylineからヒートマップを生成し、角度を比較"""
    # Sample GT lines.json から1つ取得
    dataset_root = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
    sample_dir = dataset_root / "sample5" / "C2"
    lines_json = sample_dir / "lines.json"

    if not lines_json.exists():
        print(f"Not found: {lines_json}")
        return

    with open(lines_json) as f:
        lines_data = json.load(f)

    # Slice 37を使用（最初にユーザーが見た画像）
    slice_data = lines_data.get("37")
    if not slice_data:
        print("Slice 37 not found")
        return

    image_size = 224
    sigma = 2.5

    print("=== 実際のGTデータでテスト ===\n")

    for line_key in ["line_1", "line_2", "line_3", "line_4"]:
        gt_pts = slice_data.get(line_key)
        if not gt_pts or len(gt_pts) < 2:
            continue

        print(f"--- {line_key} ---")
        print(f"GT points: {gt_pts[0]} -> {gt_pts[-1]}")

        # GT抽出
        gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size)
        print(f"GT params: phi={np.degrees(gt_phi):.2f}度, rho={gt_rho:.4f}")

        # GTからヒートマップ生成
        heatmap = generate_gaussian_heatmap(gt_pts, (image_size, image_size), sigma)

        # 予測抽出（ヒートマップから）
        hm_torch = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_params, confidence = extract_pred_line_params_batch(hm_torch, image_size)
        pred_phi = pred_params[0, 0, 0].item()
        pred_rho = pred_params[0, 0, 1].item()

        print(f"Pred params (from heatmap): phi={np.degrees(pred_phi):.2f}度, rho={pred_rho:.4f}")
        print(f"Angle diff: {abs(np.degrees(gt_phi - pred_phi)):.2f}度")
        print(f"Rho diff: {abs(gt_rho - pred_rho):.4f}")

        # 期待値：角度差 < 5度（ヒートマップのぼかしによる誤差を考慮）
        angle_diff = abs(np.degrees(gt_phi - pred_phi))
        if angle_diff < 5:
            print("✓ OK")
        else:
            print(f"✗ NG (expected < 5°)")

        print()


if __name__ == "__main__":
    test_with_real_gt()
