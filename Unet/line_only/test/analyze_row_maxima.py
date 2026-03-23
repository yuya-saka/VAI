"""
row_maxima の詳細分析

画像を読み込んで、数値的な特性を分析
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_row_maxima_from_saved_images():
    """
    保存されたヒートマップ画像から詳細分析
    """
    import sys
    import torch
    import yaml

    unet_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(unet_dir))

    from line_only.line_detection import detect_line_moments
    from line_only.line_losses import extract_gt_line_params, extract_pred_line_params_batch
    from line_only.train_heat import (
        load_config,
        kfold_split_samples,
        create_data_loaders,
        TinyUNet,
    )

    # 設定とモデルをロード
    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    dataset_root_str = cfg.get("data", {}).get("root_dir", "")
    dataset_root = Path(dataset_root_str) if dataset_root_str else Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ分割
    all_samples = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("sample")]
    train_samples, val_samples, test_samples = kfold_split_samples(all_samples, n_folds=5, test_fold=0, seed=42)

    # データローダー
    _, _, test_loader = create_data_loaders(
        train_samples, val_samples, test_samples,
        dataset_root,
        cfg.get("data", {}).get("vertebra_group", "ALL"),
        image_size,
        float(cfg.get("data", {}).get("sigma", 2.5)),
        42, cfg,
    )

    # モデルをロード
    model_cfg = cfg.get("model", {})
    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    ckpt_dir = unet_dir / cfg.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")
    ckpt_files = list(ckpt_dir.glob("best_fold*.pt")) + list(ckpt_dir.glob("fold*_*.pth"))
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)

    model = TinyUNet(in_ch=in_ch, out_ch=out_ch, feats=feats, dropout=dropout)
    checkpoint = torch.load(latest_ckpt, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # sample22_C1_slice038 を探す
    target_sample = "sample22"
    target_vertebra = "C1"
    target_slice = 38

    selected_batch = None
    selected_idx = None

    for batch in test_loader:
        for i in range(len(batch["sample"])):
            if (
                batch["sample"][i] == target_sample
                and batch["vertebra"][i] == target_vertebra
                and int(batch["slice_idx"][i]) == target_slice
            ):
                selected_batch = batch
                selected_idx = i
                break
        if selected_batch is not None:
            break

    if selected_batch is None:
        print("Sample not found")
        return

    # 推論
    x = selected_batch["image"].to(device).float()
    with torch.no_grad():
        pred = torch.sigmoid(model(x))

    heatmaps = pred[selected_idx].cpu().numpy()  # (4, H, W)

    # GTをロード
    lines_json_path = dataset_root / target_sample / target_vertebra / "lines.json"
    with open(lines_json_path) as f:
        lines_data = json.load(f)
    slice_data = lines_data.get(str(target_slice), {})

    # 詳細分析
    print("=" * 80)
    print("Row Maxima 詳細分析")
    print("=" * 80)

    for ch in range(4):
        line_name = f"line_{ch + 1}"
        gt_pts = slice_data.get(line_name)
        if not gt_pts or len(gt_pts) < 2:
            continue

        heatmap = heatmaps[ch]
        H, W = heatmap.shape

        print(f"\n{line_name}:")
        print("-" * 60)

        # 1. 行ごとの最大値インデックス
        row_maxima = [np.argmax(row) for row in heatmap]
        row_maxima = np.array(row_maxima)

        # 2. 行ごとの最大値（強度）
        row_max_values = [np.max(row) for row in heatmap]
        row_max_values = np.array(row_max_values)

        # 3. 有効な行（最大値が閾値以上）
        threshold = 0.15
        valid_rows = row_max_values > threshold
        n_valid = valid_rows.sum()

        print(f"  有効な行数（max > {threshold}）: {n_valid} / {H}")

        if n_valid > 0:
            valid_maxima = row_maxima[valid_rows]
            valid_indices = np.where(valid_rows)[0]

            # 有効範囲での統計
            print(f"  有効範囲の列位置: {valid_maxima.min()} - {valid_maxima.max()}")
            print(f"  有効範囲の行位置: {valid_indices.min()} - {valid_indices.max()}")

            # 差分の統計
            if len(valid_maxima) > 1:
                diffs = np.diff(valid_maxima)
                print(f"  連続する有効行間の差分:")
                print(f"    Mean: {np.mean(diffs):.2f}")
                print(f"    Std:  {np.std(diffs):.2f}")
                print(f"    Min:  {np.min(diffs):.2f}")
                print(f"    Max:  {np.max(diffs):.2f}")

        # 4. 全ピクセルの加重平均（モーメント法と同じ）
        y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
        x_grid = np.arange(W, dtype=np.float64) - W / 2.0
        Y, X = np.meshgrid(y_grid, x_grid, indexing="ij")

        M00 = heatmap.sum()
        cx_math = (heatmap * X).sum() / M00
        cy_math = (heatmap * Y).sum() / M00

        # 画像座標系での重心
        cx_img = cx_math + W / 2.0
        cy_img = -cy_math + H / 2.0

        print(f"  重心（画像座標）: ({cx_img:.2f}, {cy_img:.2f})")

        # 5. GT線の情報
        gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size)
        gt_angle_deg = np.degrees(gt_phi)

        # 予測線の情報
        hm_torch = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_params, confidence = extract_pred_line_params_batch(hm_torch, image_size)
        pred_phi = pred_params[0, 0, 0].item()
        pred_angle_deg = np.degrees(pred_phi)

        print(f"  GT角度:   {gt_angle_deg:.2f}度")
        print(f"  予測角度: {pred_angle_deg:.2f}度")
        print(f"  差分:     {abs(gt_angle_deg - pred_angle_deg):.2f}度")

        # 6. ヒートマップの質量分布
        print(f"  ヒートマップ統計:")
        print(f"    総質量(M00): {M00:.2f}")
        print(f"    最大値: {heatmap.max():.4f}")
        print(f"    平均値: {heatmap.mean():.4f}")
        print(f"    閾値以上の比率: {(heatmap > threshold).sum() / (H * W) * 100:.2f}%")


if __name__ == "__main__":
    analyze_row_maxima_from_saved_images()
