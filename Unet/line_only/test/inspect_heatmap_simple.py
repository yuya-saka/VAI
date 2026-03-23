"""
ヒートマップテンソルの詳細検証（簡易版）

既存の出力結果から検証を行う
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


def load_existing_results():
    """
    既存の予測結果を検索

    戻り値:
        list of dict: 予測結果ファイルのリスト
    """
    # 最新のoutputsディレクトリを探す
    outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    # test_results_* ディレクトリを探す
    test_dirs = sorted(outputs_dir.glob("test_results_*"), reverse=True)
    if not test_dirs:
        raise FileNotFoundError("No test_results_* directories found")

    latest_dir = test_dirs[0]
    print(f"Using results from: {latest_dir}")

    # PRED_lines.jsonファイルを探す
    pred_files = list(latest_dir.glob("*_PRED_lines.json"))
    if not pred_files:
        raise FileNotFoundError("No PRED_lines.json files found")

    return latest_dir, pred_files


def load_heatmap_from_checkpoint():
    """
    チェックポイントからモデルをロードし、サンプルの推論を実行

    戻り値:
        dict: {
            "heatmaps": (4, H, W) numpy array,
            "ct_image": (H, W) numpy array,
            "sample_name": str,
            "gt_lines": dict
        }
    """
    import sys
    import torch
    import yaml

    # Unet/line_only ディレクトリをインポートパスに追加
    unet_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(unet_dir))

    from line_only.train_heat import (
        load_config,
        resolve_dataset_root,
        kfold_split_samples,
        create_data_loaders,
        TinyUNet,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定ファイルをロード
    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    # Use root_dir instead of dataset_root
    dataset_root_str = cfg.get("data", {}).get("root_dir", "")
    dataset_root = Path(dataset_root_str) if dataset_root_str else resolve_dataset_root("")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # データ分割
    all_samples = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("sample")]

    # Fold 0のデータ分割
    fold = 0
    train_samples, val_samples, test_samples = kfold_split_samples(all_samples, n_folds=5, test_fold=fold, seed=42)

    # データローダーを作成
    _, _, test_loader = create_data_loaders(
        train_samples,
        val_samples,
        test_samples,
        dataset_root,
        cfg.get("data", {}).get("vertebra_group", "ALL"),
        image_size,
        float(cfg.get("data", {}).get("sigma", 2.5)),
        42,
        cfg,
    )

    # モデルをロード
    model_cfg = cfg.get("model", {})
    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    # Use checkpoint_dir from config
    ckpt_dir = unet_dir / cfg.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")
    ckpt_files = list(ckpt_dir.glob("best_fold*.pt")) + list(ckpt_dir.glob("fold*_*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest_ckpt}")

    model = TinyUNet(in_ch=in_ch, out_ch=out_ch, feats=feats, dropout=dropout)
    checkpoint = torch.load(latest_ckpt, map_location=device)
    # Try different possible keys
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # テストデータから1サンプル取得（sample5_C2_slice036を優先）
    target_sample = "sample5"
    target_vertebra = "C2"
    target_slice = 36

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
        # 見つからない場合は最初のバッチを使用
        selected_batch = next(iter(test_loader))
        selected_idx = 0
        target_sample = selected_batch["sample"][0]
        target_vertebra = selected_batch["vertebra"][0]
        target_slice = int(selected_batch["slice_idx"][0])

    print(f"\nSelected sample: {target_sample}_{target_vertebra}_slice{target_slice:03d}")

    # 入力画像
    x = selected_batch["image"].to(device).float()
    ct_img = x[selected_idx, 0].cpu().numpy()

    # 推論
    with torch.no_grad():
        pred = torch.sigmoid(model(x))

    heatmaps = pred[selected_idx].cpu().numpy()  # (4, H, W)

    # GTをロード
    lines_json_path = dataset_root / target_sample / target_vertebra / "lines.json"
    if not lines_json_path.exists():
        raise FileNotFoundError(f"GT file not found: {lines_json_path}")

    with open(lines_json_path) as f:
        lines_data = json.load(f)

    slice_data = lines_data.get(str(target_slice), {})

    return {
        "heatmaps": heatmaps,
        "ct_image": ct_img,
        "sample_name": f"{target_sample}_{target_vertebra}_slice{target_slice:03d}",
        "gt_lines": slice_data,
        "image_size": image_size,
    }


def plot_row_maxima(heatmap, line_name, save_path):
    """
    ヒートマップの行ごとの最大値インデックスをプロット

    引数:
        heatmap: (H, W) numpy array
        line_name: str (e.g., "line_1")
        save_path: Path
    """
    H, W = heatmap.shape

    # 各行の最大値インデックスを抽出
    row_maxima = []
    for row in range(H):
        row_data = heatmap[row, :]
        max_idx = np.argmax(row_data)
        row_maxima.append(max_idx)

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ヒートマップ表示
    im = ax1.imshow(heatmap, cmap="hot", origin="upper")
    ax1.set_title(f"{line_name} - Heatmap")
    ax1.set_xlabel("Column (x)")
    ax1.set_ylabel("Row (y)")
    plt.colorbar(im, ax=ax1)

    # 行ごとの最大値インデックス（画像座標系で重ね合わせ）
    ax1.plot(row_maxima, range(H), "c-", linewidth=1, alpha=0.7, label="Row maxima")
    ax1.legend()

    # 最大値インデックスの推移（滑らかさの確認）
    ax2.plot(range(H), row_maxima, "b-", marker=".", markersize=2)
    ax2.set_title(f"{line_name} - Row Maxima Progression")
    ax2.set_xlabel("Row index (y)")
    ax2.set_ylabel("Column index of max value (x)")
    ax2.grid(True, alpha=0.3)

    # 滑らかさの数値評価（差分の標準偏差）
    diffs = np.diff(row_maxima)
    smoothness = np.std(diffs)
    ax2.text(
        0.02, 0.98, f"Smoothness (std of diff): {smoothness:.2f}",
        transform=ax2.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Row maxima plot saved: {save_path.name}")
    print(f"  Smoothness (std): {smoothness:.2f}")

    return smoothness


def extract_skeleton(heatmap, threshold=0.15):
    """
    ヒートマップから中心線を抽出（skeletonize）

    引数:
        heatmap: (H, W) numpy array [0, 1]
        threshold: 二値化閾値

    戻り値:
        skeleton: (H, W) bool array
    """
    # 二値化
    binary = heatmap > threshold

    # スケルトン化
    skeleton = skeletonize(binary)

    return skeleton


def plot_skeleton_comparison(heatmap, gt_pts, line_name, save_path):
    """
    ヒートマップ・スケルトン・GT線を比較表示

    引数:
        heatmap: (H, W) numpy array [0, 1]
        gt_pts: [[x, y], ...] GT折れ線（画像座標系）
        line_name: str
        save_path: Path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. ヒートマップ
    ax1 = axes[0]
    im = ax1.imshow(heatmap, cmap="hot", origin="upper")
    ax1.set_title(f"{line_name} - Heatmap")
    plt.colorbar(im, ax=ax1)

    # GT線をプロット（画像座標系）
    if gt_pts and len(gt_pts) >= 2:
        gt_arr = np.array(gt_pts)
        ax1.plot(gt_arr[:, 0], gt_arr[:, 1], "g-", linewidth=2, label="GT polyline")

    ax1.legend()
    ax1.set_xlabel("x (col)")
    ax1.set_ylabel("y (row)")

    # 2. スケルトン
    skeleton = extract_skeleton(heatmap, threshold=0.15)
    ax2 = axes[1]
    ax2.imshow(skeleton, cmap="gray", origin="upper")
    ax2.set_title(f"{line_name} - Skeleton")

    # GT線をプロット
    if gt_pts and len(gt_pts) >= 2:
        gt_arr = np.array(gt_pts)
        ax2.plot(gt_arr[:, 0], gt_arr[:, 1], "g-", linewidth=2, alpha=0.7, label="GT")

    ax2.legend()
    ax2.set_xlabel("x (col)")
    ax2.set_ylabel("y (row)")

    # 3. 重ね合わせ（ヒートマップ + スケルトン + GT線）
    ax3 = axes[2]
    # ヒートマップを背景として薄く表示
    ax3.imshow(heatmap, cmap="hot", origin="upper", alpha=0.4)
    # スケルトンを重ねる
    skeleton_overlay = np.ma.masked_where(~skeleton, skeleton)
    ax3.imshow(skeleton_overlay, cmap="Blues", origin="upper", alpha=0.6)

    # GT線
    if gt_pts and len(gt_pts) >= 2:
        gt_arr = np.array(gt_pts)
        ax3.plot(gt_arr[:, 0], gt_arr[:, 1], "g-", linewidth=2, label="GT")

    ax3.set_title(f"{line_name} - Overlay")
    ax3.legend()
    ax3.set_xlabel("x (col)")
    ax3.set_ylabel("y (row)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Skeleton comparison saved: {save_path.name}")


def main():
    """メイン処理"""
    print("=" * 80)
    print("ヒートマップテンソル詳細検証")
    print("=" * 80)

    # 出力ディレクトリ
    out_dir = Path(__file__).parent / "heatmap_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデルから推論を実行してヒートマップを取得
    print("\nLoading model and running inference...")
    data = load_heatmap_from_checkpoint()

    heatmaps = data["heatmaps"]  # (4, H, W)
    ct_image = data["ct_image"]  # (H, W)
    sample_name = data["sample_name"]
    gt_lines = data["gt_lines"]
    image_size = data["image_size"]

    print(f"Processing: {sample_name}")
    print(f"Heatmap shape: {heatmaps.shape}")
    print("-" * 80)

    # 検証結果を格納
    all_smoothness = {}

    # 各直線について処理
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        print(f"\n{line_name}:")

        # GT折れ線
        gt_pts = gt_lines.get(line_name)
        if not gt_pts or len(gt_pts) < 2:
            print(f"  No GT for {line_name}")
            continue

        # ヒートマップ
        heatmap = heatmaps[ch]

        # 1. 行ごとの最大値インデックスをプロット
        smoothness = plot_row_maxima(
            heatmap,
            line_name,
            out_dir / f"{sample_name}_{line_name}_row_maxima.png"
        )
        all_smoothness[line_name] = smoothness

        # 2. スケルトン比較
        plot_skeleton_comparison(
            heatmap,
            gt_pts,
            line_name,
            out_dir / f"{sample_name}_{line_name}_skeleton.png"
        )

    # 統計サマリー
    print("\n" + "=" * 80)
    print("Smoothness Summary (lower = smoother):")
    for line_name, smooth in all_smoothness.items():
        print(f"  {line_name}: {smooth:.2f}")
    print("=" * 80)
    print(f"\n検証完了。結果は {out_dir} に保存されました。")
    print("=" * 80)


if __name__ == "__main__":
    main()
