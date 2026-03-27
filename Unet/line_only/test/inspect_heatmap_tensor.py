"""
ヒートマップテンソルの詳細検証

目的：
1. モデル出力テンソルの可視化
2. 行ごとの最大値インデックスのプロット（滑らかさの確認）
3. skeletonizeによる中心線抽出と予測線の比較
4. 座標対応の数値検証
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import skeletonize

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from line_only.utils.detection import detect_line_moments
from line_only.utils.losses import extract_gt_line_params, extract_pred_line_params_batch


def load_model_and_data():
    """
    訓練済みモデルとサンプルデータをロード

    戻り値:
        model: 訓練済みU-Net
        sample_data: テストサンプル
        device: torch device
    """
    import yaml

    # Import from Unet directory
    unet_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(unet_dir))
    from data import build_dataloaders
    from model import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定ファイルをロード
    config_path = unet_dir / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # データローダーを作成
    _, _, test_loader = build_dataloaders(cfg, fold=0)

    # モデルをロード（最新のチェックポイント）
    ckpt_dir = unet_dir / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("fold0_*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest_ckpt}")

    model = UNet(in_channels=1, out_channels=4)
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # テストデータから1バッチ取得
    sample_batch = next(iter(test_loader))

    return model, sample_batch, test_loader, device, cfg


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
        transform=ax2.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Row maxima plot saved to: {save_path}")
    print(f"  Smoothness (std): {smoothness:.2f}")


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


def plot_skeleton_comparison(heatmap, pred_endpoints, gt_pts, line_name, save_path, image_size=224):
    """
    ヒートマップ・スケルトン・予測線・GT線を比較表示

    引数:
        heatmap: (H, W) numpy array [0, 1]
        pred_endpoints: [[x1, y1], [x2, y2]] 予測線の端点（画像座標系）
        gt_pts: [[x, y], ...] GT折れ線（画像座標系）
        line_name: str
        save_path: Path
        image_size: int
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

    # 予測線をプロット（画像座標系）
    if pred_endpoints and len(pred_endpoints) == 2:
        x1, y1 = pred_endpoints[0]
        x2, y2 = pred_endpoints[1]
        ax1.plot([x1, x2], [y1, y2], "c-", linewidth=2, label="Pred line")

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

    # 予測線をプロット
    if pred_endpoints and len(pred_endpoints) == 2:
        x1, y1 = pred_endpoints[0]
        x2, y2 = pred_endpoints[1]
        ax2.plot([x1, x2], [y1, y2], "c-", linewidth=2, alpha=0.7, label="Pred")

    ax2.legend()
    ax2.set_xlabel("x (col)")
    ax2.set_ylabel("y (row)")

    # 3. 重ね合わせ（ヒートマップ + スケルトン + 予測線 + GT線）
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

    # 予測線
    if pred_endpoints and len(pred_endpoints) == 2:
        x1, y1 = pred_endpoints[0]
        x2, y2 = pred_endpoints[1]
        ax3.plot([x1, x2], [y1, y2], "c-", linewidth=2, label="Pred")

    ax3.set_title(f"{line_name} - Overlay")
    ax3.legend()
    ax3.set_xlabel("x (col)")
    ax3.set_ylabel("y (row)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Skeleton comparison saved to: {save_path}")


def numerical_verification(heatmap, pred_phi, pred_rho, gt_phi, gt_rho, line_name, image_size=224):
    """
    座標対応の数値検証

    引数:
        heatmap: (H, W) numpy array [0, 1]
        pred_phi, pred_rho: 予測パラメータ（ラジアン、正規化済み）
        gt_phi, gt_rho: GTパラメータ（ラジアン、正規化済み）
        line_name: str
        image_size: int

    戻り値:
        verification_dict: 検証結果の辞書
    """
    H, W = heatmap.shape
    assert H == W == image_size, f"Unexpected heatmap size: {heatmap.shape}"

    # 数学座標系での座標グリッド（モデルと同じ）
    y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
    x_grid = np.arange(W, dtype=np.float64) - W / 2.0
    Y, X = np.meshgrid(y_grid, x_grid, indexing="ij")

    # 重心（数学座標系）
    M00 = heatmap.sum()
    if M00 < 1e-6:
        return {"error": "Heatmap mass too small"}

    cx_math = (heatmap * X).sum() / M00
    cy_math = (heatmap * Y).sum() / M00

    # 画像座標系での重心
    cx_img = cx_math + image_size / 2.0
    cy_img = -cy_math + image_size / 2.0

    # GTと予測の法線ベクトル
    gt_nx = np.cos(gt_phi)
    gt_ny = np.sin(gt_phi)
    pred_nx = np.cos(pred_phi)
    pred_ny = np.sin(pred_phi)

    # 法線の角度差（度）
    angle_diff = abs(np.degrees(gt_phi - pred_phi))
    angle_diff_180 = min(angle_diff, 180.0 - angle_diff)  # 180度周期

    # ρの差（非正規化）
    D = np.sqrt(image_size**2 + image_size**2)
    gt_rho_px = gt_rho * D
    pred_rho_px = pred_rho * D
    rho_diff_px = abs(gt_rho_px - pred_rho_px)

    # 重心からGT直線への距離（数学座標系）
    gt_dist_from_centroid = abs(gt_nx * cx_math + gt_ny * cy_math - gt_rho_px)

    # 重心から予測直線への距離（数学座標系）
    pred_dist_from_centroid = abs(pred_nx * cx_math + pred_ny * cy_math - pred_rho_px)

    result = {
        "line_name": line_name,
        "centroid_math": [float(cx_math), float(cy_math)],
        "centroid_img": [float(cx_img), float(cy_img)],
        "gt_phi_deg": float(np.degrees(gt_phi)),
        "pred_phi_deg": float(np.degrees(pred_phi)),
        "angle_diff_deg": float(angle_diff_180),
        "gt_rho_px": float(gt_rho_px),
        "pred_rho_px": float(pred_rho_px),
        "rho_diff_px": float(rho_diff_px),
        "gt_dist_from_centroid_px": float(gt_dist_from_centroid),
        "pred_dist_from_centroid_px": float(pred_dist_from_centroid),
        "heatmap_mass": float(M00),
    }

    return result


def main():
    """メイン処理"""
    print("=" * 80)
    print("ヒートマップテンソル詳細検証")
    print("=" * 80)

    # モデルとデータをロード
    model, sample_batch, test_loader, device, cfg = load_model_and_data()

    # 出力ディレクトリ
    out_dir = Path(__file__).parent / "heatmap_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # データセットルート
    dataset_root = Path(cfg["data"]["dataset_root"])

    # サンプル選択（sample5_C2_slice036を優先）
    target_sample = "sample5"
    target_vertebra = "C2"
    target_slice = 36

    # バッチをイテレートして対象サンプルを探す
    sample_idx = None
    for i in range(len(sample_batch["sample"])):
        if (
            sample_batch["sample"][i] == target_sample
            and sample_batch["vertebra"][i] == target_vertebra
            and int(sample_batch["slice_idx"][i]) == target_slice
        ):
            sample_idx = i
            break

    # 見つからない場合は最初のサンプルを使用
    if sample_idx is None:
        print(f"Target sample {target_sample}_{target_vertebra}_slice{target_slice:03d} not found in first batch.")
        print("Using first sample in batch instead.")
        sample_idx = 0
        target_sample = sample_batch["sample"][0]
        target_vertebra = sample_batch["vertebra"][0]
        target_slice = int(sample_batch["slice_idx"][0])

    print(f"\nProcessing: {target_sample}_{target_vertebra}_slice{target_slice:03d}")
    print("-" * 80)

    # 入力画像とGTを取得
    x = sample_batch["image"].to(device).float()
    ct_img = x[sample_idx, 0].cpu().numpy()  # (H, W)

    # モデル推論
    with torch.no_grad():
        pred = torch.sigmoid(model(x))

    # ヒートマップを取得
    heatmaps = pred[sample_idx].cpu().numpy()  # (4, H, W)

    # GTを取得
    lines_json_path = dataset_root / target_sample / target_vertebra / "lines.json"
    if not lines_json_path.exists():
        print(f"GT file not found: {lines_json_path}")
        return

    with open(lines_json_path) as f:
        lines_data = json.load(f)

    slice_data = lines_data.get(str(target_slice), {})

    image_size = int(cfg["data"]["image_size"])

    # 予測パラメータを抽出
    pred_params_batch, confidence_batch = extract_pred_line_params_batch(
        pred[sample_idx:sample_idx+1], image_size
    )
    pred_params = pred_params_batch[0].cpu().numpy()  # (4, 2)

    # 検証結果を格納
    all_verifications = []

    # 各直線について処理
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        print(f"\n{line_name}:")

        # GT折れ線
        gt_pts = slice_data.get(line_name)
        if not gt_pts or len(gt_pts) < 2:
            print(f"  No GT for {line_name}")
            continue

        # GTパラメータ
        gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size)
        if np.isnan(gt_phi):
            print(f"  Invalid GT for {line_name}")
            continue

        # 予測パラメータ
        pred_phi = pred_params[ch, 0]
        pred_rho = pred_params[ch, 1]

        print(f"  GT:   φ={np.degrees(gt_phi):.2f}°, ρ={gt_rho:.4f}")
        print(f"  Pred: φ={np.degrees(pred_phi):.2f}°, ρ={pred_rho:.4f}")

        # ヒートマップ
        heatmap = heatmaps[ch]

        # 1. 行ごとの最大値インデックスをプロット
        plot_row_maxima(
            heatmap,
            line_name,
            out_dir / f"{target_sample}_{target_vertebra}_slice{target_slice:03d}_{line_name}_row_maxima.png"
        )

        # 2. momentsで端点を計算
        pred_line_info = detect_line_moments(heatmap, length_px=None, extend_ratio=1.10)
        if pred_line_info is None:
            print(f"  Failed to detect line from heatmap")
            continue

        pred_endpoints = pred_line_info.get("endpoints")

        # 3. スケルトン比較
        plot_skeleton_comparison(
            heatmap,
            pred_endpoints,
            gt_pts,
            line_name,
            out_dir / f"{target_sample}_{target_vertebra}_slice{target_slice:03d}_{line_name}_skeleton.png",
            image_size
        )

        # 4. 数値検証
        verification = numerical_verification(
            heatmap, pred_phi, pred_rho, gt_phi, gt_rho, line_name, image_size
        )
        all_verifications.append(verification)

        print(f"  Angle diff: {verification['angle_diff_deg']:.2f}°")
        print(f"  Rho diff: {verification['rho_diff_px']:.2f} px")
        print(f"  Centroid (math): ({verification['centroid_math'][0]:.2f}, {verification['centroid_math'][1]:.2f})")
        print(f"  Centroid (img): ({verification['centroid_img'][0]:.2f}, {verification['centroid_img'][1]:.2f})")

    # 検証結果を保存
    verification_path = out_dir / f"{target_sample}_{target_vertebra}_slice{target_slice:03d}_verification.json"
    with open(verification_path, "w") as f:
        json.dump(all_verifications, f, indent=2)

    print("\n" + "=" * 80)
    print(f"検証完了。結果は {out_dir} に保存されました。")
    print(f"検証JSON: {verification_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
