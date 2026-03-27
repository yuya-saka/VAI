"""
チャンネル間クロストーク分析

各チャンネルのヒートマップが他のチャンネルのGT線位置に
どれだけ反応しているかを定量的に分析する
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.src.data_utils import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
)
from line_only.src.model import TinyUNet


def create_line_mask(gt_points, image_size=224, thickness=5):
    """
    GT線の周辺にマスクを作成

    引数:
        gt_points: 線を定義する点のリスト [[x1,y1], [x2,y2], ...]
        image_size: 画像サイズ
        thickness: 線の太さ（片側ピクセル数）

    戻り値:
        (H, W) のブールマスク
    """
    import cv2
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    pts = np.array(gt_points, dtype=np.int32)

    for i in range(len(pts) - 1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, thickness=thickness*2+1)

    return mask > 0


def analyze_crosstalk(heatmaps, gt_lines, image_size=224):
    """
    各チャンネルの反応が正しい位置/間違った位置にどれだけあるかを分析

    引数:
        heatmaps: (4, H, W) 予測ヒートマップ
        gt_lines: {"line_1": [[x,y], ...], ...} GT線

    戻り値:
        分析結果の辞書
    """
    results = {}
    line_names = ["line_1", "line_2", "line_3", "line_4"]

    # 各チャンネルのGT線マスクを作成
    gt_masks = {}
    for ch, line_name in enumerate(line_names):
        gt_pts = gt_lines.get(line_name)
        if gt_pts and len(gt_pts) >= 2:
            gt_masks[line_name] = create_line_mask(gt_pts, image_size)
        else:
            gt_masks[line_name] = None

    # 各チャンネルを分析
    for ch, line_name in enumerate(line_names):
        heatmap = heatmaps[ch]
        own_mask = gt_masks[line_name]

        if own_mask is None:
            continue

        # 自分のGT位置での反応
        own_response = heatmap[own_mask].sum()
        own_max = heatmap[own_mask].max() if own_mask.any() else 0

        # 他のチャンネルのGT位置での反応
        other_responses = {}
        for other_ch, other_name in enumerate(line_names):
            if other_ch == ch:
                continue
            other_mask = gt_masks[other_name]
            if other_mask is not None and other_mask.any():
                # 自分のマスクと重ならない部分のみ
                other_only = other_mask & ~own_mask
                if other_only.any():
                    other_responses[other_name] = {
                        "sum": float(heatmap[other_only].sum()),
                        "max": float(heatmap[other_only].max()),
                        "mean": float(heatmap[other_only].mean()),
                    }

        # 総質量
        total_mass = float(heatmap.sum())

        results[line_name] = {
            "total_mass": total_mass,
            "own_response": {
                "sum": float(own_response),
                "max": float(own_max),
                "ratio": float(own_response / total_mass) if total_mass > 0 else 0,
            },
            "crosstalk": other_responses,
            "crosstalk_total": sum(r["sum"] for r in other_responses.values()),
            "crosstalk_ratio": sum(r["sum"] for r in other_responses.values()) / total_mass if total_mass > 0 else 0,
        }

    return results


def main():
    """複数サンプルでクロストーク分析を実行"""

    # 設定とモデルをロード
    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    sigma = float(cfg.get("data", {}).get("sigma", 2.5))
    dataset_root_str = cfg.get("data", {}).get("root_dir", "")
    dataset_root = Path(dataset_root_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ分割
    all_samples = [
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("sample")
    ]
    train_samples, val_samples, test_samples = kfold_split_samples(
        all_samples, n_folds=5, test_fold=0, seed=42
    )

    # データローダー
    _, _, test_loader = create_data_loaders(
        train_samples, val_samples, test_samples,
        dataset_root,
        cfg.get("data", {}).get("vertebra_group", "ALL"),
        image_size, sigma, 42, cfg,
    )

    # モデルをロード
    model_cfg = cfg.get("model", {})
    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    ckpt_dir = unet_dir / cfg.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")
    ckpt_files = list(ckpt_dir.glob("best_fold*.pt")) + list(ckpt_dir.glob("fold*_*.pth"))

    if not ckpt_files:
        print("チェックポイントファイルが見つかりません")
        return

    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f"モデルをロード: {latest_ckpt.name}")

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

    print("\n" + "=" * 80)
    print("チャンネル間クロストーク分析")
    print("=" * 80)

    # 集計用
    all_results = []

    processed = 0
    max_samples = 30

    for batch in test_loader:
        if processed >= max_samples:
            break

        x = batch["image"].to(device).float()
        with torch.no_grad():
            pred = torch.sigmoid(model(x))

        pred_np = pred.cpu().numpy()

        B = pred_np.shape[0]
        for i in range(B):
            if processed >= max_samples:
                break

            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])

            # GT lines.jsonをロード
            lines_json_path = dataset_root / sample / vertebra / "lines.json"
            if not lines_json_path.exists():
                continue

            with open(lines_json_path) as f:
                lines_data = json.load(f)

            slice_data = lines_data.get(str(slice_idx), {})
            if not slice_data:
                continue

            # クロストーク分析
            results = analyze_crosstalk(pred_np[i], slice_data, image_size)

            if results:
                all_results.append({
                    "name": f"{sample}_{vertebra}_slice{slice_idx:03d}",
                    "results": results,
                })
                processed += 1

    # 結果を表示
    print("\n" + "-" * 80)
    print("サンプルごとのクロストーク率 (他のGT線位置での反応 / 総質量)")
    print("-" * 80)

    crosstalk_rates = {f"line_{i+1}": [] for i in range(4)}

    for item in all_results:
        print(f"\n{item['name']}:")
        for line_name, data in item["results"].items():
            own_ratio = data["own_response"]["ratio"]
            cross_ratio = data["crosstalk_ratio"]
            crosstalk_rates[line_name].append(cross_ratio)

            print(f"  {line_name}:")
            print(f"    自分のGT位置: {own_ratio*100:.1f}%")
            print(f"    他のGT位置:   {cross_ratio*100:.1f}%")
            if data["crosstalk"]:
                for other_name, other_data in data["crosstalk"].items():
                    print(f"      → {other_name}: max={other_data['max']:.3f}, mean={other_data['mean']:.4f}")

    # 統計サマリー
    print("\n" + "=" * 80)
    print("クロストーク率の統計（全サンプル平均）")
    print("=" * 80)

    for line_name, rates in crosstalk_rates.items():
        if rates:
            arr = np.array(rates)
            print(f"{line_name}:")
            print(f"  平均: {arr.mean()*100:.2f}%")
            print(f"  最大: {arr.max()*100:.2f}%")
            print(f"  標準偏差: {arr.std()*100:.2f}%")


if __name__ == "__main__":
    main()
