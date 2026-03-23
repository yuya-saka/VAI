"""
実際の予測ヒートマップを診断

- 実際のクロストークレベルを測定
- ヒートマップ品質を評価
- 角度誤差の真の原因を特定
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

unet_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(unet_dir))

from line_only.train_heat import (
    load_config,
    kfold_split_samples,
    create_data_loaders,
    TinyUNet,
)
from line_only.line_losses import extract_pred_line_params_batch, extract_gt_line_params


def measure_crosstalk_level(heatmaps, gt_lines_dict, image_size=224):
    """
    実際のクロストークレベルを測定

    各チャンネルが他のチャンネルのGT位置でどれくらい反応しているか

    引数:
        heatmaps: (4, H, W) 予測ヒートマップ
        gt_lines_dict: {"line_1": [[x,y], ...], ...}
        image_size: 画像サイズ

    戻り値:
        crosstalk_matrix: (4, 4) matrix[i, j] = channel i の response at line j's position
    """
    C, H, W = heatmaps.shape
    crosstalk = np.zeros((C, C))

    for ch_i in range(C):
        for ch_j in range(C):
            line_name = f"line_{ch_j + 1}"
            gt_pts = gt_lines_dict.get(line_name)

            if gt_pts is None or len(gt_pts) < 2:
                crosstalk[ch_i, ch_j] = np.nan
                continue

            # GTライン上のピクセルをサンプリング
            pts = np.array(gt_pts)
            xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
            ys = np.clip(pts[:, 1].astype(int), 0, H - 1)

            # チャンネルiのヒートマップのGTライン上の平均値
            values = heatmaps[ch_i, ys, xs]
            crosstalk[ch_i, ch_j] = values.mean()

    return crosstalk


def diagnose_sample(model, sample_batch, dataset_root, device):
    """1サンプルを詳細診断"""

    x = sample_batch["image"].to(device).float()
    sample_name = sample_batch["sample"][0]
    vertebra = sample_batch["vertebra"][0]
    slice_idx = int(sample_batch["slice_idx"][0])

    # 予測
    with torch.no_grad():
        pred = torch.sigmoid(model(x))

    pred_np = pred[0].cpu().numpy()  # (4, H, W)

    # GT lines.json をロード
    lines_json_path = dataset_root / sample_name / vertebra / "lines.json"
    if not lines_json_path.exists():
        return None

    with open(lines_json_path) as f:
        lines_data = json.load(f)

    slice_data = lines_data.get(str(slice_idx), {})
    if not slice_data:
        return None

    # クロストーク測定
    crosstalk_matrix = measure_crosstalk_level(pred_np, slice_data, image_size=224)

    # 各チャンネルの品質評価
    channel_quality = []

    for ch in range(4):
        line_name = f"line_{ch + 1}"
        gt_pts = slice_data.get(line_name)

        hm = pred_np[ch]

        # GT角度
        gt_phi, gt_rho = extract_gt_line_params(gt_pts, image_size=224)

        # 予測角度
        pred_params, conf = extract_pred_line_params_batch(
            torch.from_numpy(hm).unsqueeze(0).unsqueeze(0), image_size=224
        )
        pred_phi = pred_params[0, 0, 0].item()
        pred_phi_deg = np.degrees(pred_phi) % 180
        gt_phi_deg = np.degrees(gt_phi) % 180

        # 角度誤差
        angle_error = abs(pred_phi_deg - gt_phi_deg)
        angle_error = min(angle_error, 180 - angle_error)

        # ヒートマップ品質指標
        peak_value = hm.max()
        mean_value = hm.mean()
        mass = hm.sum()

        # クロストーク（自分以外のチャンネルへの漏れ）
        self_response = crosstalk_matrix[ch, ch]
        other_responses = [crosstalk_matrix[ch, j] for j in range(4) if j != ch]
        avg_crosstalk = np.nanmean(other_responses) if other_responses else 0

        channel_quality.append({
            "channel": ch + 1,
            "angle_error": angle_error,
            "peak": peak_value,
            "mean": mean_value,
            "mass": mass,
            "self_response": self_response,
            "avg_crosstalk": avg_crosstalk,
            "crosstalk_ratio": avg_crosstalk / self_response if self_response > 0 else np.nan,
            "confidence": conf[0, 0].item(),
        })

    return {
        "sample": sample_name,
        "vertebra": vertebra,
        "slice_idx": slice_idx,
        "crosstalk_matrix": crosstalk_matrix,
        "channel_quality": channel_quality,
    }


def main():
    """複数サンプルを診断"""

    cfg = load_config()
    image_size = int(cfg.get("data", {}).get("image_size", 224))
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
        image_size, 2.5, 42, cfg,
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

    print("="*80)
    print("実際のヒートマップ診断")
    print("="*80)

    all_diagnostics = []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 10:  # 最初の10サンプル
            break

        diag = diagnose_sample(model, batch, dataset_root, device)
        if diag is None:
            continue

        all_diagnostics.append(diag)

        print(f"\n{diag['sample']}_{diag['vertebra']}_slice{diag['slice_idx']:03d}:")
        print(f"  Crosstalk Matrix (rows=output_ch, cols=GT_line):")
        for i in range(4):
            row_str = "  " + " ".join(f"{diag['crosstalk_matrix'][i, j]:6.3f}" for j in range(4))
            print(row_str)

        print(f"\n  Channel Quality:")
        for cq in diag['channel_quality']:
            print(f"    Ch{cq['channel']}: "
                  f"Error={cq['angle_error']:5.1f}° "
                  f"Peak={cq['peak']:.3f} "
                  f"Crosstalk={cq['crosstalk_ratio']*100:4.1f}% "
                  f"Conf={cq['confidence']:.3f}")

    # 統計サマリー
    print("\n" + "="*80)
    print("統計サマリー")
    print("="*80)

    all_errors = [cq['angle_error'] for d in all_diagnostics for cq in d['channel_quality']]
    all_crosstalk_ratios = [
        cq['crosstalk_ratio'] * 100
        for d in all_diagnostics
        for cq in d['channel_quality']
        if not np.isnan(cq['crosstalk_ratio'])
    ]

    print(f"角度誤差: 平均={np.mean(all_errors):.1f}° 中央値={np.median(all_errors):.1f}° "
          f"最大={np.max(all_errors):.1f}°")
    print(f"クロストーク比: 平均={np.mean(all_crosstalk_ratios):.1f}% "
          f"中央値={np.median(all_crosstalk_ratios):.1f}% "
          f"最大={np.max(all_crosstalk_ratios):.1f}%")


if __name__ == "__main__":
    main()
