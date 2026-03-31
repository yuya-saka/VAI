"""
全テストサンプルに対してしきい値を変えながら評価指標を計算するスクリプト

使用方法:
    uv run python Unet/eval_threshold_sweep.py \
        --checkpoint Unet/outputs/regularization/sig3.5/checkpoints/best_fold1.pt

出力:
    コンソールに各しきい値での指標テーブル
    .claude/docs/ に結果ファイルを保存
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Unet.line_only.src.data_utils import (
    kfold_split_samples,
    resolve_dataset_root,
    set_seed,
)
from Unet.line_only.src.dataset import PngLineDataset, _is_sample_valid_png
from Unet.line_only.src.model import TinyUNet, VERTEBRA_TO_IDX
from Unet.line_only.utils.detection import LinesJsonCache
from Unet.line_only.utils import losses as line_losses
from Unet.line_only.utils import metrics as line_metrics
from torch.utils.data import DataLoader


THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]


@torch.no_grad()
def collect_predictions(model, loader, device, cache: LinesJsonCache, image_size: int):
    """
    全バッチの予測を収集して返す（しきい値なしで保存）

    戻り値:
        records: list of dict {
            "pred_hm": (4,H,W) numpy,
            "gt_lines": dict,
            "sample": str,
            "vertebra": str,
            "slice_idx": int,
        }
    """
    model.eval()
    records = []
    for batch in loader:
        x = batch["image"].to(device).float()
        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device, dtype=torch.long,
        )
        pred = torch.sigmoid(model(x, v_idx))
        pr_np = pred.cpu().numpy()

        B = pr_np.shape[0]
        for i in range(B):
            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_idx) or {}
            records.append({
                "pred_hm": pr_np[i],        # (4,H,W)
                "gt_lines": gt_lines,
                "sample": sample,
                "vertebra": vertebra,
                "slice_idx": slice_idx,
            })
    return records


def compute_metrics_for_threshold(records, threshold: float, image_size: int) -> dict:
    """
    収集済み予測に対して指定しきい値で指標を計算

    戻り値:
        dict with keys: angle_deg, rho_px, perp_px, n_valid, n_total, detection_rate
    """
    angle_errors = []
    rho_errors = []
    perp_dists = []
    n_total = 0
    n_valid = 0

    ch_keys = ["line_1", "line_2", "line_3", "line_4"]

    for rec in records:
        hm4 = rec["pred_hm"]      # (4,H,W) numpy
        gt_lines = rec["gt_lines"]

        # しきい値適用してパラメータ抽出（バッチサイズ1）
        hm_t = torch.tensor(hm4, dtype=torch.float32).unsqueeze(0)   # (1,4,H,W)
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            hm_t, image_size, threshold=threshold
        )
        pred_params_np = pred_params[0].numpy()   # (4,2)
        conf_np = confidence[0].numpy()           # (4,)

        for c in range(4):
            k = ch_keys[c]
            gt_pts = gt_lines.get(k, None)
            gt_phi, gt_rho = line_losses.extract_gt_line_params(gt_pts, image_size)
            n_total += 1

            if np.isnan(gt_phi):
                continue  # GT なし

            pred_conf = conf_np[c]
            if pred_conf <= 0:
                continue  # 検出なし

            n_valid += 1
            pred_phi = pred_params_np[c, 0]
            pred_rho = pred_params_np[c, 1]

            gt_t = torch.tensor([[gt_phi, gt_rho]], dtype=torch.float32)
            pr_t = torch.tensor([[pred_phi, pred_rho]], dtype=torch.float32)
            mask = torch.tensor([True])

            angle_err = line_metrics.compute_angle_error(pr_t, gt_t, mask)
            rho_err = line_metrics.compute_rho_error(pr_t, gt_t, image_size, mask)
            perp_dist = line_metrics.compute_perpendicular_distance(
                gt_pts, pred_phi, pred_rho, image_size
            )

            angle_errors.append(angle_err)
            rho_errors.append(rho_err)
            if not np.isnan(perp_dist):
                perp_dists.append(perp_dist)

    def _mean(v):
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    def _max(v):
        v = [x for x in v if not np.isnan(x)]
        return float(np.max(v)) if v else float("nan")

    return {
        "threshold": threshold,
        "angle_deg": _mean(angle_errors),
        "angle_deg_max": _max(angle_errors),
        "rho_px": _mean(rho_errors),
        "perp_px": _mean(perp_dists),
        "n_valid": n_valid,
        "n_total": n_total,
        "detection_rate": n_valid / max(1, n_total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    device_id = args.gpu
    if device_id is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_id}")
    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] device: {device}")

    # チェックポイント読み込み
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # モデル作成
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model = TinyUNet(
        in_ch=int(model_cfg.get("in_channels", 2)),
        out_ch=int(model_cfg.get("out_channels", 4)),
        feats=tuple(model_cfg.get("features", [16, 32, 64, 128])),
        dropout=float(model_cfg.get("dropout", 0.0)),
        num_vertebra=int(model_cfg.get("num_vertebra", 7)) if model_cfg.get("use_vertebra_conditioning") else 0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print("[INFO] モデル読み込み完了")

    # データ設定
    root_dir = resolve_dataset_root(data_cfg.get("root_dir", ""))
    group = data_cfg.get("group", "ALL")
    image_size = int(data_cfg.get("image_size", 224))
    sigma = float(data_cfg.get("sigma", 4.0))
    n_folds = int(data_cfg.get("n_folds", 5))
    test_fold = int(data_cfg.get("test_fold", 0))
    seed = int(data_cfg.get("random_seed", 42))
    set_seed(seed)

    # テストサンプルを決定
    sample_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sample")])
    valid_samples = [d.name for d in sample_dirs if _is_sample_valid_png(d, group)]
    _, _, test_samples = kfold_split_samples(valid_samples, n_folds=n_folds, test_fold=test_fold, seed=seed)
    print(f"[INFO] テストサンプル数: {len(test_samples)}")
    print(f"[INFO] テストサンプル: {test_samples}")

    # データローダー作成
    test_ds = PngLineDataset(root_dir, test_samples, group=group, image_size=image_size, sigma=sigma)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    # GT lineキャッシュ
    cache = LinesJsonCache(root_dir)

    # 全テストサンプルの予測を収集（1回のみ推論）
    print("[INFO] 推論中...")
    records = collect_predictions(model, test_loader, device, cache, image_size)
    print(f"[INFO] {len(records)} スライス分の予測を収集完了")

    # 各しきい値で指標を計算
    print("\n" + "=" * 95)
    print(f"{'threshold':>10} | {'mean angle':>10} | {'max angle':>9} | {'rho(px)':>8} | {'perp(px)':>9} | {'detect%':>8} | {'n_valid':>7}")
    print("=" * 95)

    results = []
    baseline = None
    for thr in args.thresholds:
        m = compute_metrics_for_threshold(records, thr, image_size)
        results.append(m)
        if baseline is None:
            baseline = m

        da = m["angle_deg"] - baseline["angle_deg"]
        dp = m["perp_px"] - baseline["perp_px"]
        sign_a = "+" if da >= 0 else ""
        sign_p = "+" if dp >= 0 else ""

        print(
            f"{m['threshold']:>10.2f} | "
            f"{m['angle_deg']:>8.2f}° | "
            f"{m['angle_deg_max']:>8.2f}° | "
            f"{m['rho_px']:>7.2f}px | "
            f"{m['perp_px']:>8.2f}px | "
            f"{m['detection_rate']:>7.1%} | "
            f"{m['n_valid']:>7d}"
            f"  Δangle={sign_a}{da:.2f}° Δmax={m['angle_deg_max'] - baseline['angle_deg_max']:+.2f}°"
        )

    print("=" * 85)

    # 最良のしきい値を報告
    valid_results = [r for r in results if not np.isnan(r["angle_deg"])]
    best_angle = min(valid_results, key=lambda r: r["angle_deg"])
    best_perp = min(valid_results, key=lambda r: r["perp_px"])
    print(f"\n[最良 angle error] thr={best_angle['threshold']:.2f}: {best_angle['angle_deg']:.2f}°")
    print(f"[最良 perp dist]   thr={best_perp['threshold']:.2f}: {best_perp['perp_px']:.2f}px")
    print(f"[現在の設定]       thr=0.20 → angle={next(r for r in results if r['threshold']==0.20)['angle_deg']:.2f}°  perp={next(r for r in results if r['threshold']==0.20)['perp_px']:.2f}px")

    # 結果をファイルに保存
    out_path = Path(".claude/docs/codex") / f"threshold_sweep_fold{test_fold}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Threshold sweep: {ckpt_path}\n")
        f.write(f"Test fold: {test_fold}, n_slices: {len(records)}\n\n")
        f.write(f"{'threshold':>10} | {'angle(deg)':>10} | {'rho(px)':>8} | {'perp(px)':>9} | {'detect%':>8} | n_valid\n")
        f.write("-" * 75 + "\n")
        for m in results:
            f.write(
                f"{m['threshold']:>10.2f} | {m['angle_deg']:>9.2f}° | "
                f"{m['rho_px']:>7.2f}px | {m['perp_px']:>8.2f}px | "
                f"{m['detection_rate']:>7.1%} | {m['n_valid']}\n"
            )
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
