"""
角度誤差が大きいケースを画像で確認するスクリプト

使用方法:
    uv run python Unet/debug_worst_cases.py \
        --checkpoint Unet/outputs/regularization/sig3.5/checkpoints/best_fold1.pt \
        --threshold 0.5 --top_n 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Unet.line_only.src.data_utils import kfold_split_samples, resolve_dataset_root, set_seed
from Unet.line_only.src.dataset import PngLineDataset, _is_sample_valid_png
from Unet.line_only.src.model import TinyUNet, VERTEBRA_TO_IDX
from Unet.line_only.utils import losses as line_losses
from Unet.line_only.utils import metrics as line_metrics
from Unet.line_only.utils.detection import LinesJsonCache, detect_line_moments, line_extent
from Unet.line_only.utils.visualization import LINE_COLORS


def make_worst_case_image(ct01, hm4, threshold, gt_lines, pred_errors):
    """
    4チャンネルヒートマップ + 検出線 + GT線 を1枚にまとめる
    各チャンネルに角度誤差を表示
    """
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    ct_bgr = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    ch_keys = ["line_1", "line_2", "line_3", "line_4"]
    alpha = 0.55
    tiles = []

    for c in range(4):
        k = ch_keys[c]
        hm_raw = hm4[c].astype(np.float64)
        hm_thr = np.where(hm_raw >= threshold, hm_raw, 0.0)

        hm_u8 = (np.clip(hm_thr, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        tile = cv2.addWeighted(ct_bgr.copy(), 1 - alpha, heat, alpha, 0)

        # 検出線（白）
        gt_pts = gt_lines.get(k, None)
        Lgt = line_extent(gt_pts) if gt_pts else None
        if Lgt and Lgt <= 1e-6:
            Lgt = None
        info = detect_line_moments(hm_raw, length_px=Lgt, threshold=threshold)
        if info is not None:
            (x1, y1), (x2, y2) = info["endpoints"]
            cv2.line(tile, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 255, 255), 2)
            cx, cy = info["centroid"]
            cv2.circle(tile, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)

        # GT線（チャンネルカラー）
        if gt_pts and len(gt_pts) >= 2:
            color = LINE_COLORS.get(k, (200, 200, 200))
            pts_i32 = np.array(gt_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(tile, [pts_i32], isClosed=False, color=color, thickness=2)

        # ラベル
        err = pred_errors.get(k)
        err_str = f"{err:.1f}deg" if err is not None and not np.isnan(err) else "N/A"
        label = f"CH{c+1} err={err_str}"
        color_label = (0, 80, 255) if err is not None and not np.isnan(err) and err > 30 else (255, 255, 255)
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_label, 2, cv2.LINE_AA)
        tiles.append(tile)

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]

    # モデル
    mc = cfg.get("model", {})
    model = TinyUNet(
        in_ch=int(mc.get("in_channels", 2)),
        out_ch=int(mc.get("out_channels", 4)),
        feats=tuple(mc.get("features", [16, 32, 64, 128])),
        dropout=float(mc.get("dropout", 0.0)),
        num_vertebra=int(mc.get("num_vertebra", 7)) if mc.get("use_vertebra_conditioning") else 0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # データ
    dc = cfg.get("data", {})
    root_dir = resolve_dataset_root(dc.get("root_dir", ""))
    group = dc.get("group", "ALL")
    image_size = int(dc.get("image_size", 224))
    sigma = float(dc.get("sigma", 4.0))
    n_folds = int(dc.get("n_folds", 5))
    test_fold = int(dc.get("test_fold", 0))
    seed = int(dc.get("random_seed", 42))
    set_seed(seed)

    sample_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sample")])
    valid_samples = [d.name for d in sample_dirs if _is_sample_valid_png(d, group)]
    _, _, test_samples = kfold_split_samples(valid_samples, n_folds=n_folds, test_fold=test_fold, seed=seed)

    test_ds = PngLineDataset(root_dir, test_samples, group=group, image_size=image_size, sigma=sigma)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)
    cache = LinesJsonCache(root_dir)

    ch_keys = ["line_1", "line_2", "line_3", "line_4"]

    # 全サンプル推論して誤差を収集
    all_errors = []   # list of (angle_err, sample, vertebra, slice_idx, ch_key, pred_hm_4ch, ct01, gt_lines)

    for batch in test_loader:
        x = batch["image"].to(device).float()
        v_idx = torch.as_tensor(
            [VERTEBRA_TO_IDX.get(v, 0) for v in batch["vertebra"]],
            device=device, dtype=torch.long,
        )
        pred = torch.sigmoid(model(x, v_idx))
        hm_t = pred
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            hm_t, image_size, threshold=args.threshold
        )
        pr_np = pred.cpu().numpy()
        pred_params_np = pred_params.cpu().numpy()
        conf_np = confidence.cpu().numpy()
        x_np = x.cpu().numpy()

        B = pr_np.shape[0]
        for i in range(B):
            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])
            ct01 = x_np[i, 0]
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_idx) or {}

            # チャンネルごとの誤差
            ch_errors = {}
            for c in range(4):
                k = ch_keys[c]
                gt_pts = gt_lines.get(k, None)
                gt_phi, gt_rho = line_losses.extract_gt_line_params(gt_pts, image_size)
                if np.isnan(gt_phi):
                    ch_errors[k] = None
                    continue
                if conf_np[i, c] <= 0:
                    ch_errors[k] = float("nan")
                    continue
                pred_phi = pred_params_np[i, c, 0]
                pred_rho = pred_params_np[i, c, 1]
                gt_t = torch.tensor([[gt_phi, gt_rho]], dtype=torch.float32)
                pr_t = torch.tensor([[pred_phi, pred_rho]], dtype=torch.float32)
                mask = torch.tensor([True])
                err = line_metrics.compute_angle_error(pr_t, gt_t, mask)
                ch_errors[k] = err

                all_errors.append({
                    "angle_err": err,
                    "sample": sample,
                    "vertebra": vertebra,
                    "slice_idx": slice_idx,
                    "ch_key": k,
                    "pred_hm4": pr_np[i],
                    "ct01": ct01,
                    "gt_lines": gt_lines,
                    "ch_errors": None,  # 後で付与
                })

            # ch_errors を全エントリに付与（同一スライスのもの）
            for e in all_errors[-sum(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in ch_errors.values() if v is not None):]:
                pass
            # 直近のエントリに ch_errors を付与
            for e in all_errors:
                if e["sample"] == sample and e["vertebra"] == vertebra and e["slice_idx"] == slice_idx:
                    e["ch_errors"] = ch_errors

    # 角度誤差でソート
    valid_errors = [e for e in all_errors if not np.isnan(e["angle_err"])]
    valid_errors.sort(key=lambda e: e["angle_err"], reverse=True)

    print(f"[INFO] 総チャンネル数: {len(all_errors)}, 有効: {len(valid_errors)}")
    print(f"\n[上位 {args.top_n} の角度誤差 (thr={args.threshold})]")
    print(f"{'rank':>4} | {'angle_err':>10} | {'sample':>12} | {'vertebra':>8} | {'slice':>5} | {'channel':>8}")
    print("-" * 65)

    # 出力ディレクトリ
    ckpt_path = Path(args.checkpoint)
    fold_str = ckpt_path.stem.replace("best_", "")
    out_dir = ckpt_path.parent.parent / "vis" / fold_str / f"worst_thr{str(args.threshold).replace('.', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 重複スライスをまとめて上位を画像保存
    seen_slices = {}
    saved = 0

    for rank, e in enumerate(valid_errors[:args.top_n * 4], 1):
        print(f"{rank:>4} | {e['angle_err']:>9.2f}° | {e['sample']:>12} | {e['vertebra']:>8} | {e['slice_idx']:>5} | {e['ch_key']:>8}")

        # スライスキーでまとめて1枚の画像に
        slice_key = f"{e['sample']}_{e['vertebra']}_slice{e['slice_idx']:03d}"
        if slice_key not in seen_slices and saved < args.top_n:
            ch_errors = e.get("ch_errors") or {}
            img = make_worst_case_image(
                e["ct01"], e["pred_hm4"], args.threshold, e["gt_lines"], ch_errors
            )
            # タイトルバー追加
            title_bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
            max_err = max((v for v in ch_errors.values() if v is not None and not np.isnan(v)), default=0.0)
            title = f"#{saved+1} {slice_key}  max_angle={max_err:.1f}deg  thr={args.threshold}"
            cv2.putText(title_bar, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 200), 1, cv2.LINE_AA)
            img = np.concatenate([title_bar, img], axis=0)
            cv2.imwrite(str(out_dir / f"rank{saved+1:02d}_{slice_key}.png"), img)
            seen_slices[slice_key] = True
            saved += 1

        if saved >= args.top_n:
            break

    print(f"\n[保存] {saved} 枚 → {out_dir}/")


if __name__ == "__main__":
    main()
