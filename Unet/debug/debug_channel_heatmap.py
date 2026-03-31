"""
チャンネルごとのヒートマップとしきい値処理後の出力を可視化するデバッグスクリプト

使用方法:
    uv run python Unet/debug_channel_heatmap.py \
        --checkpoint Unet/outputs/regularization/sig3.5/checkpoints/best_fold1.pt \
        --sample sample15.2 --vertebra C2 --slice 46

出力:
    Unet/outputs/regularization/sig3.5/vis/fold1/debug/
        {name}_ch_raw.png         : 4チャンネルの生ヒートマップ（2x2グリッド）
        {name}_ch_thr{T}.png      : 各しきい値で処理したヒートマップ（2x2グリッド）
        {name}_ch_lines_thr{T}.png: しきい値処理後に検出した直線オーバーレイ
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# パスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Unet.line_only.src.dataset import PngLineDataset
from Unet.line_only.src.model import TinyUNet, VERTEBRA_TO_IDX
from Unet.line_only.utils.detection import detect_line_moments, LinesJsonCache
from Unet.line_only.utils.visualization import LINE_COLORS


def _tile_channel_grid(ct_u8_bgr: np.ndarray, hm4: np.ndarray, alpha: float = 0.6, title_prefix: str = "") -> np.ndarray:
    """4チャンネルヒートマップを2x2グリッドに描画"""
    tiles = []
    for c in range(4):
        hm = np.clip(hm4[c], 0, 1)
        hm_u8 = (hm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(ct_u8_bgr.copy(), 1 - alpha, heat_color, alpha, 0)
        label = f"{title_prefix}CH{c+1} max={hm.max():.3f}"
        cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(out)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


def _tile_channel_with_lines(
    ct_u8_bgr: np.ndarray,
    hm4: np.ndarray,
    threshold: float,
    gt_lines: dict,
    alpha: float = 0.6,
) -> np.ndarray:
    """各チャンネルをしきい値処理し、検出した直線とGT線を描画した2x2グリッド"""
    tiles = []
    ch_keys = ["line_1", "line_2", "line_3", "line_4"]
    for c in range(4):
        hm_raw = hm4[c].astype(np.float64)
        hm_thr = np.where(hm_raw >= threshold, hm_raw, 0.0)

        hm_u8 = (np.clip(hm_thr, 0, 1) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(ct_u8_bgr.copy(), 1 - alpha, heat_color, alpha, 0)

        # 検出した直線（白）
        info = detect_line_moments(hm_raw, threshold=threshold)
        if info is not None:
            ep = info["endpoints"]
            (x1, y1), (x2, y2) = ep
            cv2.line(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 255, 255), 2)
            cx, cy = info["centroid"]
            cv2.circle(out, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)

        # GT線（チャンネルに対応する色）
        k = ch_keys[c]
        gt_pts = gt_lines.get(k, None)
        if gt_pts is not None and len(gt_pts) >= 2:
            color = LINE_COLORS.get(k, (200, 200, 200))
            pts_i32 = np.array(gt_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts_i32], isClosed=False, color=color, thickness=2)

        # 残留mass情報
        M00_raw = hm_raw.sum()
        M00_thr = hm_thr.sum()
        ratio = M00_thr / (M00_raw + 1e-12)
        label = f"CH{c+1} thr={threshold:.2f} kept={ratio:.1%}"
        cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(out)

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


def main():
    parser = argparse.ArgumentParser(description="チャンネルごとのヒートマップデバッグ")
    parser.add_argument("--checkpoint", required=True, help="チェックポイントファイルパス (.pt)")
    parser.add_argument("--sample", required=True, help="サンプル名 (例: sample15.2)")
    parser.add_argument("--vertebra", required=True, help="椎体名 (例: C2)")
    parser.add_argument("--slice", type=int, required=True, help="スライスインデックス (例: 46)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.30],
                        help="確認するしきい値リスト")
    parser.add_argument("--out_dir", default=None, help="出力ディレクトリ（未指定時はcheckpointと同階層に自動設定）")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] device: {device}")

    # チェックポイント読み込み
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    print(f"[INFO] experiment: {cfg.get('experiment', {})}")

    # モデル作成・重み読み込み
    model_cfg = cfg.get("model", {})
    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))
    use_vert = bool(model_cfg.get("use_vertebra_conditioning", False))
    num_vert = int(model_cfg.get("num_vertebra", 7)) if use_vert else 0

    model = TinyUNet(in_ch=in_ch, out_ch=out_ch, feats=feats, dropout=dropout, num_vertebra=num_vert).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[INFO] モデルを読み込みました")

    # データ設定
    data_cfg = cfg.get("data", {})
    root_dir = Path(data_cfg.get("root_dir", ""))
    group = data_cfg.get("group", "ALL")
    image_size = int(data_cfg.get("image_size", 224))
    sigma = float(data_cfg.get("sigma", 4.0))
    print(f"[INFO] dataset root: {root_dir}, sigma={sigma}, image_size={image_size}")

    # 対象スライスをデータセットから検索
    ds = PngLineDataset(root_dir, [args.sample], group=group, image_size=image_size, sigma=sigma)
    target_item = None
    for i in range(len(ds)):
        item = ds[i]
        if item["vertebra"] == args.vertebra and int(item["slice_idx"]) == args.slice:
            target_item = item
            break

    if target_item is None:
        print(f"[ERROR] {args.sample}/{args.vertebra}/slice{args.slice:03d} が見つかりませんでした")
        sys.exit(1)

    print(f"[INFO] スライスを発見: {target_item['sample']}/{target_item['vertebra']}/slice{int(target_item['slice_idx']):03d}")

    # 推論
    x = target_item["image"].unsqueeze(0).float().to(device)  # (1,C,H,W)
    v_idx = torch.tensor([VERTEBRA_TO_IDX.get(args.vertebra, 0)], device=device, dtype=torch.long)

    with torch.no_grad():
        pred = torch.sigmoid(model(x, v_idx))

    pr_np = pred[0].cpu().numpy()   # (4,H,W)
    ct01 = target_item["image"][0].numpy()   # (H,W) CT
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    ct_bgr = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    # GT線の取得
    cache = LinesJsonCache(root_dir)
    gt_lines = cache.get_lines_for_slice(args.sample, args.vertebra, args.slice) or {}

    # 出力ディレクトリ
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # checkpointから vis/../debug/ に自動設定
        out_dir = ckpt_path.parent.parent / "vis" / f"fold{_infer_fold(ckpt_path)}" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.sample}_{args.vertebra}_slice{args.slice:03d}"
    print(f"[INFO] 出力先: {out_dir}")

    # 1. 生ヒートマップ（しきい値なし）
    grid_raw = _tile_channel_grid(ct_bgr, pr_np, title_prefix="RAW ")
    cv2.imwrite(str(out_dir / f"{name}_ch_raw.png"), grid_raw)
    print(f"  保存: {name}_ch_raw.png")

    # 2. 各しきい値でしきい値処理後ヒートマップ + 検出線
    for thr in args.thresholds:
        grid_thr = _tile_channel_with_lines(ct_bgr, pr_np, thr, gt_lines)
        thr_str = f"{thr:.2f}".replace(".", "")
        fname = f"{name}_ch_thr{thr_str}.png"
        cv2.imwrite(str(out_dir / fname), grid_thr)
        print(f"  保存: {fname}")

    # 3. チャンネルごとの値の統計をコンソールに出力
    print("\n[チャンネルごとのヒートマップ統計]")
    ch_keys = ["line_1", "line_2", "line_3", "line_4"]
    for c in range(4):
        hm = pr_np[c]
        k = ch_keys[c]
        thr_val = 0.2
        mask = hm >= thr_val
        print(f"  CH{c+1} ({k}):")
        print(f"    max={hm.max():.4f}  mean={hm.mean():.4f}  pixels>{thr_val}: {mask.sum()} / {hm.size}")
        # しきい値なしで検出
        info_no_thr = detect_line_moments(hm, threshold=None)
        info_thr02 = detect_line_moments(hm, threshold=0.2)
        if info_no_thr:
            print(f"    検出（thr=None）: centroid={info_no_thr['centroid']}, angle={info_no_thr['angle_deg']:.1f}°")
        if info_thr02:
            print(f"    検出（thr=0.20）: centroid={info_thr02['centroid']}, angle={info_thr02['angle_deg']:.1f}°")
        else:
            print(f"    検出（thr=0.20）: 検出なし（マス不足）")

    print(f"\n[完了] {out_dir}/")


def _infer_fold(ckpt_path: Path) -> str:
    """チェックポイントファイル名からfold番号を推測"""
    name = ckpt_path.stem  # e.g. "best_fold1"
    for part in name.split("_"):
        if part.startswith("fold"):
            return part[4:]
    return "0"


if __name__ == "__main__":
    main()
