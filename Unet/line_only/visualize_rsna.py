"""RSNAデータセットの線予測結果を可視化するスクリプト。

各サンプルにつき1枚のPNGを出力する:
  横に4パネル:
    [1] CT原画像
    [2] 4chヒートマップグリッド（2×2）
    [3] 予測線オーバーレイ
    [4] 4領域分割マップ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_here = Path(__file__).resolve().parent   # line_only/
_unet = _here.parent                      # Unet/
_root = _unet.parent                      # VAI/
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from line_only.src.model import VERTEBRA_TO_IDX, TinyUNet  # noqa: E402
from line_only.utils.detection import detect_line_moments, line_extent, moments_to_phi_rho  # noqa: E402
from data_preprocessing.segmentation_dataset.generate_region_mask import (  # noqa: E402
    generate_region_mask,
)

PROJECT_ROOT = _unet.parent
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
METADATA_DIR = RSNA_DATA_DIR / "processing_metadata"
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / "Unet" / "outputs" / "line_20260616"
    / "sig4.0_ALL(CC適用)" / "checkpoints"
)
DEFAULT_TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
IMAGE_SIZE = 224
CT_CENTER_CH = 2
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
FALLBACK_LINE_LENGTH_PX = 80.0

# 4本線の描画色 (BGR)
LINE_COLORS = {
    "line_1": (0, 220, 0),    # 緑
    "line_2": (0, 60, 255),   # 赤
    "line_3": (255, 100, 0),  # 青
    "line_4": (0, 220, 220),  # 黄
}

# 4領域の塗り色 (BGR) - region_eval.py と同一定義
REGION_COLORS_BGR = (
    (0, 0, 0),        # 0: 背景
    (0, 200, 0),      # 1: 椎体 body
    (0, 0, 200),      # 2: 右椎間孔 right foramen
    (200, 0, 0),      # 3: 左椎間孔 left foramen
    (0, 200, 200),    # 4: 後方要素 posterior
)


# -------------------------
# モデル読み込み
# -------------------------
def load_models(ckpt_dir: Path, n_folds: int, device: torch.device) -> list[TinyUNet]:
    """全foldのTinyUNetをロードする。"""
    models: list[TinyUNet] = []
    for fold in range(n_folds):
        p = ckpt_dir / f"best_fold{fold}.pt"
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location=device, weights_only=True)
        cfg = ckpt.get("cfg", {})
        mc = cfg.get("model", {})
        m = TinyUNet(
            in_ch=int(mc.get("in_channels", 2)),
            out_ch=int(mc.get("out_channels", 4)),
            feats=tuple(mc.get("features", [16, 32, 64, 128])),
            dropout=0.0,
            num_vertebra=int(mc.get("num_vertebra", 7))
            if mc.get("use_vertebra_conditioning", False) else 0,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def compute_average_training_line_lengths(dataset_dir: Path) -> dict[str, float]:
    """学習データの lines.json から line別平均長を計算する。"""
    sums = {key: 0.0 for key in LINE_KEYS}
    counts = {key: 0 for key in LINE_KEYS}

    for lines_path in sorted(dataset_dir.glob("sample*/C*/lines.json")):
        try:
            data = json.loads(lines_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for slice_data in data.values():
            if not isinstance(slice_data, dict):
                continue
            for key in LINE_KEYS:
                length = line_extent(slice_data.get(key))
                if length <= 1e-6:
                    continue
                sums[key] += length
                counts[key] += 1

    averages: dict[str, float] = {}
    for key in LINE_KEYS:
        if counts[key] == 0:
            print(f"[WARN] {key} の学習線長を計算できません。fallback={FALLBACK_LINE_LENGTH_PX}px")
            averages[key] = FALLBACK_LINE_LENGTH_PX
            continue
        averages[key] = sums[key] / counts[key]
    return averages


# -------------------------
# データ読み込み・推論
# -------------------------
def find_max_area_index(meta: dict, vertebra: str) -> int | None:
    """メタデータからmax_area_forcedプレーンのインデックスを返す。"""
    for plane in meta["vertebrae"][vertebra]["classifier_planes"]["planes"]:
        if plane.get("max_area_forced", False):
            return int(plane["sequence_index"])
    return None


@torch.no_grad()
def infer(
    models: list[TinyUNet],
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    vertebra: str,
    device: torch.device,
    average_line_lengths: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """アンサンブル推論で(4,H,W)ヒートマップと4本線情報を返す。"""
    x = torch.from_numpy(
        np.stack([ct_slice, mask_slice], axis=0)
    ).unsqueeze(0).to(device)
    vidx = torch.tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0)], device=device, dtype=torch.long,
    )

    hm_sum: torch.Tensor | None = None
    for model in models:
        out = torch.sigmoid(model(x, vidx))
        hm_sum = out if hm_sum is None else hm_sum + out
    hm = (hm_sum / len(models)).cpu().numpy()[0]  # (4, H, W)

    threshold = {"mode": "adaptive", "min": 0.10, "peak_ratio": 0.4}
    lines: dict[str, Any] = {}
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        line_length = average_line_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX)
        result = detect_line_moments(
            hm[ch],
            length_px=line_length,
            extend_ratio=1.0,
            clip=False,
            threshold=threshold,
        )
        if result is None:
            lines[line_name] = None
            continue
        phi, rho = moments_to_phi_rho(result, IMAGE_SIZE)
        lines[line_name] = {
            "endpoints": result["endpoints"],
            "centroid": result["centroid"],
            "angle_deg": result["angle_deg"],
            "phi_rad": phi,
            "rho_normalized": rho,
            "line_length_px": line_length,
        }
    return hm, lines


# -------------------------
# 平均長線分
# -------------------------

def lines_to_polylines(lines: dict[str, Any]) -> dict[str, list] | None:
    """各ヒートマップ由来の無限線を、学習平均長の線分として返す。"""
    polylines: dict[str, list] = {}
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None or info.get("endpoints") is None:
            return None
        polylines[key] = [
            list(info["endpoints"][0]),
            list(info["endpoints"][1]),
        ]
    return polylines


# -------------------------
# 4領域マスク生成・描画
# -------------------------

def draw_region_overlay(
    ct: np.ndarray,
    lines: dict[str, Any],
    mask_slice: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """generate_region_mask を使って4領域を描画した画像 (H,W,3) を返す。

    ヒートマップ由来の無限線を学習平均長で線分化し、
    領域マスクと描画線の入力を一致させる。
    """
    ct_u8 = (np.clip(ct, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    polylines = lines_to_polylines(lines)
    if polylines is None:
        return base

    vertebra_mask = (mask_slice > 0.5).astype(np.uint8)
    try:
        seg, _ = generate_region_mask(
            line_1=polylines["line_1"],
            line_2=polylines["line_2"],
            line_3=polylines["line_3"],
            line_4=polylines["line_4"],
            vertebra_mask=vertebra_mask,
        )
    except Exception as e:
        print(f"[WARN] generate_region_mask 失敗: {e}")
        return base

    label = np.argmax(seg, axis=0).astype(np.uint8)

    ct_f = base.astype(np.float32)
    color_layer = np.zeros_like(ct_f)
    fg = label > 0
    for lbl, bgr in enumerate(REGION_COLORS_BGR):
        color_layer[label == lbl] = bgr

    blended = ct_f.copy()
    blended[fg] = ct_f[fg] * (1 - alpha) + color_layer[fg] * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 領域マスクと同じ平均長線分で線を描画
    for k, pts in polylines.items():
        c = LINE_COLORS.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = pts
        cv2.line(blended,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), c, 2)

    return blended


# -------------------------
# ヒートマップグリッド
# -------------------------
def make_heatmap_grid(ct: np.ndarray, heatmaps: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """4チャンネルを2×2グリッドで表示 (2H, 2W, 3)。"""
    H, W = ct.shape
    ct_u8 = (np.clip(ct, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
    labels = ["line_1", "line_2", "line_3", "line_4"]
    tiles = []
    for c in range(4):
        hm = np.clip(heatmaps[c], 0, 1)
        heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tile = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
        color = LINE_COLORS[labels[c]]
        cv2.putText(tile, labels[c], (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        tiles.append(tile)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    # メインパネル (H,W) に合わせてリサイズ
    return cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)


# -------------------------
# 線オーバーレイ
# -------------------------
def make_line_overlay(ct: np.ndarray, lines: dict[str, Any]) -> np.ndarray:
    """予測線をCT上に描画 (H,W,3)。"""
    ct_u8 = (np.clip(ct, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    polylines = lines_to_polylines(lines)
    if polylines is None:
        return img

    for k, pts in polylines.items():
        c = LINE_COLORS.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = pts
        cv2.line(img,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), c, 2)
        info = lines.get(k)
        if info is not None:
            cx, cy = info.get("centroid", [None, None])
            if cx is not None and cy is not None:
                cv2.circle(img, (int(round(cx)), int(round(cy))), 3, c, -1)

    return img


# -------------------------
# 1サンプル可視化
# -------------------------
def visualize_sample(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    heatmaps: np.ndarray,
    lines: dict[str, Any],
    study_uid: str,
    vertebra: str,
    plane_index: int,
    save_path: Path,
) -> None:
    """4パネル横並びPNGを保存する。"""
    H, W = IMAGE_SIZE, IMAGE_SIZE

    # パネル1: CT原画像
    ct_u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
    p1 = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, "CT", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # パネル2: ヒートマップグリッド
    p2 = make_heatmap_grid(ct_slice, heatmaps)
    cv2.putText(p2, "Heatmaps", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # パネル3: 予測線
    p3 = make_line_overlay(ct_slice, lines)
    cv2.putText(p3, "Pred Lines", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # パネル4: 4領域分割
    p4 = draw_region_overlay(ct_slice, lines, mask_slice)
    cv2.putText(p4, "4 Regions", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 区切り線
    sep = np.full((H, 3, 3), 80, dtype=np.uint8)
    canvas = np.concatenate([p1, sep, p2, sep, p3, sep, p4], axis=1)

    # タイトルバー
    title_h = 28
    title_bar = np.zeros((title_h, canvas.shape[1], 3), dtype=np.uint8)
    title = f"{study_uid[:30]}  {vertebra}  plane={plane_index}"
    cv2.putText(title_bar, title, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    final = np.concatenate([title_bar, canvas], axis=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final)


# -------------------------
# メイン
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA線予測の可視化")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        default=DEFAULT_TRAINING_DATASET_DIR,
        help="学習に使った lines.json を含む dataset ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "rsna_line_vis",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--study-ids", nargs="*", default=None,
        help="可視化対象スタディUID（省略時は先頭N件）",
    )
    parser.add_argument("--n-studies", type=int, default=5, help="study-ids省略時の件数")
    parser.add_argument("--vertebrae", nargs="*", default=None,
                        help="対象椎体（省略時は全C1-C7）")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了")
    average_line_lengths = compute_average_training_line_lengths(args.training_dataset_dir)
    print("[INFO] 学習データ平均線長(px): " + ", ".join(
        f"{key}={value:.1f}" for key, value in average_line_lengths.items()
    ))

    vertebrae = args.vertebrae or [f"C{i}" for i in range(1, 8)]

    if args.study_ids:
        study_uids = args.study_ids
    else:
        study_uids = sorted(
            d.name for d in args.fracture_dataset_dir.iterdir() if d.is_dir()
        )[: args.n_studies]
    print(f"[INFO] 対象スタディ: {len(study_uids)} 件 × 椎体 {vertebrae}")

    for study_uid in study_uids:
        meta_path = args.metadata_dir / f"{study_uid}.json"
        if not meta_path.exists():
            print(f"[SKIP] メタデータなし: {study_uid}")
            continue
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        study_dir = args.fracture_dataset_dir / study_uid

        for vertebra in vertebrae:
            plane_index = None
            for plane in meta["vertebrae"][vertebra]["classifier_planes"]["planes"]:
                if plane.get("max_area_forced", False):
                    plane_index = int(plane["sequence_index"])
                    break
            if plane_index is None:
                print(f"[SKIP] max_area_forcedなし: {study_uid}/{vertebra}")
                continue

            ct_path = study_dir / vertebra / "ct.npy"
            mask_path = study_dir / vertebra / "vertebra_mask.npy"
            if not ct_path.exists() or not mask_path.exists():
                continue

            ct_vol = np.load(ct_path, allow_pickle=False)
            mask_vol = np.load(mask_path, allow_pickle=False)

            ct_slice = ct_vol[plane_index, CT_CENTER_CH].astype(np.float32) / 255.0
            mask_slice = mask_vol[plane_index].astype(np.float32)

            heatmaps, lines = infer(
                models,
                ct_slice,
                mask_slice,
                vertebra,
                device,
                average_line_lengths,
            )

            save_path = args.output_dir / study_uid / f"{vertebra}_plane{plane_index:02d}.png"
            visualize_sample(ct_slice, mask_slice, heatmaps, lines, study_uid, vertebra, plane_index, save_path)
            resolved_save_path = save_path.resolve()
            try:
                display_path = resolved_save_path.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = resolved_save_path
            print(f"[SAVED] {display_path}")

    print(f"\n[DONE] {args.output_dir}")


if __name__ == "__main__":
    main()
