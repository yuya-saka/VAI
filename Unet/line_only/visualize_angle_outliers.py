"""C1/C2/C7の線角度分布と外れ値サンプルを可視化するスクリプト。

出力:
  angle_dist_C1C2C7.png  - 3レベル × 4線の角度ヒストグラム
  outliers_Cx_line_y.png - 外れ値サンプルのCT画像パネル
  normal_Cx_line_y.png   - 正常サンプルのCT画像パネル（比較用）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np
import torch

_here = Path(__file__).resolve().parent
_unet = _here.parent
_root = _unet.parent
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from line_only.src.model import VERTEBRA_TO_IDX, TinyUNet  # noqa: E402
from line_only.utils.detection import (  # noqa: E402
    detect_line_moments,
    line_extent,
    moments_to_phi_rho,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (  # noqa: E402
    generate_region_mask,
)

PROJECT_ROOT = _root
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
ANGLE_STATS_DIR = PROJECT_ROOT / "Unet" / "outputs" / "angle_stats"
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / "Unet" / "outputs" / "line_20260616"
    / "sig4.0_ALL(CC適用)" / "checkpoints"
)
DEFAULT_TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
TARGET_LEVELS = ["C1", "C2", "C7"]
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
IMAGE_SIZE = 224
FALLBACK_LINE_LENGTH_PX = 80.0
CENTER_SLICE = 2

LINE_COLORS_BGR = {
    "line_1": (0, 220, 0),
    "line_2": (0, 60, 255),
    "line_3": (255, 100, 0),
    "line_4": (0, 220, 220),
}
LINE_COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in LINE_COLORS_BGR.items()}
REGION_COLORS_BGR = (
    (0, 0, 0), (0, 200, 0), (0, 0, 200), (200, 0, 0), (0, 200, 200),
)


def load_models(ckpt_dir: Path, n_folds: int, device: torch.device) -> list[TinyUNet]:
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
    return {
        key: sums[key] / counts[key] if counts[key] > 0 else FALLBACK_LINE_LENGTH_PX
        for key in LINE_KEYS
    }


@torch.no_grad()
def infer(
    models: list[TinyUNet],
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    vertebra: str,
    device: torch.device,
    avg_lengths: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """(4,H,W) ヒートマップと4本線情報を返す。"""
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
    hm = (hm_sum / len(models)).cpu().numpy()[0]

    threshold = {"mode": "adaptive", "min": 0.10, "peak_ratio": 0.4}
    lines: dict[str, Any] = {}
    for ch in range(4):
        line_name = f"line_{ch + 1}"
        result = detect_line_moments(
            hm[ch],
            length_px=avg_lengths.get(line_name, FALLBACK_LINE_LENGTH_PX),
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
            "angle_deg": float(result["angle_deg"]),
            "phi_rad": phi,
            "rho_normalized": rho,
        }
    return hm, lines


def draw_panel(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    lines: dict[str, Any],
    avg_lengths: dict[str, float],
    highlight_line: str | None = None,
    alpha: float = 0.40,
) -> np.ndarray:
    """CT + 4線 + 4領域分割の1パネル(H, 2W+sep, 3)。"""
    H, W = IMAGE_SIZE, IMAGE_SIZE
    ct_u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    # 左: CT + 線
    left = base.copy()
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None or info.get("endpoints") is None:
            continue
        color = LINE_COLORS_BGR[key]
        thickness = 3 if key == highlight_line else 2
        (x1, y1), (x2, y2) = info["endpoints"]
        cv2.line(left, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color, thickness)
        if key == highlight_line:
            ang = info.get("angle_deg", 0)
            cv2.putText(left, f"{ang:.1f}°", (4, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 右: 4領域
    polylines: dict[str, list] = {}
    ok = True
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None or info.get("endpoints") is None:
            ok = False
            break
        polylines[key] = [list(info["endpoints"][0]), list(info["endpoints"][1])]

    right = base.copy()
    if ok:
        vertebra_mask = (mask_slice > 0.5).astype(np.uint8)
        try:
            seg, _ = generate_region_mask(
                line_1=polylines["line_1"], line_2=polylines["line_2"],
                line_3=polylines["line_3"], line_4=polylines["line_4"],
                vertebra_mask=vertebra_mask,
            )
            label = np.argmax(seg, axis=0).astype(np.uint8)
            ct_f = base.astype(np.float32)
            color_layer = np.zeros_like(ct_f)
            for lbl, bgr in enumerate(REGION_COLORS_BGR):
                color_layer[label == lbl] = bgr
            fg = label > 0
            right = ct_f.copy()
            right[fg] = ct_f[fg] * (1 - alpha) + color_layer[fg] * alpha
            right = np.clip(right, 0, 255).astype(np.uint8)
            # 線を重ねる
            for key, pts in polylines.items():
                c = LINE_COLORS_BGR[key]
                (x1, y1), (x2, y2) = pts
                cv2.line(right, (int(round(x1)), int(round(y1))),
                         (int(round(x2)), int(round(y2))), c, 2)
        except Exception:
            pass

    sep = np.full((H, 3, 3), 60, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


# ---- Phase 1: 角度ヒストグラム ----

def plot_angle_histograms(
    raw_csv: Path,
    output_path: Path,
) -> None:
    """C1/C2/C7 × 4線の角度ヒストグラムを3×4グリッドで出力。"""
    # CSVから角度を読み込む
    data: dict[str, dict[str, list[float]]] = {
        level: {key: [] for key in LINE_KEYS}
        for level in TARGET_LEVELS
    }
    with raw_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level = row["vertebra"]
            if level not in TARGET_LEVELS:
                continue
            for key in LINE_KEYS:
                val = row.get(key, "")
                if val:
                    data[level][key].append(float(val))

    fig, axes = plt.subplots(
        len(TARGET_LEVELS), len(LINE_KEYS),
        figsize=(16, 10),
        sharex=False,
    )
    bins = np.arange(0, 185, 5)

    for row_i, level in enumerate(TARGET_LEVELS):
        for col_i, line_key in enumerate(LINE_KEYS):
            ax = axes[row_i][col_i]
            vals = data[level][line_key]
            if not vals:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            arr = np.array(vals)
            color_rgb = tuple(c / 255 for c in LINE_COLORS_RGB[line_key])

            # IQRで外れ値判定
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            outlier_mask = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
            normal_vals = arr[~outlier_mask]
            outlier_vals = arr[outlier_mask]

            ax.hist(normal_vals, bins=bins, color=color_rgb, alpha=0.8,
                    label=f"正常 n={len(normal_vals)}")
            if len(outlier_vals) > 0:
                ax.hist(outlier_vals, bins=bins, color="red", alpha=0.7,
                        label=f"外れ値 n={len(outlier_vals)} ({len(outlier_vals)/len(arr)*100:.1f}%)")

            ax.axvline(float(np.median(arr)), color="black", linestyle="--",
                       linewidth=1.2, label=f"中央値 {np.median(arr):.1f}°")
            ax.set_title(f"{level} / {line_key}", fontsize=10)
            ax.set_xlabel("angle (deg)")
            ax.set_ylabel("count")
            ax.set_xlim(0, 180)
            ax.legend(fontsize=7)

    fig.suptitle("C1/C2/C7 線角度分布（中央スライス、全データ）", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path.relative_to(PROJECT_ROOT)}", flush=True)


# ---- Phase 2: 外れ値 / 正常サンプルの可視化 ----

def collect_outlier_and_normal_ids(
    raw_csv: Path,
    level: str,
    line_key: str,
    n_samples: int = 6,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """IQR法で外れ値/正常のstudy_uidを返す。"""
    rows: list[tuple[str, float]] = []
    with raw_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["vertebra"] != level:
                continue
            val = row.get(line_key, "")
            if val:
                rows.append((row["study_uid"], float(val)))
    if not rows:
        return [], []

    vals = np.array([r[1] for r in rows])
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    outlier_mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)

    rng = np.random.RandomState(seed)
    outlier_ids = [rows[i][0] for i in np.where(outlier_mask)[0]]
    normal_ids = [rows[i][0] for i in np.where(~outlier_mask)[0]]

    if len(outlier_ids) > n_samples:
        outlier_ids = [outlier_ids[i] for i in rng.choice(
            len(outlier_ids), n_samples, replace=False
        )]
    if len(normal_ids) > n_samples:
        normal_ids = [normal_ids[i] for i in rng.choice(
            len(normal_ids), n_samples, replace=False
        )]
    return outlier_ids, normal_ids


def build_sample_grid(
    study_ids: list[str],
    level: str,
    highlight_line: str,
    models: list[TinyUNet],
    device: torch.device,
    avg_lengths: dict[str, float],
    label: str,
) -> np.ndarray | None:
    """複数スタディのパネルを横に並べてグリッド画像を返す。"""
    panels = []
    for study_id in study_ids:
        level_dir = FRACTURE_DATASET_DIR / study_id / level
        seg_ct_path = level_dir / "seg_ct.npy"
        seg_mask_path = level_dir / "seg_vertebra_mask.npy"
        if not seg_ct_path.exists():
            continue
        seg_ct = np.load(seg_ct_path)
        seg_mask = np.load(seg_mask_path)
        ct_f = seg_ct[CENTER_SLICE].astype(np.float32) / 255.0
        mask_f = seg_mask[CENTER_SLICE].astype(np.float32)
        _, lines = infer(models, ct_f, mask_f, level, device, avg_lengths)
        panel = draw_panel(ct_f, mask_f, lines, avg_lengths, highlight_line)

        # 角度をタイトルに追加
        info = lines.get(highlight_line)
        ang_str = f"{info['angle_deg']:.1f}°" if info else "None"
        title_h = 22
        title_bar = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
        short_uid = study_id.split(".")[-1]
        cv2.putText(title_bar, f"{short_uid} {ang_str}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        panels.append(np.concatenate([title_bar, panel], axis=0))

    if not panels:
        return None

    sep = np.full((panels[0].shape[0], 4, 3), 40, dtype=np.uint8)
    cols = [panels[0]]
    for p in panels[1:]:
        cols.append(sep)
        cols.append(p)
    canvas = np.concatenate(cols, axis=1)

    header_h = 30
    header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
    color = (100, 220, 100) if label == "正常" else (80, 80, 255)
    cv2.putText(header, f"[{label}] {level} / {highlight_line}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return np.concatenate([header, canvas], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="C1/C2/C7 角度外れ値可視化")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=DEFAULT_TRAINING_DATASET_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "angle_outlier_vis",
    )
    parser.add_argument(
        "--raw-csv", type=Path,
        default=ANGLE_STATS_DIR / "angles_raw.csv",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=6,
                        help="外れ値/正常のサンプル数（各）")
    # 外れ値率が特に高い line を対象にする
    parser.add_argument(
        "--target-lines", nargs="*",
        default=["line_2", "line_4"],
        help="可視化対象の線（デフォルト: line_2 line_4）",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    # ---- Phase 1: ヒストグラム ----
    print("\n=== Phase 1: 角度ヒストグラム ===", flush=True)
    hist_path = args.output_dir / "angle_dist_C1C2C7.png"
    plot_angle_histograms(args.raw_csv, hist_path)

    # ---- Phase 2: サンプル可視化（モデル必要） ----
    print("\n=== Phase 2: 外れ値/正常サンプル可視化 ===", flush=True)
    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了", flush=True)
    avg_lengths = compute_average_training_line_lengths(args.training_dataset_dir)

    for level in TARGET_LEVELS:
        for line_key in args.target_lines:
            print(f"  {level} / {line_key} ...", flush=True)
            outlier_ids, normal_ids = collect_outlier_and_normal_ids(
                args.raw_csv, level, line_key, args.n_samples,
            )
            print(
                f"    外れ値: {len(outlier_ids)}, 正常: {len(normal_ids)}",
                flush=True,
            )

            grids = []
            if outlier_ids:
                g = build_sample_grid(
                    outlier_ids, level, line_key,
                    models, device, avg_lengths, "外れ値",
                )
                if g is not None:
                    grids.append(g)

            if normal_ids:
                g = build_sample_grid(
                    normal_ids, level, line_key,
                    models, device, avg_lengths, "正常",
                )
                if g is not None:
                    grids.append(g)

            if not grids:
                continue

            sep_h = np.full((6, grids[0].shape[1], 3), 80, dtype=np.uint8)
            combined = grids[0]
            for g in grids[1:]:
                # 幅を合わせる（短い方をゼロパディング）
                max_w = max(combined.shape[1], g.shape[1])
                def pad_w(img: np.ndarray, w: int) -> np.ndarray:
                    if img.shape[1] >= w:
                        return img
                    pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
                    return np.concatenate([img, pad], axis=1)
                combined = np.concatenate([pad_w(combined, max_w), sep_h[:, :max_w], pad_w(g, max_w)], axis=0)

            save_path = args.output_dir / f"{level}_{line_key}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), combined)
            print(f"    [SAVED] {save_path.relative_to(PROJECT_ROOT)}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
