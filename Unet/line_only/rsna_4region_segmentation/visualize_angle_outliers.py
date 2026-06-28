"""C1/C2/C7 の線角度分布と外れ値サンプルを可視化するスクリプト。

出力:
  angle_dist_C1C2C7.png  - 3 レベル × 4 線の角度ヒストグラム
  {level}_{line}.png     - 外れ値/正常サンプルの CT パネル
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib_fontja  # noqa: E402, F401
import numpy as np
import torch

from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    IMAGE_SIZE,
    LINE_COLORS_BGR,
    LINE_COLORS_RGB,
    LINE_KEYS,
    PROJECT_ROOT,
    TRAINING_DATASET_DIR,
)
from .inference import predict_single_slice
from .model_io import compute_avg_line_lengths, load_models
from .visualization import (
    concat_with_separator,
    ct_to_bgr,
    draw_lines_on_image,
    lines_to_polylines,
    make_region_overlay,
)

TARGET_LEVELS = ["C1", "C2", "C7"]
ANGLE_STATS_DIR = PROJECT_ROOT / "Unet" / "outputs" / "angle_stats"


# ---- Phase 1: 角度ヒストグラム ----


def plot_angle_histograms(raw_csv: Path, output_path: Path) -> None:
    """C1/C2/C7 × 4 線の角度ヒストグラムを 3×4 グリッドで出力。"""
    data: dict[str, dict[str, list[float]]] = {
        level: {key: [] for key in LINE_KEYS} for level in TARGET_LEVELS
    }
    with raw_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            level = row["vertebra"]
            if level not in TARGET_LEVELS:
                continue
            for key in LINE_KEYS:
                val = row.get(key, "")
                if val:
                    data[level][key].append(float(val))

    fig, axes = plt.subplots(len(TARGET_LEVELS), len(LINE_KEYS), figsize=(16, 10))
    bins = np.arange(0, 185, 5)

    for row_i, level in enumerate(TARGET_LEVELS):
        for col_i, line_key in enumerate(LINE_KEYS):
            ax = axes[row_i][col_i]
            vals = data[level][line_key]
            if not vals:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            arr = np.array(vals)
            color_rgb = tuple(c / 255 for c in LINE_COLORS_RGB[line_key])
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            outlier_mask = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)

            ax.hist(
                arr[~outlier_mask],
                bins=bins,
                color=color_rgb,
                alpha=0.8,
                label=f"正常 n={int((~outlier_mask).sum())}",
            )
            if outlier_mask.any():
                ax.hist(
                    arr[outlier_mask],
                    bins=bins,
                    color="red",
                    alpha=0.7,
                    label=f"外れ値 n={int(outlier_mask.sum())} ({outlier_mask.mean() * 100:.1f}%)",
                )

            ax.axvline(
                float(np.median(arr)),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"中央値 {np.median(arr):.1f}°",
            )
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
    """IQR 法で外れ値/正常の study_uid を返す。"""
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
        outlier_ids = [
            outlier_ids[i]
            for i in rng.choice(len(outlier_ids), n_samples, replace=False)
        ]
    if len(normal_ids) > n_samples:
        normal_ids = [
            normal_ids[i] for i in rng.choice(len(normal_ids), n_samples, replace=False)
        ]
    return outlier_ids, normal_ids


def _draw_panel(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    lines: dict,
    highlight_line: str | None = None,
    alpha: float = 0.40,
) -> np.ndarray:
    """CT + 線 | 4 領域のパネル (H, 2W+sep, 3) を返す。"""
    left = ct_to_bgr(ct_slice)
    draw_lines_on_image(left, lines, highlight_line=highlight_line)
    if highlight_line:
        info = lines.get(highlight_line)
        if info is not None:
            ang = (
                info.angle_deg
                if hasattr(info, "angle_deg")
                else info.get("angle_deg", 0)
            )
            cv2.putText(
                left,
                f"{ang:.1f}",
                (4, IMAGE_SIZE - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                LINE_COLORS_BGR[highlight_line],
                1,
                cv2.LINE_AA,
            )

    right = ct_to_bgr(ct_slice)
    polylines = lines_to_polylines(lines)
    if polylines is not None:
        vertebra_mask = (mask_slice > 0.5).astype(np.uint8)
        try:
            seg, _ = generate_region_mask(
                line_1=polylines["line_1"],
                line_2=polylines["line_2"],
                line_3=polylines["line_3"],
                line_4=polylines["line_4"],
                vertebra_mask=vertebra_mask,
            )
            label = np.argmax(seg, axis=0).astype(np.uint8)
            ct_u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
            right = make_region_overlay(ct_u8, label, alpha)
            for k, pts in polylines.items():
                c = LINE_COLORS_BGR[k]
                (x1, y1), (x2, y2) = pts
                cv2.line(
                    right,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    c,
                    2,
                )
        except Exception:
            pass

    return concat_with_separator([left, right], axis=1, sep_width=3)


def build_sample_grid(
    study_ids: list[str],
    level: str,
    highlight_line: str,
    models: list,
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
        ct_f = seg_ct[CENTER_CHANNEL].astype(np.float32) / 255.0
        mask_f = seg_mask[CENTER_CHANNEL].astype(np.float32)
        _, lines = predict_single_slice(
            models, ct_f, mask_f, level, device, avg_lengths
        )
        panel = _draw_panel(ct_f, mask_f, lines, highlight_line)

        info = lines.get(highlight_line)
        ang_str = f"{info.angle_deg:.1f}" if info else "None"
        short_uid = study_id.split(".")[-1]
        title_h = 22
        title_bar = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_bar,
            f"{short_uid} {ang_str}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.concatenate([title_bar, panel], axis=0))

    if not panels:
        return None

    canvas = concat_with_separator(panels, axis=1, sep_width=4, sep_value=40)
    header_h = 30
    header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
    color = (100, 220, 100) if label == "正常" else (80, 80, 255)
    cv2.putText(
        header,
        f"[{label}] {level} / {highlight_line}",
        (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, canvas], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="C1/C2/C7 角度外れ値可視化")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=TRAINING_DATASET_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "angle_outlier_vis",
    )
    parser.add_argument(
        "--raw-csv", type=Path, default=ANGLE_STATS_DIR / "angles_raw.csv"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--target-lines", nargs="*", default=["line_2", "line_4"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("\n=== Phase 1: 角度ヒストグラム ===", flush=True)
    plot_angle_histograms(args.raw_csv, args.output_dir / "angle_dist_C1C2C7.png")

    print("\n=== Phase 2: 外れ値/正常サンプル可視化 ===", flush=True)
    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了", flush=True)
    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)

    for level in TARGET_LEVELS:
        for line_key in args.target_lines:
            print(f"  {level} / {line_key} ...", flush=True)
            outlier_ids, normal_ids = collect_outlier_and_normal_ids(
                args.raw_csv,
                level,
                line_key,
                args.n_samples,
            )

            grids = []
            for ids, lbl in [(outlier_ids, "外れ値"), (normal_ids, "正常")]:
                if ids:
                    g = build_sample_grid(
                        ids, level, line_key, models, device, avg_lengths, lbl
                    )
                    if g is not None:
                        grids.append(g)

            if not grids:
                continue

            # 幅を揃えて縦結合
            max_w = max(g.shape[1] for g in grids)
            padded = []
            for g in grids:
                if g.shape[1] < max_w:
                    pad = np.zeros((g.shape[0], max_w - g.shape[1], 3), dtype=np.uint8)
                    g = np.concatenate([g, pad], axis=1)
                padded.append(g)
            sep_h = np.full((6, max_w, 3), 80, dtype=np.uint8)
            combined = padded[0]
            for g in padded[1:]:
                combined = np.concatenate([combined, sep_h, g], axis=0)

            save_path = args.output_dir / f"{level}_{line_key}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), combined)
            print(f"    [SAVED] {save_path.relative_to(PROJECT_ROOT)}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
