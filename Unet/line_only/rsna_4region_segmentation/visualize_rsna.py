"""RSNA データセットの線予測結果を可視化するスクリプト。

各サンプルにつき 1 枚の PNG を出力する:
  [1] CT 原画像  [2] 4ch ヒートマップ  [3] 予測線  [4] 4 領域分割
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

from .constants import (
    CENTER_CHANNEL,
    DEFAULT_CKPT_DIR,
    FRACTURE_DATASET_DIR,
    LINE_COLORS_BGR,
    PROCESSING_METADATA_DIR,
    PROJECT_ROOT,
    TRAINING_DATASET_DIR,
    VERTEBRA_LEVELS,
)
from .inference import predict_single_slice
from .metadata import find_max_area_plane_index, load_metadata
from .model_io import compute_avg_line_lengths, load_models
from .visualization import (
    add_title_bar,
    concat_with_separator,
    ct_to_bgr,
    draw_lines_on_image,
    lines_to_polylines,
    make_heatmap_grid,
    make_region_overlay,
)


def _draw_region_overlay(
    ct_f32: np.ndarray,
    lines: dict[str, Any],
    mask_slice: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """4 領域オーバーレイ + 線を描画した画像を返す。"""
    base = ct_to_bgr(ct_f32)
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
    except Exception:
        return base

    label = np.argmax(seg, axis=0).astype(np.uint8)
    ct_u8 = (np.clip(ct_f32, 0, 1) * 255).astype(np.uint8)
    blended = make_region_overlay(ct_u8, label, alpha)

    for k, pts in polylines.items():
        c = LINE_COLORS_BGR.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = pts
        cv2.line(
            blended,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
    return blended


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
    """4 パネル横並び PNG を保存する。"""
    p1 = ct_to_bgr(ct_slice)
    cv2.putText(
        p1,
        "CT",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    p2 = make_heatmap_grid(ct_slice, heatmaps)
    cv2.putText(
        p2,
        "Heatmaps",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    p3 = ct_to_bgr(ct_slice)
    draw_lines_on_image(p3, lines, draw_centroids=True)
    cv2.putText(
        p3,
        "Pred Lines",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    p4 = _draw_region_overlay(ct_slice, lines, mask_slice)
    cv2.putText(
        p4,
        "4 Regions",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    canvas = concat_with_separator([p1, p2, p3, p4], axis=1, sep_width=3, sep_value=80)
    title = f"{study_uid[:30]}  {vertebra}  plane={plane_index}"
    final = add_title_bar(canvas, title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA 線予測の可視化")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument("--metadata-dir", type=Path, default=PROCESSING_METADATA_DIR)
    parser.add_argument(
        "--training-dataset-dir", type=Path, default=TRAINING_DATASET_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "rsna_line_vis",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--study-ids", nargs="*", default=None)
    parser.add_argument("--n-studies", type=int, default=5)
    parser.add_argument("--vertebrae", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    models = load_models(args.checkpoint_dir, args.n_folds, device)
    print(f"[INFO] {len(models)} fold モデル読み込み完了")
    avg_lengths = compute_avg_line_lengths(args.training_dataset_dir)

    vertebrae = args.vertebrae or VERTEBRA_LEVELS

    if args.study_ids:
        study_uids = args.study_ids
    else:
        study_uids = sorted(
            d.name for d in args.fracture_dataset_dir.iterdir() if d.is_dir()
        )[: args.n_studies]
    print(f"[INFO] 対象スタディ: {len(study_uids)} 件 × 椎体 {vertebrae}")

    for study_uid in study_uids:
        metadata = load_metadata(study_uid, args.metadata_dir)
        if metadata is None:
            print(f"[SKIP] メタデータなし: {study_uid}")
            continue

        study_dir = args.fracture_dataset_dir / study_uid
        for vertebra in vertebrae:
            plane_index = find_max_area_plane_index(metadata, vertebra)
            if plane_index is None:
                continue

            ct_path = study_dir / vertebra / "ct.npy"
            mask_path = study_dir / vertebra / "vertebra_mask.npy"
            if not ct_path.exists() or not mask_path.exists():
                continue

            ct_vol = np.load(ct_path, allow_pickle=False)
            mask_vol = np.load(mask_path, allow_pickle=False)
            ct_slice = ct_vol[plane_index, CENTER_CHANNEL].astype(np.float32) / 255.0
            mask_slice = mask_vol[plane_index].astype(np.float32)

            heatmaps, lines = predict_single_slice(
                models,
                ct_slice,
                mask_slice,
                vertebra,
                device,
                avg_lengths,
            )

            save_path = (
                args.output_dir / study_uid / f"{vertebra}_plane{plane_index:02d}.png"
            )
            visualize_sample(
                ct_slice,
                mask_slice,
                heatmaps,
                lines,
                study_uid,
                vertebra,
                plane_index,
                save_path,
            )
            print(f"[SAVED] {save_path.relative_to(PROJECT_ROOT)}")

    print(f"\n[DONE] {args.output_dir}")


if __name__ == "__main__":
    main()
