"""SDF 4 領域分割の可視化。

region_4class.npy の結果を CT 画像に重ねて表示する。
各椎体レベルの 15 プレーンを横に並べて 1 枚の PNG に保存。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    FRACTURE_DATASET_DIR,
    PROCESSING_METADATA_DIR,
    PROJECT_ROOT,
    VERTEBRA_LEVELS,
)
from .metadata import load_max_area_indices
from .visualization import add_title_bar, concat_with_separator, make_region_overlay


def visualize_study_level(
    study_id: str,
    vertebra: str,
    level_dir: Path,
    output_dir: Path,
    max_area_idx: int,
) -> Path | None:
    """1 椎体レベルの 15 プレーンを横並びで可視化して PNG に保存する。"""
    ct_path = level_dir / "ct.npy"
    region_path = level_dir / "region_4class.npy"
    if not ct_path.exists() or not region_path.exists():
        return None

    ct = np.load(ct_path)
    region = np.load(region_path)

    panels = []
    for i in range(15):
        overlay = make_region_overlay(ct[i, 2], region[i])
        if i == max_area_idx:
            cv2.rectangle(overlay, (0, 0), (223, 223), (0, 255, 255), 3)
        cv2.putText(
            overlay,
            str(i),
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(overlay)

    row = concat_with_separator(panels, axis=1)
    short_id = study_id.split(".")[-1]
    final = add_title_bar(row, f"{short_id}  {vertebra}  (yellow=max_area)")

    out_path = output_dir / f"{short_id}_{vertebra}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SDF 4 領域分割の可視化")
    parser.add_argument(
        "--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR
    )
    parser.add_argument("--metadata-dir", type=Path, default=PROCESSING_METADATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Unet" / "outputs" / "sdf_segmentation_vis",
    )
    parser.add_argument("--n-studies", type=int, default=5)
    parser.add_argument("--study-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.study_id:
        study_ids = [args.study_id]
    else:
        all_studies = sorted(
            d.name for d in args.fracture_dataset_dir.iterdir() if d.is_dir()
        )
        has_region = [
            s
            for s in all_studies
            if any(
                (args.fracture_dataset_dir / s / lv / "region_4class.npy").exists()
                for lv in VERTEBRA_LEVELS
            )
        ]
        rng = np.random.RandomState(args.seed)
        n = min(args.n_studies, len(has_region))
        idx = rng.choice(len(has_region), size=n, replace=False)
        study_ids = [has_region[i] for i in sorted(idx)]

    print(f"[INFO] 可視化対象: {len(study_ids)} study")
    saved = []
    for study_id in study_ids:
        max_area_indices = load_max_area_indices(study_id, args.metadata_dir)
        for vertebra in VERTEBRA_LEVELS:
            level_dir = args.fracture_dataset_dir / study_id / vertebra
            out = visualize_study_level(
                study_id,
                vertebra,
                level_dir,
                args.output_dir,
                max_area_indices[vertebra],
            )
            if out:
                saved.append(out)

    print(f"[DONE] {len(saved)} 枚保存 → {args.output_dir}")
    for p in saved:
        print(f"  {p.resolve().relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
