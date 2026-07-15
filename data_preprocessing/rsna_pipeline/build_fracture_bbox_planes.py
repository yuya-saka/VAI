"""Precompute fracture bbox -> classifier-plane projections into a single CSV.

Each RSNA bounding box is assigned to exactly one fracture-positive vertebra by
the preprocessing pipeline. The complete row-level assignment is persisted in
``processing_metadata/*.json`` as ``assigned_bbox_slice_numbers`` for each
level. Re-deriving the level from contiguous groups or overlapping physical
ranges is forbidden because both approaches lose the canonical assignment.

This script reads the ground-truth assignment, projects every bbox row onto the
owning level's 15 classifier planes (the same geometry used to build
``fracture_dataset``), and writes one row per bbox into
``fracture_bbox_planes.csv``. Downstream tools (the annotation server, viewers)
just read this CSV and never recompute the assignment.

Usage:
    uv run python data_preprocessing/rsna_pipeline/build_fracture_bbox_planes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
META_DIR = DATA_DIR / "processing_metadata"
BBOX_CSV = DATA_DIR / "train_bounding_boxes.csv"
OUTPUT_CSV = DATA_DIR / "fracture_bbox_planes.csv"

SAMPLING_PIXEL_SPACING_MM = 0.4
OUTPUT_SIZE = 224
HALF_PX = (OUTPUT_SIZE - 1) / 2.0  # 111.5
LEVEL_COLS = tuple(f"C{i}" for i in range(1, 8))

OUTPUT_COLUMNS = [
    "study_id",
    "level",
    "plane_index",
    "slice_number",
    "x",
    "y",
    "width",
    "height",
    "row_min",
    "col_min",
    "row_max",
    "col_max",
]


class BboxProjectionError(ValueError):
    """Raised when canonical bbox assignment or projection data is invalid."""


def _build_slice_mapping(slices_meta: list[dict]) -> dict[int, int]:
    """Map slice_number -> slices_meta index (file stem first, instance fallback)."""
    result: dict[int, int] = {}
    for idx, s in enumerate(slices_meta):
        stem = Path(s["source_file"]).stem
        try:
            result[int(stem)] = idx
        except ValueError:
            pass
        if s["instance_number"] is not None:
            result.setdefault(int(s["instance_number"]), idx)
    return result


def _assigned_slice_to_level(meta: dict) -> dict[int, str]:
    """Read the canonical {bbox_slice_number: level} assignment."""
    mapping: dict[int, str] = {}
    for level in LEVEL_COLS:
        vertebra = meta.get("vertebrae", {}).get(level)
        if not vertebra:
            continue
        classifier_planes = vertebra["classifier_planes"]
        if "assigned_bbox_slice_numbers" not in classifier_planes:
            raise BboxProjectionError(
                f"Metadata is missing assigned_bbox_slice_numbers for {level}"
            )
        for raw_slice_number in classifier_planes["assigned_bbox_slice_numbers"]:
            slice_number = int(raw_slice_number)
            previous_level = mapping.get(slice_number)
            if previous_level is not None:
                raise BboxProjectionError(
                    f"BBox slice {slice_number} is assigned to both "
                    f"{previous_level} and {level}"
                )
            mapping[slice_number] = level
    return mapping


def _project_bbox_to_plane(
    x: float,
    y: float,
    w: float,
    h: float,
    image_position_lps: list[float],
    row_dir: np.ndarray,
    col_dir: np.ndarray,
    ps_row: float,
    ps_col: float,
    plane: dict,
) -> tuple[float, float, float, float]:
    """Project a DICOM bbox onto a classifier plane's 224x224 space.

    Returns (row_min, col_min, row_max, col_max) in pixels (unclipped).
    """
    image_pos = np.array(image_position_lps, dtype=np.float64)
    center = np.array(plane["center_lps_mm"], dtype=np.float64)
    row_basis = np.array(plane["row_basis_lps"], dtype=np.float64)
    col_basis = np.array(plane["column_basis_lps"], dtype=np.float64)

    # train_bounding_boxes: x = column direction, y = row direction.
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    rows_px: list[float] = []
    cols_px: list[float] = []
    for cx, cy in corners:
        lps = image_pos + cx * ps_col * row_dir + cy * ps_row * col_dir
        delta = lps - center
        rows_px.append(
            float(np.dot(delta, col_basis)) / SAMPLING_PIXEL_SPACING_MM + HALF_PX
        )
        cols_px.append(
            float(np.dot(delta, row_basis)) / SAMPLING_PIXEL_SPACING_MM + HALF_PX
        )
    return min(rows_px), min(cols_px), max(rows_px), max(cols_px)


def _process_study(
    study_id: str,
    study_bboxes: pd.DataFrame,
    meta: dict,
) -> list[dict]:
    """Project one study's canonically assigned bbox rows."""
    rows: list[dict] = []

    slices_meta = meta["dicom_geometry"]["slices"]
    row_dir = np.array(meta["dicom_geometry"]["row_direction_lps"], dtype=np.float64)
    col_dir = np.array(meta["dicom_geometry"]["column_direction_lps"], dtype=np.float64)
    ps_row, ps_col = meta["dicom_geometry"]["pixel_spacing_row_column_mm"]

    sn_to_idx = _build_slice_mapping(slices_meta)
    slice_to_level = _assigned_slice_to_level(meta)

    planes_by_level = {
        level: meta["vertebrae"][level]["classifier_planes"]["planes"]
        for level in LEVEL_COLS
        if meta.get("vertebrae", {}).get(level)
    }
    plane_positions_by_level = {
        level: np.array([p["position_mm"] for p in planes])
        for level, planes in planes_by_level.items()
    }
    # 各レベルのnormal_lps（椎体ごとに±35°まで傾く）。nearest-plane判定は、
    # スキャナの生z(slice_position_mm)ではなく必ずこのレベル固有の傾いた
    # normalへの投影値で行う。生zとの単純比較は傾きが大きい椎体で数十mmずれ、
    # bboxが端のプレーン(p0等、骨マスクがほぼ無い)に誤集約される。
    normal_by_level = {
        level: np.array(planes[0]["normal_lps"], dtype=np.float64)
        for level, planes in planes_by_level.items()
    }

    for _, bbox in study_bboxes.iterrows():
        slice_number = int(bbox["slice_number"])
        level = slice_to_level.get(slice_number)
        if level is None:
            raise BboxProjectionError(
                f"{study_id}: bbox slice {slice_number} has no canonical level"
            )
        planes = planes_by_level[level]
        plane_positions = plane_positions_by_level[level]
        normal = normal_by_level[level]
        idx = sn_to_idx.get(slice_number)
        if idx is None:
            raise BboxProjectionError(
                f"{study_id}: bbox slice {slice_number} has no DICOM geometry"
            )
        image_pos = np.array(
            slices_meta[idx]["image_position_lps_mm"], dtype=np.float64
        )
        center_x = float(bbox["x"]) + float(bbox["width"]) / 2.0
        center_y = float(bbox["y"]) + float(bbox["height"]) / 2.0
        bbox_point_lps = (
            image_pos + center_x * ps_col * row_dir + center_y * ps_row * col_dir
        )
        projected_position = float(np.dot(bbox_point_lps, normal))
        plane_index = int(np.argmin(np.abs(plane_positions - projected_position)))
        r_min, c_min, r_max, c_max = _project_bbox_to_plane(
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["width"]),
            float(bbox["height"]),
            image_pos,
            row_dir,
            col_dir,
            ps_row,
            ps_col,
            planes[plane_index],
        )
        rows.append(
            {
                "study_id": study_id,
                "level": level,
                "plane_index": plane_index,
                "slice_number": slice_number,
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "width": float(bbox["width"]),
                "height": float(bbox["height"]),
                "row_min": r_min,
                "col_min": c_min,
                "row_max": r_max,
                "col_max": c_max,
            }
        )
    return rows


def build(bbox_csv: Path, meta_dir: Path, output_csv: Path) -> None:
    """Build the fracture bbox -> plane projection CSV."""
    bbox_df = pd.read_csv(bbox_csv)
    studies = sorted(bbox_df["StudyInstanceUID"].unique())

    all_rows: list[dict] = []
    for study_id in studies:
        meta_path = meta_dir / f"{study_id}.json"
        if not meta_path.exists():
            raise BboxProjectionError(f"Metadata not found: {meta_path}")
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_id]
        all_rows.extend(_process_study(study_id, study_bboxes, meta))

    out_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    out_df = out_df.sort_values(["study_id", "level", "plane_index", "slice_number"])
    out_df = out_df.reset_index(drop=True)
    if len(out_df) != len(bbox_df):
        raise BboxProjectionError(
            f"Projected row count mismatch: {len(out_df)} != {len(bbox_df)}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_csv.with_suffix(f"{output_csv.suffix}.tmp")
    out_df.to_csv(temporary_path, index=False)
    temporary_path.replace(output_csv)

    print(f"studies processed: {len(studies)}")
    print(f"bbox rows written: {len(out_df)}")
    print(f"study/level pairs: {out_df.groupby(['study_id', 'level']).ngroups}")
    print(f"output: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fracture bbox plane projections CSV"
    )
    parser.add_argument("--bbox-csv", type=Path, default=BBOX_CSV)
    parser.add_argument("--meta-dir", type=Path, default=META_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    build(args.bbox_csv, args.meta_dir, args.output_csv)


if __name__ == "__main__":
    main()
