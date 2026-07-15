"""Visualize projected fracture bboxes over classifier region masks.

The canonical vertebra assignment and 224x224 projection are read exclusively
from ``fracture_bbox_planes.csv``. This viewer must not derive a level from raw
DICOM ranges or recompute bbox geometry independently.

Usage:
    uv run python -m \
        data_preprocessing.rsna_pipeline.visualize_fracture_region_bbox
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np
import pandas as pd

DATA_DIR = Path("data/rsna_data")
FRACTURE_DATASET_DIR = DATA_DIR / "fracture_dataset"
BBOX_PLANES_CSV = DATA_DIR / "fracture_bbox_planes.csv"
OUTPUT_DIR = Path("Unet/outputs/fracture_region_bbox_vis")

OUTPUT_SIZE = 224
REGION_COLORS_BGR = {
    1: (255, 80, 80),
    2: (80, 255, 80),
    3: (80, 80, 255),
    4: (255, 255, 80),
}


def compute_share_r(
    region_mask: np.ndarray,
    vertebra_mask: np.ndarray,
    bbox_mask: np.ndarray,
) -> dict[int, float]:
    """Return each region's share inside the bbox-vertebra intersection."""
    valid = (bbox_mask > 0) & (vertebra_mask > 0)
    total_valid = int(valid.sum())
    if total_valid == 0:
        return {region: 0.0 for region in range(1, 5)}
    return {
        region: int(((region_mask == region) & valid).sum()) / total_valid
        for region in range(1, 5)
    }


def make_region_overlay(
    ct_plane: np.ndarray,
    region_mask: np.ndarray,
    vertebra_mask: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """Return a BGR CT image with anatomical region colors."""
    bgr = cv2.cvtColor(ct_plane, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    for region, color in REGION_COLORS_BGR.items():
        mask = (region_mask == region) & (vertebra_mask > 0)
        overlay[mask] = color
    return cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)


def group_bboxes_by_plane(
    bbox_planes: pd.DataFrame,
    plane_count: int = 15,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Group projected bbox rectangles by classifier plane index."""
    grouped: dict[int, list[tuple[float, float, float, float]]] = {
        index: [] for index in range(plane_count)
    }
    for _, row in bbox_planes.iterrows():
        plane_index = int(row["plane_index"])
        if plane_index not in grouped:
            raise ValueError(f"Invalid classifier plane index: {plane_index}")
        grouped[plane_index].append(
            (
                float(row["row_min"]),
                float(row["col_min"]),
                float(row["row_max"]),
                float(row["col_max"]),
            )
        )
    return grouped


def visualize_study_level(
    study_id: str,
    level: str,
    bbox_planes: pd.DataFrame,
    fracture_dataset_dir: Path,
    output_dir: Path,
    n_planes_to_show: int = 5,
) -> Path | None:
    """Write one multi-plane visualization for a study-level pair."""
    level_dir = fracture_dataset_dir / study_id / level
    ct = np.load(level_dir / "ct.npy")
    vertebra_mask = np.load(level_dir / "vertebra_mask.npy")
    region_mask = np.load(level_dir / "region_4class.npy")

    plane_bboxes = group_bboxes_by_plane(bbox_planes, plane_count=ct.shape[0])
    planes_with_bbox = [index for index, values in plane_bboxes.items() if values]
    if not planes_with_bbox:
        return None

    step = max(1, len(planes_with_bbox) // n_planes_to_show)
    selected = planes_with_bbox[::step][:n_planes_to_show]
    figure, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]
    figure.suptitle(f"{study_id.split('.')[-1]} / {level}", fontsize=11)

    for axis, plane_index in zip(axes, selected, strict=True):
        visualization = make_region_overlay(
            ct[plane_index, 2],
            region_mask[plane_index],
            vertebra_mask[plane_index],
        )
        axis.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        axis.set_title(f"plane {plane_index}", fontsize=8)
        axis.axis("off")

        bbox_mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
        for row_min, col_min, row_max, col_max in plane_bboxes[plane_index]:
            row_start = max(0, int(np.floor(row_min)))
            column_start = max(0, int(np.floor(col_min)))
            row_stop = min(OUTPUT_SIZE, int(np.ceil(row_max)))
            column_stop = min(OUTPUT_SIZE, int(np.ceil(col_max)))
            if row_start < row_stop and column_start < column_stop:
                bbox_mask[row_start:row_stop, column_start:column_stop] = 1
            axis.add_patch(
                plt.Rectangle(
                    (column_start, row_start),
                    max(0, column_stop - column_start),
                    max(0, row_stop - row_start),
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                )
            )

        shares = compute_share_r(
            region_mask[plane_index],
            vertebra_mask[plane_index],
            bbox_mask,
        )
        share_text = " ".join(
            f"R{region}:{share:.2f}"
            for region, share in shares.items()
            if share > 0
        )
        axis.set_xlabel(share_text, fontsize=7)

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{study_id.split('.')[-1]}_{level}.png"
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_targets(
    bbox_planes: pd.DataFrame,
    fracture_dataset_dir: Path,
) -> list[tuple[str, str]]:
    """Return bbox-bearing study-level pairs with region masks."""
    pairs = bbox_planes[["study_id", "level"]].drop_duplicates()
    return [
        (str(row["study_id"]), str(row["level"]))
        for _, row in pairs.iterrows()
        if (
            fracture_dataset_dir
            / str(row["study_id"])
            / str(row["level"])
            / "region_4class.npy"
        ).is_file()
    ]


def clear_generated_visualizations(output_dir: Path) -> int:
    """Remove PNG files previously generated in the dedicated output directory."""
    if not output_dir.is_dir():
        return 0
    paths = tuple(output_dir.glob("*.png"))
    for path in paths:
        path.unlink()
    return len(paths)


def main() -> None:
    """Generate a deterministic random sample of bbox-region visualizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-planes-csv", type=Path, default=BBOX_PLANES_CSV)
    parser.add_argument("--fracture-dataset-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--clean-output", action="store_true")
    arguments = parser.parse_args()
    if arguments.samples <= 0:
        raise ValueError("Sample count must be positive")

    bbox_planes = pd.read_csv(arguments.bbox_planes_csv)
    targets = build_targets(bbox_planes, arguments.fracture_dataset_dir)
    print(f"targets: {len(targets)}")
    if arguments.clean_output:
        removed_count = clear_generated_visualizations(arguments.output_dir)
        print(f"removed stale visualizations: {removed_count}")

    random_generator = np.random.default_rng(42)
    sample_count = min(arguments.samples, len(targets))
    indices = random_generator.choice(len(targets), size=sample_count, replace=False)
    for index in sorted(indices):
        study_id, level = targets[index]
        level_bboxes = bbox_planes[
            (bbox_planes["study_id"] == study_id)
            & (bbox_planes["level"] == level)
        ]
        output_path = visualize_study_level(
            study_id,
            level,
            level_bboxes,
            arguments.fracture_dataset_dir,
            arguments.output_dir,
        )
        if output_path is not None:
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
