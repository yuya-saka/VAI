"""Visualize four regions and corrected bbox on bbox-centred CT planes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np

DATA_DIR = Path("data/rsna_data")
DATASET_DIR = DATA_DIR / "bbox_centered_dataset"
OUTPUT_DIR = Path("Unet/outputs/bbox_centered_vis")

REGION_COLORS_RGB = {
    1: np.array([100, 149, 237], dtype=np.float32),
    2: np.array([50, 205, 50], dtype=np.float32),
    3: np.array([220, 80, 80], dtype=np.float32),
    4: np.array([255, 215, 0], dtype=np.float32),
}
REGION_NAMES = {
    1: "body",
    2: "right_foramen",
    3: "left_foramen",
    4: "posterior",
}


def visualize_run(run_directory: Path, output_path: Path) -> Path:
    """Render bbox-bearing planes from one bbox-centred run."""
    ct = np.load(run_directory / "ct.npy", allow_pickle=False)
    vertebra_mask = np.load(run_directory / "vertebra_mask.npy", allow_pickle=False)
    occupancy = np.load(
        run_directory / "bbox_corrected_occupancy.npy", allow_pickle=False
    )
    region_path = run_directory / "region_4class.npy"
    region_mask = (
        np.load(region_path, allow_pickle=False) if region_path.is_file() else None
    )
    metadata = json.loads((run_directory / "metadata.json").read_text())
    contours = json.loads((run_directory / "bbox_corrected_contours.json").read_text())
    nonzero = np.flatnonzero(occupancy.sum(axis=(1, 2)) > 0).tolist()
    selected = _selected_planes(nonzero, int(metadata["center_plane_index"]))
    figure, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4.5))
    if len(selected) == 1:
        axes = [axes]
    contour_by_plane = {
        int(item["plane_index"]): item["components"] for item in contours
    }
    for axis, plane_index in zip(axes, selected, strict=True):
        image = cv2.cvtColor(ct[plane_index, 2], cv2.COLOR_GRAY2RGB)
        mask = vertebra_mask[plane_index] > 0
        if region_mask is None:
            image[mask] = np.asarray(
                0.8 * image[mask] + 0.2 * np.array([80, 180, 255]),
                dtype=np.uint8,
            )
        else:
            image = _overlay_regions(image, region_mask[plane_index])
        axis.imshow(image)
        axis.imshow(
            occupancy[plane_index],
            cmap="Reds",
            alpha=np.asarray(occupancy[plane_index], dtype=np.float32) / 510.0,
            vmin=0,
            vmax=255,
        )
        for component in contour_by_plane.get(plane_index, []):
            polygon = _simplify_contour(component)
            closed = np.vstack((polygon, polygon[0]))
            axis.plot(closed[:, 1], closed[:, 0], color="yellow", linewidth=1.5)
        axis.set_title(
            f"plane {plane_index}\narea={occupancy[plane_index].sum() / 255:.1f}px"
        )
        axis.axis("off")
    figure.suptitle(
        f"{metadata['study_id'].split('.')[-1]} / {metadata['level']} / "
        f"run {metadata['bbox_run_id']} / geometry={metadata['geometry_mode']}"
    )
    if region_mask is not None:
        figure.legend(
            handles=[
                mpatches.Patch(
                    color=REGION_COLORS_RGB[label] / 255.0,
                    label=f"{label}: {REGION_NAMES[label]}",
                )
                for label in REGION_COLORS_RGB
            ],
            loc="lower center",
            ncol=4,
            frameon=False,
        )
    else:
        figure.text(
            0.5,
            0.02,
            "4領域maskなし（既存QC除外）",
            ha="center",
            color="darkred",
        )
    figure.tight_layout(rect=(0, 0.12 if region_mask is not None else 0.05, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _overlay_regions(
    image: np.ndarray,
    region_mask: np.ndarray,
    *,
    alpha: float = 0.38,
) -> np.ndarray:
    """Overlay the four anatomical regions on one RGB CT image."""
    output = image.astype(np.float32)
    for label, color in REGION_COLORS_RGB.items():
        selected = region_mask == label
        output[selected] = (1.0 - alpha) * output[selected] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _simplify_contour(
    component: list[list[float]],
    *,
    tolerance_px: float = 0.75,
) -> np.ndarray:
    """Remove supersampling stair-steps while retaining the union contour."""
    polygon = np.asarray(component, dtype=np.float32)
    if len(polygon) < 3:
        raise ValueError("BBox contour must have at least three vertices")
    simplified = cv2.approxPolyDP(
        polygon[:, ::-1].reshape(-1, 1, 2),
        epsilon=tolerance_px,
        closed=True,
    ).reshape(-1, 2)
    return np.asarray(simplified[:, ::-1], dtype=np.float64)


def _selected_planes(nonzero: list[int], center_index: int) -> list[int]:
    if not nonzero:
        raise ValueError("Corrected bbox occupancy is empty")
    candidates = sorted(set((*nonzero, center_index)))
    if len(candidates) <= 5:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, 5)
    selected = {candidates[int(round(position))] for position in positions}
    selected.add(center_index)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bbox-centred views")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    runs = sorted(arguments.dataset_dir.glob("*/*/run_*"))
    if arguments.samples <= 0:
        raise ValueError("Sample count must be positive")
    rng = np.random.default_rng(arguments.seed)
    if len(runs) > arguments.samples:
        indices = sorted(
            rng.choice(len(runs), size=arguments.samples, replace=False).tolist()
        )
        runs = [runs[index] for index in indices]
    for run_directory in runs:
        relative = run_directory.relative_to(arguments.dataset_dir)
        study_id, level, run_id = relative.parts
        output_path = arguments.output_dir / (
            f"{study_id.split('.')[-1]}_{level}_{run_id}.png"
        )
        print(visualize_run(run_directory, output_path))


if __name__ == "__main__":
    main()
