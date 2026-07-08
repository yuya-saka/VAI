"""Inference and plotting helpers for the Stage2 region-head notebook."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_MATPLOTLIB_CACHE = Path("/tmp") / f"vai-matplotlib-{os.getuid()}"
_MATPLOTLIB_CACHE.mkdir(mode=0o700, parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from train_models.stage2.src.model import (
    STAGE2_ARCHITECTURE_VERSION,
    Stage2Model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "train_models/stage2/outputs/baseline/v1"
DEFAULT_BBOX_CSV_PATH = PROJECT_ROOT / "data/rsna_data/train_bounding_boxes.csv"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data/rsna_data/processing_metadata"
REGION_NAMES = ("body", "right_foramen", "left_foramen", "posterior")
REGION_LABELS = ("Body", "Right foramen", "Left foramen", "Posterior")
REGION_COLORS = np.asarray(
    [
        [0.95, 0.25, 0.20],
        [0.10, 0.70, 0.95],
        [0.20, 0.85, 0.35],
        [1.00, 0.65, 0.10],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RegionInference:
    """Per-plane and aggregate predictions from one Stage2 checkpoint."""

    primary_plane_probabilities: NDArray[np.float32]
    primary_probability: float
    region_probabilities: NDArray[np.float32]
    region_valid: NDArray[np.bool_]
    any_region_probabilities: NDArray[np.float32]
    region_evidence: NDArray[np.float32]
    region_probability: float


def load_effective_config(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> dict[str, Any]:
    """Load the effective configuration saved with a Stage2 experiment."""
    config_path = outputs_dir / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"invalid config: {config_path}")
    return config


def resolve_dataset_dir(
    config: dict[str, Any],
    dataset_dir: Path | None = None,
) -> Path:
    """Resolve the original Stage2 dataset directory."""
    if dataset_dir is not None:
        return dataset_dir
    configured = Path(str(config["data"]["dataset_dir"]))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def build_model(config: dict[str, Any], device: torch.device) -> Stage2Model:
    """Construct the Stage2 model without requesting pretrained weights."""
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    return Stage2Model(
        backbone=str(model_config.get("backbone", "tf_efficientnetv2_s")),
        in_chans=int(data_config.get("in_channels", 6)),
        n_slices=int(data_config.get("n_slices", 15)),
        n_regions=int(data_config.get("n_regions", 4)),
        drop_rate=float(model_config.get("drop_rate", 0.0)),
        drop_path_rate=float(model_config.get("drop_path_rate", 0.0)),
        drop_rate_last=float(model_config.get("drop_rate_last", 0.3)),
        lstm_hidden=int(model_config.get("lstm_hidden", 256)),
        lstm_layers=int(model_config.get("lstm_layers", 2)),
        fpn_channels=int(model_config.get("fpn_channels", 256)),
        region_hidden=int(model_config.get("region_hidden", 256)),
        region_layers=int(model_config.get("region_layers", 2)),
        region_dropout=float(model_config.get("region_dropout", 0.3)),
        pretrained=False,
        force_primary_fp32=bool(training_config.get("force_primary_fp32", True)),
    ).to(device)


def load_model(
    fold: int,
    device: torch.device,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> tuple[Stage2Model, dict[str, Any]]:
    """Load one fold's AUROC-best Stage2 checkpoint."""
    config = load_effective_config(outputs_dir)
    model = build_model(config, device)
    checkpoint_path = outputs_dir / f"fold{fold}" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture_version = checkpoint.get("architecture_version")
    if architecture_version != STAGE2_ARCHITECTURE_VERSION:
        raise ValueError(
            f"incompatible architecture_version={architecture_version!r}; "
            f"expected {STAGE2_ARCHITECTURE_VERSION}"
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def build_oof_table(
    fold: int,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return one fold's OOF predictions with region-head categories."""
    frame = pd.read_csv(outputs_dir / "oof_predictions.csv")
    frame = frame[frame["fold"] == fold].copy()
    positive = frame["region_pred_prob"] >= threshold
    frame["region_category"] = np.select(
        [
            (frame["label"] == 1) & positive,
            (frame["label"] == 1) & ~positive,
            (frame["label"] == 0) & positive,
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return frame.reset_index(drop=True)


def select_samples(
    oof_table: pd.DataFrame,
    category: str,
    limit: int,
    bbox_only: bool = False,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> pd.DataFrame:
    """Select representative samples for one region-head outcome category."""
    if category not in {"TP", "FN", "FP", "TN"}:
        raise ValueError(f"invalid category: {category}")
    ascending = category in {"FN", "TN"}
    selected = oof_table[oof_table["region_category"] == category].copy()
    if bbox_only:
        selected = selected[
            selected.apply(
                lambda row: bool(
                    get_bbox_plane_indices(
                        str(row["study_uid"]),
                        str(row["vertebra"]),
                        metadata_dir,
                    )
                ),
                axis=1,
            )
        ]
    return (
        selected.sort_values("region_pred_prob", ascending=ascending)
        .head(limit)
        .reset_index(drop=True)
    )


@lru_cache(maxsize=4096)
def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load and cache one preprocessing metadata document."""
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata if isinstance(metadata, dict) else {}


def get_bbox_plane_indices(
    study_uid: str,
    vertebra: str,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> set[int]:
    """Return classifier-plane indices intersecting a source-space bbox."""
    metadata = _load_metadata(metadata_dir / f"{study_uid}.json")
    vertebra_data = metadata.get("vertebrae", {}).get(vertebra, {})
    planes = vertebra_data.get("classifier_planes", {}).get("planes", [])
    return {
        int(plane["sequence_index"])
        for plane in planes
        if plane.get("bbox_slice_numbers")
    }


@lru_cache(maxsize=4)
def _load_bbox_table(bbox_csv_path: Path) -> pd.DataFrame:
    """Load and cache source-space fracture bounding boxes."""
    if not bbox_csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(bbox_csv_path)


def get_bbox_overlays(
    study_uid: str,
    vertebra: str,
    bbox_csv_path: Path = DEFAULT_BBOX_CSV_PATH,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> dict[int, list[NDArray[np.float64]]]:
    """Project source DICOM bounding boxes onto the 15 classifier planes."""
    metadata = _load_metadata(metadata_dir / f"{study_uid}.json")
    vertebra_data = metadata.get("vertebrae", {}).get(vertebra, {})
    planes = vertebra_data.get("classifier_planes", {}).get("planes", [])
    bbox_table = _load_bbox_table(bbox_csv_path)
    if not planes or bbox_table.empty:
        return {}

    study_boxes = bbox_table[bbox_table["StudyInstanceUID"].astype(str) == study_uid]
    if study_boxes.empty:
        return {}

    sampling = vertebra_data.get("sampling", {})
    output_size = sampling.get("output_size_row_column", [224, 224])
    output_spacing = sampling.get("pixel_spacing_row_column_mm", [0.4, 0.4])
    height, width = int(output_size[0]), int(output_size[1])
    row_spacing, column_spacing = float(output_spacing[0]), float(output_spacing[1])

    dicom_geometry = metadata.get("dicom_geometry", {})
    row_direction = np.asarray(
        dicom_geometry.get("row_direction_lps"), dtype=np.float64
    )
    column_direction = np.asarray(
        dicom_geometry.get("column_direction_lps"), dtype=np.float64
    )
    pixel_spacing = dicom_geometry.get("pixel_spacing_row_column_mm", [1.0, 1.0])
    dicom_row_spacing = float(pixel_spacing[0])
    dicom_column_spacing = float(pixel_spacing[1])
    origins = {
        int(Path(slice_data["source_file"]).stem): np.asarray(
            slice_data["image_position_lps_mm"],
            dtype=np.float64,
        )
        for slice_data in dicom_geometry.get("slices", [])
    }

    overlays: dict[int, list[NDArray[np.float64]]] = {}
    for plane in planes:
        sequence_index = int(plane["sequence_index"])
        slice_numbers = {int(value) for value in plane.get("bbox_slice_numbers", [])}
        if not slice_numbers:
            continue

        center = np.asarray(plane["center_lps_mm"], dtype=np.float64)
        plane_row = np.asarray(plane["row_basis_lps"], dtype=np.float64)
        plane_column = np.asarray(plane["column_basis_lps"], dtype=np.float64)
        matching_boxes = study_boxes[study_boxes["slice_number"].isin(slice_numbers)]
        for _, row in matching_boxes.iterrows():
            slice_number = int(row["slice_number"])
            origin = origins.get(slice_number)
            if origin is None:
                continue

            x_min = float(row["x"])
            y_min = float(row["y"])
            x_max = x_min + float(row["width"])
            y_max = y_min + float(row["height"])
            corners = np.asarray(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ],
                dtype=np.float64,
            )
            patient_points = np.asarray(
                [
                    origin
                    + x * dicom_column_spacing * row_direction
                    + y * dicom_row_spacing * column_direction
                    for x, y in corners
                ],
                dtype=np.float64,
            )
            deltas = patient_points - center
            column_offsets = deltas @ plane_row
            row_offsets = deltas @ plane_column
            polygon = np.stack(
                [
                    column_offsets / column_spacing + (width - 1) / 2.0,
                    row_offsets / row_spacing + (height - 1) / 2.0,
                ],
                axis=1,
            )
            overlays.setdefault(sequence_index, []).append(polygon)
    return overlays


def load_sample_arrays(
    study_uid: str,
    vertebra: str,
    dataset_dir: Path,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """Load CT, vertebra mask, and four-class region mask arrays."""
    sample_dir = dataset_dir / study_uid / vertebra
    ct = np.load(sample_dir / "ct.npy", allow_pickle=False)
    vertebra_mask = np.load(sample_dir / "vertebra_mask.npy", allow_pickle=False)
    region_mask = np.load(sample_dir / "region_4class.npy", allow_pickle=False)
    if ct.shape[:2] != (15, 5):
        raise ValueError(f"invalid CT shape: {ct.shape}")
    if vertebra_mask.shape != (15, 224, 224):
        raise ValueError(f"invalid vertebra mask shape: {vertebra_mask.shape}")
    if region_mask.shape != (15, 224, 224):
        raise ValueError(f"invalid region mask shape: {region_mask.shape}")
    return ct, vertebra_mask, region_mask


def summarize_region_logits(
    primary_logits: NDArray[np.floating[Any]],
    region_logits: NDArray[np.floating[Any]],
    region_valid: NDArray[np.bool_],
) -> RegionInference:
    """Convert raw logits into the probabilities visualized by the notebook."""
    if region_logits.shape != region_valid.shape:
        raise ValueError("region_logits and region_valid must have identical shapes")
    if region_logits.ndim != 2 or region_logits.shape[1] != len(REGION_NAMES):
        raise ValueError("region logits must have shape [planes, 4]")
    if primary_logits.shape != (region_logits.shape[0],):
        raise ValueError("primary logits must have shape [planes]")

    primary_probabilities = (
        1.0 / (1.0 + np.exp(-np.clip(primary_logits, -60.0, 60.0)))
    ).astype(np.float32)
    probabilities = (1.0 / (1.0 + np.exp(-np.clip(region_logits, -60.0, 60.0)))).astype(
        np.float32
    )
    survival = np.prod(np.where(region_valid, 1.0 - probabilities, 1.0), axis=1)
    any_region = (1.0 - survival).astype(np.float32)
    plane_valid = region_valid.any(axis=1)

    valid_counts = region_valid.sum(axis=0)
    evidence = np.divide(
        np.where(region_valid, probabilities, 0.0).sum(axis=0),
        valid_counts,
        out=np.zeros(len(REGION_NAMES), dtype=np.float32),
        where=valid_counts > 0,
    ).astype(np.float32)
    region_probability = (
        float(any_region[plane_valid].mean()) if plane_valid.any() else 0.0
    )
    return RegionInference(
        primary_plane_probabilities=primary_probabilities,
        primary_probability=float(primary_probabilities.mean()),
        region_probabilities=probabilities,
        region_valid=region_valid,
        any_region_probabilities=any_region,
        region_evidence=evidence,
        region_probability=region_probability,
    )


def infer_regions(
    model: Stage2Model,
    ct: NDArray[np.uint8],
    vertebra_mask: NDArray[np.uint8],
    region_mask: NDArray[np.uint8],
    device: torch.device,
) -> RegionInference:
    """Run one sample and return all values needed for visualization."""
    images = np.concatenate([ct, vertebra_mask[:, None]], axis=1)
    image_tensor = torch.from_numpy(images).unsqueeze(0).to(device).float() / 255.0
    region_tensor = torch.from_numpy(region_mask).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(image_tensor, region_tensor)
    if output.region_logits is None or output.region_plane_valid is None:
        raise RuntimeError("model did not return region outputs")
    return summarize_region_logits(
        output.slice_logits[0].float().cpu().numpy(),
        output.region_logits[0].float().cpu().numpy(),
        output.region_plane_valid[0].cpu().numpy(),
    )


def compose_region_overlay(
    ct_plane: NDArray[np.integer[Any]],
    region_mask: NDArray[np.integer[Any]],
    region_probabilities: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    """Blend fixed region colors with alpha proportional to model evidence."""
    if ct_plane.shape != region_mask.shape:
        raise ValueError("ct_plane and region_mask must have identical shapes")
    if region_probabilities.shape != (len(REGION_NAMES),):
        raise ValueError("region_probabilities must have shape [4]")

    gray = np.clip(ct_plane.astype(np.float32) / 255.0, 0.0, 1.0)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    for region_index, color in enumerate(REGION_COLORS, start=1):
        selected = region_mask == region_index
        if not selected.any():
            continue
        probability = float(np.clip(region_probabilities[region_index - 1], 0.0, 1.0))
        alpha = 0.12 + 0.58 * probability
        overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def compose_single_region_overlay(
    ct_plane: NDArray[np.integer[Any]],
    region_mask: NDArray[np.integer[Any]],
    region_index: int,
    score: float,
) -> NDArray[np.float32]:
    """Overlay one anatomical region with opacity proportional to its score."""
    if ct_plane.shape != region_mask.shape:
        raise ValueError("ct_plane and region_mask must have identical shapes")
    if not 0 <= region_index < len(REGION_NAMES):
        raise ValueError(f"invalid region_index: {region_index}")

    gray = np.clip(ct_plane.astype(np.float32) / 255.0, 0.0, 1.0)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    selected = region_mask == region_index + 1
    if not selected.any():
        return overlay
    alpha = 0.12 + 0.58 * float(np.clip(score, 0.0, 1.0))
    color = REGION_COLORS[region_index]
    overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def _draw_bbox_polygons(
    axis: plt.Axes,
    polygons: list[NDArray[np.float64]],
) -> None:
    """Draw projected fracture bounding boxes on one classifier plane."""
    for polygon in polygons:
        closed_polygon = np.vstack([polygon, polygon[0]])
        axis.plot(
            closed_polygon[:, 0],
            closed_polygon[:, 1],
            color="yellow",
            linewidth=2.2,
        )


def build_bbox_region_figure(
    study_uid: str,
    vertebra: str,
    label: int,
    ct: NDArray[np.uint8],
    region_mask: NDArray[np.uint8],
    inference: RegionInference,
    bbox_overlays: dict[int, list[NDArray[np.float64]]],
    detection_threshold: float = 0.5,
) -> Figure:
    """Show four separate region scores for every bbox-intersecting plane."""
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("detection_threshold must be in [0, 1]")
    plane_indices = sorted(bbox_overlays)
    if not plane_indices:
        raise ValueError("bbox_overlays must contain at least one plane")

    figure, axes = plt.subplots(
        len(plane_indices),
        len(REGION_NAMES),
        figsize=(
            4.0 * len(REGION_NAMES),
            4.0 * len(plane_indices) + 1.4,
        ),
        squeeze=False,
    )
    for row_index, plane_index in enumerate(plane_indices):
        for region_index, (region_label, color) in enumerate(
            zip(REGION_LABELS, REGION_COLORS, strict=True)
        ):
            axis = axes[row_index, region_index]
            score = float(inference.region_probabilities[plane_index, region_index])
            valid = bool(inference.region_valid[plane_index, region_index])
            detected = valid and score >= detection_threshold
            overlay = compose_single_region_overlay(
                ct[plane_index, 2],
                region_mask[plane_index],
                region_index,
                score,
            )
            axis.imshow(overlay)
            region_pixels = region_mask[plane_index] == region_index + 1
            if region_pixels.any():
                axis.contour(
                    region_pixels,
                    levels=[0.5],
                    colors=[color],
                    linewidths=1.2,
                )
            _draw_bbox_polygons(axis, bbox_overlays[plane_index])
            status = "DETECTED" if detected else "not detected"
            score_text = f"{score:.3f}" if valid else "N/A"
            axis.set_title(
                f"plane {plane_index:02d} | {region_label}\n"
                f"score={score_text} | {status}",
                fontsize=9,
                color=color if detected else "black",
                fontweight="bold" if detected else "normal",
            )
            for spine in axis.spines.values():
                spine.set_edgecolor(color if detected else "#777777")
                spine.set_linewidth(2.5 if detected else 0.8)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f"{study_uid} / {vertebra}  GT={label}  "
        f"region={inference.region_probability:.3f}  "
        f"threshold={detection_threshold:.2f}\n"
        "Yellow: projected GT bbox | score: weakly supervised region evidence",
        fontsize=12,
        y=0.98,
    )
    top = 0.72 if len(plane_indices) == 1 else 0.86
    figure.subplots_adjust(top=top, hspace=0.35, wspace=0.08)
    return figure


def build_region_figure(
    study_uid: str,
    vertebra: str,
    label: int,
    ct: NDArray[np.uint8],
    region_mask: NDArray[np.uint8],
    inference: RegionInference,
    display_indices: list[int] | None = None,
    bbox_overlays: dict[int, list[NDArray[np.float64]]] | None = None,
) -> Figure:
    """Build CT overlays and per-plane region evidence curves."""
    indices = list(range(ct.shape[0])) if display_indices is None else display_indices
    bbox_overlays = {} if bbox_overlays is None else bbox_overlays
    if not indices:
        raise ValueError("display_indices must not be empty")
    columns = min(5, len(indices))
    rows = int(np.ceil(len(indices) / columns))
    figure = plt.figure(figsize=(3.7 * columns, 3.4 * rows + 3.2))
    grid = figure.add_gridspec(rows + 1, columns, height_ratios=[1.0] * rows + [0.8])

    for position, plane_index in enumerate(indices):
        axis = figure.add_subplot(grid[position // columns, position % columns])
        probabilities = inference.region_probabilities[plane_index]
        overlay = compose_region_overlay(
            ct[plane_index, 2],
            region_mask[plane_index],
            probabilities,
        )
        axis.imshow(overlay)
        for region_index, color in enumerate(REGION_COLORS, start=1):
            if np.any(region_mask[plane_index] == region_index):
                axis.contour(
                    region_mask[plane_index] == region_index,
                    levels=[0.5],
                    colors=[color],
                    linewidths=0.7,
                )
        _draw_bbox_polygons(axis, bbox_overlays.get(plane_index, []))
        valid = inference.region_valid[plane_index]
        strongest = int(np.argmax(np.where(valid, probabilities, -1.0)))
        bbox_tag = " [bbox]" if plane_index in bbox_overlays else ""
        axis.set_title(
            f"plane {plane_index:02d}{bbox_tag}  "
            f"any={inference.any_region_probabilities[plane_index]:.2f}\n"
            f"max={REGION_LABELS[strongest]} {probabilities[strongest]:.2f}",
            fontsize=8,
        )
        axis.axis("off")

    for position in range(len(indices), rows * columns):
        axis = figure.add_subplot(grid[position // columns, position % columns])
        axis.axis("off")

    curve_axis = figure.add_subplot(grid[rows, :])
    plane_numbers = np.arange(ct.shape[0])
    for region_index, (label_name, color) in enumerate(
        zip(REGION_LABELS, REGION_COLORS, strict=True)
    ):
        values = np.where(
            inference.region_valid[:, region_index],
            inference.region_probabilities[:, region_index],
            np.nan,
        )
        curve_axis.plot(
            plane_numbers,
            values,
            marker="o",
            linewidth=1.5,
            color=color,
            label=f"{label_name} (mean={inference.region_evidence[region_index]:.2f})",
        )
    curve_axis.plot(
        plane_numbers,
        inference.any_region_probabilities,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Any region (Noisy-OR)",
    )
    curve_axis.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    curve_axis.set(xlabel="Plane index", ylabel="Evidence", ylim=(-0.02, 1.02))
    curve_axis.set_xticks(plane_numbers)
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(ncol=3, fontsize=8, loc="upper center")

    legend_handles = [
        Patch(facecolor=color, label=label_name)
        for label_name, color in zip(REGION_LABELS, REGION_COLORS, strict=True)
    ]
    if bbox_overlays:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="yellow", linewidth=2.0, label="GT bbox")
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
    )
    figure.suptitle(
        f"{study_uid} / {vertebra}  GT={label}  "
        f"primary={inference.primary_probability:.3f}  "
        f"region={inference.region_probability:.3f}",
        fontsize=12,
    )
    figure.subplots_adjust(top=0.93, bottom=0.09, hspace=0.35, wspace=0.08)
    return figure


def top_plane_indices(inference: RegionInference, count: int) -> list[int]:
    """Return the highest Noisy-OR planes in anatomical sequence order."""
    if count <= 0:
        raise ValueError("count must be positive")
    selected = np.argsort(inference.any_region_probabilities)[-count:]
    return sorted(int(index) for index in selected)
