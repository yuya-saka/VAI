"""Grad-CAM and anatomical-region summaries for Baseline 0."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from fracture_detection.baseline0.modeling.model import Baseline0Model
from fracture_detection.common.constants import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
    INPUT_MANIFEST_CSV,
    N_PLANES,
    N_REGIONS,
    REGION_COLUMNS,
)
from fracture_detection.core.factory import build_model

FloatArray = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class GradCamBatch:
    """Grad-CAM output for a batch of vertebra bags."""

    cams: FloatArray
    plane_probabilities: FloatArray
    bag_probabilities: FloatArray


def load_baseline0_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Baseline0Model:
    """Load a Baseline 0 model without requesting pretrained weights."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"Invalid Baseline 0 checkpoint: {checkpoint_path}")
    model_config = copy.deepcopy(config)
    model_config["model"]["pretrained"] = False
    model = build_model(model_config)
    if not isinstance(model, Baseline0Model):
        raise TypeError("Checkpoint does not describe a Baseline 0 model")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_bag_arrays(
    dataset_dir: Path,
    study_id: str,
    level: str,
) -> tuple[UInt8Array, UInt8Array, UInt8Array]:
    """Load the CT, whole mask, and four-region mask for one bag."""
    bag_dir = dataset_dir / study_id / level
    ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
    whole_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
    region_mask = np.load(bag_dir / "region_4class.npy", allow_pickle=False)
    if ct.shape != EXPECTED_CT_SHAPE or ct.dtype != np.uint8:
        raise ValueError(f"Invalid CT array in {bag_dir}: {ct.shape}, {ct.dtype}")
    if whole_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Invalid whole mask in {bag_dir}: {whole_mask.shape}")
    if region_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Invalid region mask in {bag_dir}: {region_mask.shape}")
    region_values = set(np.unique(region_mask).tolist())
    if not region_values.issubset(set(range(N_REGIONS + 1))):
        raise ValueError(f"Invalid region values in {bag_dir}: {region_values}")
    return (
        cast(UInt8Array, ct),
        cast(UInt8Array, whole_mask),
        cast(UInt8Array, region_mask),
    )


def prepare_inputs(ct: UInt8Array, whole_mask: UInt8Array) -> Tensor:
    """Build the normalized six-channel Baseline 0 input tensor."""
    if ct.shape != EXPECTED_CT_SHAPE or whole_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError("CT or whole-mask shape does not match the fixed contract")
    mask_channel = np.rint((whole_mask > 0).astype(np.float32) * 255.0).astype(
        np.uint8
    )[:, None]
    values = np.concatenate([ct, mask_channel], axis=1)
    return torch.from_numpy(np.ascontiguousarray(values)).float().div(255.0)


def compute_gradcam(
    model: Baseline0Model,
    inputs: Tensor,
    device: torch.device,
) -> GradCamBatch:
    """Explain each bag score at the final spatial CNN feature map.

    The target is the exact bag readout, mean(sigmoid(plane_logits)). Summing the
    targets over a batch preserves independent per-sample gradients because the
    model has no cross-sample operations in evaluation mode.
    """
    if inputs.ndim != 5 or inputs.shape[1] != N_PLANES or inputs.shape[2] != 6:
        raise ValueError(f"Expected [B,{N_PLANES},6,H,W], got {inputs.shape}")
    target_layer = cast(nn.Module, model.encoder.bn2)
    activation_holder: dict[str, Tensor] = {}

    def capture_activation(
        _module: nn.Module,
        _arguments: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor")
        output.retain_grad()
        activation_holder["value"] = output

    handle = target_layer.register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    moved = inputs.to(device)
    try:
        with torch.enable_grad():
            logits = model(moved)
            plane_probabilities = logits.sigmoid()
            bag_probabilities = plane_probabilities.mean(dim=1)
            bag_probabilities.sum().backward()
        activations = activation_holder.get("value")
        if activations is None or activations.grad is None:
            raise RuntimeError("Grad-CAM activation or gradient was not captured")
        batch_size, plane_count = logits.shape
        expected_rows = batch_size * plane_count
        if activations.shape[0] != expected_rows:
            raise ValueError(
                "Target activation does not preserve the flattened plane dimension"
            )
        gradients = activations.grad
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        low_resolution = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        resized = F.interpolate(
            low_resolution,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cams = resized.reshape(batch_size, plane_count, *inputs.shape[-2:])
        return GradCamBatch(
            cams=cams.detach().float().cpu().numpy().astype(np.float32),
            plane_probabilities=plane_probabilities.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
            bag_probabilities=bag_probabilities.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32),
        )
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def anatomical_attention_metrics(
    cams: FloatArray,
    whole_mask: UInt8Array,
    region_mask: UInt8Array,
) -> dict[str, float | int | bool]:
    """Measure CAM mass and area-corrected density in each anatomical region."""
    if cams.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Invalid CAM shape: {cams.shape}")
    if whole_mask.shape != cams.shape or region_mask.shape != cams.shape:
        raise ValueError("CAM and mask shapes must match")
    if not np.isfinite(cams).all() or np.any(cams < 0):
        raise ValueError("CAM values must be finite and non-negative")

    whole = whole_mask > 0
    total_mass = float(cams.sum(dtype=np.float64))
    whole_mass = float(cams[whole].sum(dtype=np.float64))
    whole_area = int(whole.sum())
    image_area = int(whole.size)
    whole_density = whole_mass / whole_area if whole_area else float("nan")
    image_density = total_mass / image_area if image_area else float("nan")
    metrics: dict[str, float | int | bool] = {
        "cam_total": total_mass,
        "cam_zero": total_mass <= 0.0,
        "vertebra_area_fraction": _safe_ratio(whole_area, image_area),
        "in_vertebra_mass_fraction": _safe_ratio(whole_mass, total_mass),
        "vertebra_density_enrichment": _safe_ratio(whole_density, image_density),
        "outside_vertebra_mass_fraction": _safe_ratio(
            total_mass - whole_mass, total_mass
        ),
    }

    assigned = np.zeros_like(whole, dtype=bool)
    for region_index, region_column in enumerate(REGION_COLUMNS, start=1):
        region = (region_mask == region_index) & whole
        assigned |= region
        region_area = int(region.sum())
        region_mass = float(cams[region].sum(dtype=np.float64))
        region_density = region_mass / region_area if region_area else float("nan")
        metrics[f"{region_column}_area_fraction"] = _safe_ratio(region_area, whole_area)
        metrics[f"{region_column}_mass_fraction"] = _safe_ratio(region_mass, whole_mass)
        metrics[f"{region_column}_density_enrichment"] = _safe_ratio(
            region_density, whole_density
        )

    unassigned_mass = float(cams[whole & ~assigned].sum(dtype=np.float64))
    plane_masses = cams.sum(axis=(1, 2), dtype=np.float64)
    peak_plane = int(np.argmax(plane_masses))
    metrics.update(
        unassigned_vertebra_mass_fraction=_safe_ratio(unassigned_mass, whole_mass),
        peak_attention_plane=peak_plane,
        peak_attention_plane_fraction=_safe_ratio(
            float(plane_masses[peak_plane]), total_mass
        ),
    )
    return metrics


def load_oof_predictions(experiment_dir: Path) -> pd.DataFrame:
    """Load and validate all five formal outer-fold prediction tables."""
    frames: list[pd.DataFrame] = []
    required = {
        "study_id",
        "level",
        "fold",
        "vertebra_target",
        "vertebra_score",
        "vertebra_prediction",
    }
    for fold in range(5):
        path = experiment_dir / f"outer{fold}" / "outer_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"OOF prediction file does not exist: {path}")
        frame = pd.read_csv(path, dtype={"study_id": str, "level": str})
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing OOF columns in {path}: {sorted(missing)}")
        if not frame["fold"].eq(fold).all():
            raise ValueError(f"Prediction file contains another fold: {path}")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError("OOF predictions contain duplicate bags")
    manifest = pd.read_csv(
        INPUT_MANIFEST_CSV,
        dtype={"study_id": str, "level": str},
    )
    target_columns = ["study_id", "level", *REGION_COLUMNS]
    predictions = predictions.merge(
        manifest[target_columns],
        on=["study_id", "level"],
        how="left",
        validate="one_to_one",
    )
    if predictions[list(REGION_COLUMNS)].isna().any().any():
        raise ValueError("OOF predictions do not match the frozen input manifest")
    predictions["category"] = np.select(
        [
            predictions["vertebra_target"].eq(1)
            & predictions["vertebra_prediction"].eq(1),
            predictions["vertebra_target"].eq(0)
            & predictions["vertebra_prediction"].eq(1),
            predictions["vertebra_target"].eq(1)
            & predictions["vertebra_prediction"].eq(0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    return predictions


def select_stratified_high_scores(
    predictions: pd.DataFrame,
    categories: tuple[str, ...],
    samples_per_stratum: int,
) -> pd.DataFrame:
    """Select top scores within every requested category/fold/level stratum."""
    if samples_per_stratum < 1:
        raise ValueError("samples_per_stratum must be at least one")
    unknown = set(categories) - {"TP", "FP", "FN", "TN"}
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    selected = (
        predictions[predictions["category"].isin(categories)]
        .sort_values(
            ["category", "fold", "level", "vertebra_score", "study_id"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["category", "fold", "level"], sort=True, as_index=False)
        .head(samples_per_stratum)
    )
    return selected.sort_values(
        ["fold", "category", "level", "vertebra_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)
