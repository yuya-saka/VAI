"""Metrics and region-collapse diagnostics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from train_models.stage1.utils.metrics import compute_level_metrics, compute_oof_metrics

REGION_NAMES = ("body", "right_foramen", "left_foramen", "posterior")


def region_names_for(n_regions: int) -> tuple[str, ...]:
    """Human-readable names for the model's active region axis.

    The canonical 4 anatomical names apply to the masked region-pooling
    path; other region counts (currently just the single global-FPN-pooled
    pseudo-region ablation) fall back to generic positional names.
    """
    if n_regions == len(REGION_NAMES):
        return REGION_NAMES
    return tuple(f"region_{index}" for index in range(n_regions))


def compute_metrics(
    labels: NDArray,
    probabilities: NDArray,
    groups: NDArray,
    vertebrae: NDArray,
) -> dict:
    """Compute Stage1-compatible overall and per-level metrics."""
    return {
        **compute_oof_metrics(labels, probabilities, groups),
        "per_vertebra": compute_level_metrics(labels, probabilities, vertebrae),
    }


def compute_region_diagnostics(
    region_logits: NDArray,
    region_plane_valid: NDArray,
    plane_valid: NDArray,
) -> dict[str, float | dict[str, float] | list[float]]:
    """Summarize region evidence utilization and collapse indicators.

    Args:
        region_logits: shape (N, S, R) fp32 per-plane region logits.
        region_plane_valid: shape (N, S, R) bool, True where a (plane, region)
            pair had nonzero mask area.
        plane_valid: shape (N, S) bool, True where at least one region was
            valid on that plane.

    Region scores are weakly supervised evidence, not calibrated
    probabilities (see the Codex architecture review), so diagnostics are
    named accordingly rather than as region-level fracture probabilities.
    """
    n_regions = region_logits.shape[-1]
    region_names = region_names_for(n_regions)
    region_probabilities = 1.0 / (1.0 + np.exp(-region_logits))
    valid_weights = region_plane_valid.astype(float)

    region_evidence: dict[str, float] = {}
    for region_index, region_name in enumerate(region_names):
        weights = valid_weights[..., region_index]
        numerator = float((region_probabilities[..., region_index] * weights).sum())
        denominator = max(float(weights.sum()), 1e-12)
        region_evidence[region_name] = numerator / denominator

    flat_plane_valid = plane_valid.reshape(-1)
    flat_logits = region_logits.reshape(-1, n_regions)[flat_plane_valid]
    flat_valid = region_plane_valid.reshape(-1, n_regions)[flat_plane_valid]

    # With a single region (the global-FPN-pooled ablation), the "which
    # region fired" softmax/entropy diagnostics below are undefined (there is
    # nothing to distribute across, and normalizing by log(1)=0 would divide
    # by zero), so they degrade to the same NaN placeholders as the no-data case.
    if flat_logits.shape[0] == 0 or n_regions < 2:
        return {
            "region_evidence": region_evidence,
            "argmax_distribution": [float("nan")] * n_regions,
            "mean_entropy": float("nan"),
            "normalized_entropy": float("nan"),
            "valid_plane_rate": float(np.mean(plane_valid)),
            "valid_region_plane_rate": float(np.mean(region_plane_valid)),
        }

    masked_logits = np.where(flat_valid, flat_logits, -np.inf)
    shifted = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    exponentials = np.where(flat_valid, np.exp(shifted), 0.0)
    probabilities = exponentials / np.maximum(
        exponentials.sum(axis=1, keepdims=True), 1e-12
    )
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0),
        axis=1,
    )
    argmax = np.argmax(masked_logits, axis=1)
    distribution = [
        float(np.mean(argmax == region_index)) for region_index in range(n_regions)
    ]

    return {
        "region_evidence": region_evidence,
        "argmax_distribution": distribution,
        "mean_entropy": float(np.mean(entropy)),
        "normalized_entropy": float(np.mean(entropy) / math.log(n_regions)),
        "valid_plane_rate": float(np.mean(plane_valid)),
        "valid_region_plane_rate": float(np.mean(region_plane_valid)),
    }
