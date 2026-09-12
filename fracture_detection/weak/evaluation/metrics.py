"""OOF evaluation: region macro AP, whole AP/AUROC, and probability quality.

Reuses ``baseline0.evaluation.metrics`` directly for the AP/AUROC/bootstrap-CI
primitives (``region_metrics``, ``evaluate_vertebra_prediction_frame``) --
this module does not reimplement them. Brier score and ECE for MODEL OUTPUT
calibration do not exist anywhere else in this repository (the only existing
Brier usage is baseline0's pseudo-label CAM-calibration audit, a different
purpose), so they are written here.

Per ``fracture_detection/REGION_MIL_DESIGN.md`` section 8, four populations
are evaluated separately and must never be conflated (see
``feedback_verify_metric_population_before_comparing``):

- conditional localization: the annotated-positive (A) bags only, all four
  regions, OOF q.
- whole detection: every bag's OOF p_whole (natural distribution).
- region detection: annotated-positive (A) UNION whole-negative (N) bags'
  OOF q -- weak-positive (U) bags are NEVER treated as negative here.
- probability quality: BCE/Brier/ECE computed over each population above.

Every returned dict carries n/positives/prevalence so a caller can verify
which population a number came from before comparing it to another.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fracture_detection.baseline0.data.constants import REGION_COLUMNS
from fracture_detection.baseline0.evaluation.metrics import (
    evaluate_vertebra_prediction_frame,
    region_metrics,
)
from fracture_detection.weak.data_pipeline.groups import (
    ANNOTATED_GROUP,
    GROUP_COLUMN,
    NEGATIVE_GROUP,
)

Q_COLUMNS = tuple(f"q_{index}" for index in range(1, 5))

_DEFAULT_ECE_BINS = 10
_ECE_EPSILON = 1e-12


def brier_score(targets: NDArray[Any], scores: NDArray[Any]) -> float:
    """Mean squared error between binary targets and predicted probability."""
    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    return float(np.mean((score_array - target_array) ** 2))


def binary_cross_entropy(targets: NDArray[Any], scores: NDArray[Any]) -> float:
    """Mean BCE, with scores clamped away from 0/1 to keep log() finite."""
    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.clip(
        np.asarray(scores, dtype=np.float64), _ECE_EPSILON, 1.0 - _ECE_EPSILON
    )
    return float(
        -np.mean(
            target_array * np.log(score_array)
            + (1.0 - target_array) * np.log(1.0 - score_array)
        )
    )


def expected_calibration_error(
    targets: NDArray[Any], scores: NDArray[Any], n_bins: int = _DEFAULT_ECE_BINS
) -> float:
    """Equal-width-bin ECE: sum_b (|bin_b|/N) * |mean(score) - mean(target)|."""
    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if len(target_array) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(score_array, bin_edges[1:-1]), 0, n_bins - 1)
    total = len(target_array)
    error = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / total) * abs(
            score_array[in_bin].mean() - target_array[in_bin].mean()
        )
    return float(error)


def probability_quality_metrics(
    targets: NDArray[Any], scores: NDArray[Any]
) -> dict[str, float]:
    """BCE, Brier, and ECE for one target/score population."""
    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(len(target_array)),
        "positives": int(target_array.sum()),
        "prevalence": float(target_array.mean()) if len(target_array) else float("nan"),
        "bce": binary_cross_entropy(target_array, score_array),
        "brier": brier_score(target_array, score_array),
        "ece": expected_calibration_error(target_array, score_array),
    }


def _macro_average_precision(region_result: dict[str, Any]) -> float:
    values = [metrics["average_precision"] for metrics in region_result.values()]
    finite = [value for value in values if not np.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_conditional_localization(
    predictions: pd.DataFrame, n_bootstrap: int = 1000
) -> dict[str, Any]:
    """Region-wise AP/AUROC/probability-quality over the A bags only."""
    annotated = predictions[predictions[GROUP_COLUMN] == ANNOTATED_GROUP]
    return _region_population_metrics(annotated, n_bootstrap)


def evaluate_region_detection(
    predictions: pd.DataFrame, n_bootstrap: int = 1000
) -> dict[str, Any]:
    """Region-wise AP/AUROC/probability-quality over A union N bags."""
    subset = predictions[
        predictions[GROUP_COLUMN].isin((ANNOTATED_GROUP, NEGATIVE_GROUP))
    ]
    return _region_population_metrics(subset, n_bootstrap)


def _region_population_metrics(
    subset: pd.DataFrame, n_bootstrap: int
) -> dict[str, Any]:
    if subset.empty:
        raise ValueError("評価対象の母集団が空です")
    region_targets = subset[list(REGION_COLUMNS)].to_numpy()
    region_scores = subset[list(Q_COLUMNS)].to_numpy()
    regions = region_metrics(
        region_targets,
        region_scores,
        groups=subset["study_id"].to_numpy(),
        n_bootstrap=n_bootstrap,
    )
    quality = {
        region_column: probability_quality_metrics(
            subset[region_column].to_numpy(), subset[q_column].to_numpy()
        )
        for region_column, q_column in zip(REGION_COLUMNS, Q_COLUMNS, strict=True)
    }
    return {
        "n": int(len(subset)),
        "regions": regions,
        "macro_average_precision": _macro_average_precision(regions),
        "probability_quality": quality,
    }


def evaluate_whole_detection(
    predictions: pd.DataFrame, n_bootstrap: int = 1000
) -> dict[str, Any]:
    """Whole-bag AP/AUROC/probability-quality over every bag (natural distribution)."""
    frame = predictions.rename(columns={"p_whole": "vertebra_score"})
    metrics = evaluate_vertebra_prediction_frame(frame, n_bootstrap=n_bootstrap)
    quality = probability_quality_metrics(
        predictions["vertebra_target"].to_numpy(), predictions["p_whole"].to_numpy()
    )
    return {**metrics, "probability_quality": quality}


def evaluate_weak_oof(
    predictions: pd.DataFrame, n_bootstrap: int = 1000
) -> dict[str, Any]:
    """Combine all three populations into one OOF report."""
    required = {"study_id", "level", GROUP_COLUMN, "vertebra_target", "p_whole"}
    required.update(REGION_COLUMNS)
    required.update(Q_COLUMNS)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"予測表に必要な列がありません: {sorted(missing)}")
    return {
        "conditional_localization": evaluate_conditional_localization(
            predictions, n_bootstrap
        ),
        "region_detection": evaluate_region_detection(predictions, n_bootstrap),
        "whole_detection": evaluate_whole_detection(predictions, n_bootstrap),
    }
