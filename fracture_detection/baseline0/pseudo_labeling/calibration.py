"""Shared logit-share Platt calibration for CAM pseudo-region targets.

Pure functions only: converting view-level CAM density enrichment into a
calibrated soft probability. CAM computation, file I/O, and CLI argument
handling belong to ``gradcam.py`` and ``cli/generate_pseudo_labels.py``.

The frozen transform (``.claude/docs/REGION_MODEL_DESIGN_JA.md`` Section 5,
``.claude/docs/work-logs/2026-09/2026-09-01-cam-soft-bce-implementation-plan.md``
Section 0.2)::

    s_vr = e_vr / sum_j(e_vj)            # share within one TTA view
    s_r  = mean_v(s_vr)                  # views are averaged AFTER sharing
    x_r  = logit(clip(s_r, floor, 1-floor))
    q*_r = sigmoid(a_k * x_r + b_k)      # one shared (a_k, b_k) for all regions
    q_r  = q*_r                          if sum_j(q*_j) >= 1
           q*_r / sum_j(q*_j)            otherwise

``a_k, b_k`` are fit once per student outer fold ``k``, using only that
student's training-fold bags whose four human region targets are all valid
and whose bag is whole-positive. Region identifiers, per-region intercepts,
bag probability, CAM total, and cardinality are deliberately excluded from
the fit: see the "打ち切った検討" section of
``.claude/docs/work-logs/2026-09/2026-09-01-pseudo-label-construction-redesign.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

FloatArray = NDArray[np.float64]

SHARE_FLOOR = 0.01
REGULARIZATION_C = 1.0
N_REGIONS = 4
PROBABILITY_EPSILON = 1e-6


def enrichment_to_share(enrichment: FloatArray) -> FloatArray:
    """Normalize one or more views' non-negative region enrichment into a share.

    ``enrichment`` has shape ``(..., n_regions)``. Every row must be finite,
    non-negative, and sum to a positive value; a single region can score
    exactly zero, but a bag whose whole-vertebra CAM mass vanished cannot be
    shared and must be excluded by the caller before this is called.
    """
    array = np.asarray(enrichment, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("enrichment must be finite")
    if np.any(array < 0):
        raise ValueError("enrichment must be non-negative")
    totals = array.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("enrichment must sum to a positive value per row")
    return array / totals


def average_view_shares(view_shares: FloatArray) -> FloatArray:
    """Average already-shared views over their leading axis.

    ``view_shares`` has shape ``(n_views, ..., n_regions)`` with every row
    summing to one. Averaging shares — rather than averaging raw enrichment
    and sharing once — keeps one high-magnitude view from dominating.
    """
    array = np.asarray(view_shares, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("view_shares needs a leading view axis")
    row_sums = array.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("every view must already be a region share summing to one")
    return array.mean(axis=0)


def shares_to_logit_features(
    shares: FloatArray, share_floor: float = SHARE_FLOOR
) -> FloatArray:
    """Clip a region share away from 0/1 and take its logit."""
    array = np.asarray(shares, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("shares must be finite")
    if share_floor <= 0 or share_floor >= 0.5:
        raise ValueError("share_floor must lie in (0, 0.5)")
    clipped = np.clip(array, share_floor, 1.0 - share_floor)
    return np.asarray(logit(clipped), dtype=np.float64)


@dataclass(frozen=True)
class SharedLogitShareCalibration:
    """One outer fold's shared ``(slope, intercept)`` logit-share Platt map."""

    student_outer_fold: int
    slope: float
    intercept: float
    n_fit_bags: int
    n_fit_studies: int
    share_floor: float = SHARE_FLOOR


def fit_shared_logit_share_calibration(
    shares: FloatArray,
    targets: NDArray[np.int_],
    study_ids: NDArray[np.str_] | list[str],
    student_outer_fold: int,
    share_floor: float = SHARE_FLOOR,
    regularization_c: float = REGULARIZATION_C,
) -> SharedLogitShareCalibration:
    """Fit one region-agnostic logistic map on complete whole-positive cells.

    ``shares`` and ``targets`` have shape ``(n_bags, 4)``; every row of
    ``targets`` must already be hard 0/1 human labels for all four regions.
    Callers are responsible for restricting the input to whole-positive,
    fully human-annotated, student-train-fold-only bags before calling this;
    this function only fits and validates, it does not select the population.
    """
    shares_array = np.asarray(shares, dtype=np.float64)
    targets_array = np.asarray(targets)
    if shares_array.ndim != 2 or shares_array.shape[1] != N_REGIONS:
        raise ValueError("shares must have shape (n_bags, 4)")
    if shares_array.shape != targets_array.shape:
        raise ValueError("shares and targets must have the same shape")
    if shares_array.shape[0] < 2:
        raise ValueError("need at least two bags to fit a calibration map")
    unique_targets = set(np.unique(targets_array).tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("targets must be hard 0/1 labels")
    study_array = np.asarray(list(study_ids))
    if study_array.shape[0] != shares_array.shape[0]:
        raise ValueError("study_ids must have one entry per bag")

    features = shares_to_logit_features(shares_array, share_floor).reshape(-1, 1)
    labels = targets_array.reshape(-1)
    model = LogisticRegression(C=regularization_c, max_iter=10_000, solver="lbfgs")
    model.fit(features, labels)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("calibration fit produced a non-finite coefficient")
    if slope <= 0:
        raise ValueError(
            "calibration slope must be positive for outer fold "
            f"{student_outer_fold}, got {slope:.6f}"
        )
    return SharedLogitShareCalibration(
        student_outer_fold=student_outer_fold,
        slope=slope,
        intercept=intercept,
        n_fit_bags=int(shares_array.shape[0]),
        n_fit_studies=int(np.unique(study_array).size),
        share_floor=share_floor,
    )


def apply_shared_logit_share_calibration(
    shares: FloatArray, calibration: SharedLogitShareCalibration
) -> FloatArray:
    """Map averaged shares to calibrated probabilities with one fold's map."""
    features = shares_to_logit_features(shares, calibration.share_floor)
    probabilities = expit(calibration.slope * features + calibration.intercept)
    return np.asarray(probabilities, dtype=np.float64)


def project_sum_at_least_one(probabilities: FloatArray) -> FloatArray:
    """Rescale rows whose regional probabilities sum below one.

    Every whole-positive bag has at least one true positive region, so the
    generated targets should not collectively claim less than that.
    Rescaling — rather than renormalizing every row — leaves an
    already-coherent row untouched and preserves each row's within-bag rank.
    """
    array = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    sums = array.sum(axis=-1, keepdims=True)
    needs_projection = sums < 1.0
    return np.where(needs_projection, array / sums, array)


def cross_validated_calibration_quality(
    shares: FloatArray,
    targets: NDArray[np.int_],
    study_ids: NDArray[np.str_] | list[str],
    student_outer_fold: int,
    n_splits: int = 5,
    share_floor: float = SHARE_FLOOR,
    regularization_c: float = REGULARIZATION_C,
) -> dict[str, float]:
    """Study-grouped OOF quality of the shared logit-share map, for metadata only.

    This never selects the coefficients used at generation time — those are
    fit once on the entire fit population by
    ``fit_shared_logit_share_calibration`` — it only reports whether the
    complete-only fit population looks like it generalizes across studies.
    """
    shares_array = np.asarray(shares, dtype=np.float64)
    targets_array = np.asarray(targets)
    study_array = np.asarray(list(study_ids))
    unique_studies = np.unique(study_array)
    splits = min(n_splits, unique_studies.size)

    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None
    while splits >= 2:
        candidate = list(
            GroupKFold(n_splits=splits).split(shares_array, groups=study_array)
        )
        if all(len(train_index) >= 2 for train_index, _ in candidate):
            fold_indices = candidate
            break
        splits -= 1
    if fold_indices is None:
        return {"oof_available": 0.0, "n_studies": float(unique_studies.size)}

    oof = np.full_like(shares_array, np.nan, dtype=np.float64)
    for train_index, test_index in fold_indices:
        fold_calibration = fit_shared_logit_share_calibration(
            shares_array[train_index],
            targets_array[train_index],
            study_array[train_index],
            student_outer_fold,
            share_floor,
            regularization_c,
        )
        oof[test_index] = apply_shared_logit_share_calibration(
            shares_array[test_index], fold_calibration
        )
    if not np.isfinite(oof).all():
        raise ValueError("OOF calibration audit produced non-finite predictions")

    flat_targets = targets_array.reshape(-1)
    flat_probabilities = np.clip(
        oof.reshape(-1), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    return {
        "oof_available": 1.0,
        "n_studies": float(unique_studies.size),
        "oof_ap": float(average_precision_score(flat_targets, flat_probabilities)),
        "oof_brier": float(brier_score_loss(flat_targets, flat_probabilities)),
        "oof_log_loss": float(log_loss(flat_targets, flat_probabilities)),
    }


def summarize_probability_distribution(probabilities: FloatArray) -> dict[str, float]:
    """Quantile and projection-rate statistics for generation metadata."""
    array = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite")
    sums = array.sum(axis=-1)
    quantile_levels = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    q_quantiles = np.quantile(array.reshape(-1), quantile_levels)
    sum_quantiles = np.quantile(sums, quantile_levels)
    summary: dict[str, float] = {
        "n_bags": float(array.shape[0]),
        "n_cells": float(array.size),
        "q_mean": float(array.mean()),
        "sum_mean": float(sums.mean()),
        "projected_fraction": float((sums < 1.0 - 1e-9).mean()),
    }
    for level, value in zip(quantile_levels, q_quantiles, strict=True):
        summary[f"q_p{int(level * 100)}"] = float(value)
    for level, value in zip(quantile_levels, sum_quantiles, strict=True):
        summary[f"sum_p{int(level * 100)}"] = float(value)
    return summary
