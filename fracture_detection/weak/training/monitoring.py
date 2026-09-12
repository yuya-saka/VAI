"""Diagnostic-only monitoring: no automatic collapse stop.

Reference design: ``fracture_detection/REGION_MIL_DESIGN.md`` section 8 lists
q's mean/std, sum(q), all-high rate, inter-region correlation, positive-bag
argmax distribution, and negative-bag false positives as things to WATCH, not
thresholds to enforce. Unlike ``region_branch/training/monitoring.py``
(which halts training after 3 collapsed epochs), this module intentionally
has no ``CollapseMonitor``-equivalent -- the accepted design records these
diagnostics every GT-pass and leaves the stop decision to the person running
the comparison, per "診断だけでモデルを有効と判定しない" (section 8).

These diagnostics run over whatever prediction set the caller passes in
(typically the natural-distribution inner-fold predictions collected for that
GT-pass) rather than a separately sampled fixed subset, since this package
has no CAM/pseudo-label student-teacher comparison to hold constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from fracture_detection.weak.data_pipeline.constants import REGION_NAMES

STANDARD_DEVIATION_FLOOR = 1e-8
ALL_HIGH_THRESHOLD = 0.8


@dataclass(frozen=True)
class DiagnosticRecord:
    """One GT-pass's collapse/calibration-shape diagnostics."""

    gt_pass: int
    q_mean: dict[str, float]
    q_std: dict[str, float]
    sum_q_mean: float
    all_high_rate: float
    inter_region_spearman: dict[str, float]
    first_pc_explained_variance: float
    positive_argmax_share: dict[str, float]
    negative_false_positive_rate: dict[str, float]

    def as_row(self) -> dict[str, Any]:
        """Flatten to one CSV row."""
        row: dict[str, Any] = {
            "gt_pass": self.gt_pass,
            "sum_q_mean": self.sum_q_mean,
            "all_high_rate": self.all_high_rate,
            "first_pc_explained_variance": self.first_pc_explained_variance,
        }
        row.update({f"q_mean_{name}": value for name, value in self.q_mean.items()})
        row.update({f"q_std_{name}": value for name, value in self.q_std.items()})
        row.update(
            {
                f"spearman_{key}": value
                for key, value in self.inter_region_spearman.items()
            }
        )
        row.update(
            {
                f"positive_argmax_{name}": value
                for name, value in self.positive_argmax_share.items()
            }
        )
        row.update(
            {
                f"negative_fp_rate_{name}": value
                for name, value in self.negative_false_positive_rate.items()
            }
        )
        return row


def compute_pass_diagnostics(
    gt_pass: int,
    region_logits: np.ndarray,
    group_id: np.ndarray,
    negative_group_id: int,
    positive_group_ids: tuple[int, ...],
    all_high_threshold: float = ALL_HIGH_THRESHOLD,
) -> DiagnosticRecord:
    """Compute one GT-pass's q/correlation/argmax/false-positive diagnostics.

    Args:
        region_logits: [N, N_REGIONS] raw region logits (any prediction set,
            typically the inner-fold natural-distribution predictions).
        group_id: [N] bag group id, see data_pipeline.groups.GROUP_IDS.
        negative_group_id: the integer id for whole-negative bags.
        positive_group_ids: integer ids counted as "positive" for the argmax
            distribution (both annotated and weak positive).
    """
    if region_logits.ndim != 2 or region_logits.shape[1] != len(REGION_NAMES):
        raise ValueError(f"region_logitsは[N,{len(REGION_NAMES)}]が必要です")
    if region_logits.shape[0] != group_id.shape[0]:
        raise ValueError("region_logitsとgroup_idの件数が一致しません")

    q = 1.0 / (1.0 + np.exp(-region_logits))
    q_mean = {name: float(q[:, i].mean()) for i, name in enumerate(REGION_NAMES)}
    q_std = {name: float(q[:, i].std()) for i, name in enumerate(REGION_NAMES)}
    sum_q_mean = float(q.sum(axis=1).mean())
    all_high_rate = float((q >= all_high_threshold).all(axis=1).mean())

    inter_region: dict[str, float] = {}
    for i in range(len(REGION_NAMES)):
        for j in range(i + 1, len(REGION_NAMES)):
            key = f"{REGION_NAMES[i]}_vs_{REGION_NAMES[j]}"
            inter_region[key] = _safe_spearman(q[:, i], q[:, j])

    standard_deviation = q.std(axis=0)
    standardized = (q - q.mean(axis=0)) / np.clip(
        standard_deviation, STANDARD_DEVIATION_FLOOR, None
    )
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigen_sum = float(eigenvalues.sum())
    explained_variance = (
        float(eigenvalues.max() / eigen_sum) if eigen_sum > 0 else float("nan")
    )

    positive_mask = np.isin(group_id, positive_group_ids)
    positive_argmax_share = _argmax_share(q[positive_mask])

    negative_mask = group_id == negative_group_id
    negative_q = q[negative_mask]
    negative_false_positive_rate = {
        name: float((negative_q[:, i] >= 0.5).mean())
        if negative_q.shape[0]
        else float("nan")
        for i, name in enumerate(REGION_NAMES)
    }

    return DiagnosticRecord(
        gt_pass=gt_pass,
        q_mean=q_mean,
        q_std=q_std,
        sum_q_mean=sum_q_mean,
        all_high_rate=all_high_rate,
        inter_region_spearman=inter_region,
        first_pc_explained_variance=explained_variance,
        positive_argmax_share=positive_argmax_share,
        negative_false_positive_rate=negative_false_positive_rate,
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _argmax_share(positive_q: np.ndarray) -> dict[str, float]:
    if positive_q.shape[0] == 0:
        return {name: float("nan") for name in REGION_NAMES}
    argmax_indices = positive_q.argmax(axis=1)
    counts = np.bincount(argmax_indices, minlength=len(REGION_NAMES))
    shares = counts / positive_q.shape[0]
    return {name: float(shares[i]) for i, name in enumerate(REGION_NAMES)}
