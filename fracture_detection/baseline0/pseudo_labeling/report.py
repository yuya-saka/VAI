"""Summaries for the Baseline 0 Grad-CAM generation-stage audit.

Consumes the long score table produced by ``cli/cam_audit.py`` and produces the
three tables the kill criteria are read from: teacher memorization, mask
boundary sensitivity, and horizontal-flip TTA stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from fracture_detection.baseline0.data.constants import REGION_COLUMNS, REGION_NAMES
from fracture_detection.baseline0.evaluation.metrics import (
    cluster_bootstrap_interval,
    safe_auroc,
    safe_average_precision,
)
from fracture_detection.baseline0.pseudo_labeling.cam_audit import (
    gate_perturbation_names,
)

IDENTITY = "identity"
NO_TTA = "none"
HFLIP_TTA = "hflip"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_823
SCORE_EPSILON = 1e-6

# Kill criteria from .claude/docs/codex/20260823-pseudo-label-mtl-design.md
MEMORIZATION_AUROC_GATE = 0.05
MEMORIZATION_SMD_GATE = 0.25
MEMORIZATION_MIN_FAILING_REGIONS = 2
PERTURBATION_SPEARMAN_GATE = 0.80
PERTURBATION_ARGMAX_GATE = 0.10


def score_column(region_column: str) -> str:
    """Name of the CAM score column for one region."""
    return f"{region_column}_score"


def _finite_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """AUROC over the rows whose score is defined.

    A perturbed mask can erase a small region, and a bag can have zero CAM mass,
    which leaves the density ratio undefined. Those rows are dropped rather than
    imputed so that an undefined score never counts as a low one.
    """
    usable = np.isfinite(scores)
    if usable.sum() < 2:
        return float("nan")
    picked = targets[usable]
    if picked.min() == picked.max():
        return float("nan")
    return safe_auroc(picked, scores[usable])


def select_role_frame(scores: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per bag for the requested teacher role.

    ``outer`` and ``inner`` are unique by construction: each fold is the outer
    fold of exactly one Baseline 0 run and the inner fold of exactly one other.
    ``train`` bags have three in-sample teachers, so the lowest teacher index is
    taken. Averaging the three would denoise the score and inflate its AUROC
    relative to the single-teacher ``outer`` score, which is the comparison the
    memorization gate depends on.
    """
    subset = scores[
        scores["role"].eq(role)
        & scores["perturbation"].eq(IDENTITY)
        & scores["tta"].eq(NO_TTA)
    ]
    if subset.empty:
        raise ValueError(f"No identity/no-TTA rows for role {role!r}")
    selected = (
        subset.sort_values("teacher")
        .groupby(["study_id", "level"], as_index=False, sort=True)
        .head(1)
    )
    return selected.sort_values(["study_id", "level"]).reset_index(drop=True)


def _paired_auroc_difference_interval(
    targets: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    """Patient-clustered percentile interval for ``AUROC(a) - AUROC(b)``."""
    unique_groups = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        rows = np.concatenate([index_by_group[group] for group in sampled])
        picked = targets[rows]
        if picked.min() == picked.max():
            continue
        difference = safe_auroc(picked, scores_a[rows]) - safe_auroc(
            picked, scores_b[rows]
        )
        if np.isfinite(difference):
            draws.append(difference)
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _standardized_mean_difference(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """SMD of log scores, pooled standard deviation."""
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if values_a.size < 2 or values_b.size < 2:
        return float("nan")
    log_a = np.log(np.clip(values_a, SCORE_EPSILON, None))
    log_b = np.log(np.clip(values_b, SCORE_EPSILON, None))
    pooled = np.sqrt((log_a.var(ddof=1) + log_b.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0.0:
        return float("nan")
    return float((log_a.mean() - log_b.mean()) / pooled)


def memorization_table(
    scores: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_SAMPLES
) -> pd.DataFrame:
    """Compare in-sample (``train``) against held-out (``outer``) teacher CAMs."""
    train = select_role_frame(scores, "train")
    inner = select_role_frame(scores, "inner")
    outer = select_role_frame(scores, "outer")
    keys = ["study_id", "level"]
    if not (train[keys].equals(outer[keys]) and inner[keys].equals(outer[keys])):
        raise ValueError("Role frames do not cover the same bags")

    rows: list[dict[str, object]] = []
    for index, region_column in enumerate(REGION_COLUMNS):
        valid = outer[f"{region_column}_target_valid"].to_numpy(dtype=bool)
        targets = outer.loc[valid, region_column].to_numpy(dtype=float)
        groups = outer.loc[valid, "study_id"].to_numpy()
        train_scores = train.loc[valid, score_column(region_column)].to_numpy(float)
        inner_scores = inner.loc[valid, score_column(region_column)].to_numpy(float)
        outer_scores = outer.loc[valid, score_column(region_column)].to_numpy(float)
        train_auroc = _finite_auroc(targets, train_scores)
        outer_auroc = _finite_auroc(targets, outer_scores)
        low, high = _paired_auroc_difference_interval(
            targets, train_scores, outer_scores, groups, n_bootstrap
        )
        rows.append(
            {
                "region": region_column,
                "region_name": REGION_NAMES[index],
                "n_positive": int(targets.sum()),
                "n_negative": int((1 - targets).sum()),
                "auroc_train": train_auroc,
                "auroc_inner": _finite_auroc(targets, inner_scores),
                "auroc_outer": outer_auroc,
                "auroc_difference": train_auroc - outer_auroc,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "smd_positive": _standardized_mean_difference(
                    train_scores[targets == 1.0], outer_scores[targets == 1.0]
                ),
                "smd_negative": _standardized_mean_difference(
                    train_scores[targets == 0.0], outer_scores[targets == 0.0]
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["auroc_gate_failed"] = table["auroc_difference"].abs() > (
        MEMORIZATION_AUROC_GATE
    )
    table["smd_gate_failed"] = (
        table[["smd_positive", "smd_negative"]].abs() > MEMORIZATION_SMD_GATE
    ).any(axis=1)
    return table


def _cluster_interval(
    values: np.ndarray, groups: np.ndarray, n_bootstrap: int
) -> tuple[float, float]:
    """Patient-clustered percentile interval for the mean of ``values``."""
    unique_groups = np.unique(groups)
    if unique_groups.size == 0:
        return float("nan"), float("nan")
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        rows = np.concatenate([index_by_group[group] for group in sampled])
        draws.append(float(values[rows].mean()))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def localization_table(
    scores: pd.DataFrame,
    role: str = "outer",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    """Held-out CAM localization per region, with a patient-clustered interval.

    This reproduces the headline teacher numbers from one command instead of a
    scratch script, so the pseudo-label decision rests on a repo artefact.
    """
    frame = select_role_frame(scores, role)
    rows: list[dict[str, object]] = []
    for index, region_column in enumerate(REGION_COLUMNS):
        valid = frame[f"{region_column}_target_valid"].to_numpy(dtype=bool)
        targets = frame.loc[valid, region_column].to_numpy(dtype=float)
        region_scores = frame.loc[valid, score_column(region_column)].to_numpy(float)
        groups = frame.loc[valid, "study_id"].to_numpy()
        usable = np.isfinite(region_scores)
        low, high = cluster_bootstrap_interval(
            targets[usable],
            region_scores[usable],
            groups[usable],
            safe_auroc,
            n_bootstrap=n_bootstrap,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "region": region_column,
                "region_name": REGION_NAMES[index],
                "n_positive": int(targets.sum()),
                "n_negative": int((1 - targets).sum()),
                "n_undefined": int((~usable).sum()),
                "auroc": _finite_auroc(targets, region_scores),
                "auroc_ci_low": low,
                "auroc_ci_high": high,
                "average_precision": safe_average_precision(
                    targets[usable], region_scores[usable]
                ),
            }
        )
    return pd.DataFrame(rows)


def laterality_summary(
    scores: pd.DataFrame,
    role: str = "outer",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    """Correct-side win rate on bags where exactly one transverse foramen is broken.

    R2 and R3 are the mirror-image transverse foramina, so a bag labelled for one
    but not the other is a direct test of whether the CAM picks the right side.
    This is the endpoint closest to the clinical motivation (vertebral artery
    injury risk), which is why it is reported separately from the region AUROCs.
    """
    frame = select_role_frame(scores, role)
    both_valid = frame["region_2_target_valid"].to_numpy(dtype=bool) & frame[
        "region_3_target_valid"
    ].to_numpy(dtype=bool)
    right_label = frame["region_2"].to_numpy(dtype=int)
    left_label = frame["region_3"].to_numpy(dtype=int)
    right_score = frame[score_column("region_2")].to_numpy(dtype=float)
    left_score = frame[score_column("region_3")].to_numpy(dtype=float)
    selected = (
        both_valid
        & (right_label != left_label)
        & np.isfinite(right_score)
        & np.isfinite(left_score)
    )
    if not selected.any():
        return {"n_bags": 0, "win_rate": float("nan")}
    correct = np.where(
        right_label[selected] == 1,
        right_score[selected] > left_score[selected],
        left_score[selected] > right_score[selected],
    ).astype(float)
    groups = frame.loc[selected, "study_id"].to_numpy()
    low, high = _cluster_interval(correct, groups, n_bootstrap)
    return {
        "n_bags": int(selected.sum()),
        "n_studies": int(np.unique(groups).size),
        "win_rate": float(correct.mean()),
        "win_rate_ci_low": low,
        "win_rate_ci_high": high,
    }


def _argmax_change_rate(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    columns = [score_column(name) for name in REGION_COLUMNS]
    left = reference[columns].to_numpy(dtype=float)
    right = candidate[columns].to_numpy(dtype=float)
    usable = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    if not usable.any():
        return float("nan")
    return float((left[usable].argmax(axis=1) != right[usable].argmax(axis=1)).mean())


def _stability_row(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> list[dict[str, object]]:
    keys = ["study_id", "level"]
    if not reference[keys].equals(candidate[keys]):
        raise ValueError(f"Bag order differs between identity and {label}")
    rows: list[dict[str, object]] = []
    for index, region_column in enumerate(REGION_COLUMNS):
        column = score_column(region_column)
        left = reference[column].to_numpy(dtype=float)
        right = candidate[column].to_numpy(dtype=float)
        usable = np.isfinite(left) & np.isfinite(right)
        correlation = (
            float(spearmanr(left[usable], right[usable]).statistic)
            if usable.sum() > 2
            else float("nan")
        )
        valid = reference[f"{region_column}_target_valid"].to_numpy(dtype=bool)
        targets = reference.loc[valid, region_column].to_numpy(dtype=float)
        rows.append(
            {
                "variant": label,
                "region": region_column,
                "region_name": REGION_NAMES[index],
                "spearman_vs_identity": correlation,
                "undefined_fraction": float((~np.isfinite(right)).mean()),
                "auroc": _finite_auroc(targets, right[valid]),
                "auroc_identity": _finite_auroc(targets, left[valid]),
            }
        )
    change = _argmax_change_rate(reference, candidate)
    for row in rows:
        row["argmax_change_rate"] = change
    return rows


def _identity_reference(scores: pd.DataFrame, role: str, tta: str) -> pd.DataFrame:
    subset = scores[
        scores["role"].eq(role)
        & scores["perturbation"].eq(IDENTITY)
        & scores["tta"].eq(tta)
    ]
    return subset.sort_values(["study_id", "level"]).reset_index(drop=True)


def perturbation_table(scores: pd.DataFrame, role: str = "outer") -> pd.DataFrame:
    """Rank stability of the region score under four-region mask perturbations."""
    reference = _identity_reference(scores, role, NO_TTA)
    if reference.empty:
        raise ValueError(f"No identity rows for role {role!r}")
    rows: list[dict[str, object]] = []
    for name in scores["perturbation"].unique():
        if name == IDENTITY:
            continue
        candidate = (
            scores[
                scores["role"].eq(role)
                & scores["perturbation"].eq(name)
                & scores["tta"].eq(NO_TTA)
            ]
            .sort_values(["study_id", "level"])
            .reset_index(drop=True)
        )
        rows.extend(_stability_row(reference, candidate, str(name)))
    table = pd.DataFrame(rows)
    table["spearman_gate_failed"] = (
        table["spearman_vs_identity"] < PERTURBATION_SPEARMAN_GATE
    )
    table["argmax_gate_failed"] = table["argmax_change_rate"] > (
        PERTURBATION_ARGMAX_GATE
    )
    return table


def tta_table(scores: pd.DataFrame, role: str = "outer") -> pd.DataFrame:
    """Rank stability of the region score under a laterality-safe horizontal flip."""
    reference = _identity_reference(scores, role, NO_TTA)
    candidate = _identity_reference(scores, role, HFLIP_TTA)
    if candidate.empty:
        raise ValueError("No horizontal-flip TTA rows were generated")
    table = pd.DataFrame(_stability_row(reference, candidate, HFLIP_TTA))
    table["spearman_gate_failed"] = (
        table["spearman_vs_identity"] < PERTURBATION_SPEARMAN_GATE
    )
    table["argmax_gate_failed"] = table["argmax_change_rate"] > (
        PERTURBATION_ARGMAX_GATE
    )
    return table


def audit_verdict(
    memorization: pd.DataFrame,
    perturbation: pd.DataFrame,
) -> dict[str, object]:
    """Apply the pre-registered kill criteria to the two stability tables."""
    failing_auroc = int(memorization["auroc_gate_failed"].sum())
    failing_smd = int(memorization["smd_gate_failed"].sum())
    memorization_failed = (
        failing_auroc >= MEMORIZATION_MIN_FAILING_REGIONS
        or failing_smd >= MEMORIZATION_MIN_FAILING_REGIONS
    )
    gate_rows = perturbation[perturbation["variant"].isin(gate_perturbation_names())]
    perturbation_failed = bool(
        gate_rows["spearman_gate_failed"].any() or gate_rows["argmax_gate_failed"].any()
    )
    return {
        "memorization_regions_failing_auroc": failing_auroc,
        "memorization_regions_failing_smd": failing_smd,
        "memorization_gate_failed": memorization_failed,
        "perturbation_gate_failed": perturbation_failed,
        "proceed_to_pseudo_label_generation": not (
            memorization_failed or perturbation_failed
        ),
    }
