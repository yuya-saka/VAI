"""pooled OOF・固定順序検定・領域floor gate。"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fracture_detection.common.constants import REGION_COLUMNS
from fracture_detection.common.metrics import safe_auroc, safe_average_precision

Metric = Callable[[NDArray[Any], NDArray[Any]], float]
KEYS = ["study_id", "level", "fold", "vertebra_target"]


def collect_oof_predictions(
    run_root: Path,
    manifest: pd.DataFrame,
    *,
    filename: str = "outer_predictions.csv",
    expected_frozen_manifest_sha256: str | None = None,
) -> pd.DataFrame:
    """5 outer予測を集め、凍結manifestと1対1整合を検証する。"""
    frames: list[pd.DataFrame] = []
    for outer_fold in range(5):
        path = run_root / f"outer{outer_fold}" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"study_id": str, "level": str})
        if set(frame["fold"].astype(int)) != {outer_fold}:
            raise ValueError(f"outer{outer_fold}予測に別foldが混入しています")
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if expected_frozen_manifest_sha256 is not None:
        if "frozen_manifest_sha256" not in pooled:
            raise ValueError("OOF予測にfrozen manifest hashがありません")
        hashes = set(pooled["frozen_manifest_sha256"].astype(str))
        if hashes != {expected_frozen_manifest_sha256}:
            raise ValueError(f"OOF予測のfrozen manifest hashが不一致です: {hashes}")
    if pooled.duplicated(["study_id", "level"]).any():
        raise ValueError("OOF予測にstudy/level重複があります")
    expected = manifest[["study_id", "level", "fold", "vertebra_target"]].copy()
    actual = pooled[KEYS].copy()
    expected["fold"] = expected["fold"].astype(int)
    actual["fold"] = actual["fold"].astype(int)
    if (
        not expected.sort_values(KEYS)
        .reset_index(drop=True)
        .equals(actual.sort_values(KEYS).reset_index(drop=True))
    ):
        raise ValueError("OOF予測が凍結input manifestと一致しません")
    return pooled.sort_values(["study_id", "level"]).reset_index(drop=True)


def paired_cluster_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    score_column: str,
    target_column: str,
    metric: Metric,
    n_bootstrap: int = 10_000,
    seed: int = 20260818,
) -> dict[str, float]:
    """patient clusterを共有resampleしたpaired metric差を返す。"""
    aligned = _align_pair(left, right, score_column, target_column)
    targets = aligned[target_column].to_numpy()
    left_scores = aligned[f"{score_column}_left"].to_numpy()
    right_scores = aligned[f"{score_column}_right"].to_numpy()
    groups = aligned["study_id"].astype(str).to_numpy()
    estimate = metric(targets, left_scores) - metric(targets, right_scores)
    samples = _cluster_difference_samples(
        targets,
        left_scores,
        right_scores,
        groups,
        metric,
        n_bootstrap,
        seed,
    )
    return {
        "estimate": float(estimate),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "one_sided_p": float((1 + np.count_nonzero(samples <= 0)) / (n_bootstrap + 1)),
    }


def fixed_sequence_whole_tests(
    predictions: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 10_000,
) -> dict[str, object]:
    """H1→H2の固定順序を適用する。"""
    h1 = paired_cluster_bootstrap_difference(
        predictions["baseline1_b"],
        predictions["control_b"],
        score_column="vertebra_score",
        target_column="vertebra_target",
        metric=safe_auroc,
        n_bootstrap=n_bootstrap,
        seed=20260818,
    )
    h1_confirmed = h1["ci_lower"] > 0
    h2 = paired_cluster_bootstrap_difference(
        predictions["proposed_max"],
        predictions["proposed_max_beta0"],
        score_column="vertebra_score",
        target_column="vertebra_target",
        metric=safe_auroc,
        n_bootstrap=n_bootstrap,
        seed=20260819,
    )
    return {
        "H1": {**h1, "confirmatory": True, "rejected_null": h1_confirmed},
        "H2": {
            **h2,
            "confirmatory": h1_confirmed,
            "rejected_null": bool(h1_confirmed and h2["ci_lower"] > 0),
        },
    }


def region_floor_gate(
    proposed_b: pd.DataFrame,
    floor: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
) -> dict[str, object]:
    """Proposed–Bだけを4領域cross-fitted floorと比較しHolm補正する。"""
    annotated = proposed_b[proposed_b["has_region_target"].astype(bool)].copy()
    results: dict[str, dict[str, float]] = {}
    for region in REGION_COLUMNS:
        left = annotated.rename(
            columns={f"{region}_score": "score", f"{region}_target": "target"}
        )
        right = floor.rename(
            columns={f"{region}_floor_score": "score", region: "target"}
        )
        results[region] = paired_cluster_bootstrap_difference(
            left,
            right,
            score_column="score",
            target_column="target",
            metric=safe_average_precision,
            n_bootstrap=n_bootstrap,
            seed=20260820,
        )
    adjusted = holm_adjust(
        {region: value["one_sided_p"] for region, value in results.items()}
    )
    return {
        region: {
            **results[region],
            "holm_adjusted_p": adjusted[region],
            "passes_floor": adjusted[region] < 0.05 and results[region]["estimate"] > 0,
        }
        for region in REGION_COLUMNS
    }


def region_pair_differences(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
) -> dict[str, dict[str, float]]:
    """同じannotated母集団で領域別APのpaired差を返す。"""
    left = left_predictions[left_predictions["has_region_target"].astype(bool)]
    right = right_predictions[right_predictions["has_region_target"].astype(bool)]
    results: dict[str, dict[str, float]] = {}
    for region in REGION_COLUMNS:
        left_region = left.rename(
            columns={f"{region}_score": "score", f"{region}_target": "target"}
        )
        right_region = right.rename(
            columns={f"{region}_score": "score", f"{region}_target": "target"}
        )
        results[region] = paired_cluster_bootstrap_difference(
            left_region,
            right_region,
            score_column="score",
            target_column="target",
            metric=safe_average_precision,
            n_bootstrap=n_bootstrap,
            seed=20260821,
        )
    return results


def region_ap_sensitivity(
    predictions: pd.DataFrame, *, n_bootstrap: int = 10_000
) -> dict[str, object]:
    """within-level rankとR2/R3 swap negative controlを返す。"""
    annotated = predictions[predictions["has_region_target"].astype(bool)].copy()
    within_level: dict[str, float] = {}
    raw: dict[str, float] = {}
    for region in REGION_COLUMNS:
        target = annotated[f"{region}_target"].to_numpy()
        score = annotated[f"{region}_score"].to_numpy()
        ranks = annotated.groupby("level")[f"{region}_score"].rank(
            method="average", pct=True
        )
        raw[region] = safe_average_precision(target, score)
        within_level[region] = safe_average_precision(target, ranks.to_numpy())
    swap_results: dict[str, object] = {}
    for region, swapped_region, seed in (
        ("region_2", "region_3", 20260822),
        ("region_3", "region_2", 20260823),
    ):
        target_column = f"{region}_target"
        correct = annotated[
            ["study_id", "level", target_column, f"{region}_score"]
        ].rename(columns={f"{region}_score": "score", target_column: "target"})
        swapped = annotated[
            ["study_id", "level", target_column, f"{swapped_region}_score"]
        ].rename(columns={f"{swapped_region}_score": "score", target_column: "target"})
        swap_results[region] = {
            "correct_ap": safe_average_precision(
                correct["target"].to_numpy(), correct["score"].to_numpy()
            ),
            "swapped_ap": safe_average_precision(
                swapped["target"].to_numpy(), swapped["score"].to_numpy()
            ),
            "paired_difference": paired_cluster_bootstrap_difference(
                correct,
                swapped,
                score_column="score",
                target_column="target",
                metric=safe_average_precision,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
        }
    return {
        "raw_ap": raw,
        "within_level_percentile_ap": within_level,
        "r2_r3_swap_negative_control": swap_results,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-valueを返す。"""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"p-valueが不正です: {name}={value}")
        running = max(running, min((count - index) * value, 1.0))
        adjusted[name] = running
    return adjusted


def _align_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    score_column: str,
    target_column: str,
) -> pd.DataFrame:
    keys = ["study_id", "level"]
    required = {*keys, score_column, target_column}
    if required - set(left) or required - set(right):
        raise ValueError("paired比較に必要な列がありません")
    merged = left[[*keys, target_column, score_column]].merge(
        right[[*keys, target_column, score_column]],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("paired比較のbag集合が一致しません")
    left_target = merged[f"{target_column}_left"].to_numpy()
    right_target = merged[f"{target_column}_right"].to_numpy()
    if not np.array_equal(left_target, right_target):
        raise ValueError("paired比較のtargetが一致しません")
    return merged.rename(columns={f"{target_column}_left": target_column}).drop(
        columns=f"{target_column}_right"
    )


def _cluster_difference_samples(
    targets: NDArray[Any],
    left_scores: NDArray[Any],
    right_scores: NDArray[Any],
    groups: NDArray[Any],
    metric: Metric,
    n_bootstrap: int,
    seed: int,
) -> NDArray[np.float64]:
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        row_indices = np.concatenate([indices[group] for group in selected])
        samples[iteration] = metric(
            targets[row_indices], left_scores[row_indices]
        ) - metric(targets[row_indices], right_scores[row_indices])
    finite = samples[np.isfinite(samples)]
    if len(finite) != n_bootstrap:
        raise FloatingPointError("bootstrap metric差に非有限値があります")
    return finite
