"""Evaluate paired Stage4 confirmatory arms from five-seed pooled OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGION_PROBABILITY_COLUMNS = tuple(f"region_prob_r{index}" for index in range(1, 5))
REGION_TARGET_COLUMNS = tuple(f"region_target_r{index}" for index in range(1, 5))
KEY_COLUMNS = ("study_uid", "vertebra", "fold")


def load_seed_ensemble(
    arm_dir: Path,
    seeds: list[int],
) -> pd.DataFrame:
    """Validate seed alignment and average probabilities bag by bag."""
    if not seeds:
        raise ValueError("at least one seed is required")
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = arm_dir / f"seed{seed}" / "oof_predictions.csv"
        frame = pd.read_csv(path, dtype={"study_uid": str, "vertebra": str})
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"duplicate OOF bags: {path}")
        frames.append(frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True))

    reference = frames[0]
    invariant_columns = [
        *KEY_COLUMNS,
        "label",
        "region_supervised",
        *REGION_TARGET_COLUMNS,
    ]
    for seed, frame in zip(seeds[1:], frames[1:], strict=True):
        try:
            pd.testing.assert_frame_equal(
                reference.loc[:, invariant_columns],
                frame.loc[:, invariant_columns],
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(f"seed {seed} OOF rows do not align") from error

    result = reference.copy()
    probability_columns = ["pred_prob", *REGION_PROBABILITY_COLUMNS]
    for column in probability_columns:
        result[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames],
            axis=0,
        )
    return result


def _region_frame(frame: pd.DataFrame, exclude_c2: bool) -> pd.DataFrame:
    selected = frame.loc[frame["region_supervised"].astype(bool)].copy()
    if exclude_c2:
        selected = selected.loc[selected["vertebra"] != "C2"].copy()
    return selected


def region_metrics(
    frame: pd.DataFrame,
    exclude_c2: bool = False,
    percentile_rank: bool = False,
) -> dict[str, Any]:
    """Compute pooled region AP on strong validation bags."""
    selected = _region_frame(frame, exclude_c2)
    probabilities = selected.loc[:, REGION_PROBABILITY_COLUMNS].to_numpy(
        dtype=np.float64
    )
    if percentile_rank:
        ranked = selected.loc[:, ["fold"]].copy()
        for column in REGION_PROBABILITY_COLUMNS:
            ranked[column] = selected.groupby("fold")[column].rank(
                method="average",
                pct=True,
            )
        probabilities = ranked.loc[:, REGION_PROBABILITY_COLUMNS].to_numpy(
            dtype=np.float64
        )
    targets = selected.loc[:, REGION_TARGET_COLUMNS].to_numpy(dtype=np.int8)
    per_region = {
        f"R{index + 1}": float(
            average_precision_score(targets[:, index], probabilities[:, index])
        )
        for index in range(4)
    }
    return {
        "n_bags": len(selected),
        "per_region_ap": per_region,
        "macro_ap": float(np.mean(list(per_region.values()))),
    }


def fold_region_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute diagnostic-only fold AP values."""
    selected = _region_frame(frame, exclude_c2=False)
    return {
        str(fold): region_metrics(
            selected.loc[selected["fold"] == fold],
            exclude_c2=False,
        )
        for fold in sorted(selected["fold"].unique())
    }


def vertebra_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute pooled vertebra AUROC and AUPRC."""
    targets = frame["label"].to_numpy(dtype=np.int8)
    probabilities = frame["pred_prob"].to_numpy(dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
    }


def _paired_frames(
    weak: pd.DataFrame,
    mixed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weak_sorted = weak.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    mixed_sorted = mixed.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    invariant = [*KEY_COLUMNS, "label"]
    try:
        pd.testing.assert_frame_equal(
            weak_sorted.loc[:, invariant],
            mixed_sorted.loc[:, invariant],
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("Weak-only and Mixed OOF bags do not align") from error
    return weak_sorted, mixed_sorted


def _cluster_groups(frame: pd.DataFrame) -> list[list[np.ndarray]]:
    groups: list[list[np.ndarray]] = []
    for fold in sorted(frame["fold"].unique()):
        fold_indices = np.flatnonzero(frame["fold"].to_numpy() == fold)
        fold_studies = frame.iloc[fold_indices]["study_uid"].astype(str)
        studies = fold_studies.unique()
        groups.append(
            [fold_indices[fold_studies.to_numpy() == study] for study in studies]
        )
    return groups


def _cluster_bootstrap_indices(
    groups: list[list[np.ndarray]],
    random_generator: np.random.Generator,
) -> np.ndarray:
    indices: list[np.ndarray] = []
    for fold_groups in groups:
        sampled = random_generator.integers(
            0,
            len(fold_groups),
            size=len(fold_groups),
        )
        indices.extend(fold_groups[index] for index in sampled)
    return np.concatenate(indices)


def _fast_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_targets = targets[order].astype(np.float64, copy=False)
    sorted_probabilities = probabilities[order]
    positives = sorted_targets.sum()
    if positives == 0:
        return 0.0
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(sorted_probabilities)),
        len(sorted_probabilities) - 1,
    ]
    true_positives = np.cumsum(sorted_targets)[threshold_ends]
    precisions = true_positives / (threshold_ends + 1)
    recalls = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recalls]) * precisions))


def _fast_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_targets = targets[order].astype(np.int8, copy=False)
    sorted_probabilities = probabilities[order]
    positives = int(sorted_targets.sum())
    negatives = len(sorted_targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(sorted_probabilities)),
        len(sorted_probabilities) - 1,
    ]
    threshold_starts = np.r_[0, threshold_ends[:-1] + 1]
    cumulative_positives = np.cumsum(sorted_targets)[threshold_ends]
    group_positives = np.diff(np.r_[0, cumulative_positives])
    group_sizes = threshold_ends - threshold_starts + 1
    group_negatives = group_sizes - group_positives
    negatives_before = np.cumsum(group_negatives) - group_negatives
    concordant = np.sum(group_positives * (negatives_before + 0.5 * group_negatives))
    return float(concordant / (positives * negatives))


def _region_macro_ap_arrays(
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> float:
    return float(
        np.mean(
            [
                _fast_average_precision(
                    targets[indices, region_index],
                    probabilities[indices, region_index],
                )
                for region_index in range(4)
            ]
        )
    )


def paired_region_bootstrap(
    weak: pd.DataFrame,
    mixed: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired region macro-AP differences within outer folds."""
    weak_region, mixed_region = _paired_frames(
        _region_frame(weak, exclude_c2=False),
        _region_frame(mixed, exclude_c2=False),
    )
    random_generator = np.random.default_rng(seed)
    groups = _cluster_groups(weak_region)
    targets = weak_region.loc[:, REGION_TARGET_COLUMNS].to_numpy(dtype=np.int8)
    weak_probabilities = weak_region.loc[:, REGION_PROBABILITY_COLUMNS].to_numpy(
        dtype=np.float64
    )
    mixed_probabilities = mixed_region.loc[:, REGION_PROBABILITY_COLUMNS].to_numpy(
        dtype=np.float64
    )
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = _cluster_bootstrap_indices(groups, random_generator)
        deltas[iteration] = _region_macro_ap_arrays(
            targets,
            mixed_probabilities,
            indices,
        ) - _region_macro_ap_arrays(
            targets,
            weak_probabilities,
            indices,
        )
    weak_point = region_metrics(weak_region)["macro_ap"]
    mixed_point = region_metrics(mixed_region)["macro_ap"]
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_macro_ap": float(mixed_point - weak_point),
        "ci95": [float(lower), float(upper)],
        "iterations": iterations,
        "superior": bool(lower > 0.0),
    }


def paired_vertebra_bootstrap(
    weak: pd.DataFrame,
    mixed: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired vertebra AUROC/AUPRC safety differences."""
    weak_aligned, mixed_aligned = _paired_frames(weak, mixed)
    random_generator = np.random.default_rng(seed)
    groups = _cluster_groups(weak_aligned)
    targets = weak_aligned["label"].to_numpy(dtype=np.int8)
    weak_probabilities = weak_aligned["pred_prob"].to_numpy(dtype=np.float64)
    mixed_probabilities = mixed_aligned["pred_prob"].to_numpy(dtype=np.float64)
    auroc_deltas = np.empty(iterations, dtype=np.float64)
    auprc_deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = _cluster_bootstrap_indices(groups, random_generator)
        sampled_targets = targets[indices]
        weak_probability = weak_probabilities[indices]
        mixed_probability = mixed_probabilities[indices]
        auroc_deltas[iteration] = _fast_auroc(
            sampled_targets,
            mixed_probability,
        ) - _fast_auroc(sampled_targets, weak_probability)
        auprc_deltas[iteration] = _fast_average_precision(
            sampled_targets,
            mixed_probability,
        ) - _fast_average_precision(sampled_targets, weak_probability)
    weak_point = vertebra_metrics(weak_aligned)
    mixed_point = vertebra_metrics(mixed_aligned)
    auroc_lower, auroc_upper = np.quantile(auroc_deltas, [0.025, 0.975])
    auprc_lower, auprc_upper = np.quantile(auprc_deltas, [0.025, 0.975])
    auroc_delta = mixed_point["auroc"] - weak_point["auroc"]
    auprc_delta = mixed_point["auprc"] - weak_point["auprc"]
    return {
        "delta_auroc": float(auroc_delta),
        "delta_auroc_ci95": [float(auroc_lower), float(auroc_upper)],
        "delta_auprc": float(auprc_delta),
        "delta_auprc_ci95": [float(auprc_lower), float(auprc_upper)],
        "iterations": iterations,
        "pass": bool(auroc_lower > -0.010 and auprc_delta >= -0.020),
    }


def build_report(
    weak: pd.DataFrame,
    mixed: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the complete Stage4 confirmatory report."""
    report: dict[str, Any] = {
        "weak_only": {
            "region_all": region_metrics(weak),
            "region_excluding_c2": region_metrics(weak, exclude_c2=True),
            "region_fold_diagnostic": fold_region_metrics(weak),
            "vertebra": vertebra_metrics(weak),
        },
        "mixed": {
            "region_all": region_metrics(mixed),
            "region_excluding_c2": region_metrics(mixed, exclude_c2=True),
            "region_fold_diagnostic": fold_region_metrics(mixed),
            "vertebra": vertebra_metrics(mixed),
        },
        "primary_hypothesis": paired_region_bootstrap(
            weak,
            mixed,
            iterations=iterations,
            seed=seed,
        ),
        "vertebra_safety_gate": paired_vertebra_bootstrap(
            weak,
            mixed,
            iterations=iterations,
            seed=seed,
        ),
    }
    for arm_name, frame in (("weak_only", weak), ("mixed", mixed)):
        raw_macro = report[arm_name]["region_all"]["macro_ap"]
        ranked = region_metrics(frame, percentile_rank=True)
        difference = abs(ranked["macro_ap"] - raw_macro)
        report[arm_name]["fold_percentile_rank_sensitivity"] = {
            **ranked,
            "absolute_difference_from_raw": difference,
            "fold_scale_sensitive": bool(difference >= 0.03),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    output_root = ROOT / "train_models/stage4/outputs"
    parser.add_argument(
        "--weak-dir",
        type=Path,
        default=output_root / "stage4_weak_only_v2",
    )
    parser.add_argument(
        "--mixed-dir",
        type=Path,
        default=output_root / "stage4_mixed_v2",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=output_root / "stage4_confirmatory_report_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    weak = load_seed_ensemble(arguments.weak_dir, arguments.seeds)
    mixed = load_seed_ensemble(arguments.mixed_dir, arguments.seeds)
    report = build_report(
        weak,
        mixed,
        iterations=arguments.iterations,
        seed=arguments.bootstrap_seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
