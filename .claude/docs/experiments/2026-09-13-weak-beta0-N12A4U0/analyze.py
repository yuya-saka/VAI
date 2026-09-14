"""Compare weak beta=0 with N/A/U=12/4/0 against beta=0 8/4/4 and Baseline 0.

All metrics are recomputed from each run's outer_predictions.csv. The helper
functions (whole_metrics, region_macro_ap, ensembles) are reused from the
previous experiment's analyze.py so both write-ups use identical definitions.
Differences are paired on the same vertebrae and bootstrapped by study.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUTPUT_DIR = Path(__file__).resolve().parent
REPO = OUTPUT_DIR.parents[3]
# Run as a plain script from the docs tree, so the repo root is not on sys.path.
sys.path.insert(0, str(REPO))

from fracture_detection.weak.evaluation.metrics import (  # noqa: E402
    expected_calibration_error,
)
NEW_DIR = REPO / "fracture_detection/weak/outputs/09_13/test_v2_beta=0_N12A4U0"
OLD_DIR = REPO / "fracture_detection/weak/outputs/09_12/test_v2_beta=0"
BASELINE_DIR = REPO / "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加"
PREVIOUS_ANALYZE = (
    OUTPUT_DIR.parent / "2026-09-12-weak-test-v2-beta0" / "analyze.py"
)
N_FOLDS = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260913
REGION_COLUMNS = ["q_1", "q_2", "q_3", "q_4"]


def _load_previous_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("previous_analyze", PREVIOUS_ANALYZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_previous_helpers()


def paired_fold(outer_fold: int) -> pd.DataFrame:
    new = helpers.load_predictions(NEW_DIR, outer_fold)
    old = helpers.load_predictions(OLD_DIR, outer_fold)
    baseline = helpers.load_predictions(BASELINE_DIR, outer_fold)
    keys = ["study_id", "level"]
    paired = (
        new.merge(
            old[[*keys, "p_whole", *REGION_COLUMNS, "vertebra_target"]],
            on=keys,
            validate="one_to_one",
            suffixes=("", "_old"),
        )
        .merge(
            baseline[[*keys, "vertebra_score", "vertebra_target"]],
            on=keys,
            validate="one_to_one",
            suffixes=("", "_baseline"),
        )
    )
    assert len(paired) == len(new) == len(old) == len(baseline)
    assert paired.vertebra_target.equals(paired.vertebra_target_old)
    assert paired.vertebra_target.equals(paired.vertebra_target_baseline)
    paired["outer_fold"] = outer_fold
    return paired


def whole_with_calibration(targets: pd.Series, scores: pd.Series) -> dict[str, float]:
    metrics = helpers.whole_metrics(targets, scores, int(targets.sum()))
    return {
        **metrics,
        "ece": expected_calibration_error(targets.to_numpy(), scores.to_numpy()),
        "mean_score": float(scores.mean()),
        "prevalence": float(targets.mean()),
    }


def old_region_frame(frame: pd.DataFrame) -> pd.DataFrame:
    old = frame[["bag_group", "region_1", "region_2", "region_3", "region_4"]].copy()
    for column in REGION_COLUMNS:
        old[column] = frame[f"{column}_old"]
    return old


def run_summary(frame: pd.DataFrame) -> dict[str, object]:
    targets = frame.vertebra_target
    return {
        "whole": {
            "new_12_4_0": whole_with_calibration(targets, frame.p_whole),
            "old_8_4_4": whole_with_calibration(targets, frame.p_whole_old),
            "baseline0": whole_with_calibration(targets, frame.vertebra_score),
        },
        "region": {
            "new_12_4_0": helpers.region_macro_ap(frame),
            "old_8_4_4": helpers.region_macro_ap(old_region_frame(frame)),
        },
    }


def pass_summary(outer_fold: int) -> dict[str, object]:
    result = {}
    for name, root in (("new_12_4_0", NEW_DIR), ("old_8_4_4", OLD_DIR)):
        history = pd.read_csv(root / f"outer{outer_fold}/history.csv")
        best = json.loads((root / f"outer{outer_fold}/fold_metrics.json").read_text())[
            "best_gt_pass"
        ]
        result[name] = {
            "best_gt_pass": int(best),
            "stopped_gt_pass": int(history.gt_pass.max()),
            "new_lr_after_best_pass": float(
                history.loc[history.gt_pass == best, "new_lr"].iloc[0]
            ),
        }
    return result


def _study_rows(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(rows) for rows in frame.groupby("study_id", sort=True).indices.values()
    ]


def _resample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    chosen = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[index] for index in chosen])


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "fraction_above_zero": float((array > 0).mean()),
        "n_valid_replicates": len(array),
    }


def paired_bootstrap(frame: pd.DataFrame) -> dict[str, object]:
    """Study-clustered paired bootstrap of pooled differences."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = _study_rows(frame.reset_index(drop=True))
    targets = frame.vertebra_target.to_numpy()
    new = frame.p_whole.to_numpy()
    old = frame.p_whole_old.to_numpy()
    baseline = frame.vertebra_score.to_numpy()
    ap_new_old, ap_new_base, ap_old_base, auc_new_old = [], [], [], []
    for _ in range(N_BOOTSTRAP):
        rows = _resample(groups, rng)
        y = targets[rows]
        ap_new = average_precision_score(y, new[rows])
        ap_old = average_precision_score(y, old[rows])
        ap_base = average_precision_score(y, baseline[rows])
        ap_new_old.append(ap_new - ap_old)
        ap_new_base.append(ap_new - ap_base)
        ap_old_base.append(ap_old - ap_base)
        auc_new_old.append(roc_auc_score(y, new[rows]) - roc_auc_score(y, old[rows]))

    annotated = frame[frame.bag_group == "annotated_positive"].reset_index(drop=True)
    annotated_groups = _study_rows(annotated)
    region_diff = []
    for _ in range(N_BOOTSTRAP):
        sample = annotated.iloc[_resample(annotated_groups, rng)]
        new_ap, old_ap = [], []
        valid = True
        for region in range(1, 5):
            labels = sample[f"region_{region}"]
            if labels.nunique() < 2:
                valid = False
                break
            new_ap.append(average_precision_score(labels, sample[f"q_{region}"]))
            old_ap.append(average_precision_score(labels, sample[f"q_{region}_old"]))
        if valid:
            region_diff.append(float(np.mean(new_ap) - np.mean(old_ap)))

    return {
        "whole_ap_new_minus_old": _interval(ap_new_old),
        "whole_ap_new_minus_baseline0": _interval(ap_new_base),
        "whole_ap_old_minus_baseline0": _interval(ap_old_base),
        "whole_auroc_new_minus_old": _interval(auc_new_old),
        "region_macro_ap_new_minus_old": _interval(region_diff),
    }


def ensembles(frames: dict[int, pd.DataFrame]) -> dict[str, object]:
    result = {}
    for name, score_column, q_suffix in (
        ("new_12_4_0", "p_whole", ""),
        ("old_8_4_4", "p_whole_old", "_old"),
    ):
        logit_input = {
            fold: frame[["study_id", "level", "vertebra_score", "vertebra_target"]].assign(
                p_whole=frame[score_column]
            )
            for fold, frame in frames.items()
        }
        max_input = {
            fold: frame[["study_id", "level", "vertebra_score", "vertebra_target"]].assign(
                p_whole=frame[score_column],
                **{column: frame[f"{column}{q_suffix}"] for column in REGION_COLUMNS},
            )
            for fold, frame in frames.items()
        }
        logit = helpers.ensemble_analysis(logit_input)
        maximum = helpers.max_combo_analysis(max_input)
        result[name] = {
            "logit_cv_selected_alpha_per_fold": logit["cv_selected_alpha_per_fold"],
            "logit_cv_pooled": logit["cv_selected_pooled"],
            "max_combo_pooled": maximum["pooled_max_combo"],
            "max_combo_per_fold_ap": {
                fold: metrics["max_combo"]["ap"]
                for fold, metrics in maximum["per_fold"].items()
            },
        }
    return result


def main() -> None:
    frames = {fold: paired_fold(fold) for fold in range(N_FOLDS)}
    pooled = pd.concat(frames.values(), ignore_index=True)
    result = {
        "per_fold": {
            fold: {**run_summary(frame), "passes": pass_summary(fold)}
            for fold, frame in frames.items()
        },
        "pooled": run_summary(pooled),
        "paired_study_bootstrap": paired_bootstrap(pooled),
        "ensembles_with_baseline0": ensembles(frames),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
