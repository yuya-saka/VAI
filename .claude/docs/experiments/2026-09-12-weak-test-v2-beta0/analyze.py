"""Compare weak/test_v2_beta=0 (5 outer folds) against Baseline 0 and the
matched beta=1 (test_v2) outer0 control.

Reuses the same recomputation approach as
`.claude/docs/experiments/2026-09-11-weak-test-v1/analyze.py`: outer metrics
are recomputed from `outer_predictions.csv`, not read from
`fold_metrics.json.best_metrics` (which is an inner-validation snapshot).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[4]
BETA0_DIR = REPO / "fracture_detection/weak/outputs/09_12/test_v2_beta=0"
BETA1_DIR = REPO / "fracture_detection/weak/outputs/09_12/test_v2"
BASELINE_DIR = REPO / "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加"
OUTPUT_DIR = Path(__file__).resolve().parent


def whole_metrics(targets: pd.Series, scores: pd.Series, top_k: int) -> dict[str, float]:
    negatives = scores[targets == 0]
    positives = scores[targets == 1]
    ranking = np.argsort(-scores.to_numpy(), kind="stable")
    top100 = min(100, len(targets))
    return {
        "n": int(len(targets)),
        "positives": int(targets.sum()),
        "ap": float(average_precision_score(targets, scores)),
        "auroc": float(roc_auc_score(targets, scores)),
        "unweighted_bce": float(log_loss(targets, scores)),
        "negative_score_mean": float(negatives.mean()),
        "negative_score_p90": float(negatives.quantile(0.9)),
        "negative_score_ge_0_5": int((negatives >= 0.5).sum()),
        "positive_score_mean": float(positives.mean()),
        "positive_score_median": float(positives.median()),
        "positives_in_top100": int(targets.iloc[ranking[:top100]].sum()),
        f"precision_at_top{top_k}": float(targets.iloc[ranking[:top_k]].mean()),
    }


def region_macro_ap(frame: pd.DataFrame) -> dict[str, float]:
    annotated = frame.query('bag_group == "annotated_positive"')
    region_ap = [
        float(average_precision_score(annotated[f"region_{r}"], annotated[f"q_{r}"]))
        for r in range(1, 5)
    ]
    return {
        "n": int(len(annotated)),
        "region_ap": region_ap,
        "macro_ap": float(np.mean(region_ap)),
    }


def load_predictions(root: Path, outer_fold: int) -> pd.DataFrame:
    return pd.read_csv(
        root / f"outer{outer_fold}/outer_predictions.csv",
        dtype={"study_id": str, "level": str},
    )


_EPS = 1e-6


def logit(probability: pd.Series) -> np.ndarray:
    clipped = probability.to_numpy().clip(_EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def logit_ensemble(weak: pd.Series, baseline: pd.Series, alpha: float) -> np.ndarray:
    """alpha=1 -> pure weak/beta0, alpha=0 -> pure baseline0."""
    combined_logit = alpha * logit(weak) + (1.0 - alpha) * logit(baseline)
    return 1.0 / (1.0 + np.exp(-combined_logit))


def select_alpha_by_ap(
    targets: pd.Series, weak: pd.Series, baseline: pd.Series, grid: np.ndarray
) -> tuple[float, float]:
    """Grid-search alpha maximizing AP on the given (training) population."""
    best_alpha, best_ap = 0.5, -1.0
    for alpha in grid:
        ap = average_precision_score(targets, logit_ensemble(weak, baseline, alpha))
        if ap > best_ap:
            best_alpha, best_ap = float(alpha), float(ap)
    return best_alpha, best_ap


def ensemble_analysis(
    per_fold_paired: dict[int, pd.DataFrame],
) -> dict[str, object]:
    """Logit-space ensemble of weak/beta0 p_whole and baseline0 vertebra_score.

    Reports a naive fixed alpha=0.5 blend and a leave-one-fold-out alpha
    selection (alpha chosen on the pooled OTHER 4 folds, applied to the held
    -out fold) so the reported cross-validated numbers are not tuned in
    sample on the same fold they are evaluated on.
    """
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    fixed_alpha_per_fold = {}
    cv_frames = []
    cv_alpha_per_fold = {}
    for outer_fold, frame in per_fold_paired.items():
        n_positive = int(frame.vertebra_target.sum())
        fixed_scores = logit_ensemble(frame.p_whole, frame.vertebra_score, 0.5)
        fixed_alpha_per_fold[outer_fold] = whole_metrics(
            frame.vertebra_target, pd.Series(fixed_scores), n_positive
        )

        other_folds = pd.concat(
            [f for fold, f in per_fold_paired.items() if fold != outer_fold],
            ignore_index=True,
        )
        alpha, train_ap = select_alpha_by_ap(
            other_folds.vertebra_target,
            other_folds.p_whole,
            other_folds.vertebra_score,
            grid,
        )
        cv_alpha_per_fold[outer_fold] = {"alpha": alpha, "other_folds_ap": train_ap}
        held_out = frame.copy()
        held_out["cv_ensemble_score"] = logit_ensemble(
            frame.p_whole, frame.vertebra_score, alpha
        )
        cv_frames.append(held_out)

    cv_pooled = pd.concat(cv_frames, ignore_index=True)
    n_positive_pooled = int(cv_pooled.vertebra_target.sum())
    cv_pooled_metrics = whole_metrics(
        cv_pooled.vertebra_target, cv_pooled.cv_ensemble_score, n_positive_pooled
    )

    # Exploratory, in-sample upper bound: alpha chosen on the full 5-fold pool.
    exploratory_alpha, exploratory_ap = select_alpha_by_ap(
        cv_pooled.vertebra_target, cv_pooled.p_whole, cv_pooled.vertebra_score, grid
    )
    exploratory_scores = logit_ensemble(
        cv_pooled.p_whole, cv_pooled.vertebra_score, exploratory_alpha
    )
    exploratory_metrics = whole_metrics(
        cv_pooled.vertebra_target, pd.Series(exploratory_scores), n_positive_pooled
    )

    return {
        "fixed_alpha_0_5_per_fold": fixed_alpha_per_fold,
        "cv_selected_alpha_per_fold": cv_alpha_per_fold,
        "cv_selected_pooled": cv_pooled_metrics,
        "exploratory_in_sample_best_alpha": exploratory_alpha,
        "exploratory_in_sample_pooled": exploratory_metrics,
    }


def disagreement_analysis(pooled: pd.DataFrame) -> dict[str, object]:
    """Where beta0's p_whole and baseline0's vertebra_score disagree.

    Explains the ensemble gain: beta0 gets dense negative-direction gradient
    on every negative bag regardless of beta (the negative_bag_loss term is
    not beta-weighted), while positive-direction gradient only reaches the
    ~20% GT-annotated bags when beta=0. Baseline0 has the opposite balance
    (dense positive AND negative supervision, no region structure).
    """
    negative = pooled[pooled.vertebra_target == 0]
    positive = pooled[pooled.vertebra_target == 1]
    low_threshold = 0.3
    return {
        "logit_space_pearson_correlation": {
            "negative": float(
                np.corrcoef(logit(negative.p_whole), logit(negative.vertebra_score))[
                    0, 1
                ]
            ),
            "positive": float(
                np.corrcoef(logit(positive.p_whole), logit(positive.vertebra_score))[
                    0, 1
                ]
            ),
        },
        "negative_false_positive_disagreement_at_0_5": {
            "n_negative": len(negative),
            "both_ge_0_5_unfixable": int(
                ((negative.p_whole >= 0.5) & (negative.vertebra_score >= 0.5)).sum()
            ),
            "beta0_only_ge_0_5": int(
                ((negative.p_whole >= 0.5) & (negative.vertebra_score < 0.5)).sum()
            ),
            "baseline0_only_ge_0_5": int(
                ((negative.p_whole < 0.5) & (negative.vertebra_score >= 0.5)).sum()
            ),
        },
        "positive_weak_score_disagreement_below_0_3": {
            "n_positive": len(positive),
            "both_confident": int(
                ((positive.p_whole >= low_threshold)
                & (positive.vertebra_score >= low_threshold)).sum()
            ),
            "baseline0_rescues_beta0_weak": int(
                ((positive.p_whole < low_threshold)
                & (positive.vertebra_score >= low_threshold)).sum()
            ),
            "beta0_rescues_baseline0_weak": int(
                ((positive.p_whole >= low_threshold)
                & (positive.vertebra_score < low_threshold)).sum()
            ),
            "both_weak_unfixable": int(
                ((positive.p_whole < low_threshold)
                & (positive.vertebra_score < low_threshold)).sum()
            ),
        },
        "mean_scores": {
            "negative_beta0": float(negative.p_whole.mean()),
            "negative_baseline0": float(negative.vertebra_score.mean()),
            "positive_beta0": float(positive.p_whole.mean()),
            "positive_baseline0": float(positive.vertebra_score.mean()),
        },
    }


def max_combo_analysis(per_fold_full: dict[int, pd.DataFrame]) -> dict[str, object]:
    """Parameter-free max(vertebra_score, q_1..q_4) ensemble, no tuning.

    Contrast against max(q_1..q_4) alone (no baseline0) to show the region
    scores alone are not a good whole detector -- the gain comes from using
    baseline0 as the primary signal with region scores as a recall safety
    net, not from the region max on its own.
    """
    region_columns = ["q_1", "q_2", "q_3", "q_4"]
    per_fold_metrics = {}
    frames = []
    for outer_fold, frame in per_fold_full.items():
        n_positive = int(frame.vertebra_target.sum())
        region_max = frame[region_columns].to_numpy().max(axis=1)
        combo = np.maximum(frame.vertebra_score.to_numpy(), region_max)
        per_fold_metrics[outer_fold] = {
            "max_combo": whole_metrics(
                frame.vertebra_target, pd.Series(combo), n_positive
            ),
            "max_region_only": whole_metrics(
                frame.vertebra_target, pd.Series(region_max), n_positive
            ),
        }
        enriched = frame.copy()
        enriched["max_combo_score"] = combo
        frames.append(enriched)

    pooled = pd.concat(frames, ignore_index=True)
    n_positive_pooled = int(pooled.vertebra_target.sum())
    pooled_region_max = pooled[region_columns].to_numpy().max(axis=1)
    return {
        "per_fold": per_fold_metrics,
        "pooled_max_combo": whole_metrics(
            pooled.vertebra_target, pooled.max_combo_score, n_positive_pooled
        ),
        "pooled_max_region_only": whole_metrics(
            pooled.vertebra_target, pd.Series(pooled_region_max), n_positive_pooled
        ),
    }


def main() -> None:
    beta0_frames = []
    baseline_frames = []
    per_fold = {}
    per_fold_paired = {}
    per_fold_full = {}
    for outer_fold in range(5):
        beta0 = load_predictions(BETA0_DIR, outer_fold)
        baseline = load_predictions(BASELINE_DIR, outer_fold)
        paired = beta0.merge(
            baseline[["study_id", "level", "vertebra_score", "vertebra_target"]],
            on=["study_id", "level"],
            validate="one_to_one",
            suffixes=("", "_baseline"),
        )
        assert len(paired) == len(beta0) == len(baseline)
        assert paired.vertebra_target.equals(paired.vertebra_target_baseline)
        per_fold_paired[outer_fold] = paired[
            ["study_id", "level", "p_whole", "vertebra_score", "vertebra_target"]
        ]
        per_fold_full[outer_fold] = paired[
            [
                "study_id",
                "level",
                "p_whole",
                "q_1",
                "q_2",
                "q_3",
                "q_4",
                "vertebra_score",
                "vertebra_target",
            ]
        ]
        n_positive = int(paired.vertebra_target.sum())
        per_fold[outer_fold] = {
            "n": len(paired),
            "n_positive": n_positive,
            "best_gt_pass": json.loads(
                (BETA0_DIR / f"outer{outer_fold}/fold_metrics.json").read_text()
            )["best_gt_pass"],
            "beta0_whole": whole_metrics(
                paired.vertebra_target, paired.p_whole, n_positive
            ),
            "baseline0_whole": whole_metrics(
                paired.vertebra_target, paired.vertebra_score, n_positive
            ),
            "beta0_region": region_macro_ap(beta0),
        }
        beta0_frames.append(beta0)
        baseline_frames.append(
            baseline[["study_id", "level", "vertebra_score", "vertebra_target"]]
        )

    pooled_beta0 = pd.concat(beta0_frames, ignore_index=True)
    pooled_baseline = pd.concat(baseline_frames, ignore_index=True)
    pooled = pooled_beta0.merge(
        pooled_baseline,
        on=["study_id", "level"],
        validate="one_to_one",
        suffixes=("", "_baseline"),
    )
    assert len(pooled) == len(pooled_beta0)
    n_positive_pooled = int(pooled.vertebra_target.sum())
    pooled_result = {
        "n": len(pooled),
        "n_positive": n_positive_pooled,
        "beta0_whole": whole_metrics(
            pooled.vertebra_target, pooled.p_whole, n_positive_pooled
        ),
        "baseline0_whole": whole_metrics(
            pooled.vertebra_target, pooled.vertebra_score, n_positive_pooled
        ),
        "beta0_region": region_macro_ap(pooled_beta0),
    }

    # Matched outer0-only ablation: identical architecture/LSE/tau/8-4-4 batches,
    # differing only in beta (1.0 vs 0.0) and patience_gt_passes (10 vs 15).
    beta0_outer0 = load_predictions(BETA0_DIR, 0)
    beta1_outer0 = load_predictions(BETA1_DIR, 0)
    matched = beta0_outer0.merge(
        beta1_outer0[["study_id", "level", "p_whole", "vertebra_target"]],
        on=["study_id", "level"],
        validate="one_to_one",
        suffixes=("_beta0", "_beta1"),
    )
    assert matched.vertebra_target_beta0.equals(matched.vertebra_target_beta1)
    n_positive_outer0 = int(matched.vertebra_target_beta0.sum())
    matched_result = {
        "n": len(matched),
        "n_positive": n_positive_outer0,
        "beta0_whole": whole_metrics(
            matched.vertebra_target_beta0, matched.p_whole_beta0, n_positive_outer0
        ),
        "beta1_whole": whole_metrics(
            matched.vertebra_target_beta1, matched.p_whole_beta1, n_positive_outer0
        ),
        "beta0_region": region_macro_ap(beta0_outer0),
        "beta1_region": region_macro_ap(beta1_outer0),
    }

    ensemble_result = ensemble_analysis(per_fold_paired)
    disagreement_result = disagreement_analysis(pooled)
    max_combo_result = max_combo_analysis(per_fold_full)

    result = {
        "per_fold": per_fold,
        "pooled_5fold_beta0_vs_baseline0": pooled_result,
        "matched_outer0_beta0_vs_beta1": matched_result,
        "ensemble_beta0_x_baseline0_logit_weighted": ensemble_result,
        "disagreement_beta0_vs_baseline0": disagreement_result,
        "ensemble_max_combo_beta0_x_baseline0": max_combo_result,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
