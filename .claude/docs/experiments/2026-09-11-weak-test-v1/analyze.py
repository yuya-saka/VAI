"""Recompute the saved outer predictions and plot inner learning curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    repository = output_dir.parents[3]
    weak_dir = repository / "fracture_detection/weak/outputs/09_10/test_v1/outer0"
    baseline_dir = (
        repository
        / "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加/outer0"
    )
    history = pd.read_csv(weak_dir / "history.csv")
    predictions = pd.read_csv(weak_dir / "outer_predictions.csv")
    baseline = pd.read_csv(baseline_dir / "outer_predictions.csv")
    paired = predictions.merge(
        baseline,
        on=["study_id", "level"],
        validate="one_to_one",
        suffixes=("_weak", "_baseline"),
    )
    assert len(paired) == len(predictions) == len(baseline)
    assert paired.vertebra_target_weak.equals(paired.vertebra_target_baseline)
    assert predictions.region_observed_all.all()
    best_pass = int(
        json.loads((weak_dir / "fold_metrics.json").read_text())["best_gt_pass"]
    )
    targets = paired.vertebra_target_weak
    whole_results = {}
    for name, scores in {
        "baseline0": paired.vertebra_score,
        "weak": paired.p_whole,
    }.items():
        negatives = scores[targets == 0]
        ranking = np.argsort(-scores.to_numpy(), kind="stable")
        whole_results[name] = {
            "n": len(paired),
            "positives": int(targets.sum()),
            "ap": float(average_precision_score(targets, scores)),
            "auroc": float(roc_auc_score(targets, scores)),
            "unweighted_bce": float(log_loss(targets, scores)),
            "negative_score_mean": float(negatives.mean()),
            "negative_score_p90": float(negatives.quantile(0.9)),
            "precision_at_top262": float(targets.iloc[ranking[:262]].mean()),
        }
    region_results = {}
    for name, frame in {
        "annotated_positive_only": predictions.query(
            'bag_group == "annotated_positive"'
        ),
        "annotated_positive_and_negative": predictions.query(
            'bag_group != "weak_positive"'
        ),
    }.items():
        region_ap = [
            float(
                average_precision_score(frame[f"region_{region}"], frame[f"q_{region}"])
            )
            for region in range(1, 5)
        ]
        region_bce = [
            float(log_loss(frame[f"region_{region}"], frame[f"q_{region}"]))
            for region in range(1, 5)
        ]
        region_results[name] = {
            "n": len(frame),
            "region_ap": region_ap,
            "macro_ap": float(np.mean(region_ap)),
            "region_bce": region_bce,
            "mean_bce": float(np.mean(region_bce)),
        }
    weak_positive = predictions.query('bag_group == "weak_positive"')
    negative = predictions.query('bag_group == "negative"')
    negative_max = negative[[f"q_{region}" for region in range(1, 5)]].max(axis=1)
    maximum_scores = predictions[[f"q_{region}" for region in range(1, 5)]].max(axis=1)
    result = {
        "best_gt_pass": best_pass,
        "inner_at_best_pass": history.set_index("gt_pass")
        .loc[best_pass]
        .filter(like="val_")
        .to_dict(),
        "outer_whole": whole_results,
        "outer_regions": region_results,
        "outer_weak_positive": {
            "n": len(weak_positive),
            "median_p_whole": float(weak_positive.p_whole.median()),
            "fraction_p_whole_above_0_95": float((weak_positive.p_whole > 0.95).mean()),
            "mean_positive_or_loss": float(-np.log(weak_positive.p_whole).mean()),
        },
        "outer_negative": {
            "n": len(negative),
            "p_whole_at_least_0_5": int((negative.p_whole >= 0.5).sum()),
            "max_q_at_least_0_5": int((negative_max >= 0.5).sum()),
            "p_whole_at_least_0_5_with_all_q_below_0_5": int(
                ((negative.p_whole >= 0.5) & (negative_max < 0.5)).sum()
            ),
        },
        "exploratory_outer_max_q_ap_not_a_selected_model": float(
            average_precision_score(predictions.vertebra_target, maximum_scores)
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    passes = history.gt_pass
    axes[0, 0].plot(passes, history.train_annotated_loss / 4, label="Train, augmented")
    axes[0, 0].plot(passes, history.val_region_bce_annotated, label="Inner, 53 GT bags")
    axes[0, 0].set(title="GT region BCE (mean per cell)", ylabel="BCE")
    axes[0, 1].plot(
        passes,
        history.val_region_macro_ap,
        color="tab:orange",
        label="Inner, 53 GT bags",
    )
    axes[0, 1].set(title="Conditional localization", ylabel="Region macro AP")
    axes[1, 0].plot(passes, history.train_weak_loss, label="Train, augmented")
    axes[1, 0].plot(
        passes, history.val_weak_loss, label="Inner, 215 weak-positive bags"
    )
    axes[1, 0].set(title="Weak-positive noisy-OR loss", ylabel="Mean -log(p_whole)")
    axes[1, 1].plot(
        passes,
        history.val_whole_average_precision,
        color="tab:orange",
        label="Weak model, inner",
    )
    baseline_metrics = json.loads((baseline_dir / "fold_metrics.json").read_text())
    baseline_inner_ap = baseline_metrics["auroc_checkpoint"]["validation"][
        "average_precision"
    ]
    axes[1, 1].axhline(
        baseline_inner_ap,
        color="tab:blue",
        linestyle=":",
        label="Baseline0, same inner",
    )
    axes[1, 1].set(title="Whole detection (2,687 inner bags)", ylabel="Whole AP")
    for axis in axes.flat:
        axis.axvline(
            best_pass, color="gray", linestyle="--", alpha=0.7, label="Selected pass 35"
        )
        axis.set_xlabel("GT-pass (40 optimizer steps)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "weak/test_v1, outer0 run: training and INNER validation", fontsize=14
    )
    figure.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
