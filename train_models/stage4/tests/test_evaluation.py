from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from train_models.stage4.scripts.stage4_evaluate import (
    _fast_auroc,
    _fast_average_precision,
    build_report,
    load_seed_ensemble,
)


def _predictions(mixed: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for patient in range(4):
            positive = patient % 2
            row: dict[str, object] = {
                "study_uid": f"f{fold}-p{patient}",
                "vertebra": f"C{patient + 2}",
                "fold": fold,
                "label": positive,
                "pred_prob": 0.8 if positive else 0.2,
                "region_supervised": True,
            }
            for region in range(1, 5):
                target = int((patient + region) % 2 == 0)
                weak_probability = 0.5
                row[f"region_target_r{region}"] = target
                row[f"region_prob_r{region}"] = (
                    (0.9 if target else 0.1) if mixed else weak_probability
                )
                row[f"region_valid_r{region}"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_seed_ensemble_averages_bag_probabilities(tmp_path: Path) -> None:
    arm_dir = tmp_path / "arm"
    first = _predictions()
    second = _predictions()
    first["pred_prob"] = 0.2
    second["pred_prob"] = 0.6
    for seed, frame in ((42, first), (43, second)):
        seed_dir = arm_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        frame.to_csv(seed_dir / "oof_predictions.csv", index=False)

    ensemble = load_seed_ensemble(arm_dir, [42, 43])

    np.testing.assert_allclose(ensemble["pred_prob"], 0.4)


def test_build_report_detects_region_improvement_and_safety() -> None:
    weak = _predictions(mixed=False)
    mixed = _predictions(mixed=True)

    report = build_report(weak, mixed, iterations=100, seed=42)

    assert report["primary_hypothesis"]["delta_macro_ap"] > 0
    assert report["primary_hypothesis"]["superior"] is True
    assert report["vertebra_safety_gate"]["delta_auroc"] == 0
    assert report["vertebra_safety_gate"]["pass"] is True
    assert report["weak_only"]["region_all"]["n_bags"] == 20
    assert report["weak_only"]["region_excluding_c2"]["n_bags"] == 15


def test_fast_bootstrap_metrics_match_sklearn_with_ties() -> None:
    targets = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.8, 0.8, 0.4, 0.6, 0.4])

    assert _fast_average_precision(targets, probabilities) == pytest.approx(
        average_precision_score(targets, probabilities)
    )
    assert _fast_auroc(targets, probabilities) == pytest.approx(
        roc_auc_score(targets, probabilities)
    )
