from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fracture_detection.common.metrics import (
    binary_decision_metrics,
    evaluate_prediction_frame,
    evaluate_vertebra_prediction_frame,
    select_f1_threshold,
    threshold_metrics,
)


def test_threshold_metrics_reports_precision_recall_and_f1() -> None:
    targets = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])

    metrics = threshold_metrics(targets, scores, threshold=0.5)

    assert metrics == {
        "threshold": 0.5,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 1,
        "predicted_positives": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_select_f1_threshold_uses_validation_scores() -> None:
    targets = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    metrics = select_f1_threshold(targets, scores)

    assert metrics["threshold"] == 0.7
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)


def test_select_f1_threshold_breaks_ties_toward_higher_threshold() -> None:
    targets = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    metrics = select_f1_threshold(targets, scores)

    assert metrics["threshold"] == 0.9
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_binary_decision_metrics_rejects_non_binary_predictions() -> None:
    with pytest.raises(ValueError, match="predictionは0/1"):
        binary_decision_metrics(np.array([0, 1]), np.array([0, 2]))


def test_evaluate_prediction_frame_reports_perfect_predictions() -> None:
    region_targets = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    for index, targets in enumerate(region_targets):
        row: dict[str, object] = {
            "study_id": f"study-{index}",
            "level": f"C{index + 1}",
            "vertebra_target": index % 2,
            "vertebra_score": 0.9 if index % 2 else 0.1,
            "has_region_target": True,
        }
        for region_index in range(4):
            region_number = region_index + 1
            target = targets[region_index]
            row[f"region_{region_number}_target"] = target
            row[f"region_{region_number}_score"] = 0.9 if target else 0.1
        rows.append(row)
    predictions = pd.DataFrame(rows)

    metrics = evaluate_prediction_frame(predictions, n_bootstrap=20)

    assert metrics["vertebra"]["auroc"] == 1.0
    assert set(metrics["regions"]) == {
        "region_1",
        "region_2",
        "region_3",
        "region_4",
    }
    assert all(
        region_metrics["average_precision"] == 1.0
        for region_metrics in metrics["regions"].values()
    )
    assert "side_balanced_accuracy" not in metrics


def test_evaluate_vertebra_prediction_frame_requires_no_region_scores() -> None:
    predictions = pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "vertebra_target": 0,
                "vertebra_score": 0.1,
            },
            {
                "study_id": "study-b",
                "level": "C2",
                "vertebra_target": 1,
                "vertebra_score": 0.9,
            },
        ]
    )

    metrics = evaluate_vertebra_prediction_frame(predictions, n_bootstrap=20)

    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_evaluate_vertebra_prediction_frame_includes_saved_decisions() -> None:
    predictions = pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "vertebra_target": 0,
                "vertebra_score": 0.6,
                "vertebra_prediction": 1,
            },
            {
                "study_id": "study-b",
                "level": "C2",
                "vertebra_target": 1,
                "vertebra_score": 0.7,
                "vertebra_prediction": 1,
            },
        ]
    )

    metrics = evaluate_vertebra_prediction_frame(predictions, n_bootstrap=20)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)
