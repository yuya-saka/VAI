from __future__ import annotations

import numpy as np
import pandas as pd

from fracture_detection.common.metrics import evaluate_prediction_frame


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
    assert metrics["regions"]["macro_average_precision"] == 1.0
    assert metrics["side_balanced_accuracy"] == 1.0
