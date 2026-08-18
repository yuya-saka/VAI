from __future__ import annotations

import pandas as pd
import pytest

from fracture_detection.baseline0.cli.evaluate import validate_outer_prediction_frame


def _expected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"study_id": "study-a", "level": "C1", "fold": 0, "vertebra_target": 0},
            {"study_id": "study-b", "level": "C2", "fold": 0, "vertebra_target": 1},
        ]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "fold": 0,
                "vertebra_target": 0,
                "vertebra_score": 0.1,
                "decision_threshold": 0.6,
                "vertebra_prediction": 0,
            },
            {
                "study_id": "study-b",
                "level": "C2",
                "fold": 0,
                "vertebra_target": 1,
                "vertebra_score": 0.9,
                "decision_threshold": 0.6,
                "vertebra_prediction": 1,
            },
        ]
    )


def test_validate_outer_prediction_frame_accepts_exact_outer_ids() -> None:
    predictions = _predictions()

    validated = validate_outer_prediction_frame(predictions, _expected(), 0)

    assert len(validated) == 2


def test_validate_outer_prediction_frame_rejects_missing_id() -> None:
    predictions = _predictions().iloc[:1].copy()

    with pytest.raises(ValueError, match="一致しません"):
        validate_outer_prediction_frame(predictions, _expected(), 0)


def test_validate_outer_prediction_frame_rejects_changed_outer_threshold() -> None:
    predictions = _predictions()
    predictions.loc[1, "decision_threshold"] = 0.7

    with pytest.raises(ValueError, match="閾値が一定"):
        validate_outer_prediction_frame(predictions, _expected(), 0)


def test_validate_outer_prediction_frame_rejects_inconsistent_decision() -> None:
    predictions = _predictions()
    predictions.loc[0, "vertebra_prediction"] = 1

    with pytest.raises(ValueError, match="閾値適用結果"):
        validate_outer_prediction_frame(predictions, _expected(), 0)
