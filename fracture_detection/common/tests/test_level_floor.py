from __future__ import annotations

import pandas as pd
import pytest

from fracture_detection.common.constants import INPUT_MANIFEST_CSV
from fracture_detection.common.level_floor import build_cross_fitted_level_floor
from fracture_detection.common.splits import resolve_nested_folds


def test_cross_fitted_floor_uses_only_three_training_folds() -> None:
    manifest = pd.read_csv(INPUT_MANIFEST_CSV, dtype={"study_id": str, "level": str})

    floor = build_cross_fitted_level_floor(manifest)
    row = floor.iloc[0]
    assignment = resolve_nested_folds(int(row["fold"]))
    training = manifest[
        manifest["has_region_target"].astype(bool)
        & manifest["fold"].isin(assignment.train_folds)
        & manifest["level"].eq(row["level"])
    ]
    expected = (training["region_1"].sum() + 0.5) / (len(training) + 1.0)

    assert len(floor) == 268
    assert floor["study_id"].nunique() == 160
    assert row["region_1_floor_score"] == pytest.approx(expected)
    assert int(row["inner_fold"]) == assignment.inner_fold
