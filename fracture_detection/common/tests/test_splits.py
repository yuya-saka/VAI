from __future__ import annotations

import pandas as pd
import pytest

from fracture_detection.common.splits import (
    resolve_nested_folds,
    split_nested_manifest,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"study_id": f"study-{fold}", "level": "C1", "fold": fold}
            for fold in range(5)
        ]
    )


@pytest.mark.parametrize(
    ("outer_fold", "inner_fold", "train_folds"),
    [
        (0, 1, (2, 3, 4)),
        (1, 2, (0, 3, 4)),
        (4, 0, (1, 2, 3)),
    ],
)
def test_resolve_nested_folds_uses_cyclic_inner(
    outer_fold: int, inner_fold: int, train_folds: tuple[int, ...]
) -> None:
    assignment = resolve_nested_folds(outer_fold)

    assert assignment.inner_fold == inner_fold
    assert assignment.train_folds == train_folds


def test_split_nested_manifest_is_disjoint_and_complete() -> None:
    manifest = _manifest()

    train, inner, outer = split_nested_manifest(manifest, outer_fold=4)

    assert set(train["fold"]) == {1, 2, 3}
    assert inner["fold"].tolist() == [0]
    assert outer["fold"].tolist() == [4]
    assert len(train) + len(inner) + len(outer) == len(manifest)


def test_split_nested_manifest_rejects_patient_leakage() -> None:
    manifest = _manifest()
    manifest.loc[1, "study_id"] = manifest.loc[0, "study_id"]
    manifest.loc[1, "level"] = "C2"

    with pytest.raises(ValueError, match="複数fold"):
        split_nested_manifest(manifest, outer_fold=0)
