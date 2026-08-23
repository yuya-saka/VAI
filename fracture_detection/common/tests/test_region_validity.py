from __future__ import annotations

import pandas as pd
import pytest

from fracture_detection.common.constants import REGION_TARGET_VALID_COLUMNS
from fracture_detection.common.region_validity import attach_region_target_validity


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": "complete",
                "level": "C1",
                "has_region_target": True,
                "region_1": 1,
                "region_2": 0,
                "region_3": 0,
                "region_4": 0,
            },
            {
                "study_id": "partial",
                "level": "C2",
                "has_region_target": True,
                "region_1": 0,
                "region_2": 1,
                "region_3": 0,
                "region_4": 1,
            },
            {
                "study_id": "unannotated",
                "level": "C3",
                "has_region_target": False,
                "region_1": 0,
                "region_2": 0,
                "region_3": 0,
                "region_4": 0,
            },
        ]
    )


def test_attach_region_target_validity_preserves_only_observed_partial_targets() -> (
    None
):
    coverage = pd.DataFrame(
        [
            {"study_id": "complete", "level": "C1", "annotation_complete": True},
            {"study_id": "partial", "level": "C2", "annotation_complete": False},
        ]
    )

    result = attach_region_target_validity(_manifest(), coverage)

    assert result.loc[0, list(REGION_TARGET_VALID_COLUMNS)].tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert result.loc[1, list(REGION_TARGET_VALID_COLUMNS)].tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert result.loc[2, list(REGION_TARGET_VALID_COLUMNS)].tolist() == [
        False,
        False,
        False,
        False,
    ]


def test_attach_region_target_validity_rejects_missing_annotated_coverage() -> None:
    coverage = pd.DataFrame(
        [{"study_id": "complete", "level": "C1", "annotation_complete": True}]
    )
    with pytest.raises(ValueError, match="missing for an annotated bag"):
        attach_region_target_validity(_manifest(), coverage)


def test_attach_region_target_validity_rejects_duplicate_coverage() -> None:
    coverage = pd.DataFrame(
        [
            {"study_id": "complete", "level": "C1", "annotation_complete": True},
            {"study_id": "complete", "level": "C1", "annotation_complete": True},
        ]
    )
    with pytest.raises(ValueError, match="duplicate bags"):
        attach_region_target_validity(_manifest(), coverage)
