from __future__ import annotations

import pandas as pd

from fracture_detection.common.constants import REGION_TARGET_VALID_COLUMNS
from fracture_detection.common.manifest import (
    apply_quality_exclusions,
    assemble_manifest,
    build_manifest,
)


def test_assemble_manifest_keeps_only_complete_bags() -> None:
    inventory = pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "ct_bytes": 1,
                "vertebra_mask_bytes": 1,
                "region_4class_bytes": 1,
            },
            {
                "study_id": "study-b",
                "level": "C2",
                "ct_bytes": 1,
                "vertebra_mask_bytes": 1,
                "region_4class_bytes": 1,
            },
            {
                "study_id": "study-c",
                "level": "C3",
                "ct_bytes": 1,
                "vertebra_mask_bytes": 1,
                "region_4class_bytes": -1,
            },
        ]
    )
    vertebra_labels = pd.DataFrame(
        [
            {"study_id": "study-a", "level": "C1", "fractured": 1},
            {"study_id": "study-b", "level": "C2", "fractured": 0},
            {"study_id": "study-c", "level": "C3", "fractured": 1},
        ]
    )
    region_labels = pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "region_1": 1,
                "region_2": 0,
                "region_3": 0,
                "region_4": 1,
            }
        ]
    )
    folds = pd.DataFrame(
        [
            {"study_id": "study-a", "fold": 0},
            {"study_id": "study-b", "fold": 1},
            {"study_id": "study-c", "fold": 2},
        ]
    )

    manifest = assemble_manifest(inventory, vertebra_labels, region_labels, folds)

    assert manifest[["study_id", "level"]].values.tolist() == [
        ["study-a", "C1"],
        ["study-b", "C2"],
    ]
    assert manifest["has_region_target"].tolist() == [True, False]
    assert manifest.loc[1, "region_1"] == 0


def test_apply_quality_exclusions_removes_studies_and_levels() -> None:
    manifest = pd.DataFrame(
        [
            {"study_id": "study-a", "level": "C1"},
            {"study_id": "study-a", "level": "C2"},
            {"study_id": "study-b", "level": "C1"},
            {"study_id": "study-c", "level": "C3"},
        ]
    )
    excluded_studies = pd.DataFrame([{"study_uid": "study-a"}])
    excluded_levels = pd.DataFrame([{"study_uid": "study-b", "vertebra": "C1"}])

    filtered = apply_quality_exclusions(manifest, excluded_studies, excluded_levels)

    assert filtered[["study_id", "level"]].values.tolist() == [["study-c", "C3"]]
    assert filtered.attrs["quality_exclusions"] == {
        "rows_before": 4,
        "rows_removed": 3,
        "rows_after": 1,
        "listed_studies": 1,
        "listed_levels": 1,
    }


def test_frozen_manifest_matches_stage1_quality_filtered_population() -> None:
    manifest = build_manifest()

    assert len(manifest) == 13_432
    assert manifest["study_id"].nunique() == 2_009
    assert int(manifest["vertebra_target"].sum()) == 1_332
    assert int(manifest["has_region_target"].sum()) == 268
    annotated = manifest[manifest["has_region_target"].astype(bool)]
    complete = annotated[annotated["annotation_complete"].astype(bool)]
    partial = annotated[~annotated["annotation_complete"].astype(bool)]
    assert len(complete) == 235
    assert len(partial) == 33
    assert complete[list(REGION_TARGET_VALID_COLUMNS)].all().all()
    assert manifest[list(REGION_TARGET_VALID_COLUMNS)].sum().tolist() == [
        245,
        243,
        244,
        251,
    ]
    for region_index, validity_column in enumerate(
        REGION_TARGET_VALID_COLUMNS, start=1
    ):
        target_column = f"region_{region_index}"
        assert partial.loc[partial[target_column].eq(1), validity_column].all()
        assert not partial.loc[partial[target_column].eq(0), validity_column].any()
    assert manifest.attrs["quality_exclusions"]["rows_removed"] == 496
