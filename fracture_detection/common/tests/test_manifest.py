from __future__ import annotations

import pandas as pd

from fracture_detection.common.manifest import assemble_manifest


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
