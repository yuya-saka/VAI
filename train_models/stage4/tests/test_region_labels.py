from pathlib import Path

import numpy as np
import pandas as pd

from train_models.stage4.src.region_labels import (
    load_region_labels,
    region_supervision_of,
)

ROOT = Path(__file__).resolve().parents[3]


def test_load_region_labels_or_aggregates_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "study_id": "s1",
                "level": "C3",
                "run_id": "run_00",
                "region_1": 1,
                "region_2": 0,
                "region_3": 0,
                "region_4": 0,
            },
            {
                "study_id": "s1",
                "level": "C3",
                "run_id": "run_01",
                "region_1": 0,
                "region_2": 1,
                "region_3": 0,
                "region_4": 1,
            },
        ]
    ).to_csv(csv_path, index=False)

    labels = load_region_labels(csv_path)

    np.testing.assert_array_equal(labels[("s1", "C3")], [1, 1, 0, 1])
    assert labels[("s1", "C3")].dtype == np.int8


def test_region_supervision_of_classifies_all_three_types() -> None:
    labels = {("s1", "C3"): np.asarray([1, 0, 0, 0], dtype=np.int8)}

    assert region_supervision_of(0, ("s1", "C3"), labels) == "negative"
    assert region_supervision_of(1, ("s1", "C3"), labels) == "strong"
    assert region_supervision_of(1, ("s2", "C4"), labels) == "weak"


def test_production_region_labels_cover_268_bags() -> None:
    labels = load_region_labels(
        ROOT / "data/rsna_data/fracture_region_labels_dicom.csv"
    )

    assert len(labels) == 268
    assert all(label.shape == (4,) for label in labels.values())
