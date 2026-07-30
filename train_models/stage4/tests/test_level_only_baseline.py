from pathlib import Path

import pytest

from train_models.stage4.scripts.stage4_level_only_baseline import (
    compute_level_only_oof,
    summarize_macro_ap,
)

ROOT = Path(__file__).resolve().parents[3]


def test_level_only_production_baseline_matches_preregistered_values() -> None:
    records = compute_level_only_oof(
        ROOT / "data/rsna_data/fracture_region_labels_dicom.csv",
        ROOT / "data/rsna_data/stage4_folds.csv",
    )

    all_bags = summarize_macro_ap(records)
    excluding_c2 = summarize_macro_ap(records, exclude_c2=True)

    assert all_bags["n_bags"] == 268
    assert excluding_c2["n_bags"] == 231
    assert all_bags["macro_ap"] == pytest.approx(0.458, abs=0.002)
    assert excluding_c2["macro_ap"] == pytest.approx(0.345, abs=0.002)
