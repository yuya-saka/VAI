from pathlib import Path

import pytest

from train_models.stage4.src.stage4_folds import (
    load_stage4_fold_map,
    split_by_stage4_fold,
)

ROOT = Path(__file__).resolve().parents[3]


def test_load_stage4_fold_map_verifies_frozen_manifest() -> None:
    fold_map = load_stage4_fold_map(ROOT / "data/rsna_data/stage4_folds.csv")

    assert len(fold_map) == 2009
    assert set(fold_map.values()) == set(range(5))


def test_load_stage4_fold_map_rejects_unknown_hash(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    path.write_text("study_id,fold\ns1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_stage4_fold_map(path)


def test_split_by_stage4_fold_uses_study_assignments() -> None:
    items = [
        {"study_uid": "s1", "vertebra": "C2"},
        {"study_uid": "s1", "vertebra": "C3"},
        {"study_uid": "s2", "vertebra": "C4"},
    ]

    train, valid = split_by_stage4_fold(items, {"s1": 0, "s2": 1}, val_fold=0)

    assert [item["study_uid"] for item in train] == ["s2"]
    assert [item["study_uid"] for item in valid] == ["s1", "s1"]
