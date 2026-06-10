"""data_utils.pyの患者単位CV分割テスト。"""

from __future__ import annotations

from learning.src.data_utils import split_bags_cv


def _make_bags(n_patients: int = 20) -> list[dict]:
    bags = []
    for patient_index in range(n_patients):
        patient_id = f"sample{patient_index}"
        for vertebra_index in range(3):
            bags.append({
                "patient_id": patient_id,
                "sample": patient_id,
                "vertebra": f"C{vertebra_index + 1}",
                "label": int((patient_index + vertebra_index) % 4 == 0),
            })
    return bags


def _patient_ids(bags: list[dict]) -> set[str]:
    return {str(bag["patient_id"]) for bag in bags}


def test_train_and_validation_have_no_patient_overlap() -> None:
    """各foldのtrainとvalで患者が重複しない。"""
    train, val = split_bags_cv(_make_bags(), n_splits=5, val_fold=0)

    assert _patient_ids(train).isdisjoint(_patient_ids(val))


def test_cv_validation_folds_cover_all_patients_once() -> None:
    """全val foldを合わせると全患者を1回ずつ含む。"""
    bags = _make_bags()
    validation_patients: list[str] = []

    for fold in range(5):
        _, val = split_bags_cv(bags, n_splits=5, val_fold=fold)
        validation_patients.extend(_patient_ids(val))

    assert len(validation_patients) == len(set(validation_patients))
    assert set(validation_patients) == _patient_ids(bags)
