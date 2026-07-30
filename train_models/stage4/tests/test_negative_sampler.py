import json
from collections import Counter
from pathlib import Path

from train_models.stage4.src.negative_sampler import NegativeRegionSampler


def _item(study_uid: str, vertebra: str, supervision: str) -> dict[str, object]:
    return {
        "study_uid": study_uid,
        "vertebra": vertebra,
        "label": int(supervision != "negative"),
        "region_supervision": supervision,
    }


def test_negative_sampler_matches_ratio_levels_and_writes_manifest(
    tmp_path: Path,
) -> None:
    strong = [
        _item("p1", "C3", "strong"),
        _item("p2", "C3", "strong"),
        _item("p3", "C5", "strong"),
    ]
    negatives = [
        _item(f"n{index}", level, "negative")
        for index, level in enumerate(["C3"] * 6 + ["C5"] * 5)
    ]
    sampler = NegativeRegionSampler(strong, negatives, tmp_path)

    sampled = sampler.sample(epoch=0)

    assert len(sampled) == len(strong)
    assert Counter(item["vertebra"] for item in sampled) == Counter(
        item["vertebra"] for item in strong
    )
    assert len({item["study_uid"] for item in sampled}) == len(sampled)
    manifest_path = tmp_path / "negative_manifest_epoch0.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert len(manifest["bags"]) == len(strong)


def test_negative_sampler_changes_selection_between_epochs(tmp_path: Path) -> None:
    strong = [_item(f"p{index}", "C4", "strong") for index in range(4)]
    negatives = [_item(f"n{index}", "C4", "negative") for index in range(20)]
    sampler = NegativeRegionSampler(strong, negatives, tmp_path)

    epoch_zero = {
        (item["study_uid"], item["vertebra"]) for item in sampler.sample(epoch=0)
    }
    epoch_one = {
        (item["study_uid"], item["vertebra"]) for item in sampler.sample(epoch=1)
    }

    assert epoch_zero != epoch_one
    assert (tmp_path / "negative_manifest_epoch1.json").exists()
