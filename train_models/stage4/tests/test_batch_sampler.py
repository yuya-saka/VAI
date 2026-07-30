import json
from collections import Counter
from pathlib import Path

from train_models.stage4.src.batch_sampler import Stage4StratifiedBatchSampler
from train_models.stage4.src.negative_sampler import NegativeRegionSampler


def _item(
    study_uid: str,
    vertebra: str,
    supervision: str,
) -> dict[str, object]:
    return {
        "study_uid": study_uid,
        "vertebra": vertebra,
        "label": int(supervision != "negative"),
        "region_supervision": supervision,
    }


def test_batch_sampler_enforces_strata_and_selected_negative_supervision(
    tmp_path: Path,
) -> None:
    strong = [
        _item("s1", "C3", "strong"),
        _item("s2", "C4", "strong"),
    ]
    weak = [_item(f"w{index}", "C5", "weak") for index in range(4)]
    negatives = [
        _item(f"n{index}", level, "negative")
        for index, level in enumerate(["C3", "C4"] + ["C5"] * 14)
    ]
    items = [*strong, *weak, *negatives]
    negative_sampler = NegativeRegionSampler(
        strong,
        negatives,
        tmp_path,
    )
    sampler = Stage4StratifiedBatchSampler(
        items,
        negative_sampler,
        batch_size=8,
        strong_per_batch=2,
        weak_per_batch=2,
        negative_per_batch=4,
    )

    sampler.set_epoch(0)
    batches = list(sampler)

    assert len(batches) == 3
    supervised_strong_count = 0
    supervised_negative_count = 0
    supervised_negative_keys: set[tuple[str, str]] = set()
    for batch in batches:
        strata = Counter(items[index]["region_supervision"] for index, _ in batch)
        assert strata == {"strong": 2, "weak": 2, "negative": 4}
        supervised_strong_count += sum(
            supervised
            for index, supervised in batch
            if items[index]["region_supervision"] == "strong"
        )
        supervised_negative_count += sum(
            supervised
            for index, supervised in batch
            if items[index]["region_supervision"] == "negative"
        )
        supervised_negative_keys.update(
            (str(items[index]["study_uid"]), str(items[index]["vertebra"]))
            for index, supervised in batch
            if supervised and items[index]["region_supervision"] == "negative"
        )
    assert supervised_strong_count == supervised_negative_count == 6
    assert len(supervised_negative_keys) == len(strong)

    exposure_path = tmp_path / "exposure_manifest_epoch0.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    assert exposure["region_supervision_exposures"] == {
        "negative": 6,
        "strong": 6,
    }
    assert exposure["region_supervision_group_weight"] == {
        "negative": 0.5,
        "strong": 0.5,
    }
    assert sum(exposure["vertebra_population_group_weight"].values()) == 1.0
    assert sum(exposure["negative_population_subgroup_weight"].values()) == 1.0


def test_batch_sampler_ddp_ranks_receive_disjoint_batches(tmp_path: Path) -> None:
    strong = [_item(f"s{index}", "C3", "strong") for index in range(4)]
    weak = [_item(f"w{index}", "C4", "weak") for index in range(8)]
    negatives = [_item(f"n{index}", "C3", "negative") for index in range(32)]
    items = [*strong, *weak, *negatives]

    def make_sampler(rank: int) -> Stage4StratifiedBatchSampler:
        negative_sampler = NegativeRegionSampler(
            strong,
            negatives,
            tmp_path,
            write_manifest=rank == 0,
        )
        sampler = Stage4StratifiedBatchSampler(
            items,
            negative_sampler,
            batch_size=8,
            strong_per_batch=2,
            weak_per_batch=2,
            negative_per_batch=4,
            rank=rank,
            world_size=2,
        )
        sampler.set_epoch(0)
        return sampler

    rank_zero = list(make_sampler(0))
    rank_one = list(make_sampler(1))

    assert len(rank_zero) == len(rank_one)
    assert rank_zero != rank_one
    global_batches = [*rank_zero, *rank_one]
    region_counts = Counter(
        items[index]["region_supervision"]
        for batch in global_batches
        for index, supervised in batch
        if supervised
    )
    assert region_counts == {"strong": 12, "negative": 12}
