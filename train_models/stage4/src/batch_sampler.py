"""Deterministic Stage4 strong/weak/negative batch composition."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler

from .negative_sampler import NegativeRegionSampler

SampleIndex = tuple[int, bool]


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item["study_uid"]), str(item["vertebra"])


def _cycled_indices(
    indices: list[int],
    count: int,
    random_generator: random.Random,
) -> list[int]:
    if not indices:
        raise ValueError("cannot sample an empty Stage4 stratum")
    result: list[int] = []
    while len(result) < count:
        shuffled = list(indices)
        random_generator.shuffle(shuffled)
        result.extend(shuffled[: count - len(result)])
    return result


class Stage4StratifiedBatchSampler(Sampler[list[SampleIndex]]):
    """Yield fixed-ratio batches and refresh region-supervised negatives."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        negative_sampler: NegativeRegionSampler,
        batch_size: int,
        strong_per_batch: int,
        weak_per_batch: int,
        negative_per_batch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        positive_weight: float = 2.0,
    ) -> None:
        if strong_per_batch + weak_per_batch + negative_per_batch != batch_size:
            raise ValueError("Stage4 stratum counts must sum to batch_size")
        if min(strong_per_batch, weak_per_batch, negative_per_batch) < 1:
            raise ValueError("every Stage4 stratum needs at least one batch slot")
        if strong_per_batch > negative_per_batch:
            raise ValueError(
                "negative_per_batch must cover one supervised negative per strong slot"
            )
        if rank not in range(world_size):
            raise ValueError("rank must be in [0, world_size)")
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        self.items = items
        self.negative_sampler = negative_sampler
        self.batch_size = batch_size
        self.strong_per_batch = strong_per_batch
        self.weak_per_batch = weak_per_batch
        self.negative_per_batch = negative_per_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.positive_weight = positive_weight
        self.epoch = 0
        self._batches: list[list[SampleIndex]] | None = None

        self.strong_indices = [
            index
            for index, item in enumerate(items)
            if item["region_supervision"] == "strong"
        ]
        self.weak_indices = [
            index
            for index, item in enumerate(items)
            if item["region_supervision"] == "weak"
        ]
        self.negative_indices = [
            index
            for index, item in enumerate(items)
            if item["region_supervision"] == "negative"
        ]
        if (
            not self.strong_indices
            or not self.weak_indices
            or not self.negative_indices
        ):
            raise ValueError("Stage4 training requires all three supervision strata")
        self.index_by_key = {_item_key(item): index for index, item in enumerate(items)}
        self.global_batch_count = math.ceil(len(items) / batch_size)
        if self.global_batch_count % world_size:
            self.global_batch_count += world_size - (
                self.global_batch_count % world_size
            )

    def set_epoch(self, epoch: int) -> None:
        """Rebuild deterministic batches and selected-negative supervision."""
        self.epoch = epoch
        self._batches = self._build_batches()

    def __iter__(self) -> Iterator[list[SampleIndex]]:
        if self._batches is None:
            self.set_epoch(self.epoch)
        assert self._batches is not None
        return iter(self._batches)

    def __len__(self) -> int:
        return self.global_batch_count // self.world_size

    def _build_batches(self) -> list[list[SampleIndex]]:
        random_generator = random.Random(self.seed + self.epoch)
        selected_negative_items = self.negative_sampler.sample(self.epoch)
        selected_negative_keys = {_item_key(item) for item in selected_negative_items}
        total_strong = self.global_batch_count * self.strong_per_batch
        total_weak = self.global_batch_count * self.weak_per_batch
        supervised_negative_count = total_strong
        unsupervised_negative_count = self.global_batch_count * (
            self.negative_per_batch - self.strong_per_batch
        )

        strong = _cycled_indices(self.strong_indices, total_strong, random_generator)
        weak = _cycled_indices(self.weak_indices, total_weak, random_generator)
        selected_indices = [
            self.index_by_key[key] for key in sorted(selected_negative_keys)
        ]
        supervised_negative = _cycled_indices(
            selected_indices,
            supervised_negative_count,
            random_generator,
        )
        remaining_negative_indices = [
            index
            for index in self.negative_indices
            if _item_key(self.items[index]) not in selected_negative_keys
        ]
        if unsupervised_negative_count > len(remaining_negative_indices):
            raise ValueError(
                "unsupervised negative slots exceed remaining negatives "
                "without replacement"
            )
        random_generator.shuffle(remaining_negative_indices)
        unsupervised_negative = remaining_negative_indices[:unsupervised_negative_count]

        batches: list[list[SampleIndex]] = []
        for batch_index in range(self.global_batch_count):
            batch: list[SampleIndex] = [
                *[
                    (index, True)
                    for index in strong[
                        batch_index * self.strong_per_batch : (batch_index + 1)
                        * self.strong_per_batch
                    ]
                ],
                *[
                    (index, False)
                    for index in weak[
                        batch_index * self.weak_per_batch : (batch_index + 1)
                        * self.weak_per_batch
                    ]
                ],
                *[
                    (index, True)
                    for index in supervised_negative[
                        batch_index * self.strong_per_batch : (batch_index + 1)
                        * self.strong_per_batch
                    ]
                ],
                *[
                    (index, False)
                    for index in unsupervised_negative[
                        batch_index
                        * (self.negative_per_batch - self.strong_per_batch) : (
                            batch_index + 1
                        )
                        * (self.negative_per_batch - self.strong_per_batch)
                    ]
                ],
            ]
            random_generator.shuffle(batch)
            batches.append(batch)
        self._validate_global_batches(batches, selected_negative_keys)
        if self.rank == 0 and self.negative_sampler.write_manifest:
            self._save_exposure_manifest(batches, selected_negative_keys)
        return batches[self.rank :: self.world_size]

    def _validate_global_batches(
        self,
        batches: list[list[SampleIndex]],
        selected_negative_keys: set[tuple[str, str]],
    ) -> None:
        strong_supervised = 0
        negative_supervised = 0
        observed_selected: set[tuple[str, str]] = set()
        for batch in batches:
            strata = Counter(
                str(self.items[index]["region_supervision"]) for index, _ in batch
            )
            expected = {
                "strong": self.strong_per_batch,
                "weak": self.weak_per_batch,
                "negative": self.negative_per_batch,
            }
            if strata != expected:
                raise RuntimeError(
                    f"invalid Stage4 batch composition: {dict(strata)} != {expected}"
                )
            batch_strong = sum(
                bool(supervised)
                for index, supervised in batch
                if self.items[index]["region_supervision"] == "strong"
            )
            batch_negative = sum(
                bool(supervised)
                for index, supervised in batch
                if self.items[index]["region_supervision"] == "negative"
            )
            if batch_strong != batch_negative:
                raise RuntimeError(
                    "strong and matched-negative region supervision must be 1:1 "
                    "in every global batch"
                )
            strong_supervised += batch_strong
            negative_supervised += batch_negative
            observed_selected.update(
                _item_key(self.items[index])
                for index, supervised in batch
                if supervised and self.items[index]["region_supervision"] == "negative"
            )
        if strong_supervised != negative_supervised:
            raise RuntimeError("epoch-level region supervision is not 1:1")
        if observed_selected != selected_negative_keys:
            raise RuntimeError("not every sampled negative received region supervision")

    def _save_exposure_manifest(
        self,
        batches: list[list[SampleIndex]],
        selected_negative_keys: set[tuple[str, str]],
    ) -> None:
        bag_exposures = Counter(
            str(self.items[index]["region_supervision"])
            for batch in batches
            for index, _ in batch
        )
        region_exposures = Counter(
            str(self.items[index]["region_supervision"])
            for batch in batches
            for index, supervised in batch
            if supervised
        )
        exposure_by_index = Counter(index for batch in batches for index, _ in batch)
        exposure_ranges = {}
        for name, indices in (
            ("strong", self.strong_indices),
            ("weak", self.weak_indices),
            ("negative", self.negative_indices),
        ):
            counts = [exposure_by_index[index] for index in indices]
            exposure_ranges[name] = {
                "min": min(counts),
                "max": max(counts),
            }
        n_strong = len(self.strong_indices)
        n_weak = len(self.weak_indices)
        n_negative = len(self.negative_indices)
        n_sampled_negative = len(selected_negative_keys)
        vertebra_denominator = self.positive_weight * (n_strong + n_weak) + n_negative
        payload = {
            "epoch": self.epoch,
            "global_batch_count": len(batches),
            "world_size": self.world_size,
            "unique_bags": {
                "strong": len(self.strong_indices),
                "weak": len(self.weak_indices),
                "negative": len(self.negative_indices),
                "sampled_negative": len(selected_negative_keys),
            },
            "bag_exposures": dict(bag_exposures),
            "bag_exposure_per_item_range": exposure_ranges,
            "vertebra_population_group_weight": {
                "strong": self.positive_weight * n_strong / vertebra_denominator,
                "weak": self.positive_weight * n_weak / vertebra_denominator,
                "negative": n_negative / vertebra_denominator,
            },
            "negative_population_subgroup_weight": {
                "sampled": n_sampled_negative / n_negative,
                "other": (n_negative - n_sampled_negative) / n_negative,
            },
            "region_supervision_exposures": dict(region_exposures),
            "region_supervision_group_weight": {
                "strong": 0.5,
                "negative": 0.5,
            },
        }
        path = (
            self.negative_sampler.manifest_dir
            / f"exposure_manifest_epoch{self.epoch}.json"
        )
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
