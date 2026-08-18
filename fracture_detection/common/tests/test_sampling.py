from __future__ import annotations

from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)


def test_epoch_shuffle_sampler_is_reproducible_and_epoch_specific() -> None:
    first = EpochShuffleSampler(range(8), seed=12)
    second = EpochShuffleSampler(range(8), seed=12)

    first.set_epoch(3)
    second.set_epoch(3)
    order = list(first)

    assert order == list(second)
    assert sorted(order) == list(range(8))
    first.set_epoch(4)
    assert list(first) != order


def test_annotated_cycle_sampler_continues_before_reshuffle() -> None:
    sampler = AnnotatedCycleSampler(dataset_size=5, samples_per_epoch=3, seed=7)

    sampler.set_epoch(0)
    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    first_cycle = first_epoch + second_epoch[:2]
    assert sorted(first_cycle) == list(range(5))
    assert len(second_epoch) == 3
