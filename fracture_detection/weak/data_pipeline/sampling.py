"""GT-pass batch composition over the N/A/U bag groups.

A "GT-pass" is one full traversal of the annotated-positive (A) group, with
no bag repeated within the pass -- built the same way baseline0's
``EpochShuffleSampler`` reshuffles a natural stream once per epoch, applied
here to the (much smaller) A-only index set. The negative (N) and
weak-positive (U) groups are drawn from decisive, non-repeating queues that
carry across passes rather than resetting every pass -- this is exactly
baseline0's ``AnnotatedCycleSampler`` semantics (an infinite stream sliced by
a fixed per-call sample count, reshuffled only once every full cycle),
applied to each group's local index range and translated back to the
dataset's row indices.

Every batch is ``negative_per_batch + annotated_per_batch + weak_per_batch``
except the pass's last step, whose A slice may be short when
``len(A) % annotated_per_batch != 0`` (N and U stay full-size on that step;
see ``fracture_detection/REGION_MIL_DESIGN.md`` section 6). The sampler's
entire state is a function of ``seed`` and ``pass_index`` alone, so resuming
training only requires restoring ``pass_index``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from torch.utils.data import Sampler

from fracture_detection.baseline0.data.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.weak.data_pipeline.groups import (
    ANNOTATED_GROUP,
    NEGATIVE_GROUP,
    WEAK_GROUP,
)

# Seed offsets keep each group's shuffle/cycle independent even though they
# share the same base seed.
_ANNOTATED_SEED_OFFSET = 0
_NEGATIVE_SEED_OFFSET = 1_000_003
_WEAK_SEED_OFFSET = 2_000_003


@dataclass(frozen=True)
class PassComposition:
    """Bag counts actually presented in one GT-pass, for logging."""

    pass_index: int
    steps: int
    n_negative_presented: int
    n_annotated_presented: int
    n_weak_presented: int


class GtPassBatchSampler(Sampler[list[int]]):
    """Yield configured N/A/U batches of dataset row indices per GT-pass."""

    def __init__(
        self,
        group_indices: dict[str, list[int]],
        negative_per_batch: int,
        annotated_per_batch: int,
        weak_per_batch: int,
        seed: int,
    ) -> None:
        for group in (NEGATIVE_GROUP, ANNOTATED_GROUP, WEAK_GROUP):
            if not group_indices.get(group):
                raise ValueError(f"group_indices[{group!r}] must be non-empty")
        for name, value in (
            ("negative_per_batch", negative_per_batch),
            ("annotated_per_batch", annotated_per_batch),
            ("weak_per_batch", weak_per_batch),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

        self.negative_indices = list(group_indices[NEGATIVE_GROUP])
        self.annotated_indices = list(group_indices[ANNOTATED_GROUP])
        self.weak_indices = list(group_indices[WEAK_GROUP])
        self.negative_per_batch = negative_per_batch
        self.annotated_per_batch = annotated_per_batch
        self.weak_per_batch = weak_per_batch
        self.seed = seed
        self._pass_index = 0

        self.steps_per_pass = math.ceil(
            len(self.annotated_indices) / annotated_per_batch
        )
        self._negative_cycle = AnnotatedCycleSampler(
            dataset_size=len(self.negative_indices),
            samples_per_epoch=self.steps_per_pass * negative_per_batch,
            seed=seed + _NEGATIVE_SEED_OFFSET,
        )
        self._weak_cycle = AnnotatedCycleSampler(
            dataset_size=len(self.weak_indices),
            samples_per_epoch=self.steps_per_pass * weak_per_batch,
            seed=seed + _WEAK_SEED_OFFSET,
        )

    def set_pass(self, pass_index: int) -> None:
        """Select which GT-pass the next ``__iter__`` call will produce."""
        if pass_index < 0:
            raise ValueError("pass_index must be >= 0")
        self._pass_index = pass_index

    @property
    def pass_index(self) -> int:
        """The GT-pass index the sampler is currently configured for."""
        return self._pass_index

    def __iter__(self) -> Iterator[list[int]]:
        annotated_order = _shuffled_global_indices(
            self.annotated_indices, self.seed + _ANNOTATED_SEED_OFFSET, self._pass_index
        )
        self._negative_cycle.set_epoch(self._pass_index)
        self._weak_cycle.set_epoch(self._pass_index)
        # include_metadata=False (the default) always yields plain int, but
        # the sampler's type signature is int | SampleIndex.
        negative_local = cast(list[int], list(self._negative_cycle))
        weak_local = cast(list[int], list(self._weak_cycle))

        for step in range(self.steps_per_pass):
            annotated_batch = annotated_order[
                step * self.annotated_per_batch : (step + 1) * self.annotated_per_batch
            ]
            negative_batch = [
                self.negative_indices[local_index]
                for local_index in negative_local[
                    step * self.negative_per_batch : (step + 1)
                    * self.negative_per_batch
                ]
            ]
            weak_batch = [
                self.weak_indices[local_index]
                for local_index in weak_local[
                    step * self.weak_per_batch : (step + 1) * self.weak_per_batch
                ]
            ]
            yield [*negative_batch, *annotated_batch, *weak_batch]

    def __len__(self) -> int:
        return self.steps_per_pass

    def pass_composition(self) -> PassComposition:
        """Describe the bag counts the current pass will present."""
        return PassComposition(
            pass_index=self._pass_index,
            steps=self.steps_per_pass,
            n_negative_presented=self.steps_per_pass * self.negative_per_batch,
            n_annotated_presented=len(self.annotated_indices),
            n_weak_presented=self.steps_per_pass * self.weak_per_batch,
        )


def _shuffled_global_indices(
    global_indices: list[int], seed: int, pass_index: int
) -> list[int]:
    """Full reshuffle of ``global_indices`` for one pass, no repeats."""
    sampler = EpochShuffleSampler(data_source=global_indices, seed=seed)
    sampler.set_epoch(pass_index)
    local_order = cast(list[int], list(sampler))
    return [global_indices[local_index] for local_index in local_order]
