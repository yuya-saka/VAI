"""全アームで共有する決定的なnatural・annotated sampler。"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SampleIndex:
    """datasetへepochとstream内ordinalを渡すindex。"""

    index: int
    epoch: int
    ordinal: int


class EpochShuffleSampler(Sampler[int | SampleIndex]):
    """epoch番号と共通seedだけでnatural stream順序を決める。"""

    def __init__(
        self, data_source: Sized, seed: int, include_metadata: bool = False
    ) -> None:
        if len(data_source) < 1:
            raise ValueError("natural streamのdatasetが空です")
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.include_metadata = include_metadata

    def set_epoch(self, epoch: int) -> None:
        """次に生成する順序のepochを設定する。"""
        if epoch < 0:
            raise ValueError("epochは0以上である必要があります")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int | SampleIndex]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.data_source), generator=generator).tolist()
        if not self.include_metadata:
            yield from order
            return
        for ordinal, index in enumerate(order):
            yield SampleIndex(index=index, epoch=self.epoch, ordinal=ordinal)

    def __len__(self) -> int:
        return len(self.data_source)


class AnnotatedCycleSampler(Sampler[int | SampleIndex]):
    """全件消費後だけ再shuffleするannotated streamを生成する。"""

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
        include_metadata: bool = False,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("annotated datasetが空です")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epochは1以上である必要があります")
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.include_metadata = include_metadata

    def set_epoch(self, epoch: int) -> None:
        """連続stream上の対象epochを設定する。"""
        if epoch < 0:
            raise ValueError("epochは0以上である必要があります")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int | SampleIndex]:
        start = self.epoch * self.samples_per_epoch
        stop = start + self.samples_per_epoch
        first_cycle = start // self.dataset_size
        last_cycle = math.ceil(stop / self.dataset_size)
        stream: list[int] = []
        for cycle in range(first_cycle, last_cycle):
            generator = torch.Generator()
            generator.manual_seed(self.seed + cycle)
            stream.extend(
                torch.randperm(self.dataset_size, generator=generator).tolist()
            )
        offset = start - first_cycle * self.dataset_size
        selected = stream[offset : offset + self.samples_per_epoch]
        if not self.include_metadata:
            yield from selected
            return
        for ordinal, index in enumerate(selected):
            yield SampleIndex(index=index, epoch=self.epoch, ordinal=ordinal)

    def __len__(self) -> int:
        return self.samples_per_epoch
