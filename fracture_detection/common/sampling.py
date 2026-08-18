"""全アームで共有する決定的なnatural・annotated sampler。"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """epoch番号と共通seedだけでnatural stream順序を決める。"""

    def __init__(self, data_source: Sized, seed: int) -> None:
        if len(data_source) < 1:
            raise ValueError("natural streamのdatasetが空です")
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """次に生成する順序のepochを設定する。"""
        if epoch < 0:
            raise ValueError("epochは0以上である必要があります")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class AnnotatedCycleSampler(Sampler[int]):
    """全件消費後だけ再shuffleするannotated streamを生成する。"""

    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: int) -> None:
        if dataset_size < 1:
            raise ValueError("annotated datasetが空です")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epochは1以上である必要があります")
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """連続stream上の対象epochを設定する。"""
        if epoch < 0:
            raise ValueError("epochは0以上である必要があります")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
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
        yield from stream[offset : offset + self.samples_per_epoch]

    def __len__(self) -> int:
        return self.samples_per_epoch
