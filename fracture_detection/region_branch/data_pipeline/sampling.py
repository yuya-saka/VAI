"""補助region batch用のpersistent queue DataLoader。

`AnnotatedCycleSampler`（`baseline0/data/sampling.py`）は全件消費後だけ
再shuffleする連続streamサンプラーで、そのまま3ソース（human/negative/pseudo）の
persistent queueとして再利用できる。`samples_per_epoch`をnatural streamの
batch数（`steps_per_epoch`）×そのsourceのbatch/batchで固定することで、
train loaderと同じstep数だけ必ずbatchを供給する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fracture_detection.baseline0.data.sampling import AnnotatedCycleSampler
from fracture_detection.baseline0.training.trainer import create_data_loader


def create_source_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """1 sourceについて、natural streamと同じstep数を供給するDataLoaderを作る。"""
    if batch_size < 1:
        raise ValueError("batch_sizeは1以上である必要があります")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epochは1以上である必要があります")
    sampler = AnnotatedCycleSampler(
        dataset_size=len(dataset),  # type: ignore[arg-type]
        samples_per_epoch=steps_per_epoch * batch_size,
        seed=seed,
    )
    return create_data_loader(
        dataset,
        batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
        sampler=sampler,
    )


def set_source_loader_epoch(loader: DataLoader[Any], epoch: int) -> None:
    """persistent queue samplerへepochを伝え、周回位置を進める。"""
    sampler = loader.sampler
    if not isinstance(sampler, AnnotatedCycleSampler):
        raise TypeError("source loaderにはAnnotatedCycleSamplerが必要です")
    sampler.set_epoch(epoch)


def concatenate_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """複数sourceのbatch dictをbatch次元で連結する（human+negative+pseudoの結合等）。"""
    if not batches:
        raise ValueError("batchesが空です")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches):
        raise ValueError("batch間でkeyが一致しません")
    combined: dict[str, Any] = {}
    for key in keys:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], Tensor):
            combined[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged: list[Any] = []
            for value in values:
                merged.extend(value)
            combined[key] = merged
        else:
            raise TypeError(f"連結できない型です: {key}={type(values[0])}")
    return combined


@dataclass(frozen=True)
class BatchTensors:
    """batch dictから取り出したdevice上のTensor群。"""

    inputs: Tensor
    region_mask: Tensor
    vertebra_target: Tensor
    region_targets: Tensor
    region_target_valid: Tensor
    region_scores: Tensor


def batch_tensors(batch: dict[str, Any], device: torch.device) -> BatchTensors:
    """batch dictからdevice上のTensor群を取り出す。uint8入力は[0,1]へ正規化する。"""
    inputs = _tensor(batch, "inputs").to(device, non_blocking=True)
    if inputs.dtype == torch.uint8:
        inputs = inputs.to(dtype=torch.float32).div_(255.0)
    return BatchTensors(
        inputs=inputs,
        region_mask=_tensor(batch, "region_mask").to(device, non_blocking=True),
        vertebra_target=_tensor(batch, "vertebra_target").to(device, non_blocking=True),
        region_targets=_tensor(batch, "region_targets").to(device, non_blocking=True),
        region_target_valid=_tensor(batch, "region_target_valid").to(
            device, non_blocking=True
        ),
        region_scores=_tensor(batch, "region_scores").to(device, non_blocking=True),
    )


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value
