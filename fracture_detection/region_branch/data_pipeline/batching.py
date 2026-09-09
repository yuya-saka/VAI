"""DataLoaderのbatch dictをdevice上のTensor群へ変換する汎用helper。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class BatchTensors:
    """batch dictから取り出したdevice上のTensor群。"""

    inputs: Tensor
    region_mask: Tensor
    vertebra_target: Tensor
    region_target: Tensor
    region_target_valid: Tensor
    region_hard_valid: Tensor
    region_pseudo_valid: Tensor


def batch_tensors(batch: dict[str, Any], device: torch.device) -> BatchTensors:
    """batch dictからdevice上のTensor群を取り出す。uint8入力は[0,1]へ正規化する。"""
    inputs = _tensor(batch, "inputs").to(device, non_blocking=True)
    if inputs.dtype == torch.uint8:
        inputs = inputs.to(dtype=torch.float32).div_(255.0)
    return BatchTensors(
        inputs=inputs,
        region_mask=_tensor(batch, "region_mask").to(device, non_blocking=True),
        vertebra_target=_tensor(batch, "vertebra_target").to(device, non_blocking=True),
        region_target=_tensor(batch, "region_target").to(device, non_blocking=True),
        region_target_valid=_tensor(batch, "region_target_valid").to(
            device, non_blocking=True
        ),
        region_hard_valid=_tensor(batch, "region_hard_valid").to(
            device, non_blocking=True
        ),
        region_pseudo_valid=_tensor(batch, "region_pseudo_valid").to(
            device, non_blocking=True
        ),
    )


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value
