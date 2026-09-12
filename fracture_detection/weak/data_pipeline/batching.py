"""Move a DataLoader batch dict onto device as typed tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class WeakBatch:
    """Device-resident tensors for one batch."""

    inputs: Tensor  # [B, 15, 6, H, W] float32 in [0, 1]
    region_mask: Tensor  # [B, 15, H, W] uint8, values 0..N_REGIONS
    vertebra_target: Tensor  # [B] float32
    region_target: Tensor  # [B, N_REGIONS] float32 (only meaningful for N/A)
    group_id: (
        Tensor  # [B] int64, see fracture_detection.weak.data_pipeline.groups.GROUP_IDS
    )


def batch_tensors(batch: dict[str, Any], device: torch.device) -> WeakBatch:
    """Extract device-resident tensors; uint8 inputs are scaled to [0, 1]."""
    inputs = _tensor(batch, "inputs").to(device, non_blocking=True)
    if inputs.dtype == torch.uint8:
        inputs = inputs.to(dtype=torch.float32).div_(255.0)
    return WeakBatch(
        inputs=inputs,
        region_mask=_tensor(batch, "region_mask").to(device, non_blocking=True),
        vertebra_target=_tensor(batch, "vertebra_target").to(device, non_blocking=True),
        region_target=_tensor(batch, "region_target").to(device, non_blocking=True),
        group_id=_tensor(batch, "group_id").to(device, non_blocking=True),
    )


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value
