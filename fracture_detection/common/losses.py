"""全実験アームで共有する骨折分類損失。"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from fracture_detection.common.constants import N_REGIONS


def region_bce(
    region_logits: Tensor,
    region_targets: Tensor,
    region_target_valid: Tensor,
) -> Tensor:
    """明示的にアノテーションされた領域targetだけへ標準BCEを適用する。"""
    expected_shape = (region_logits.shape[0], N_REGIONS)
    if region_logits.shape != expected_shape:
        raise ValueError(f"領域ロジット形状が不正です: {region_logits.shape}")
    if region_targets.shape != expected_shape:
        raise ValueError(f"領域target形状が不正です: {region_targets.shape}")
    if region_target_valid.shape != expected_shape:
        raise ValueError(
            f"領域target validity形状が不正です: {region_target_valid.shape}"
        )
    valid = region_target_valid.bool()
    if not bool(valid.any()):
        return region_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        region_logits[valid], region_targets.to(region_logits.dtype)[valid]
    )
