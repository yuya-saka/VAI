"""全実験アームで共有する骨折分類損失。"""

from __future__ import annotations

from torch import Tensor

from fracture_detection.core.losses import region_bce as _plane_region_bce


def region_bce(
    region_logits: Tensor,
    region_targets: Tensor,
    region_target_valid: Tensor,
) -> Tensor:
    """明示的にアノテーションされた領域targetだけへ標準BCEを適用する。"""
    if region_logits.ndim == 2:
        return _plane_region_bce(
            region_logits.unsqueeze(1), region_targets, region_target_valid
        )
    return _plane_region_bce(region_logits, region_targets, region_target_valid)
