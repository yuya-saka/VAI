"""全実験アームで共有する骨折分類損失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fracture_detection.common.constants import N_REGIONS


def region_bce(
    region_logits: Tensor,
    region_targets: Tensor,
    region_target_valid: Tensor,
    vertebra_targets: Tensor,
) -> Tensor:
    """利用可能な領域targetへ標準BCEを適用する。

    アノテーション済みbagは4領域すべてを使用する。椎体陰性は4領域すべて
    陰性と確定するため、領域アノテーションがなくてもtargetを0として使用する。
    未アノテーションの骨折陽性は領域損失から除外する。
    """
    expected_shape = (region_logits.shape[0], N_REGIONS)
    if region_logits.shape != expected_shape:
        raise ValueError(f"領域ロジット形状が不正です: {region_logits.shape}")
    if region_targets.shape != expected_shape:
        raise ValueError(f"領域target形状が不正です: {region_targets.shape}")
    if region_target_valid.shape != expected_shape:
        raise ValueError(
            f"領域target validity形状が不正です: {region_target_valid.shape}"
        )
    if vertebra_targets.shape != (region_logits.shape[0],):
        raise ValueError(f"椎体target形状が不正です: {vertebra_targets.shape}")

    entailed_negative = (vertebra_targets <= 0.5).unsqueeze(1)
    effective_valid = region_target_valid.bool() | entailed_negative
    effective_targets = torch.where(
        region_target_valid.bool(), region_targets, torch.zeros_like(region_targets)
    )
    if not bool(effective_valid.any()):
        return region_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        region_logits[effective_valid], effective_targets[effective_valid]
    )
