"""全アームで共有する面単位損失とbag readout。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from fracture_detection.common.constants import N_REGIONS


def broadcast_targets(targets: Tensor, n_planes: int) -> Tensor:
    """椎体ターゲット[B]を各面[B,N]へ複製する。"""
    if targets.ndim != 1 or n_planes < 1:
        raise ValueError("targetsは[B]、n_planesは1以上である必要があります")
    return targets.unsqueeze(1).expand(-1, n_planes)


def broadcast_bce_loss(
    plane_logits: Tensor,
    targets: Tensor,
    pos_weight: float,
) -> Tensor:
    """全アーム共通の重み合計正規化broadcast BCEを返す。"""
    if plane_logits.ndim != 2:
        raise ValueError(f"plane logitsは[B,N]が必要です: {plane_logits.shape}")
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weightは正の有限値である必要があります")
    expanded = broadcast_targets(targets, plane_logits.shape[1]).to(plane_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        plane_logits, expanded, reduction="none"
    )
    weights = torch.where(
        expanded > 0,
        plane_logits.new_tensor(pos_weight),
        plane_logits.new_tensor(1.0),
    )
    return (losses * weights).sum() / weights.sum()


def bag_probabilities(plane_logits: Tensor) -> Tensor:
    """面sigmoidの平均をbag確率として返す。"""
    if plane_logits.ndim != 2:
        raise ValueError(f"plane logitsは[B,N]が必要です: {plane_logits.shape}")
    return plane_logits.sigmoid().mean(dim=1)


def plane_max_whole_logits(region_logits: Tensor) -> Tensor:
    """方式Aとして各面の4領域logit最大値を返す。"""
    if region_logits.ndim != 3 or region_logits.shape[-1] != N_REGIONS:
        raise ValueError(
            f"region logitsは[B,N,{N_REGIONS}]が必要です: {region_logits.shape}"
        )
    return region_logits.max(dim=-1).values


def region_bce(
    region_logits: Tensor,
    region_targets: Tensor,
    region_target_valid: Tensor,
) -> Tensor:
    """bag領域targetを面へbroadcastした素のBCEを返す。"""
    if region_logits.ndim != 3 or region_logits.shape[-1] != N_REGIONS:
        raise ValueError(
            f"region logitsは[B,N,{N_REGIONS}]が必要です: {region_logits.shape}"
        )
    expected = (region_logits.shape[0], N_REGIONS)
    if region_targets.shape != expected or region_target_valid.shape != expected:
        raise ValueError("region target/validは[B,4]である必要があります")
    expanded_targets = region_targets[:, None, :].expand_as(region_logits)
    expanded_valid = region_target_valid[:, None, :].expand_as(region_logits).bool()
    if not bool(expanded_valid.any()):
        return region_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        region_logits[expanded_valid],
        expanded_targets.to(region_logits.dtype)[expanded_valid],
    )


def attention_rmse(spatial_attention: Tensor, region_masks: Tensor) -> Tensor:
    """4領域spatial attentionとarea縮小maskの登録済みRMSEを返す。"""
    if spatial_attention.ndim != 5 or spatial_attention.shape[2] != N_REGIONS:
        raise ValueError("attentionは[B,N,4,H,W]である必要があります")
    if region_masks.ndim != 5 or region_masks.shape[2] != N_REGIONS:
        raise ValueError("region masksは[B,N,4,H,W]である必要があります")
    batch_size, plane_count, _, height, width = spatial_attention.shape
    flattened_masks = region_masks.reshape(
        batch_size * plane_count, N_REGIONS, *region_masks.shape[-2:]
    ).to(dtype=spatial_attention.dtype)
    targets = F.adaptive_avg_pool2d(flattened_masks, (height, width)).reshape_as(
        spatial_attention
    )
    squared_error = (spatial_attention - targets).square()
    per_region = squared_error.sum(dim=(0, 1, 3, 4)).div(batch_size * plane_count)
    return torch.sqrt(per_region).mean()
