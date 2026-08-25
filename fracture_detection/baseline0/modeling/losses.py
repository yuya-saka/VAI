"""Baseline 0の15面へ複製したBCEとbag確率。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def broadcast_targets(targets: Tensor, n_planes: int) -> Tensor:
    """椎体target [B]を各面 [B, N]へ複製する。"""
    if targets.ndim != 1 or n_planes < 1:
        raise ValueError("targetsは[B]、n_planesは1以上である必要があります")
    return targets.unsqueeze(1).expand(-1, n_planes)


def broadcast_bce_loss(
    plane_logits: Tensor,
    targets: Tensor,
    pos_weight: float,
) -> Tensor:
    """重み合計で正規化した面単位BCEを返す。"""
    if plane_logits.ndim != 2:
        raise ValueError(f"plane logitsは[B,N]が必要です: {plane_logits.shape}")
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weightは正の有限値である必要があります")
    expanded = broadcast_targets(targets, plane_logits.shape[1]).to(plane_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        plane_logits,
        expanded,
        reduction="none",
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
