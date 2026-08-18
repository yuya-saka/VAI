"""Baseline 0の15面へ複製したBCEとbag確率。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def broadcast_targets(targets: Tensor, n_planes: int) -> Tensor:
    """椎体ターゲット[B]を各面ターゲット[B, N]へ複製する。"""
    if targets.ndim != 1:
        raise ValueError(f"targetは[B]である必要があります: {targets.shape}")
    if n_planes < 1:
        raise ValueError("n_planesは1以上である必要があります")
    return targets.unsqueeze(1).expand(-1, n_planes)


def broadcast_bce_loss(
    plane_logits: Tensor,
    targets: Tensor,
    pos_weight: float,
) -> Tensor:
    """面logitと複製した椎体ターゲットによる重み付きBCEを返す。"""
    if plane_logits.ndim != 2:
        raise ValueError(
            f"plane logitsは[B,N]である必要があります: {plane_logits.shape}"
        )
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weightは正の有限値である必要があります")
    expanded_targets = broadcast_targets(targets, plane_logits.shape[1]).to(
        dtype=plane_logits.dtype
    )
    element_losses = nn.functional.binary_cross_entropy_with_logits(
        plane_logits,
        expanded_targets,
        reduction="none",
    )
    weights = torch.where(
        expanded_targets > 0,
        plane_logits.new_tensor(pos_weight),
        plane_logits.new_tensor(1.0),
    )
    return (element_losses * weights).sum() / weights.sum()


def bag_probabilities(plane_logits: Tensor) -> Tensor:
    """面sigmoidの平均から椎体bag確率を返す。"""
    if plane_logits.ndim != 2:
        raise ValueError(
            f"plane logitsは[B,N]である必要があります: {plane_logits.shape}"
        )
    return torch.sigmoid(plane_logits).mean(dim=1)
