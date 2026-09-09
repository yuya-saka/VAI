"""region logitのbag集約と統一target損失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

BAG_PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class ConditionalRegionLosses:
    """whole陽性bag上のregion BCEとentropy-centered値。"""

    bce: Tensor
    centered: Tensor
    valid_cells: int


def region_bag_logits(
    plane_logits: Tensor, plane_valid: Tensor
) -> tuple[Tensor, Tensor]:
    """有効面だけの平均sigmoidをbag logitへ変換する。

    Args:
        plane_logits: [B, S, R]
        plane_valid: [B, S, R] bool

    Returns:
        bag_logits: [B, R]
        cell_valid: [B, R] bool。そのbag×regionに有効面が1つ以上ある場合True。
    """
    if plane_logits.shape != plane_valid.shape:
        raise ValueError("plane_logitsとplane_validのshapeが一致しません")
    probabilities = plane_logits.float().sigmoid()
    valid_float = plane_valid.to(probabilities.dtype)
    counts = valid_float.sum(dim=1)  # [B, R]
    cell_valid = counts > 0
    safe_counts = counts.clamp_min(1.0)
    mean_probability = (probabilities * valid_float).sum(dim=1) / safe_counts
    clamped = mean_probability.clamp(
        BAG_PROBABILITY_EPSILON, 1.0 - BAG_PROBABILITY_EPSILON
    )
    return torch.logit(clamped), cell_valid


def compute_conditional_region_losses(
    bag_logits: Tensor,
    target: Tensor,
    target_valid: Tensor,
    vertebra_target: Tensor,
) -> ConditionalRegionLosses:
    """GT/pseudo統一targetへwhole陽性条件付きのregion-macro BCEを計算する。

    `target`はhuman GTを優先し、human-unknown cellだけCAM soft targetで埋めた
    単一tensorである。whole-negativeはregion条件付き確率の教師にせず、親である
    whole lossだけへ残す。各regionをvalid cellで平均してからregion間を平均する。
    `centered`は`BCE(target, prediction)-H(target)`で、学習勾配は`bce`と同一。
    """
    if bag_logits.shape != target.shape or bag_logits.shape != target_valid.shape:
        raise ValueError("bag_logits/target/target_validのshapeが一致しません")
    if vertebra_target.shape != (bag_logits.shape[0],):
        raise ValueError(
            "vertebra_targetはbag_logitsのbatch次元と一致する必要があります"
        )
    valid_targets = target[target_valid]
    if valid_targets.numel() and (
        not torch.isfinite(valid_targets).all()
        or not valid_targets.ge(0.0).all()
        or not valid_targets.le(1.0).all()
    ):
        raise ValueError("target_validなregion targetは有限な[0,1]である必要があります")
    valid_vertebra_targets = torch.logical_or(
        vertebra_target.eq(0.0), vertebra_target.eq(1.0)
    )
    if not valid_vertebra_targets.all():
        raise ValueError("vertebra_targetは0/1である必要があります")

    conditional_valid = target_valid & vertebra_target[:, None].eq(1.0)
    per_cell_loss = F.binary_cross_entropy_with_logits(
        bag_logits, target, reduction="none"
    )
    target_entropy = F.binary_cross_entropy(target, target, reduction="none")
    region_counts = conditional_valid.sum(dim=0)
    valid_regions = region_counts > 0
    if not valid_regions.any():
        zero = bag_logits.sum() * 0.0
        return ConditionalRegionLosses(bce=zero, centered=zero, valid_cells=0)

    weights = conditional_valid.to(bag_logits.dtype)
    safe_counts = region_counts.clamp_min(1).to(bag_logits.dtype)
    region_bce = (per_cell_loss * weights).sum(dim=0) / safe_counts
    region_centered = ((per_cell_loss - target_entropy) * weights).sum(
        dim=0
    ) / safe_counts
    return ConditionalRegionLosses(
        bce=region_bce[valid_regions].mean(),
        centered=region_centered[valid_regions].mean(),
        valid_cells=int(conditional_valid.sum().item()),
    )
