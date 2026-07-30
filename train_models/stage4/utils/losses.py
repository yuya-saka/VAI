"""Stage4 mixed-supervision losses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_like(value: Tensor) -> Tensor:
    return value.sum() * 0.0


def compute_region_pos_weight(
    strong_labels: np.ndarray,
    n_negative_sampled: int,
) -> Tensor:
    """Compute clipped positive BCE weights after adding sampled negatives."""
    labels = np.asarray(strong_labels)
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError(f"strong_labels must have shape [N, 4], got {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError("strong_labels must not be empty")
    if n_negative_sampled < 0:
        raise ValueError("n_negative_sampled must be non-negative")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("strong_labels must be binary")
    positives = labels.sum(axis=0, dtype=np.float64)
    if np.any(positives == 0):
        raise ValueError("every region needs at least one positive strong label")
    total_bags = labels.shape[0] + n_negative_sampled
    weights = np.minimum((total_bags - positives) / positives, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def region_loss(
    region_logits: Tensor,
    region_target: Tensor,
    supervision_mask: Tensor,
    pos_weight: Tensor,
) -> Tensor:
    """Average four-region BCE over strong and sampled-negative bags only."""
    if region_logits.ndim != 2 or region_logits.shape[1] != 4:
        raise ValueError("region_logits must have shape [B, 4]")
    if region_target.shape != region_logits.shape:
        raise ValueError("region_target shape must match region_logits")
    if supervision_mask.shape != region_logits.shape[:1]:
        raise ValueError("supervision_mask must have shape [B]")
    if pos_weight.shape != (4,):
        raise ValueError("pos_weight must have shape [4]")
    supervised = supervision_mask.bool()
    if not supervised.any():
        return region_logits.sum() * 0.0
    per_region = F.binary_cross_entropy_with_logits(
        region_logits[supervised].float(),
        region_target[supervised].float(),
        pos_weight=pos_weight.to(region_logits.device, dtype=torch.float32),
        reduction="none",
    )
    return per_region.mean(dim=1).mean()


def stratified_vertebra_loss(
    vertebra_logits: Tensor,
    targets: Tensor,
    vertebra_valid: Tensor,
    region_supervision_mask: Tensor,
    n_strong: int,
    n_weak: int,
    n_negative: int,
    n_sampled_negative: int,
    positive_weight: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Estimate the Stage3 population risk from fixed Stage4 strata."""
    if vertebra_logits.shape != targets.shape:
        raise ValueError("vertebra_logits and targets must have matching shapes")
    if vertebra_valid.shape != targets.shape:
        raise ValueError("vertebra_valid and targets must have matching shapes")
    if region_supervision_mask.shape != targets.shape:
        raise ValueError(
            "region_supervision_mask and targets must have matching shapes"
        )
    if min(n_strong, n_weak, n_negative) < 1:
        raise ValueError("all Stage4 population strata must be non-empty")
    if not 0 < n_sampled_negative < n_negative:
        raise ValueError("n_sampled_negative must be in (0, n_negative)")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive")

    valid = vertebra_valid.bool()
    strong = valid & (targets > 0) & region_supervision_mask.bool()
    weak = valid & (targets > 0) & ~region_supervision_mask.bool()
    sampled_negative = valid & (targets <= 0) & region_supervision_mask.bool()
    other_negative = valid & (targets <= 0) & ~region_supervision_mask.bool()
    if (
        not strong.any()
        or not weak.any()
        or not sampled_negative.any()
        or not other_negative.any()
    ):
        raise ValueError("every stratified training batch must contain all four strata")

    per_bag = F.binary_cross_entropy_with_logits(
        vertebra_logits.float(),
        targets.float(),
        reduction="none",
    )
    strong_loss = per_bag[strong].mean()
    weak_loss = per_bag[weak].mean()
    sampled_negative_loss = per_bag[sampled_negative].mean()
    other_negative_loss = per_bag[other_negative].mean()
    negative_loss = (
        n_sampled_negative * sampled_negative_loss
        + (n_negative - n_sampled_negative) * other_negative_loss
    ) / n_negative
    denominator = positive_weight * (n_strong + n_weak) + n_negative
    total = (
        positive_weight * n_strong * strong_loss
        + positive_weight * n_weak * weak_loss
        + n_negative * negative_loss
    ) / denominator
    return total, {
        "strong_bag_loss": strong_loss,
        "weak_bag_loss": weak_loss,
        "negative_bag_loss": negative_loss,
        "sampled_negative_bag_loss": sampled_negative_loss,
        "other_negative_bag_loss": other_negative_loss,
    }


def negative_instance_loss(
    instance_logits: Tensor,
    instance_valid: Tensor,
    targets: Tensor,
    vertebra_valid: Tensor,
) -> Tensor:
    """Average zero-target instance BCE over fracture-negative bags."""
    negative_bags = (targets <= 0) & vertebra_valid.bool()
    if not negative_bags.any():
        return _zero_like(instance_logits)
    negative_logits = instance_logits[negative_bags]
    negative_valid = instance_valid[negative_bags].bool()
    per_instance = F.binary_cross_entropy_with_logits(
        negative_logits.float(),
        torch.zeros_like(negative_logits, dtype=torch.float32),
        reduction="none",
    )
    per_bag = (per_instance * negative_valid).sum(dim=(1, 2)) / negative_valid.sum(
        dim=(1, 2)
    ).clamp_min(1)
    return per_bag.mean()


def stratified_negative_instance_loss(
    instance_logits: Tensor,
    instance_valid: Tensor,
    targets: Tensor,
    vertebra_valid: Tensor,
    region_supervision_mask: Tensor,
    n_negative: int,
    n_sampled_negative: int,
) -> Tensor:
    """Restore the full-negative population after matched-negative repetition."""
    if not 0 < n_sampled_negative < n_negative:
        raise ValueError("n_sampled_negative must be in (0, n_negative)")
    negative_bags = (targets <= 0) & vertebra_valid.bool()
    sampled = negative_bags & region_supervision_mask.bool()
    other = negative_bags & ~region_supervision_mask.bool()
    if not sampled.any() or not other.any():
        raise ValueError("each training batch needs sampled and other negatives")
    negative_logits = instance_logits[negative_bags]
    negative_valid = instance_valid[negative_bags].bool()
    per_instance = F.binary_cross_entropy_with_logits(
        negative_logits.float(),
        torch.zeros_like(negative_logits, dtype=torch.float32),
        reduction="none",
    )
    per_bag = (per_instance * negative_valid).sum(dim=(1, 2)) / negative_valid.sum(
        dim=(1, 2)
    ).clamp_min(1)
    sampled_in_negative = region_supervision_mask[negative_bags].bool()
    sampled_loss = per_bag[sampled_in_negative].mean()
    other_loss = per_bag[~sampled_in_negative].mean()
    return (
        n_sampled_negative * sampled_loss
        + (n_negative - n_sampled_negative) * other_loss
    ) / n_negative


def lambda_region_schedule(epoch: int) -> float:
    """Ramp region supervision from 0.25 to 1.0 over epochs zero through four."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return 0.25 + 0.75 * min(epoch / 4.0, 1.0)
