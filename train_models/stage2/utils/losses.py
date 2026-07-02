"""Loss and mixup utilities for Stage2."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from train_models.stage2.src.model import Stage2Output


def weighted_bce(
    logits: Tensor,
    targets: Tensor,
    positive_weight: float = 2.0,
) -> Tensor:
    """Compute sample-weighted binary cross entropy."""
    logits_flat = logits.reshape(-1)
    targets_flat = targets.reshape(-1)
    losses = F.binary_cross_entropy_with_logits(
        logits_flat, targets_flat, reduction="none"
    )
    weights = torch.where(targets_flat > 0, positive_weight, 1.0)
    return (losses * weights).sum() / weights.sum()


def region_log_survival(region_logits: Tensor, region_plane_valid: Tensor) -> Tensor:
    """Return log P(no region fires) per plane, ignoring invalid regions.

    An invalid (plane, region) entry never reads its logit value: it
    contributes exactly 0 to the log-survival sum, so it can never leak a
    default sigmoid(0)=0.5 evidence into the Noisy-OR.
    """
    with torch.autocast(device_type=region_logits.device.type, enabled=False):
        log_not_region = F.logsigmoid(-region_logits.float())
        masked = torch.where(
            region_plane_valid, log_not_region, torch.zeros_like(log_not_region)
        )
        return masked.sum(dim=-1)


def region_any_probability(log_survival: Tensor) -> Tensor:
    """Stable Noisy-OR probability that at least one region fires."""
    with torch.autocast(device_type=log_survival.device.type, enabled=False):
        return (-torch.expm1(log_survival)).clamp(0.0, 1.0)


def region_slice_loss(
    region_logits: Tensor,
    region_plane_valid: Tensor,
    plane_valid: Tensor,
    targets: Tensor,
    positive_weight: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Weighted BCE of the per-plane Noisy-OR probability against the vertebra label.

    Planes with no valid region (``plane_valid`` False) contribute to neither
    the numerator nor the denominator, so they never bias the normalization.
    """
    log_survival = region_log_survival(region_logits, region_plane_valid)
    with torch.autocast(device_type=log_survival.device.type, enabled=False):
        p_any = (-torch.expm1(log_survival)).clamp(eps, 1.0 - eps)
        log_p_any = torch.log(p_any)
        targets_expanded = targets.float().unsqueeze(1).expand_as(log_survival)
        per_plane_loss = -(
            targets_expanded * log_p_any + (1.0 - targets_expanded) * log_survival
        )
        weight = torch.where(
            targets_expanded > 0, positive_weight, 1.0
        ) * plane_valid.to(per_plane_loss.dtype)
        return (per_plane_loss * weight).sum() / weight.sum().clamp_min(eps)


def stage2_loss(
    output: Stage2Output,
    targets: Tensor,
    positive_weight: float = 2.0,
    region_loss_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the Stage1-compatible primary loss plus the region auxiliary loss.

    With ``region_loss_weight <= 0`` (or no region output at all), the total
    loss depends only on ``slice_logits`` and reproduces Stage1's loss exactly.
    """
    targets_expanded = targets.float().unsqueeze(1).expand_as(output.slice_logits)
    stage1_loss = weighted_bce(output.slice_logits, targets_expanded, positive_weight)
    if region_loss_weight <= 0.0 or output.region_logits is None:
        return stage1_loss, {
            "stage1_loss": stage1_loss,
            "region_loss": torch.zeros_like(stage1_loss),
        }
    region_loss = region_slice_loss(
        output.region_logits,
        output.region_plane_valid,
        output.plane_valid,
        targets,
        positive_weight,
    )
    total = stage1_loss + region_loss_weight * region_loss
    return total, {"stage1_loss": stage1_loss, "region_loss": region_loss}


def mixup_batch(
    images: Tensor,
    region_masks: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Mix images and one-hot region occupancy with one shared permutation."""
    indices = torch.randperm(images.size(0), device=images.device)
    lam = float(np.random.uniform(0.0, 1.0))
    mixed_images = images * lam + images[indices] * (1.0 - lam)
    n_regions = 4
    one_hot = (
        F.one_hot(region_masks.long().clamp(0, n_regions), num_classes=n_regions + 1)[
            ..., 1:
        ]
        .permute(0, 1, 4, 2, 3)
        .to(images.dtype)
    )
    mixed_masks = one_hot * lam + one_hot[indices] * (1.0 - lam)
    return mixed_images, mixed_masks, targets, targets[indices], lam
