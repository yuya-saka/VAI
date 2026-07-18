"""Numerically safe hierarchical pooling and Stage3 losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_like(value: Tensor) -> Tensor:
    return value.sum() * 0.0


def masked_softmax(logits: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Return a finite masked softmax, including all-invalid rows."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        masked = torch.where(
            valid, logits.float(), torch.full_like(logits.float(), -torch.inf)
        )
        result = torch.softmax(masked, dim=dim)
        return torch.where(
            valid.any(dim=dim, keepdim=True),
            torch.nan_to_num(result),
            torch.zeros_like(result),
        )


def normalized_smoothmax(
    logits: Tensor, valid: Tensor, dim: int, tau: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate valid logits with normalized log-sum-exp and return weights."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float()
        is_valid = valid.bool()
        any_valid = is_valid.any(dim=dim)
        scaled = torch.where(
            is_valid, values * tau, torch.full_like(values, -torch.inf)
        )
        lse = torch.logsumexp(scaled, dim=dim)
        count = is_valid.sum(dim=dim).clamp_min(1).to(values.dtype)
        pooled = (lse - count.log()) / tau
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        weights = masked_softmax(values * tau, is_valid, dim)
        return pooled, weights, any_valid


def tied_attention_pool(
    logits: Tensor, valid: Tensor, temperature: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool slice logits using evidence-tied attention over dimension one."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    attention = masked_softmax(logits / temperature, valid, dim=1)
    region_valid = valid.any(dim=1)
    values = torch.where(valid, logits.float(), torch.zeros_like(logits.float()))
    pooled = (attention * values).sum(dim=1)
    pooled = torch.where(region_valid, pooled, torch.zeros_like(pooled))
    return pooled, attention, region_valid


def normalized_lse_pool(
    logits: Tensor, valid: Tensor, tau: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool slice logits with normalized LSE and expose its softmax weights."""
    return normalized_smoothmax(logits, valid, dim=1, tau=tau)


def noisy_or_pool(logits: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate logits as a stable masked Noisy-OR logit."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float()
        is_valid = valid.bool()
        any_valid = is_valid.any(dim=1)
        log_survival = torch.where(
            is_valid, F.logsigmoid(-values), torch.zeros_like(values)
        ).sum(dim=1)
        safe_survival = log_survival.clamp(min=-30.0, max=-1e-7)
        log_probability = torch.log(-torch.expm1(safe_survival))
        pooled = log_probability - safe_survival
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        probabilities = torch.where(
            is_valid, torch.sigmoid(values), torch.zeros_like(values)
        )
        weights = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        return pooled, weights, any_valid


def max_pool(logits: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate valid logits with a finite masked maximum."""
    values = logits.float()
    is_valid = valid.bool()
    any_valid = is_valid.any(dim=1)
    masked = torch.where(is_valid, values, torch.full_like(values, -torch.inf))
    pooled = masked.max(dim=1).values
    pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
    is_winner = is_valid & (masked == pooled.unsqueeze(1))
    weights = is_winner.to(values.dtype) / is_winner.sum(dim=1, keepdim=True).clamp_min(
        1
    )
    return pooled, weights, any_valid


def weighted_bce(logits: Tensor, targets: Tensor, positive_weight: float) -> Tensor:
    """Compute Stage1-compatible sample-weighted BCE."""
    if logits.numel() == 0:
        return _zero_like(logits)
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    weights = torch.where(targets > 0, positive_weight, 1.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def stage3_loss(
    vertebra_logits: Tensor,
    instance_logits: Tensor,
    instance_valid: Tensor,
    vertebra_valid: Tensor,
    targets: Tensor,
    positive_weight: float = 2.0,
    lambda_neg: float = 0.1,
    secondary_targets: Tensor | None = None,
    mixup_lambda: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return bag BCE plus vertebra-balanced negative-instance regularization."""
    valid_logits = vertebra_logits[vertebra_valid]
    valid_targets = targets[vertebra_valid]
    bag_loss = weighted_bce(valid_logits, valid_targets, positive_weight)
    if secondary_targets is None:
        negative_bags = (targets <= 0) & vertebra_valid
    else:
        if mixup_lambda is None:
            raise ValueError("mixup_lambda is required with secondary_targets")
        bag_loss = mixup_lambda * bag_loss + (1.0 - mixup_lambda) * weighted_bce(
            vertebra_logits[vertebra_valid],
            secondary_targets[vertebra_valid],
            positive_weight,
        )
        negative_bags = (targets <= 0) & (secondary_targets <= 0) & vertebra_valid
    if not negative_bags.any():
        negative_loss = _zero_like(instance_logits)
    else:
        negative_logits = instance_logits[negative_bags]
        negative_valid = instance_valid[negative_bags]
        per_instance = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction="none"
        )
        per_bag = (per_instance * negative_valid).sum(dim=(1, 2)) / negative_valid.sum(
            dim=(1, 2)
        ).clamp_min(1)
        negative_loss = per_bag.mean()
    total = bag_loss + lambda_neg * negative_loss
    return total, {
        "bag_loss": bag_loss,
        "negative_instance_loss": negative_loss,
        "weighted_negative_loss": lambda_neg * negative_loss,
    }
