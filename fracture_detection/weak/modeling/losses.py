"""Numerically stable N/A/U mixed-supervision loss.

Reference design: ``fracture_detection/REGION_MIL_DESIGN.md`` sections 3-6.

Per bag i, with region logits z_i1..z_i4 and q_ir = sigmoid(z_ir):

    N (negative):            sum_r BCEWithLogits(z_ir, 0)
    A (annotated positive):  sum_r BCEWithLogits(z_ir, t_ir)   (t_ir observed GT)
    U (weak positive):       -log(p_whole)                       (configured MIL)

``negative_bag_loss`` is always the sum of four negative-target region BCEs.
Its log-survival implementation is also identical to whole-negative BCE when
the configured aggregation is noisy-OR. Under normalized LSE it remains dense
region supervision and is not the negative BCE of the aggregated whole score.

All internal math is done in float32 (or higher) regardless of the model's
autocast dtype, per the design doc's numerical contract:

    a = sum_r logsigmoid(-z_r)          (log prod_r (1 - q_r), stable)
    negative loss  = -a
    noisy-OR positive loss = -log(1 - exp(a)) = -log1mexp(a)

Normalized LSE instead computes a temperature-scaled log-mean-exp in logit
space, followed by softplus for positive BCE. Both paths use float32 math.

BF16 sigmoid probabilities are never multiplied or subtracted directly.

The batch objective averages all per-bag losses (weak-positive terms scaled
by ``beta``) over the number of bags actually included -- a (bag, region)
that is unobserved in ANY of its needed cells is dropped entirely and
counted in ``WeakLosses.dropped_bags``, never silently treated as q=0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fracture_detection.weak.data_pipeline.groups import GROUP_IDS

_NEGATIVE_ID = GROUP_IDS["negative"]
_ANNOTATED_ID = GROUP_IDS["annotated_positive"]
_WEAK_ID = GROUP_IDS["weak_positive"]
NOISY_OR_AGGREGATION = "noisy_or"
NORMALIZED_LSE_AGGREGATION = "normalized_logsumexp"
WHOLE_AGGREGATIONS = frozenset({NOISY_OR_AGGREGATION, NORMALIZED_LSE_AGGREGATION})

# log(1 - exp(x)) switches formula near x = -log(2) for numerical accuracy
# (Machler, "Accurately Computing log(1-exp(-|a|))", 2012).
_LOG_HALF = -math.log(2.0)


def log_survival(region_logits: Tensor) -> Tensor:
    """log prod_r (1 - q_r) = sum_r logsigmoid(-z_r), computed in float32."""
    return F.logsigmoid(-region_logits.float()).sum(dim=-1)


def log1mexp(value: Tensor) -> Tensor:
    """log(1 - exp(value)) for value <= 0, stable near 0 and near -inf.

    ``torch.where(cond, f(x), g(x))`` still backpropagates through BOTH
    branches at every element (only the selected branch's gradient survives
    the final multiply-by-mask, but an unselected branch evaluated at a
    singular point can itself produce a NaN/Inf local gradient, and
    ``0 * nan == nan`` leaks it through). Each branch is therefore evaluated
    on an input clamped into its own safe domain before selection, so the
    unselected branch is never singular.
    """
    small_magnitude = value > _LOG_HALF
    safe_for_expm1 = torch.where(small_magnitude, value, value.new_full((), _LOG_HALF))
    safe_for_log1p = torch.where(small_magnitude, value.new_full((), _LOG_HALF), value)
    return torch.where(
        small_magnitude,
        torch.log(-torch.expm1(safe_for_expm1)),
        torch.log1p(-torch.exp(safe_for_log1p)),
    )


def negative_bag_loss(region_logits: Tensor) -> Tensor:
    """-log prod_r(1-q_r); identical value and gradient to sum_r BCE(z_r, 0)."""
    return -log_survival(region_logits)


def annotated_bag_loss(region_logits: Tensor, target: Tensor) -> Tensor:
    """sum_r BCEWithLogits(z_r, t_r) over all four observed cells."""
    return F.binary_cross_entropy_with_logits(
        region_logits.float(), target.float(), reduction="none"
    ).sum(dim=-1)


def normalized_logsumexp_whole_logit(
    region_logits: Tensor, lse_temperature: float
) -> Tensor:
    """Aggregate region logits with temperature-scaled normalized log-sum-exp."""
    if not math.isfinite(lse_temperature) or lse_temperature <= 0:
        raise ValueError("lse_temperature must be a finite value > 0")
    if region_logits.shape[-1] < 1:
        raise ValueError("region_logits must contain at least one region")
    logits = region_logits.float()
    return lse_temperature * (
        torch.logsumexp(logits / lse_temperature, dim=-1)
        - math.log(region_logits.shape[-1])
    )


def weak_positive_bag_loss(
    region_logits: Tensor,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> Tensor:
    """Positive bag BCE for the configured region-to-whole aggregation."""
    if whole_aggregation == NOISY_OR_AGGREGATION:
        return -log1mexp(log_survival(region_logits))
    if whole_aggregation == NORMALIZED_LSE_AGGREGATION:
        if lse_temperature is None:
            raise ValueError("normalized_logsumexp requires lse_temperature")
        whole_logit = normalized_logsumexp_whole_logit(region_logits, lse_temperature)
        return F.softplus(-whole_logit)
    raise ValueError(f"unsupported whole aggregation: {whole_aggregation}")


@dataclass(frozen=True)
class WeakLosses:
    """Batch-level loss and per-group diagnostics."""

    total: Tensor
    negative_sum: Tensor
    annotated_sum: Tensor
    weak_sum: Tensor
    n_negative: int
    n_annotated: int
    n_weak: int
    dropped_bags: int


def compute_weak_losses(
    region_logits: Tensor,
    region_observed: Tensor,
    group_id: Tensor,
    region_target: Tensor,
    beta: float,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> WeakLosses:
    """Compute the N/A/U batch objective.

    A bag is dropped from the loss (and counted in ``dropped_bags``) if any
    region it needs is unobserved: all four for N/A groups (whose loss
    touches every region), all four for U as well (the aggregation spans all
    four regions). This never substitutes q=0 for an unobserved region.
    """
    if region_logits.shape != region_target.shape:
        raise ValueError("region_logits and region_target shapes must match")
    if region_observed.shape != region_logits.shape:
        raise ValueError("region_observed shape must match region_logits")
    if group_id.shape[0] != region_logits.shape[0]:
        raise ValueError("group_id length must match batch size")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite value >= 0")

    fully_observed = region_observed.all(dim=-1)
    dropped_bags = int((~fully_observed).sum().item())

    negative_mask = fully_observed & (group_id == _NEGATIVE_ID)
    annotated_mask = fully_observed & (group_id == _ANNOTATED_ID)
    weak_mask = fully_observed & (group_id == _WEAK_ID)

    negative_losses = negative_bag_loss(region_logits[negative_mask])
    annotated_losses = annotated_bag_loss(
        region_logits[annotated_mask], region_target[annotated_mask]
    )
    weak_losses = weak_positive_bag_loss(
        region_logits[weak_mask], whole_aggregation, lse_temperature
    )

    negative_sum = negative_losses.sum()
    annotated_sum = annotated_losses.sum()
    weak_sum = weak_losses.sum()

    n_bags = int(negative_mask.sum().item() + annotated_mask.sum().item())
    n_weak = int(weak_mask.sum().item())
    total_bags = n_bags + n_weak
    if total_bags == 0:
        zero = region_logits.sum() * 0.0
        return WeakLosses(
            total=zero,
            negative_sum=negative_sum,
            annotated_sum=annotated_sum,
            weak_sum=weak_sum,
            n_negative=int(negative_mask.sum().item()),
            n_annotated=int(annotated_mask.sum().item()),
            n_weak=n_weak,
            dropped_bags=dropped_bags,
        )

    total = (negative_sum + annotated_sum + beta * weak_sum) / total_bags
    return WeakLosses(
        total=total,
        negative_sum=negative_sum,
        annotated_sum=annotated_sum,
        weak_sum=weak_sum,
        n_negative=int(negative_mask.sum().item()),
        n_annotated=int(annotated_mask.sum().item()),
        n_weak=n_weak,
        dropped_bags=dropped_bags,
    )


def whole_probability(
    region_logits: Tensor,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> Tensor:
    """Return the whole probability for the configured aggregation."""
    if whole_aggregation == NOISY_OR_AGGREGATION:
        return -torch.expm1(log_survival(region_logits))
    if whole_aggregation == NORMALIZED_LSE_AGGREGATION:
        if lse_temperature is None:
            raise ValueError("normalized_logsumexp requires lse_temperature")
        return torch.sigmoid(
            normalized_logsumexp_whole_logit(region_logits, lse_temperature)
        )
    raise ValueError(f"unsupported whole aggregation: {whole_aggregation}")


def mean_or_nan(total: float, count: int) -> float:
    """Per-bag mean loss; NaN (not 0) when the group has no bag."""
    return total / count if count > 0 else float("nan")


def mixture_objective(
    negative_mean: float,
    annotated_mean: float,
    weak_mean: float,
    negative_per_batch: int,
    annotated_per_batch: int,
    weak_per_batch: int,
    beta: float,
) -> float:
    """Per-bag N/A/U group means combined with the training batch composition.

    This equals ``compute_weak_losses(...).total`` for a batch of the
    configured composition, so one definition serves both the augmented train
    stream and the natural-distribution inner fold. A plain average over the
    inner fold would be ~90% whole-negative bags and bury the weak-positive
    term. The literal mean of one GT-pass's batch losses differs slightly
    because of the short final step.
    """
    batch_size = negative_per_batch + annotated_per_batch + weak_per_batch
    return (
        negative_per_batch * negative_mean
        + annotated_per_batch * annotated_mean
        + beta * weak_per_batch * weak_mean
    ) / batch_size
