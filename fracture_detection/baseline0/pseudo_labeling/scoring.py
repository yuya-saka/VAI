"""Pairwise-ranking pseudo-label construction from Baseline 0 Grad-CAM.

The generation-stage audit (``cam_audit.py``) showed the region CAM density is a
usable teacher signal: no in-sample memorization, and rank stability under mask
perturbation for every gate-magnitude variant. This module turns that signal
into a training target — not an absolute probability, but a soft confidence
that one bag outranks another for a given region, following the design in
``.claude/docs/codex/20260823-pseudo-label-mtl-design.md`` (Q3e/Q4):

    u_ijr = sigmoid((log C_ir - log C_jr) / T_r)

``T_r`` is a fixed, label-free scale: the interquartile range of the region's
own log-score differences within the training population it will be paired
over. It is not tuned against any human label, so it cannot reintroduce the
grid-search-on-268-bags problem that sank the earlier lambda calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

SCORE_EPSILON = 1e-6
TEMPERATURE_FLOOR = 1e-3
DEFAULT_TEMPERATURE_PAIRS = 100_000
DEFAULT_TEMPERATURE_SEED = 20_260_823


@dataclass(frozen=True)
class RegionPairBatch:
    """Flattened same-region pairs built from one fold-matched teacher batch."""

    left_bag_indices: Tensor
    right_bag_indices: Tensor
    region_indices: Tensor
    teacher_log_score_left: Tensor
    teacher_log_score_right: Tensor
    pair_counts_by_region: Tensor

    @property
    def n_pairs(self) -> int:
        """Return the number of flattened pairs."""
        return int(self.left_bag_indices.numel())


def log_score(density: NDArray[np.floating] | Tensor) -> NDArray[np.float64] | Tensor:
    """Clip a non-negative CAM density to a floor and take its log.

    Density enrichment is a ratio and can be exactly zero when no CAM mass
    lands in a region; ``log(0)`` would make every pairwise difference
    involving that bag infinite, so it is floored first.
    """
    if isinstance(density, Tensor):
        return torch.log(density.clamp_min(SCORE_EPSILON))
    return cast(NDArray[np.float64], np.log(np.clip(density, SCORE_EPSILON, None)))


def region_temperature(
    scores: NDArray[np.floating],
    n_pairs: int = DEFAULT_TEMPERATURE_PAIRS,
    seed: int = DEFAULT_TEMPERATURE_SEED,
) -> float:
    """Label-free scale for one region: the IQR of sampled pairwise log-differences.

    ``scores`` is the region's raw (non-log) density for every bag in the
    population the pairwise loss will draw from — human label and fracture
    status play no part, so the same function applies to either population
    arm. Non-finite and non-positive scores are dropped before sampling: a
    ``NaN`` (undefined density, e.g. zero CAM mass) or non-positive value
    cannot contribute a meaningful log-difference.

    Deterministic for a fixed ``(scores, seed)``: two runs of pseudo-label
    generation from the same teacher and training fold produce the same
    ``T_r``, so it never needs to be checked into a frozen artifact separately
    from the scores it was computed from.
    """
    finite = scores[np.isfinite(scores) & (scores > 0)]
    if finite.size < 2:
        raise ValueError("Need at least two positive, finite scores to set a scale")
    logs = np.log(finite)
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    rng = np.random.default_rng(seed)
    pair_count = n_pairs
    left = rng.integers(0, finite.size, size=pair_count)
    right = rng.integers(0, finite.size, size=pair_count)
    degenerate = left == right
    right[degenerate] = (right[degenerate] + 1) % finite.size
    differences = logs[left] - logs[right]
    iqr = float(np.percentile(differences, 75) - np.percentile(differences, 25))
    return max(iqr, TEMPERATURE_FLOOR)


def pairwise_confidence(
    log_score_i: Tensor, log_score_j: Tensor, temperature: float
) -> Tensor:
    """Soft target ``u_ijr``: confidence that ``i`` outranks ``j`` for one region.

    A vanishing log-score gap maps close to 0.5 (near-ties stay a weak
    target); a gap much larger than ``T_r`` saturates toward 0 or 1. This is
    what keeps a borderline CAM difference from being trained on as if it
    were as certain as a large one.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite: {temperature}")
    return torch.sigmoid((log_score_i - log_score_j) / temperature)


def pairwise_ranking_loss(
    student_logit_i: Tensor,
    student_logit_j: Tensor,
    teacher_log_score_i: Tensor,
    teacher_log_score_j: Tensor,
    temperature: float,
) -> Tensor:
    """BCE-with-logits of the student's score gap against the teacher's soft rank target.

    All four tensors are 1-D and aligned by pair index; ``student_logit_*`` are
    the student region head's raw logits (pre-sigmoid) for the same region as
    the teacher scores. Pair selection (which bags get compared, and whether a
    human label overrides the teacher target) is the caller's responsibility —
    this function only scores one already-built batch of pairs.
    """
    if student_logit_i.shape != student_logit_j.shape:
        raise ValueError("student logits for i and j must have the same shape")
    if teacher_log_score_i.shape != student_logit_i.shape:
        raise ValueError("teacher scores must align with the student logits")
    target = pairwise_confidence(
        teacher_log_score_i, teacher_log_score_j, temperature
    ).detach()
    return torch.nn.functional.binary_cross_entropy_with_logits(
        student_logit_i - student_logit_j, target
    )


def build_region_pair_batch(
    teacher_scores: Tensor,
    vertebra_targets: Tensor,
    teacher_outer_folds: Tensor,
    generator: torch.Generator,
    human_target_valid: Tensor | None = None,
) -> RegionPairBatch:
    """Build deterministic random cycles within each eligible region.

    Eligibility is restricted to fracture-positive bags with a finite,
    strictly positive CAM score. Passing ``human_target_valid`` implements
    human-over-pseudo precedence by excluding those cells; omitting it gives
    the pseudo-only arm access to every defined teacher score without reading
    the human target value. Every eligible bag appears once on each side of a
    region's directed cycle, and no self-pair is created.
    """
    _validate_pair_inputs(
        teacher_scores,
        vertebra_targets,
        teacher_outer_folds,
        human_target_valid,
    )
    n_bags, n_regions = teacher_scores.shape
    positive = vertebra_targets.eq(1).unsqueeze(1)
    eligible = positive & torch.isfinite(teacher_scores) & teacher_scores.gt(0)
    if human_target_valid is not None:
        eligible = eligible & ~human_target_valid.bool()

    left_parts: list[Tensor] = []
    right_parts: list[Tensor] = []
    region_parts: list[Tensor] = []
    pair_counts = torch.zeros(
        n_regions, dtype=torch.int64, device=teacher_scores.device
    )
    for region_index in range(n_regions):
        bag_indices = torch.nonzero(eligible[:, region_index], as_tuple=False).flatten()
        pair_count = int(bag_indices.numel())
        if pair_count < 2:
            continue
        order = torch.randperm(pair_count, generator=generator).to(bag_indices.device)
        left = bag_indices[order]
        right = left.roll(shifts=-1)
        left_parts.append(left)
        right_parts.append(right)
        region_parts.append(torch.full_like(left, region_index, dtype=torch.int64))
        pair_counts[region_index] = pair_count

    if not left_parts:
        empty_indices = torch.empty(0, dtype=torch.int64, device=teacher_scores.device)
        empty_scores = torch.empty(
            0, dtype=teacher_scores.dtype, device=teacher_scores.device
        )
        return RegionPairBatch(
            left_bag_indices=empty_indices,
            right_bag_indices=empty_indices.clone(),
            region_indices=empty_indices.clone(),
            teacher_log_score_left=empty_scores,
            teacher_log_score_right=empty_scores.clone(),
            pair_counts_by_region=pair_counts,
        )

    left_indices = torch.cat(left_parts)
    right_indices = torch.cat(right_parts)
    region_indices = torch.cat(region_parts)
    teacher_log_score_left = torch.log(
        teacher_scores[left_indices, region_indices]
    ).detach()
    teacher_log_score_right = torch.log(
        teacher_scores[right_indices, region_indices]
    ).detach()
    return RegionPairBatch(
        left_bag_indices=left_indices,
        right_bag_indices=right_indices,
        region_indices=region_indices,
        teacher_log_score_left=teacher_log_score_left,
        teacher_log_score_right=teacher_log_score_right,
        pair_counts_by_region=pair_counts,
    )


def region_balanced_pairwise_ranking_loss(
    student_region_logits: Tensor,
    pairs: RegionPairBatch,
    temperatures: Tensor,
) -> Tensor:
    """Average pairwise BCE within each region, then across active regions."""
    if student_region_logits.ndim != 2:
        raise ValueError("student_region_logits must have shape (bags, regions)")
    n_bags, n_regions = student_region_logits.shape
    if temperatures.shape != (n_regions,):
        raise ValueError("temperatures must have shape (regions,)")
    if not torch.isfinite(temperatures).all() or not temperatures.gt(0).all():
        raise ValueError("temperatures must be positive and finite")
    if temperatures.device != student_region_logits.device:
        raise ValueError("temperatures and student logits must share a device")
    _validate_region_pair_batch(pairs, n_bags, n_regions)
    pair_tensors = (
        pairs.left_bag_indices,
        pairs.right_bag_indices,
        pairs.region_indices,
        pairs.teacher_log_score_left,
        pairs.teacher_log_score_right,
        pairs.pair_counts_by_region,
    )
    if any(tensor.device != student_region_logits.device for tensor in pair_tensors):
        raise ValueError("pairs and student logits must share a device")
    if pairs.n_pairs == 0:
        return student_region_logits.sum() * 0.0

    pair_temperatures = temperatures[pairs.region_indices]
    targets = torch.sigmoid(
        (pairs.teacher_log_score_left - pairs.teacher_log_score_right)
        / pair_temperatures
    ).detach()
    left_logits = student_region_logits[pairs.left_bag_indices, pairs.region_indices]
    right_logits = student_region_logits[pairs.right_bag_indices, pairs.region_indices]
    pair_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        left_logits - right_logits,
        targets,
        reduction="none",
    )
    region_losses = [
        pair_losses[pairs.region_indices.eq(region_index)].mean()
        for region_index in range(n_regions)
        if int(pairs.pair_counts_by_region[region_index]) > 0
    ]
    return torch.stack(region_losses).mean()


def _validate_pair_inputs(
    teacher_scores: Tensor,
    vertebra_targets: Tensor,
    teacher_outer_folds: Tensor,
    human_target_valid: Tensor | None,
) -> None:
    if teacher_scores.ndim != 2 or min(teacher_scores.shape) < 1:
        raise ValueError("teacher_scores must have shape (bags, regions)")
    n_bags, n_regions = teacher_scores.shape
    if vertebra_targets.shape != (n_bags,):
        raise ValueError("vertebra_targets must have shape (bags,)")
    if teacher_outer_folds.shape != (n_bags,):
        raise ValueError("teacher_outer_folds must have shape (bags,)")
    if torch.unique(teacher_outer_folds).numel() != 1:
        raise ValueError("a pair batch must contain exactly one teacher outer fold")
    if (
        not torch.isfinite(vertebra_targets).all()
        or not (vertebra_targets.eq(0) | vertebra_targets.eq(1)).all()
    ):
        raise ValueError("vertebra_targets must be binary and finite")
    if human_target_valid is not None and human_target_valid.shape != (
        n_bags,
        n_regions,
    ):
        raise ValueError("human_target_valid must match teacher_scores")


def _validate_region_pair_batch(
    pairs: RegionPairBatch,
    n_bags: int,
    n_regions: int,
) -> None:
    pair_tensors = (
        pairs.left_bag_indices,
        pairs.right_bag_indices,
        pairs.region_indices,
        pairs.teacher_log_score_left,
        pairs.teacher_log_score_right,
    )
    if any(tensor.ndim != 1 for tensor in pair_tensors):
        raise ValueError("pair tensors must be one-dimensional")
    if any(tensor.numel() != pairs.n_pairs for tensor in pair_tensors):
        raise ValueError("pair tensors must have equal length")
    if pairs.pair_counts_by_region.shape != (n_regions,):
        raise ValueError("pair_counts_by_region must have shape (regions,)")
    index_tensors = (
        pairs.left_bag_indices,
        pairs.right_bag_indices,
        pairs.region_indices,
        pairs.pair_counts_by_region,
    )
    if any(tensor.dtype != torch.int64 for tensor in index_tensors):
        raise ValueError("pair indices and counts must use int64")
    if (
        not torch.isfinite(pairs.teacher_log_score_left).all()
        or not torch.isfinite(pairs.teacher_log_score_right).all()
    ):
        raise ValueError("teacher log scores must be finite")
    if pairs.pair_counts_by_region.lt(0).any():
        raise ValueError("pair counts must be non-negative")
    if int(pairs.pair_counts_by_region.sum()) != pairs.n_pairs:
        raise ValueError("pair counts do not match flattened pairs")
    if pairs.n_pairs == 0:
        return
    if pairs.left_bag_indices.min() < 0 or pairs.left_bag_indices.max() >= n_bags:
        raise ValueError("left bag index is out of range")
    if pairs.right_bag_indices.min() < 0 or pairs.right_bag_indices.max() >= n_bags:
        raise ValueError("right bag index is out of range")
    if pairs.region_indices.min() < 0 or pairs.region_indices.max() >= n_regions:
        raise ValueError("region index is out of range")
    observed_counts = torch.bincount(pairs.region_indices, minlength=n_regions).to(
        pairs.pair_counts_by_region.device
    )
    if not torch.equal(observed_counts, pairs.pair_counts_by_region):
        raise ValueError("per-region counts do not match region indices")
    if pairs.left_bag_indices.eq(pairs.right_bag_indices).any():
        raise ValueError("self-pairs are not allowed")
