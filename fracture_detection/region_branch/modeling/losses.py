"""region logitのbag集約とsource-balanced exact loss。

`L = L_whole + lambda*(L_exact + alpha*L_rank)`のうち、`L_whole`は
`baseline0.modeling.losses.broadcast_bce_loss`をそのまま使う。ここでは
`L_exact = 0.5*L_H + 0.5*L_N`（human-annotated / whole-negative由来）と、
`baseline0.pseudo_labeling.scoring`のpairwise ranking損失へのbag logit
アダプタを提供する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fracture_detection.baseline0.pseudo_labeling.scoring import (
    RegionPairBatch,
    build_region_pair_batch,
    region_balanced_pairwise_ranking_loss,
)

BAG_PROBABILITY_EPSILON = 1e-6


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


@dataclass(frozen=True)
class ExactLossTerms:
    """1バッチ（または複数バッチの合計）のsource別・領域別 重み付き損失と重み。"""

    human_weighted_loss: Tensor  # [R]
    human_weight: Tensor  # [R]
    negative_loss: Tensor  # [R]
    negative_count: Tensor  # [R]

    def __add__(self, other: ExactLossTerms) -> ExactLossTerms:
        return ExactLossTerms(
            human_weighted_loss=self.human_weighted_loss + other.human_weighted_loss,
            human_weight=self.human_weight + other.human_weight,
            negative_loss=self.negative_loss + other.negative_loss,
            negative_count=self.negative_count + other.negative_count,
        )


def compute_exact_loss_terms(
    bag_logits: Tensor,
    region_targets: Tensor,
    region_target_valid: Tensor,
    cell_valid: Tensor,
    vertebra_target: Tensor,
) -> ExactLossTerms:
    """1バッチのhuman/negative別・領域別 重み付きBCE和と重み和を返す。

    human: `region_target_valid`かつ`cell_valid`なセルのBCE(z,y)和。
    negative: `vertebra_target==0`なbagの`cell_valid`セルへBCE(z,0)を課した和。
    どちらも`cell_valid`（有効面が1つ以上ある）で無意味な集約結果を除外する。
    """
    if (
        bag_logits.shape != region_targets.shape
        or bag_logits.shape != region_target_valid.shape
        or bag_logits.shape != cell_valid.shape
    ):
        raise ValueError(
            "bag_logits/region_targets/region_target_valid/cell_validの"
            "shapeが一致しません"
        )
    if vertebra_target.shape != bag_logits.shape[:1]:
        raise ValueError("vertebra_targetのshapeが不正です")

    per_cell_loss = F.binary_cross_entropy_with_logits(
        bag_logits, region_targets, reduction="none"
    )
    human_mask = (region_target_valid & cell_valid).to(bag_logits.dtype)
    human_weighted_loss = (per_cell_loss * human_mask).sum(dim=0)
    human_weight = human_mask.sum(dim=0)

    negative_bag_mask = (vertebra_target.eq(0).unsqueeze(1) & cell_valid).to(
        bag_logits.dtype
    )
    negative_loss_per_cell = F.binary_cross_entropy_with_logits(
        bag_logits, torch.zeros_like(bag_logits), reduction="none"
    )
    negative_loss = (negative_loss_per_cell * negative_bag_mask).sum(dim=0)
    negative_count = negative_bag_mask.sum(dim=0)

    return ExactLossTerms(
        human_weighted_loss=human_weighted_loss,
        human_weight=human_weight,
        negative_loss=negative_loss,
        negative_count=negative_count,
    )


def combine_exact_terms(
    terms: ExactLossTerms, reference: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """領域別L_H・L_Nをactive region平均し、`L_exact = 0.5*L_H + 0.5*L_N`を返す。

    有効セルがゼロの領域はその項の領域平均から除外する（統合4領域モデルの
    領域損失を4倍にしないための平均であり、和ではない）。全領域が無効な項は
    `reference`由来の勾配を保つ0を返す。
    """
    l_h = _region_mean_or_zero(terms.human_weighted_loss, terms.human_weight, reference)
    l_n = _region_mean_or_zero(terms.negative_loss, terms.negative_count, reference)
    return 0.5 * l_h + 0.5 * l_n, l_h, l_n


def _region_mean_or_zero(
    numerator: Tensor, denominator: Tensor, reference: Tensor
) -> Tensor:
    active = denominator > 0
    if not bool(active.any()):
        return reference.sum() * 0.0
    return (numerator[active] / denominator[active]).mean()


def region_rank_loss(
    student_region_bag_logits: Tensor,
    teacher_scores: Tensor,
    vertebra_target: Tensor,
    teacher_outer_fold: Tensor,
    temperatures: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, RegionPairBatch]:
    """pseudo-positive batchのpairwise ranking損失をbag logitへ適用する。

    `baseline0.pseudo_labeling.scoring`のペア構築・損失をそのまま呼ぶ薄い
    アダプタ。pseudo pool自体がhuman-annotated bagを含まないため
    `human_target_valid`は渡さない。
    """
    pairs = build_region_pair_batch(
        teacher_scores, vertebra_target, teacher_outer_fold, generator
    )
    loss = region_balanced_pairwise_ranking_loss(
        student_region_bag_logits, pairs, temperatures
    )
    return loss, pairs
