"""masked heatmap損失と厳密平面の幾何損失。

幾何項は3つだけ。平面のパラメータが3つ（角度・オフセット・傾き）だからである。

- `angle`  : 共有法線の向き
- `rho`    : 教師スライス上での符号付きオフセット
- `tilt`   : zあたりのオフセット変化。傾きが不確かな面は垂直 (`k=0`) が教師。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .plane import PlaneFit, aligned_rho_targets, fit_plane, gt_plane_from_slices

# 3項すべてを「線の位置ずれ [px]」へ換算してから同じHuberに通す。
# 角度と傾きを無次元量のまま扱うと、重み1.0での勾配が項ごとに数千倍ずれる。
HUBER_SCALE_PX = 3.0
# 角度誤差をpxへ換算する腕の長さ。線の実効長の半分に相当する
ANGLE_ARM_RATIO = 0.25


@dataclass(frozen=True)
class PlaneLossOutput:
    """学習で利用する損失成分と予測平面。"""

    total: torch.Tensor
    heatmap: torch.Tensor
    angle: torch.Tensor
    rho: torch.Tensor
    tilt: torch.Tensor
    prediction: PlaneFit
    target: PlaneFit


def masked_heatmap_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """教師が存在する線だけでヒートマップMSEを計算する。"""
    if prediction.shape != target.shape:
        raise ValueError(
            f"予測と教師の形状が不一致です: {prediction.shape}, {target.shape}"
        )
    per_line = (prediction - target).square().mean(dim=(-1, -2))
    mask = label_mask.to(dtype=per_line.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return (per_line * mask).sum() / denominator


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """maskされた値の平均を安全に計算する。"""
    float_mask = mask.to(dtype=values.dtype)
    denominator = float_mask.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return (values * float_mask).sum() / denominator


def warmup_weight(
    epoch: int,
    start_epoch: int,
    warmup_epochs: int,
) -> float:
    """幾何損失の線形warmup係数を返す。"""
    if epoch <= start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch - start_epoch) / warmup_epochs))


def compute_plane_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    line_params_gt: torch.Tensor,
    gt_slope: torch.Tensor,
    gt_reliable: torch.Tensor,
    positions: torch.Tensor,
    image_size: int,
    plane_config: dict[str, Any],
    geometry_weight: float,
) -> PlaneLossOutput:
    """heatmap証拠と厳密平面の3パラメータ制約を統合する。"""
    heatmap_loss = masked_heatmap_mse(prediction, target, label_mask)
    predicted_plane = fit_plane(prediction, positions)
    zero = prediction.sum() * 0.0

    target_plane = gt_plane_from_slices(
        line_params_gt,
        label_mask,
        gt_slope,
        positions,
        image_size,
    )
    if not bool(plane_config.get("enabled", False)) or geometry_weight <= 0.0:
        return PlaneLossOutput(
            total=heatmap_loss,
            heatmap=heatmap_loss,
            angle=zero,
            rho=zero,
            tilt=zero,
            prediction=predicted_plane,
            target=target_plane,
        )

    surface_valid = label_mask.any(dim=1) & target_plane.valid & predicted_plane.valid

    # 角度: doubled-angleの外積は sin(2*theta)。腕の長さを掛けて線端でのpxずれにする。
    # acosと違い0付近で滑らかで、45度以上では飽和するので外れ値にも強い
    predicted_doubled = predicted_plane.doubled_normal
    target_doubled = target_plane.doubled_normal
    angle_cross = (
        predicted_doubled[..., 0] * target_doubled[..., 1]
        - predicted_doubled[..., 1] * target_doubled[..., 0]
    )
    angle_arm_px = ANGLE_ARM_RATIO * image_size
    angle_px = 0.5 * angle_cross.abs() * angle_arm_px
    angle_per_surface = F.huber_loss(
        angle_px / HUBER_SCALE_PX,
        torch.zeros_like(angle_px),
        reduction="none",
    )
    angle_loss = _masked_mean(angle_per_surface, surface_valid)

    # オフセット: 予測平面の交線を教師スライス上で比較する
    rho_target = aligned_rho_targets(
        line_params_gt,
        label_mask,
        predicted_plane.normal,
        image_size,
    )
    rho_predicted = predicted_plane.rho_at(positions)
    rho_per_slice = F.huber_loss(
        rho_predicted / HUBER_SCALE_PX,
        rho_target / HUBER_SCALE_PX,
        reduction="none",
    )
    rho_loss = _masked_mean(rho_per_slice, label_mask)

    # 傾き: v = k * n は符号表現の反転に不変。
    # スラブ端までの距離を掛けて、そこでの線の位置ずれ [px] にする
    tilt_arm_slices = float(positions.abs().max().clamp(min=1.0))
    tilt_predicted = predicted_plane.tilt_vector() * tilt_arm_slices
    tilt_target = target_plane.tilt_vector() * tilt_arm_slices
    tilt_per_surface = F.huber_loss(
        tilt_predicted / HUBER_SCALE_PX,
        tilt_target / HUBER_SCALE_PX,
        reduction="none",
    ).sum(dim=-1)
    fallback_weight = float(plane_config.get("fallback_weight", 0.25))
    tilt_weight = torch.where(
        gt_reliable,
        torch.ones_like(tilt_per_surface),
        torch.full_like(tilt_per_surface, fallback_weight),
    )
    tilt_loss = _masked_mean(tilt_per_surface * tilt_weight, surface_valid)

    weighted_geometry = (
        float(plane_config.get("angle_weight", 1.0)) * angle_loss
        + float(plane_config.get("rho_weight", 1.0)) * rho_loss
        + float(plane_config.get("tilt_weight", 1.0)) * tilt_loss
    )
    return PlaneLossOutput(
        total=heatmap_loss + geometry_weight * weighted_geometry,
        heatmap=heatmap_loss,
        angle=angle_loss,
        rho=rho_loss,
        tilt=tilt_loss,
        prediction=predicted_plane,
        target=target_plane,
    )
