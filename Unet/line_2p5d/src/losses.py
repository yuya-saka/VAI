"""中心heatmap教師と局所z方向幾何整合性の損失。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HeatmapMoments:
    """各heatmapの重心、線方向、法線、異方性confidence。"""

    centroid: torch.Tensor
    doubled_direction: torch.Tensor
    normal: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class LossOutput:
    """学習で記録する損失成分。"""

    total: torch.Tensor
    heatmap: torch.Tensor
    heatmap_mse: torch.Tensor
    angle_consistency: torch.Tensor
    position_consistency: torch.Tensor
    geometry_weight: float


def geometry_weight_at_epoch(
    epoch: int,
    enabled: bool,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    """heatmap収束後に幾何損失を線形に立ち上げる。"""
    if not enabled or epoch <= start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - start_epoch) / ramp_epochs)


def compute_heatmap_moments(
    heatmaps: torch.Tensor,
    epsilon: float = 1e-6,
) -> HeatmapMoments:
    """`(B,N,L,H,W)` heatmapから微分可能な直線momentを抽出する。"""
    if heatmaps.ndim != 5:
        raise ValueError(f"heatmap shapeが不正です: {tuple(heatmaps.shape)}")
    _, _, _, height, width = heatmaps.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    mass = heatmaps.sum(dim=(-2, -1)).clamp_min(epsilon)
    centroid_x = (heatmaps * x_grid).sum(dim=(-2, -1)) / mass
    centroid_y = (heatmaps * y_grid).sum(dim=(-2, -1)) / mass
    delta_x = x_grid - centroid_x[..., None, None]
    delta_y = y_grid - centroid_y[..., None, None]
    variance_x = (heatmaps * delta_x.square()).sum(dim=(-2, -1)) / mass
    variance_y = (heatmaps * delta_y.square()).sum(dim=(-2, -1)) / mass
    covariance_xy = (heatmaps * delta_x * delta_y).sum(dim=(-2, -1)) / mass
    difference = variance_x - variance_y
    anisotropy = torch.sqrt(
        difference.square() + 4.0 * covariance_xy.square() + epsilon
    )
    doubled_cosine = difference / anisotropy
    doubled_sine = 2.0 * covariance_xy / anisotropy
    angle = 0.5 * torch.atan2(doubled_sine, doubled_cosine)
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    confidence = anisotropy / (variance_x + variance_y + epsilon)
    return HeatmapMoments(
        centroid=torch.stack([centroid_x, centroid_y], dim=-1),
        doubled_direction=torch.stack([doubled_cosine, doubled_sine], dim=-1),
        normal=normal,
        confidence=confidence.clamp(0.0, 1.0),
    )


def _confidence_gate(confidence: torch.Tensor, minimum: float) -> torch.Tensor:
    """低異方性heatmapを連続値で弱め、重み自体はdetachする。"""
    denominator = max(1e-6, 1.0 - minimum)
    return ((confidence - minimum) / denominator).clamp(0.0, 1.0).detach()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """有効重みがない場合も勾配接続を保った0を返す。"""
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def geometry_consistency_losses(
    heatmaps: torch.Tensor,
    context_valid: torch.Tensor,
    min_confidence: float,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """隣接角度と局所位置曲率の整合性損失を返す。"""
    moments = compute_heatmap_moments(heatmaps)
    gate = _confidence_gate(moments.confidence, min_confidence)
    valid = context_valid.to(dtype=heatmaps.dtype, device=heatmaps.device)

    doubled_dot = (
        moments.doubled_direction[:, :-1] * moments.doubled_direction[:, 1:]
    ).sum(dim=-1)
    angle_values = 0.5 * (1.0 - doubled_dot.clamp(-1.0, 1.0))
    angle_weights = (
        gate[:, :-1] * gate[:, 1:] * valid[:, :-1, None] * valid[:, 1:, None]
    )
    angle_loss = _weighted_mean(angle_values, angle_weights)

    second_difference = (
        moments.centroid[:, :-2]
        - 2.0 * moments.centroid[:, 1:-1]
        + moments.centroid[:, 2:]
    )
    normal_displacement = (second_difference * moments.normal[:, 1:-1]).sum(dim=-1)
    normalized_displacement = normal_displacement / max(1.0, image_size / 4.0)
    position_values = F.smooth_l1_loss(
        normalized_displacement,
        torch.zeros_like(normalized_displacement),
        beta=0.05,
        reduction="none",
    )
    position_weights = (
        gate[:, :-2]
        * gate[:, 1:-1]
        * gate[:, 2:]
        * valid[:, :-2, None]
        * valid[:, 1:-1, None]
        * valid[:, 2:, None]
    )
    position_loss = _weighted_mean(position_values, position_weights)
    return angle_loss, position_loss


def compute_loss(
    predictions: torch.Tensor,
    target_heatmaps: torch.Tensor,
    context_valid: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
    image_size: int,
) -> LossOutput:
    """中心教師損失と、任意の局所幾何整合性損失を合成する。"""
    center = predictions.shape[1] // 2
    center_prediction = predictions[:, center]
    geometry_config = config.get("geometry", {})
    heatmap_mse = F.mse_loss(center_prediction, target_heatmaps, reduction="mean")
    heatmap_loss = heatmap_mse

    geometry_weight = geometry_weight_at_epoch(
        epoch=epoch,
        enabled=bool(geometry_config.get("enabled", False)),
        start_epoch=int(geometry_config.get("start_epoch", 0)),
        ramp_epochs=int(geometry_config.get("ramp_epochs", 0)),
    )
    if geometry_weight > 0.0:
        angle_loss, position_loss = geometry_consistency_losses(
            predictions,
            context_valid,
            min_confidence=float(geometry_config.get("min_confidence", 0.1)),
            image_size=image_size,
        )
    else:
        zero = predictions.sum() * 0.0
        angle_loss = zero
        position_loss = zero
    geometry_loss = (
        float(geometry_config.get("angle_weight", 0.01)) * angle_loss
        + float(geometry_config.get("position_weight", 0.01)) * position_loss
    )
    total = heatmap_loss + geometry_weight * geometry_loss
    return LossOutput(
        total=total,
        heatmap=heatmap_loss,
        heatmap_mse=heatmap_mse,
        angle_consistency=angle_loss,
        position_consistency=position_loss,
        geometry_weight=geometry_weight,
    )
