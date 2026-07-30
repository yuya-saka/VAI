"""masked heatmap損失とリボン幾何損失。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .ribbon import RibbonFit, compute_heatmap_moments, fit_ribbon


@dataclass(frozen=True)
class SurfaceLossOutput:
    """学習で利用する損失成分と予測fit。"""

    total: torch.Tensor
    heatmap: torch.Tensor
    angle: torch.Tensor
    centroid: torch.Tensor
    ribbon: torch.Tensor
    fit: RibbonFit


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


def compute_surface_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    ribbon_config: dict[str, Any],
    geometry_weight: float,
) -> SurfaceLossOutput:
    """heatmap証拠と1次リボン制約を統合する。"""
    heatmap_loss = masked_heatmap_mse(prediction, target, label_mask)
    prediction_fit = fit_ribbon(compute_heatmap_moments(prediction))
    zero = prediction.sum() * 0.0
    if not bool(ribbon_config.get("enabled", False)):
        return SurfaceLossOutput(
            total=heatmap_loss,
            heatmap=heatmap_loss,
            angle=zero,
            centroid=zero,
            ribbon=zero,
            fit=prediction_fit,
        )

    with torch.no_grad():
        target_fit = fit_ribbon(compute_heatmap_moments(target))
    valid_mask = label_mask & target_fit.valid
    angle_dot = (
        prediction_fit.fitted_values[..., 2:4] * target_fit.raw_values[..., 2:4]
    ).sum(dim=-1)
    angle_per_line = 1.0 - angle_dot.clamp(min=-1.0, max=1.0)
    angle_loss = _masked_mean(angle_per_line, valid_mask)

    centroid_delta = (
        prediction_fit.fitted_values[..., :2] - target_fit.raw_values[..., :2]
    )
    centroid_per_line = torch.linalg.vector_norm(centroid_delta, dim=-1)
    centroid_loss = _masked_mean(centroid_per_line, valid_mask)

    raw_centroid = prediction_fit.raw_values[..., :2]
    fitted_centroid = prediction_fit.fitted_values[..., :2]
    centroid_residual = F.smooth_l1_loss(
        raw_centroid,
        fitted_centroid,
        reduction="none",
    ).mean(dim=-1)
    raw_angle = prediction_fit.raw_values[..., 2:4]
    fitted_angle = prediction_fit.fitted_values[..., 2:4]
    angle_residual = 1.0 - (raw_angle * fitted_angle).sum(dim=-1).clamp(-1.0, 1.0)
    ribbon_loss = (centroid_residual + angle_residual).mean()

    weighted_geometry = (
        float(ribbon_config.get("angle_weight", 1.0)) * angle_loss
        + float(ribbon_config.get("centroid_weight", 1.0)) * centroid_loss
        + float(ribbon_config.get("residual_weight", 1.0)) * ribbon_loss
    )
    total = heatmap_loss + geometry_weight * weighted_geometry
    return SurfaceLossOutput(
        total=total,
        heatmap=heatmap_loss,
        angle=angle_loss,
        centroid=centroid_loss,
        ribbon=ribbon_loss,
        fit=prediction_fit,
    )
