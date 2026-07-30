"""ヒートマップmomentと1次リボンfit。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HeatmapMoments:
    """スラブ内の各ヒートマップから抽出した幾何量。"""

    confidence: torch.Tensor
    normal_x: torch.Tensor
    normal_y: torch.Tensor
    centroid_x: torch.Tensor
    centroid_y: torch.Tensor
    major_variance: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class RibbonFit:
    """4量の生値、1次fit値、係数を保持する。"""

    raw_values: torch.Tensor
    fitted_values: torch.Tensor
    intercept: torch.Tensor
    slope: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    major_variance: torch.Tensor


def compute_heatmap_moments(
    heatmaps: torch.Tensor,
    min_mass: float = 1e-6,
) -> HeatmapMoments:
    """`(B,N,4,H,W)` ヒートマップから重心と主軸法線を抽出する。"""
    if heatmaps.ndim != 5:
        raise ValueError(f"heatmapsは5次元が必要です: {heatmaps.shape}")
    working_heatmaps = heatmaps.float()
    batch_size, slab_size, line_count, height, width = working_heatmaps.shape
    flat_count = batch_size * slab_size * line_count
    flattened = working_heatmaps.reshape(flat_count, height, width)
    device = working_heatmaps.device
    dtype = working_heatmaps.dtype

    y_axis = -(torch.arange(height, device=device, dtype=dtype) - height / 2.0)
    x_axis = torch.arange(width, device=device, dtype=dtype) - width / 2.0
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    values = flattened.reshape(flat_count, -1)
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    mass = values.sum(dim=-1)
    safe_mass = mass.clamp(min=min_mass)
    centroid_x = (values * x_flat).sum(dim=-1) / safe_mass
    centroid_y = (values * y_flat).sum(dim=-1) / safe_mass

    delta_x = x_flat[None, :] - centroid_x[:, None]
    delta_y = y_flat[None, :] - centroid_y[:, None]
    covariance_xx = (values * delta_x.square()).sum(dim=-1) / safe_mass + 1e-6
    covariance_yy = (values * delta_y.square()).sum(dim=-1) / safe_mass + 1e-6
    covariance_xy = (values * delta_x * delta_y).sum(dim=-1) / safe_mass
    discriminant = torch.hypot(
        covariance_xx - covariance_yy,
        2.0 * covariance_xy,
    )
    trace = covariance_xx + covariance_yy
    eigenvalue_1 = (trace + discriminant) / 2.0
    valid = (mass >= min_mass) & (eigenvalue_1 > 1e-8)
    confidence = discriminant / (trace + 1e-8)
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))

    angle_degenerate = discriminant < 1e-8
    angle_numerator = torch.where(
        angle_degenerate,
        torch.zeros_like(covariance_xy),
        2.0 * covariance_xy,
    )
    angle_denominator = torch.where(
        angle_degenerate,
        torch.ones_like(covariance_xx),
        covariance_xx - covariance_yy,
    )
    direction_angle = 0.5 * torch.atan2(angle_numerator, angle_denominator)
    direction_x = torch.cos(direction_angle)
    direction_y = torch.sin(direction_angle)
    normal_x = -direction_y
    normal_y = direction_x
    output_shape = (batch_size, slab_size, line_count)
    return HeatmapMoments(
        confidence=confidence.reshape(output_shape),
        normal_x=normal_x.reshape(output_shape),
        normal_y=normal_y.reshape(output_shape),
        centroid_x=centroid_x.reshape(output_shape),
        centroid_y=centroid_y.reshape(output_shape),
        major_variance=eigenvalue_1.reshape(output_shape),
        valid=valid.reshape(output_shape),
    )


def doubled_angle(
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """法線を180度周期のdoubled-angleへ変換する。"""
    cosine = normal_x.square() - normal_y.square()
    sine = 2.0 * normal_x * normal_y
    norm = torch.hypot(cosine, sine).clamp(min=1e-8)
    return cosine / norm, sine / norm


def centered_positions(
    slab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """スラブ中心を0とするz座標を返す。"""
    return torch.arange(slab_size, device=device, dtype=dtype) - (slab_size - 1) / 2.0


def fit_linear_values(
    values: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z軸方向に閉形式の1次回帰を適用する。"""
    if values.ndim < 3:
        raise ValueError("valuesは少なくとも `(B,N,C)` が必要です")
    slab_size = values.shape[1]
    if positions is None:
        positions = centered_positions(slab_size, values.device, values.dtype)
    if positions.shape != (slab_size,):
        raise ValueError(f"positionsの形状が不正です: {positions.shape}")
    centered = positions - positions.mean()
    denominator = centered.square().sum().clamp(min=1e-8)
    intercept = values.mean(dim=1)
    expand_shape = (1, slab_size) + (1,) * (values.ndim - 2)
    slope = (values * centered.reshape(expand_shape)).sum(dim=1) / denominator
    fitted = intercept[:, None] + slope[:, None] * centered.reshape(expand_shape)
    return fitted, intercept, slope


def fit_ribbon(
    moments: HeatmapMoments,
    positions: torch.Tensor | None = None,
) -> RibbonFit:
    """重心とdoubled-angleを線ごとに1次fitする。"""
    cosine, sine = doubled_angle(moments.normal_x, moments.normal_y)
    values = torch.stack(
        [moments.centroid_x, moments.centroid_y, cosine, sine],
        dim=-1,
    )
    fitted, intercept, slope = fit_linear_values(values, positions)
    normalized_angle = F.normalize(fitted[..., 2:4], dim=-1, eps=1e-8)
    fitted = torch.cat([fitted[..., :2], normalized_angle], dim=-1)
    return RibbonFit(
        raw_values=values,
        fitted_values=fitted,
        intercept=intercept,
        slope=slope,
        confidence=moments.confidence,
        valid=moments.valid,
        major_variance=moments.major_variance,
    )


def normal_from_doubled(
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """doubled-angleから符号任意の単位法線を復元する。"""
    angle = 0.5 * torch.atan2(sine, cosine)
    return torch.cos(angle), torch.sin(angle)
