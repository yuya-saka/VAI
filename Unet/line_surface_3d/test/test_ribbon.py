"""1次リボンfitと周期表現のテスト。"""

from __future__ import annotations

import math

import torch
from line_surface_3d.utils.ribbon import (
    doubled_angle,
    fit_linear_values,
    normal_from_doubled,
)


def test_fit_linear_values_recovers_exact_ribbon() -> None:
    """完全な1次系列のfit残差は数値誤差範囲になる。"""
    positions = torch.arange(5, dtype=torch.float32) - 2.0
    intercept = torch.tensor([[[2.0, -1.0], [4.0, 3.0]]])
    slope = torch.tensor([[[0.5, 0.25], [-0.2, 0.1]]])
    values = intercept[:, None] + slope[:, None] * positions[None, :, None, None]
    fitted, fitted_intercept, fitted_slope = fit_linear_values(values, positions)
    assert torch.allclose(fitted, values, atol=1e-6)
    assert torch.allclose(fitted_intercept, intercept, atol=1e-6)
    assert torch.allclose(fitted_slope, slope, atol=1e-6)


def test_doubled_angle_crosses_zero_without_jump() -> None:
    """179度と1度を同じ周期上で扱える。"""
    angles = torch.deg2rad(torch.tensor([179.0, 0.0, 1.0]))
    normal_x = torch.cos(angles)
    normal_y = torch.sin(angles)
    cosine, sine = doubled_angle(normal_x, normal_y)
    values = torch.stack([cosine, sine], dim=-1)[None, :, None, :]
    fitted, _, _ = fit_linear_values(values)
    norm = torch.linalg.vector_norm(fitted, dim=-1)
    assert torch.all(norm > 0.99)
    recovered_x, recovered_y = normal_from_doubled(
        fitted[..., 0],
        fitted[..., 1],
    )
    center_angle = math.degrees(
        math.atan2(float(recovered_y[0, 1, 0]), float(recovered_x[0, 1, 0]))
    )
    assert abs(center_angle) < 1e-4
