"""masked heatmap損失とリボン損失のテスト。"""

from __future__ import annotations

import torch
from line_surface_3d.utils.losses import (
    compute_surface_loss,
    masked_heatmap_mse,
)


def _line_heatmaps(
    batch_size: int = 1,
    slab_size: int = 3,
    line_count: int = 4,
    image_size: int = 16,
) -> torch.Tensor:
    """z方向へ平行移動するGaussian線を作る。"""
    y_axis = torch.arange(image_size, dtype=torch.float32)
    x_axis = torch.arange(image_size, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    heatmaps = torch.zeros(
        batch_size,
        slab_size,
        line_count,
        image_size,
        image_size,
    )
    for slab_index in range(slab_size):
        for line_index in range(line_count):
            center_y = 3.0 + line_index * 3.0 + 0.2 * slab_index
            heatmaps[:, slab_index, line_index] = torch.exp(
                -((y_grid - center_y) ** 2) / 2.0
            )
    return heatmaps


def test_masked_heatmap_mse_ignores_unlabeled_entries() -> None:
    """教師なしスライスの誤差を無視する。"""
    prediction = torch.zeros(1, 3, 4, 4, 4)
    target = torch.zeros_like(prediction)
    target[:, 0] = 1.0
    target[:, 1:] = 100.0
    label_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    label_mask[:, 0] = True
    assert torch.isclose(
        masked_heatmap_mse(prediction, target, label_mask), torch.tensor(1.0)
    )


def test_baseline_loss_equals_masked_mse() -> None:
    """ribbon無効時はheatmap baselineと完全一致する。"""
    prediction = _line_heatmaps().requires_grad_(True)
    target = _line_heatmaps()
    label_mask = torch.ones(1, 3, 4, dtype=torch.bool)
    output = compute_surface_loss(
        prediction,
        target,
        label_mask,
        {"enabled": False},
        geometry_weight=1.0,
    )
    expected = masked_heatmap_mse(prediction, target, label_mask)
    assert torch.equal(output.total, expected)


def test_surface_loss_has_finite_gradient() -> None:
    """全損失から予測ヒートマップへ有限勾配が届く。"""
    raw_prediction = torch.logit(_line_heatmaps().clamp(0.01, 0.99)).requires_grad_(
        True
    )
    prediction = torch.sigmoid(raw_prediction)
    target = _line_heatmaps()
    label_mask = torch.ones(1, 3, 4, dtype=torch.bool)
    output = compute_surface_loss(
        prediction,
        target,
        label_mask,
        {
            "enabled": True,
            "angle_weight": 0.1,
            "centroid_weight": 0.1,
            "residual_weight": 0.1,
        },
        geometry_weight=1.0,
    )
    output.total.backward()
    assert raw_prediction.grad is not None
    assert torch.isfinite(raw_prediction.grad).all()
