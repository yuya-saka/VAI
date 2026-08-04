"""段階的な局所幾何整合性損失のテスト。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from Unet.line_2p5d.src.losses import (
    compute_loss,
    geometry_consistency_losses,
    geometry_weight_at_epoch,
)

IMAGE_SIZE = 32
SLICE_COUNT = 5


def _line_heatmaps(
    row_offsets: tuple[float, ...],
    angle_slopes: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """全line channelに同じGaussian ridgeを持つheatmapを作る。"""
    y_grid, x_grid = torch.meshgrid(
        torch.arange(IMAGE_SIZE, dtype=torch.float32),
        torch.arange(IMAGE_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    slopes = angle_slopes or (0.0,) * len(row_offsets)
    heatmaps = torch.zeros(1, len(row_offsets), 4, IMAGE_SIZE, IMAGE_SIZE)
    for index, (row_offset, slope) in enumerate(zip(row_offsets, slopes, strict=True)):
        center_line = (
            IMAGE_SIZE / 2.0 + row_offset + slope * (x_grid - IMAGE_SIZE / 2.0)
        )
        ridge = torch.exp(-0.5 * ((y_grid - center_line) / 1.5).square())
        heatmaps[0, index] = ridge
    return heatmaps


def test_geometry_schedule_can_be_disabled_and_delayed() -> None:
    """設定で無効化でき、指定epoch後だけ線形に立ち上がる。"""
    assert geometry_weight_at_epoch(100, False, 10, 5) == 0.0
    assert geometry_weight_at_epoch(10, True, 10, 5) == 0.0
    assert geometry_weight_at_epoch(12, True, 10, 5) == 0.4
    assert geometry_weight_at_epoch(20, True, 10, 5) == 1.0


def test_heatmap_loss_is_plain_mean_mse() -> None:
    """幾何無効時のheatmap損失が通常のmean MSEと完全一致する。"""
    predictions = torch.rand(2, SLICE_COUNT, 4, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.rand(2, 4, IMAGE_SIZE, IMAGE_SIZE)
    output = compute_loss(
        predictions,
        targets,
        torch.ones(2, SLICE_COUNT, dtype=torch.bool),
        {"geometry": {"enabled": False}},
        epoch=100,
        image_size=IMAGE_SIZE,
    )
    expected = F.mse_loss(
        predictions[:, SLICE_COUNT // 2],
        targets,
        reduction="mean",
    )
    assert torch.equal(output.heatmap, expected)
    assert torch.equal(output.total, expected)


def test_consistent_linear_motion_has_near_zero_geometry_loss() -> None:
    """一定速度で移動する平行線は局所整合とみなす。"""
    heatmaps = _line_heatmaps((-2.0, -1.0, 0.0, 1.0, 2.0))
    valid = torch.ones(1, SLICE_COUNT, dtype=torch.bool)
    angle_loss, position_loss = geometry_consistency_losses(
        heatmaps,
        valid,
        min_confidence=0.1,
        image_size=IMAGE_SIZE,
    )
    assert float(angle_loss) < 1e-5
    assert float(position_loss) < 1e-5


def test_single_slice_position_jump_is_penalized() -> None:
    """1枚だけ線位置が飛ぶ崩れを検出する。"""
    heatmaps = _line_heatmaps((-2.0, -1.0, 5.0, 1.0, 2.0))
    valid = torch.ones(1, SLICE_COUNT, dtype=torch.bool)
    _, position_loss = geometry_consistency_losses(
        heatmaps,
        valid,
        min_confidence=0.1,
        image_size=IMAGE_SIZE,
    )
    assert float(position_loss) > 0.01


def test_single_slice_angle_jump_is_penalized() -> None:
    """1枚だけ線角度が飛ぶ崩れを検出する。"""
    heatmaps = _line_heatmaps(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0, 0.0),
    )
    valid = torch.ones(1, SLICE_COUNT, dtype=torch.bool)
    angle_loss, _ = geometry_consistency_losses(
        heatmaps,
        valid,
        min_confidence=0.1,
        image_size=IMAGE_SIZE,
    )
    assert float(angle_loss) > 0.01


def test_active_geometry_backpropagates_to_all_slices() -> None:
    """局所整合性損失が周辺スライス予測にも勾配を送る。"""
    predictions = _line_heatmaps((-2.0, -1.0, 4.0, 1.0, 2.0)).requires_grad_(True)
    targets = predictions[:, 2].detach().clone()
    output = compute_loss(
        predictions,
        targets,
        torch.ones(1, SLICE_COUNT, dtype=torch.bool),
        {
            "geometry": {
                "enabled": True,
                "start_epoch": 0,
                "ramp_epochs": 0,
                "angle_weight": 1.0,
                "position_weight": 1.0,
                "min_confidence": 0.1,
            },
        },
        epoch=1,
        image_size=IMAGE_SIZE,
    )
    output.total.backward()
    assert predictions.grad is not None
    per_slice_gradient = predictions.grad.abs().sum(dim=(0, 2, 3, 4))
    assert torch.all(per_slice_gradient > 0)
