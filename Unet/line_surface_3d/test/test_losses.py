"""平面損失のテスト。"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from line_surface_3d.utils.losses import compute_plane_loss, masked_heatmap_mse
from line_surface_3d.utils.plane import centered_positions

SLAB_SIZE = 15
IMAGE_SIZE = 64
SIGMA = 4.0
DIAGONAL = math.sqrt(2.0) * IMAGE_SIZE


def _build_case(slope_image: float) -> dict[str, Any]:
    """既知の傾きを持つ平面から、heatmapとスライス教師を作る。"""
    positions = centered_positions(SLAB_SIZE, torch.device("cpu"), torch.float32)
    rows = torch.arange(IMAGE_SIZE).float()[:, None]
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    line_params = torch.full((1, SLAB_SIZE, 4, 2), float("nan"))
    label_mask = torch.zeros(1, SLAB_SIZE, 4, dtype=torch.bool)
    for index in range(SLAB_SIZE):
        center_y = 32.0 + slope_image * float(positions[index])
        ridge = torch.exp(-((rows - center_y) ** 2) / (2.0 * SIGMA**2))
        heatmaps[0, index, 0] = ridge.expand(IMAGE_SIZE, IMAGE_SIZE)
        if 4 <= index <= 10:
            line_params[0, index, 0, 0] = math.pi / 2.0
            line_params[0, index, 0, 1] = (32.0 - center_y) / DIAGONAL
            label_mask[0, index, 0] = True
    return {
        "heatmaps": heatmaps,
        "line_params": line_params,
        "label_mask": label_mask,
        "positions": positions,
        # 数学座標での傾き。画像行が増える方向とは符号が逆
        "slope": torch.tensor([[-slope_image, 0.0, 0.0, 0.0]]),
    }


def _loss(case: dict[str, Any], **overrides: Any) -> Any:
    """既定設定で平面損失を計算する。"""
    config: dict[str, Any] = {
        "enabled": True,
        "angle_weight": 1.0,
        "rho_weight": 1.0,
        "tilt_weight": 1.0,
        "fallback_weight": 0.25,
    }
    config.update(overrides)
    return compute_plane_loss(
        case["prediction"],
        case["heatmaps"],
        case["label_mask"],
        case["line_params"],
        case["gt_slope"],
        case["reliable"],
        case["positions"],
        IMAGE_SIZE,
        config,
        geometry_weight=1.0,
    )


def test_disabled_plane_loss_equals_heatmap_loss() -> None:
    """平面制約が無効なときはheatmap損失と一致する。"""
    case = _build_case(0.5)
    prediction = case["heatmaps"].clone()
    output = compute_plane_loss(
        prediction,
        case["heatmaps"],
        case["label_mask"],
        case["line_params"],
        case["slope"],
        torch.ones(1, 4, dtype=torch.bool),
        case["positions"],
        IMAGE_SIZE,
        {"enabled": False},
        geometry_weight=1.0,
    )
    expected = masked_heatmap_mse(prediction, case["heatmaps"], case["label_mask"])
    assert output.total.item() == pytest.approx(expected.item())
    assert output.angle.item() == 0.0
    assert output.tilt.item() == 0.0


def test_perfect_prediction_has_near_zero_geometry_loss() -> None:
    """予測が教師と一致すれば幾何項はほぼ0になる。"""
    case = _build_case(0.5)
    case["prediction"] = case["heatmaps"].clone()
    case["gt_slope"] = case["slope"]
    case["reliable"] = torch.ones(1, 4, dtype=torch.bool)
    output = _loss(case)
    assert output.angle.item() < 1e-3
    assert output.rho.item() < 1e-2
    assert output.tilt.item() < 1e-2


def test_wrong_tilt_sign_increases_tilt_loss() -> None:
    """傾きの符号が逆なら傾き項が明確に増える。"""
    case = _build_case(0.5)
    case["prediction"] = case["heatmaps"].clone()
    case["reliable"] = torch.ones(1, 4, dtype=torch.bool)

    case["gt_slope"] = case["slope"]
    correct = _loss(case).tilt.item()
    case["gt_slope"] = -case["slope"]
    flipped = _loss(case).tilt.item()
    assert flipped > correct * 5.0


def test_fallback_surfaces_are_down_weighted() -> None:
    """垂直fallback面の傾き項は重みが下がる。"""
    case = _build_case(0.5)
    case["prediction"] = case["heatmaps"].clone()
    case["gt_slope"] = torch.zeros(1, 4)

    case["reliable"] = torch.ones(1, 4, dtype=torch.bool)
    reliable_loss = _loss(case).tilt.item()
    case["reliable"] = torch.zeros(1, 4, dtype=torch.bool)
    fallback_loss = _loss(case).tilt.item()
    assert fallback_loss == pytest.approx(0.25 * reliable_loss, rel=1e-4)


def test_tilt_loss_backpropagates_into_heatmap() -> None:
    """傾き項の勾配がheatmapまで届く。

    これが成立しないと、平面制約は学習に何も効かない。
    """
    case = _build_case(0.5)
    prediction = case["heatmaps"].clone().requires_grad_(True)
    case["prediction"] = prediction
    case["gt_slope"] = -case["slope"]
    case["reliable"] = torch.ones(1, 4, dtype=torch.bool)
    output = _loss(case, angle_weight=0.0, rho_weight=0.0)
    output.tilt.backward()

    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    # 教師のない上下スライスにも勾配が届く（平面制約が外挿を縛る）
    assert float(prediction.grad[0, 0, 0].abs().sum()) > 0.0
    assert float(prediction.grad[0, -1, 0].abs().sum()) > 0.0
