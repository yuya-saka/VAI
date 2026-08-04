"""画像単位の直線抽出・誤差指標のテスト。"""

from __future__ import annotations

import math

import numpy as np

from Unet.line_2p5d.src.metrics import (
    detect_line_from_heatmap,
    detected_line_to_dict,
    line_errors,
)


def test_horizontal_heatmap_recovers_line_parameters() -> None:
    """水平Gaussian ridgeから角度と位置を復元する。"""
    image_size = 32
    y_grid, _ = np.indices((image_size, image_size), dtype=np.float64)
    heatmap = np.exp(-0.5 * ((y_grid - 12.0) / 1.5) ** 2).astype(np.float32)
    detected = detect_line_from_heatmap(
        heatmap,
        image_size,
        {"min": 0.1, "peak_ratio": 0.4},
        min_confidence=0.05,
    )
    assert detected is not None
    angle_error, rho_error = line_errors(
        detected,
        gt_angle_rad=0.0,
        gt_rho_px=-4.0,
    )
    assert angle_error < 0.1
    assert rho_error < 0.2


def test_line_error_is_invariant_to_direction_reversal() -> None:
    """直線方向の180度反転を誤差として数えない。"""
    from Unet.line_2p5d.src.metrics import DetectedLine

    prediction = DetectedLine(angle_rad=math.pi - 0.1, rho_px=2.0, confidence=1.0)
    angle_error, _ = line_errors(prediction, gt_angle_rad=-0.1, gt_rho_px=-2.0)
    assert angle_error < 1e-5


def test_rendered_line_uses_configured_extend_ratio() -> None:
    """描画線長へline_extend_ratioを適用する。"""
    from Unet.line_2p5d.src.metrics import DetectedLine

    detection = DetectedLine(
        angle_rad=0.0,
        rho_px=0.0,
        confidence=1.0,
        centroid_x=16.0,
        centroid_y=16.0,
    )
    rendered = detected_line_to_dict(detection, 10.0, 1.5, (32, 32))
    assert rendered is not None
    assert rendered["length"] == 15.0
    assert rendered["endpoints"] == [[8.5, 16.0], [23.5, 16.0]]
