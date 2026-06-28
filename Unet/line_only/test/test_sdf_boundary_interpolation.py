"""SDF境界補間後の線分復元を検証する。"""

import math

import numpy as np

from data_preprocessing.rsna_pipeline.sdf_boundary_interpolation import (
    SDFBoundaryInterpolator,
)

LINE_KEYS = tuple(f"line_{index}" for index in range(1, 5))
IMAGE_SIZE = 224


def _horizontal_phi_rho(y_image: float) -> tuple[float, float]:
    """画像上の水平線を(phi, rho)へ変換する。"""
    phi = math.pi / 2.0
    y_math = -(y_image - IMAGE_SIZE / 2.0)
    rho = y_math / (math.sqrt(2.0) * IMAGE_SIZE)
    return phi, rho


def _build_interpolator() -> SDFBoundaryInterpolator:
    """line別平均長と重心を持つ補間器を構築する。"""
    phi_rho = _horizontal_phi_rho(112.0)
    phi_rho_anchors = {
        line_key: [phi_rho, phi_rho]
        for line_key in LINE_KEYS
    }
    centroid_anchors = {
        line_key: [(80.0, 112.0), (100.0, 112.0)]
        for line_key in LINE_KEYS
    }
    line_lengths = {
        "line_1": 38.0,
        "line_2": 39.0,
        "line_3": 40.0,
        "line_4": 41.0,
    }
    return SDFBoundaryInterpolator(
        phi_rho_anchors=phi_rho_anchors,
        z_offsets=[0.0, 2.0],
        centre_idx=0,
        image_size=IMAGE_SIZE,
        centroid_anchors=centroid_anchors,
        line_lengths_px=line_lengths,
    )


def test_get_lines_at_anchor_preserves_centroid_and_average_length() -> None:
    """アンカー位置ではrsna_line_visと同じ重心・平均線長を復元する。"""
    interpolator = _build_interpolator()

    lines = interpolator.get_lines(0.0)

    assert lines is not None
    for index, line_key in enumerate(LINE_KEYS, start=1):
        endpoints = np.asarray(lines[line_key], dtype=np.float64)
        midpoint = endpoints.mean(axis=0)
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        assert np.allclose(midpoint, [80.0, 112.0])
        assert np.isclose(length, 37.0 + index)


def test_get_lines_between_anchors_interpolates_centroid() -> None:
    """アンカー間ではheatmap重心をz方向へ線形補間する。"""
    interpolator = _build_interpolator()

    lines = interpolator.get_lines(1.0)

    assert lines is not None
    for line_key in LINE_KEYS:
        endpoints = np.asarray(lines[line_key], dtype=np.float64)
        assert np.allclose(endpoints.mean(axis=0), [90.0, 112.0])
