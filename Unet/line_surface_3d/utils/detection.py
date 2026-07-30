"""集約リボンから有限長の画像座標線を再構成する。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

ThresholdSpec = float | str | dict[str, Any] | None
ADAPTIVE_THRESHOLD_MIN = 0.15
ADAPTIVE_THRESHOLD_PEAK_RATIO = 0.4


def _adaptive_threshold_value(
    heatmap: np.ndarray,
    min_threshold: float = ADAPTIVE_THRESHOLD_MIN,
    peak_ratio: float = ADAPTIVE_THRESHOLD_PEAK_RATIO,
) -> float:
    """ピーク値に応じた適応閾値を返す。"""
    peak = float(np.max(heatmap)) if heatmap.size > 0 else 0.0
    return max(float(min_threshold), float(peak_ratio) * peak)


def _resolve_threshold_value(
    heatmap: np.ndarray,
    threshold: ThresholdSpec,
) -> float | None:
    """固定または適応閾値を数値へ変換する。"""
    if threshold is None:
        return None
    if isinstance(threshold, str):
        if threshold != "adaptive":
            raise ValueError(f"未対応の閾値modeです: {threshold}")
        return _adaptive_threshold_value(heatmap)
    if isinstance(threshold, dict):
        mode = threshold.get("mode", "fixed")
        if mode != "adaptive":
            value = threshold.get("value")
            return None if value is None else float(value)
        return _adaptive_threshold_value(
            heatmap,
            min_threshold=float(threshold.get("min", ADAPTIVE_THRESHOLD_MIN)),
            peak_ratio=float(
                threshold.get("peak_ratio", ADAPTIVE_THRESHOLD_PEAK_RATIO)
            ),
        )
    return float(threshold)


def detect_line_moments(
    heatmap: np.ndarray,
    threshold: ThresholdSpec = 0.2,
    min_mass: float = 1e-6,
) -> dict[str, Any] | None:
    """line_onlyと同じCC-filter済みmoment法で直線を検出する。"""
    values = np.asarray(heatmap, dtype=np.float64)
    height, width = values.shape
    threshold_value = _resolve_threshold_value(values, threshold)
    if threshold_value is not None:
        values = np.where(values >= threshold_value, values, 0.0)

    labeled, component_count = ndimage.label((values > 0).astype(np.uint8))
    if component_count > 1:
        peak_y, peak_x = np.unravel_index(values.argmax(), values.shape)
        values = values * (labeled == labeled[peak_y, peak_x])

    mass = float(values.sum())
    if mass < min_mass:
        return None

    y_axis = -(np.arange(height, dtype=np.float64) - height / 2.0)
    x_axis = np.arange(width, dtype=np.float64) - width / 2.0
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    centroid_x = float((values * x_grid).sum() / (mass + 1e-12))
    centroid_y = float((values * y_grid).sum() / (mass + 1e-12))
    delta_x = x_grid - centroid_x
    delta_y = y_grid - centroid_y
    moment_xx = float((values * delta_x**2).sum() / (mass + 1e-12))
    moment_yy = float((values * delta_y**2).sum() / (mass + 1e-12))
    moment_xy = float((values * delta_x * delta_y).sum() / (mass + 1e-12))
    direction_angle = 0.5 * math.atan2(
        2.0 * moment_xy,
        moment_xx - moment_yy,
    )
    return {
        "centroid": [
            centroid_x + width / 2.0,
            -centroid_y + height / 2.0,
        ],
        "angle_rad": direction_angle,
    }


def moments_to_phi_rho(
    result: dict[str, Any],
    image_size: int,
) -> tuple[float, float]:
    """moment出力をline_onlyと同じ正規化 `(phi, rho)` へ変換する。"""
    direction_angle = float(result["angle_rad"])
    normal_x = -math.sin(direction_angle)
    normal_y = math.cos(direction_angle)
    if normal_y < 0 or (normal_y == 0 and normal_x < 0):
        normal_x, normal_y = -normal_x, -normal_y
    phi = math.atan2(normal_y, normal_x)
    centroid = result["centroid"]
    if not isinstance(centroid, list):
        raise TypeError("centroidの型が不正です")
    center = image_size / 2.0
    centroid_x = float(centroid[0]) - center
    centroid_y = -(float(centroid[1]) - center)
    rho = normal_x * centroid_x + normal_y * centroid_y
    diagonal = math.sqrt(image_size**2 + image_size**2)
    return float(phi), float(rho / diagonal)


def extract_pred_params_cc_batch(
    heatmaps: np.ndarray,
    image_size: int,
    threshold: ThresholdSpec = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CC-filter済み予測からbatch単位の `(phi, rho)` を返す。"""
    batch_size, channel_count = heatmaps.shape[:2]
    parameters = np.zeros((batch_size, channel_count, 2), dtype=np.float32)
    confidence = np.zeros((batch_size, channel_count), dtype=np.float32)
    for batch_index in range(batch_size):
        for channel_index in range(channel_count):
            result = detect_line_moments(
                heatmaps[batch_index, channel_index],
                threshold=threshold,
            )
            if result is None:
                continue
            parameters[batch_index, channel_index] = moments_to_phi_rho(
                result,
                image_size,
            )
            confidence[batch_index, channel_index] = 1.0
    return parameters, confidence


def extract_gt_line_params(
    polyline_points: list[list[float]] | None,
    image_size: int,
) -> tuple[float, float]:
    """GTポリラインからline_onlyと同じPCA `(phi, rho)` を抽出する。"""
    if polyline_points is None or len(polyline_points) < 2:
        return float("nan"), float("nan")
    center = image_size / 2.0
    points = np.asarray(polyline_points, dtype=np.float64)
    math_points = np.column_stack([points[:, 0] - center, -(points[:, 1] - center)])
    centroid = math_points.mean(axis=0)
    centered = math_points - centroid
    covariance = (centered.T @ centered) / max(1, len(points))
    if covariance.max() < 1e-10:
        return float("nan"), float("nan")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    normal_x, normal_y = -direction[1], direction[0]
    if normal_y < 0 or (normal_y == 0 and normal_x < 0):
        normal_x, normal_y = -normal_x, -normal_y
    phi = np.arctan2(normal_y, normal_x)
    rho = normal_x * centroid[0] + normal_y * centroid[1]
    diagonal = np.sqrt(image_size**2 + image_size**2)
    return float(phi), float(rho / diagonal)


def estimate_line_length(
    major_variance: float,
    sigma: float,
    image_size: int,
) -> float:
    """主軸分散から一様線分相当の長さを推定する。"""
    corrected_variance = max(0.0, major_variance - sigma**2)
    estimated = math.sqrt(12.0 * corrected_variance)
    return float(np.clip(estimated, 20.0, math.sqrt(2.0) * image_size))


def line_from_ribbon(
    centroid_x: float,
    centroid_y: float,
    doubled_cosine: float,
    doubled_sine: float,
    length: float,
    image_size: int,
    extend_ratio: float = 1.0,
) -> dict[str, Any]:
    """数学座標の重心・法線・長さから描画用endpointsを作る。"""
    normal_angle = 0.5 * math.atan2(doubled_sine, doubled_cosine)
    direction_x = -math.sin(normal_angle)
    direction_y = math.cos(normal_angle)
    half_length = 0.5 * length * extend_ratio
    endpoints_math = [
        (
            centroid_x - half_length * direction_x,
            centroid_y - half_length * direction_y,
        ),
        (
            centroid_x + half_length * direction_x,
            centroid_y + half_length * direction_y,
        ),
    ]
    center = image_size / 2.0
    endpoints_image = [
        [
            float(np.clip(point_x + center, 0, image_size - 1)),
            float(np.clip(-point_y + center, 0, image_size - 1)),
        ]
        for point_x, point_y in endpoints_math
    ]
    centroid_image = [
        float(centroid_x + center),
        float(-centroid_y + center),
    ]
    direction_angle = math.atan2(direction_y, direction_x)
    return {
        "centroid": centroid_image,
        "centroid_math": [float(centroid_x), float(centroid_y)],
        "normal_doubled": [float(doubled_cosine), float(doubled_sine)],
        "normal_angle_deg": float(math.degrees(normal_angle) % 180.0),
        "angle_deg": float(math.degrees(direction_angle) % 180.0),
        "dir": [float(direction_x), float(direction_y)],
        "endpoints": endpoints_image,
        "length": float(length * extend_ratio),
    }
