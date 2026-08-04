"""中心画像heatmapから直線を抽出し、画像単位誤差を測る。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedLine:
    """heatmap momentから得た画像内直線。"""

    angle_rad: float
    rho_px: float
    confidence: float
    centroid_x: float = float("nan")
    centroid_y: float = float("nan")


def _threshold_value(heatmap: np.ndarray, threshold: Any) -> float:
    """固定値またはadaptive設定から閾値を返す。"""
    if isinstance(threshold, dict):
        mode = str(threshold.get("mode", "adaptive"))
        if mode == "fixed":
            return float(threshold.get("value", threshold.get("min", 0.1)))
        if mode != "adaptive":
            raise ValueError(f"未知のheatmap threshold modeです: {mode}")
        minimum = float(threshold.get("min", 0.1))
        peak_ratio = float(threshold.get("peak_ratio", 0.4))
        return max(minimum, peak_ratio * float(heatmap.max(initial=0.0)))
    return float(threshold)


def _largest_component(values: np.ndarray) -> np.ndarray:
    """正値領域の最大連結成分だけを残す。"""
    binary = (values > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if component_count <= 1:
        return np.zeros_like(values)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return np.where(labels == largest_label, values, 0.0)


def detect_line_from_heatmap(
    heatmap: np.ndarray,
    image_size: int,
    threshold: Any,
    min_confidence: float,
) -> DetectedLine | None:
    """閾値処理と最大連結成分後のmomentから直線を抽出する。"""
    cutoff = _threshold_value(heatmap, threshold)
    filtered = np.where(heatmap >= cutoff, heatmap, 0.0).astype(np.float64)
    filtered = _largest_component(filtered)
    mass = float(filtered.sum())
    if mass <= 1e-8:
        return None
    y_grid, x_grid = np.indices(filtered.shape, dtype=np.float64)
    centroid_x = float((filtered * x_grid).sum() / mass)
    centroid_y = float((filtered * y_grid).sum() / mass)
    delta_x = x_grid - centroid_x
    delta_y = y_grid - centroid_y
    variance_x = float((filtered * delta_x**2).sum() / mass)
    variance_y = float((filtered * delta_y**2).sum() / mass)
    covariance_xy = float((filtered * delta_x * delta_y).sum() / mass)
    difference = variance_x - variance_y
    anisotropy = math.sqrt(difference**2 + 4.0 * covariance_xy**2)
    trace = variance_x + variance_y
    confidence = anisotropy / max(trace, 1e-8)
    if confidence < min_confidence:
        return None
    angle = 0.5 * math.atan2(2.0 * covariance_xy, difference) % math.pi
    normal = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    center = image_size / 2.0
    rho = float(normal @ np.asarray([centroid_x - center, centroid_y - center]))
    return DetectedLine(
        angle_rad=angle,
        rho_px=rho,
        confidence=confidence,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )


def line_extent(points_xy: list[list[float]] | None) -> float:
    """GTポリラインの最遠点間距離を返す。"""
    if points_xy is None or len(points_xy) < 2:
        return 0.0
    points = np.asarray(points_xy, dtype=np.float64)
    distances = np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=-1))
    return float(distances.max())


def detected_line_to_dict(
    detection: DetectedLine | None,
    length_px: float,
    extend_ratio: float,
    image_shape: tuple[int, int],
) -> dict[str, Any] | None:
    """検出直線をline_only互換の描画用dictへ変換する。"""
    if detection is None:
        return None
    height, width = image_shape
    rendered_length = max(0.0, float(length_px)) * float(extend_ratio)
    direction_x = math.cos(detection.angle_rad)
    direction_y = math.sin(detection.angle_rad)
    half_length = rendered_length / 2.0
    endpoint_1 = [
        float(np.clip(detection.centroid_x - half_length * direction_x, 0, width - 1)),
        float(np.clip(detection.centroid_y - half_length * direction_y, 0, height - 1)),
    ]
    endpoint_2 = [
        float(np.clip(detection.centroid_x + half_length * direction_x, 0, width - 1)),
        float(np.clip(detection.centroid_y + half_length * direction_y, 0, height - 1)),
    ]
    return {
        "centroid": [detection.centroid_x, detection.centroid_y],
        "angle_rad": detection.angle_rad,
        "angle_deg": math.degrees(detection.angle_rad) % 180.0,
        "dir": [direction_x, direction_y],
        "endpoints": [endpoint_1, endpoint_2],
        "length": rendered_length,
        "confidence": detection.confidence,
    }


def line_errors(
    prediction: DetectedLine,
    gt_angle_rad: float,
    gt_rho_px: float,
) -> tuple[float, float]:
    """180度周期の角度誤差と符号整合後のrho誤差を返す。"""
    prediction_direction = np.asarray(
        [math.cos(prediction.angle_rad), math.sin(prediction.angle_rad)]
    )
    gt_direction = np.asarray([math.cos(gt_angle_rad), math.sin(gt_angle_rad)])
    direction_dot = float(np.clip(abs(prediction_direction @ gt_direction), 0.0, 1.0))
    angle_error = math.degrees(math.acos(direction_dot))

    prediction_normal = np.asarray(
        [-math.sin(prediction.angle_rad), math.cos(prediction.angle_rad)]
    )
    gt_normal = np.asarray([-math.sin(gt_angle_rad), math.cos(gt_angle_rad)])
    aligned_rho = prediction.rho_px
    if float(prediction_normal @ gt_normal) < 0.0:
        aligned_rho = -aligned_rho
    return angle_error, abs(aligned_rho - gt_rho_px)


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """soft Dice scoreを返す。"""
    intersection = float((prediction * target).sum())
    denominator = float((prediction**2).sum() + (target**2).sum())
    return (2.0 * intersection + epsilon) / (denominator + epsilon)
