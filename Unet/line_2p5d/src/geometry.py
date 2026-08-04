"""平面を仮定しない、画像内直線の局所幾何ユーティリティ。"""

from __future__ import annotations

import math

import numpy as np


def polyline_to_line_params(
    points_xy: list[list[float]] | None,
    image_size: int,
) -> tuple[float, float]:
    """ポリラインを画像中心基準の直線角度と法線距離へ変換する。"""
    if points_xy is None or len(points_xy) < 2:
        return float("nan"), float("nan")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return float("nan"), float("nan")
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(1, len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
    normal = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    image_center = image_size / 2.0
    centered_centroid = centroid - image_center
    rho = float(normal @ centered_centroid)
    return float(angle), rho


def preprocess_polyline(
    points_xy: list[list[float]] | None,
    duplicate_threshold: float = 1.0,
) -> list[list[float]] | None:
    """近接した中間点だけを除去し、線形状を維持する。"""
    if points_xy is None or len(points_xy) < 2:
        return points_xy
    points = np.asarray(points_xy, dtype=np.float64)
    keep_indices = [0]
    for index in range(1, len(points) - 1):
        if (
            np.linalg.norm(points[index] - points[keep_indices[-1]])
            >= duplicate_threshold
        ):
            keep_indices.append(index)
    keep_indices.append(len(points) - 1)
    filtered = points[keep_indices]
    return filtered.tolist() if len(filtered) >= 2 else points_xy
