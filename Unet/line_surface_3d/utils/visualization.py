"""全高リフォーマットと線overlayの保存。"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

LINE_COLORS = ("red", "orange", "cyan", "lime")


def save_slice_overlay(
    ct_image: np.ndarray,
    lines: dict[str, list[list[float]]],
    output_path: Path,
) -> None:
    """単一CT画像へ4線を重ねて保存する。"""
    image = np.clip(ct_image * 255.0, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgr_colors = ((0, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 0))
    for line_index in range(4):
        points = lines.get(f"line_{line_index + 1}")
        if points is None or len(points) < 2:
            continue
        first = tuple(np.rint(points[0]).astype(int))
        second = tuple(np.rint(points[1]).astype(int))
        cv2.line(color, first, second, bgr_colors[line_index], 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), color)


def _horizontal_intersection(
    endpoints: list[list[float]],
    target_y: float,
) -> float | None:
    """線分を延長した直線と水平断面のx交点を返す。"""
    (first_x, first_y), (second_x, second_y) = endpoints
    denominator = second_y - first_y
    if abs(denominator) < 1e-8:
        return None
    ratio = (target_y - first_y) / denominator
    return float(first_x + ratio * (second_x - first_x))


def _vertical_intersection(
    endpoints: list[list[float]],
    target_x: float,
) -> float | None:
    """線分を延長した直線と垂直断面のy交点を返す。"""
    (first_x, first_y), (second_x, second_y) = endpoints
    denominator = second_x - first_x
    if abs(denominator) < 1e-8:
        return None
    ratio = (target_x - first_x) / denominator
    return float(first_y + ratio * (second_y - first_y))


def save_reformat_visualization(
    ct_stack: np.ndarray,
    slice_indices: list[int],
    lines_by_slice: dict[str, dict[str, list[list[float]]]],
    output_path: Path,
) -> None:
    """冠状断・矢状断画像上へ境界面の交線を描く。"""
    if ct_stack.ndim != 3:
        raise ValueError(f"ct_stackは `(Z,H,W)` が必要です: {ct_stack.shape}")
    _, height, width = ct_stack.shape
    coronal_y = height // 2
    sagittal_x = width // 2
    coronal = ct_stack[:, coronal_y, :]
    sagittal = ct_stack[:, :, sagittal_x]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(coronal, cmap="gray", aspect="auto")
    axes[0].set_title("冠状断リフォーマット")
    axes[1].imshow(sagittal, cmap="gray", aspect="auto")
    axes[1].set_title("矢状断リフォーマット")
    for line_index, color in enumerate(LINE_COLORS):
        line_name = f"line_{line_index + 1}"
        coronal_x: list[float] = []
        sagittal_y: list[float] = []
        rows_coronal: list[int] = []
        rows_sagittal: list[int] = []
        for row, slice_index in enumerate(slice_indices):
            lines = lines_by_slice.get(str(slice_index), {})
            endpoints = lines.get(line_name)
            if endpoints is None:
                continue
            x_intersection = _horizontal_intersection(endpoints, coronal_y)
            if x_intersection is not None:
                coronal_x.append(x_intersection)
                rows_coronal.append(row)
            y_intersection = _vertical_intersection(endpoints, sagittal_x)
            if y_intersection is not None:
                sagittal_y.append(y_intersection)
                rows_sagittal.append(row)
        axes[0].plot(coronal_x, rows_coronal, color=color, linewidth=1)
        axes[1].plot(sagittal_y, rows_sagittal, color=color, linewidth=1)
    for axis in axes:
        axis.set_ylabel("zスライス")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
