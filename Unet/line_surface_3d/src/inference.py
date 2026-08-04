"""全高sliding-window推論から、椎体×面ごとに平面を1枚出す。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.plane import canonical_normal, centered_positions, fit_plane
from .evaluation import SurfaceAccumulator, vertebra_indices
from .experiment import save_json
from .model import reshape_slab_heatmaps

LINE_NAMES = ("line_1", "line_2", "line_3", "line_4")


def line_endpoints_in_image(
    normal: np.ndarray,
    rho: float,
    image_size: int,
) -> list[list[float]]:
    """`n . x = rho` の直線と画像枠の交点を返す。

    平面は無限に伸びるので、描画長を推定せず画像境界で切る。
    座標は数学座標（原点=画像中心、y上向き）から画像座標へ戻す。
    """
    center = image_size / 2.0
    half = center
    normal_x, normal_y = float(normal[0]), float(normal[1])
    points: list[tuple[float, float]] = []
    if abs(normal_y) > 1e-6:
        for x_value in (-half, half):
            y_value = (rho - normal_x * x_value) / normal_y
            if -half - 1e-6 <= y_value <= half + 1e-6:
                points.append((x_value, y_value))
    if abs(normal_x) > 1e-6:
        for y_value in (-half, half):
            x_value = (rho - normal_y * y_value) / normal_x
            if -half - 1e-6 <= x_value <= half + 1e-6:
                points.append((x_value, y_value))
    if len(points) < 2:
        return []
    first = points[0]
    farthest = max(
        points[1:], key=lambda p: math.hypot(p[0] - first[0], p[1] - first[1])
    )
    return [
        [
            float(np.clip(point[0] + center, 0, image_size - 1)),
            float(np.clip(-point[1] + center, 0, image_size - 1)),
        ]
        for point in (first, farthest)
    ]


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """全窓を集約し、椎体ごとに4平面とその全高交線を保存する。"""
    model.eval()
    data_config = config["data"]
    slab_size = int(data_config["slab_size"])
    image_size = int(data_config["image_size"])
    positions = centered_positions(slab_size, device, torch.float32)

    accumulators: dict[tuple[str, str, int], SurfaceAccumulator] = {}
    slice_range: dict[tuple[str, str], tuple[int, int]] = {}
    window_count = 0

    for batch in loader:
        images = batch["image"].to(device).float()
        logits = model(images, vertebra_indices(batch, device))
        heatmaps = torch.sigmoid(reshape_slab_heatmaps(logits, slab_size)).float()
        plane = fit_plane(heatmaps, positions)

        doubled = plane.doubled_normal.cpu().numpy()
        rho_0 = plane.rho_0.cpu().numpy()
        slope = plane.slope.cpu().numpy()
        weights = plane.weight_sum.cpu().numpy()
        valid = plane.valid.cpu().numpy()
        slice_indices = batch["slice_indices"].cpu().numpy()

        for batch_index in range(heatmaps.shape[0]):
            sample = str(batch["sample"][batch_index])
            vertebra = str(batch["vertebra"][batch_index])
            window = slice_indices[batch_index]
            window_center = float(np.mean(window))
            low, high = slice_range.get(
                (sample, vertebra), (int(window.min()), int(window.max()))
            )
            slice_range[(sample, vertebra)] = (
                min(low, int(window.min())),
                max(high, int(window.max())),
            )
            for line_index in range(4):
                if not valid[batch_index, line_index]:
                    continue
                # 各窓のrho_0は窓中心の値。global z=0 基準へ揃えてから平均する
                accumulators.setdefault(
                    (sample, vertebra, line_index), SurfaceAccumulator()
                ).add(
                    doubled=doubled[batch_index, line_index],
                    rho_at_reference=float(rho_0[batch_index, line_index])
                    - float(slope[batch_index, line_index]) * window_center,
                    slope=float(slope[batch_index, line_index]),
                    weight=float(weights[batch_index, line_index]),
                )
            window_count += 1

    return _write_outputs(
        accumulators, slice_range, image_size, window_count, output_root
    )


def _write_outputs(
    accumulators: dict[tuple[str, str, int], SurfaceAccumulator],
    slice_range: dict[tuple[str, str], tuple[int, int]],
    image_size: int,
    window_count: int,
    output_root: Path,
) -> dict[str, Any]:
    """椎体ごとにplane.jsonとlines.jsonを書き出す。"""
    summaries: list[dict[str, Any]] = []
    for (sample, vertebra), (low, high) in sorted(slice_range.items()):
        planes: dict[str, dict[str, Any]] = {}
        for line_index, line_name in enumerate(LINE_NAMES):
            aggregate = accumulators.get((sample, vertebra, line_index))
            finalized = aggregate.finalize() if aggregate is not None else None
            if finalized is None:
                continue
            normal, rho_at_zero, slope = finalized
            planes[line_name] = {
                "normal": [float(normal[0]), float(normal[1])],
                "angle_deg": float(math.degrees(math.atan2(normal[1], normal[0]))),
                "rho_at_z0_px": float(rho_at_zero),
                "slope_px_per_slice": float(slope),
                "window_count": aggregate.windows if aggregate else 0,
            }
        if not planes:
            continue

        lines_output: dict[str, dict[str, list[list[float]]]] = {}
        for slice_index in range(low, high + 1):
            per_slice: dict[str, list[list[float]]] = {}
            for line_name, parameters in planes.items():
                rho = (
                    parameters["rho_at_z0_px"]
                    + parameters["slope_px_per_slice"] * slice_index
                )
                endpoints = line_endpoints_in_image(
                    np.asarray(parameters["normal"]), rho, image_size
                )
                if endpoints:
                    per_slice[line_name] = endpoints
            if per_slice:
                lines_output[str(slice_index)] = per_slice

        vertebra_dir = output_root / sample / vertebra
        save_json(vertebra_dir / "planes.json", planes)
        save_json(vertebra_dir / "lines.json", lines_output)
        summaries.append(
            {
                "sample": sample,
                "vertebra": vertebra,
                "surfaces": len(planes),
                "slice_count": len(lines_output),
            }
        )

    summary = {
        "window_count": window_count,
        "vertebra_count": len(summaries),
        "vertebrae": summaries,
    }
    save_json(output_root / "inference_summary.json", summary)
    return summary


__all__ = ["canonical_normal", "line_endpoints_in_image", "predict_loader"]
