"""平面推論の出力テスト。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from line_surface_3d.src.inference import line_endpoints_in_image, predict_loader

SLAB_SIZE = 15
IMAGE_SIZE = 64
SIGMA = 4.0


class FixedHeatmapModel(nn.Module):
    """固定ヒートマップをlogitとして返すテストモデル。"""

    def __init__(self, heatmaps: torch.Tensor) -> None:
        super().__init__()
        clipped = heatmaps.clamp(1e-6, 1.0 - 1e-6)
        self.register_buffer("fixed_logits", torch.logit(clipped).flatten(1, 2))

    def forward(
        self,
        images: torch.Tensor,
        vertebra_indices: torch.Tensor,
    ) -> torch.Tensor:
        """固定logitを返す。"""
        del images, vertebra_indices
        return self.fixed_logits


def test_line_endpoints_span_the_image() -> None:
    """水平線は画像の左右端まで伸びる。"""
    endpoints = line_endpoints_in_image(np.array([0.0, 1.0]), 0.0, IMAGE_SIZE)
    assert len(endpoints) == 2
    for point in endpoints:
        assert point[1] == pytest.approx(IMAGE_SIZE / 2.0, abs=1.0)
    assert abs(endpoints[0][0] - endpoints[1][0]) > IMAGE_SIZE * 0.9


def test_line_endpoints_follow_offset() -> None:
    """rhoが正なら線は画像の上側へ移動する。"""
    endpoints = line_endpoints_in_image(np.array([0.0, 1.0]), 10.0, IMAGE_SIZE)
    for point in endpoints:
        assert point[1] == pytest.approx(IMAGE_SIZE / 2.0 - 10.0, abs=1.0)


def test_vertical_line_endpoints() -> None:
    """垂直線でも端点が得られる。"""
    endpoints = line_endpoints_in_image(np.array([1.0, 0.0]), 5.0, IMAGE_SIZE)
    assert len(endpoints) == 2
    for point in endpoints:
        assert point[0] == pytest.approx(IMAGE_SIZE / 2.0 + 5.0, abs=1.0)


def _batch(slope_image: float, window_start: int) -> dict[str, Any]:
    """既知の平面から1窓分の推論バッチを作る。"""
    rows = torch.arange(IMAGE_SIZE).float()[:, None]
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    slice_indices = torch.arange(window_start, window_start + SLAB_SIZE)
    center_reference = float(window_start + (SLAB_SIZE - 1) / 2.0)
    for index in range(SLAB_SIZE):
        center_y = 32.0 + slope_image * (float(slice_indices[index]) - center_reference)
        ridge = torch.exp(-((rows - center_y) ** 2) / (2.0 * SIGMA**2))
        heatmaps[0, index, 0] = ridge.expand(IMAGE_SIZE, IMAGE_SIZE)
    return {
        "image": torch.zeros(1, 2 * SLAB_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        "heatmaps": heatmaps,
        "slice_indices": slice_indices[None, :],
        "sample": ["sample1"],
        "vertebra": ["C4"],
    }


def test_predict_loader_writes_one_plane_per_surface(tmp_path: Path) -> None:
    """重なり窓から椎体ごとに平面1枚を出力する。"""
    batches = [_batch(0.5, window_start=start) for start in (20, 21, 22)]
    model = FixedHeatmapModel(batches[0]["heatmaps"])
    config = {"data": {"slab_size": SLAB_SIZE, "image_size": IMAGE_SIZE}}

    summary = predict_loader(model, batches, torch.device("cpu"), config, tmp_path)
    assert summary["vertebra_count"] == 1

    planes = json.loads((tmp_path / "sample1" / "C4" / "planes.json").read_text())
    assert set(planes) == {"line_1"}
    # 数学座標では画像行の増加方向と符号が逆
    assert planes["line_1"]["slope_px_per_slice"] == pytest.approx(-0.5, abs=0.05)
    assert planes["line_1"]["window_count"] == 3

    lines = json.loads((tmp_path / "sample1" / "C4" / "lines.json").read_text())
    # 全窓が覆うglobal zすべてに交線が出る
    assert len(lines) == 22 + SLAB_SIZE - 20
    for per_slice in lines.values():
        assert len(per_slice["line_1"]) == 2


def test_predicted_lines_follow_the_tilt(tmp_path: Path) -> None:
    """出力した交線がz方向に平面の傾きどおり移動する。"""
    batches = [_batch(0.5, window_start=20)]
    model = FixedHeatmapModel(batches[0]["heatmaps"])
    config = {"data": {"slab_size": SLAB_SIZE, "image_size": IMAGE_SIZE}}
    predict_loader(model, batches, torch.device("cpu"), config, tmp_path)

    lines = json.loads((tmp_path / "sample1" / "C4" / "lines.json").read_text())
    keys = sorted(lines, key=int)
    first_y = lines[keys[0]]["line_1"][0][1]
    last_y = lines[keys[-1]]["line_1"][0][1]
    span = int(keys[-1]) - int(keys[0])
    assert (last_y - first_y) / span == pytest.approx(0.5, abs=0.05)
    assert not math.isnan(first_y)
