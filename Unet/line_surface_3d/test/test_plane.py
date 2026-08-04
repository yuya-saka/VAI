"""厳密平面の幾何テスト。

過去に壊れていた箇所の回帰確認を含む:

- 未アノテーションスライスによる傾きの減衰（旧 `fit_ribbon` で16倍）
- 線方向の重心ドリフトが傾きへ混入すること（重心x,yの独立回帰による）
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from line_surface_3d.utils.plane import (
    build_surface_plane,
    canonical_normal,
    centered_positions,
    extract_gt_line_params,
    fit_plane,
    gt_plane_from_slices,
)

SLAB_SIZE = 15
IMAGE_SIZE = 64
SIGMA = 4.0


def _segment_heatmap(
    center_y: float,
    x_start: float,
    x_end: float,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """水平線分のGaussianリッジを描く。"""
    rows = torch.arange(size).float()[:, None]
    columns = torch.arange(size).float()[None, :]
    inside = (columns >= x_start) & (columns <= x_end)
    distance = torch.where(
        inside,
        (rows - center_y).abs(),
        torch.hypot(
            rows - center_y,
            torch.minimum((columns - x_start).abs(), (columns - x_end).abs()),
        ),
    )
    return torch.exp(-(distance**2) / (2.0 * SIGMA**2))


def _slab(rows: list[float | None], spans: list[tuple[float, float]] | None = None):
    """`rows` がNoneのスライスは全ゼロ（未アノテーション相当）にする。"""
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    for index, row in enumerate(rows):
        if row is None:
            continue
        span = spans[index] if spans is not None else (8.0, IMAGE_SIZE - 8.0)
        heatmaps[0, index, 0] = _segment_heatmap(row, span[0], span[1])
    return heatmaps


def test_fit_plane_recovers_known_tilt() -> None:
    """既知の傾きを持つ平面を正しく復元する。"""
    slope_image = 0.8
    rows = [32.0 + slope_image * (index - 7) for index in range(SLAB_SIZE)]
    fitted = fit_plane(_slab(rows))
    # 画像行が増える方向は数学座標のyが減る方向
    assert fitted.slope[0, 0].item() == pytest.approx(-slope_image, abs=0.02)
    assert fitted.normal[0, 0, 1].item() == pytest.approx(1.0, abs=1e-3)
    assert fitted.rho_0[0, 0].item() == pytest.approx(0.0, abs=0.05)


def test_unannotated_slices_do_not_attenuate_tilt() -> None:
    """全ゼロスライスがあっても傾きが減衰しない。

    旧 `fit_ribbon` は `valid` を無視し、ここで16倍減衰していた。
    """
    slope_image = 1.0
    rows: list[float | None] = [None] * SLAB_SIZE
    for index in range(4, 10):
        rows[index] = 32.0 + slope_image * (index - 6.5)
    fitted = fit_plane(_slab(rows))
    assert fitted.slope[0, 0].item() == pytest.approx(-slope_image, rel=0.05)


def test_along_line_drift_does_not_affect_tilt() -> None:
    """線に沿った重心の移動は傾きへ混入しない。

    教師polylineの描画長はスライスごとに違う。重心x,yを独立に回帰すると
    この雑音が傾きへ入るため、共有法線へ射影して落とす必要がある。
    """
    rows = [32.0] * SLAB_SIZE
    spans = [(8.0 + 2.0 * index, 40.0 + 2.0 * index) for index in range(SLAB_SIZE)]
    fitted = fit_plane(_slab(rows, spans))
    assert abs(fitted.slope[0, 0].item()) < 0.02


def test_canonical_normal_is_upper_half_plane() -> None:
    """復元した法線は常に `n_y >= 0` になる。"""
    angles = torch.tensor([0.2, 1.4, 2.9, -0.5, -2.2])
    doubled = torch.stack([torch.cos(2 * angles), torch.sin(2 * angles)], dim=-1)
    normal = canonical_normal(doubled)
    assert bool((normal[:, 1] >= -1e-6).all())
    assert torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones(5))


def test_build_surface_plane_recovers_clear_tilt() -> None:
    """明確な傾きを持つ手動線からは符号付きの傾きが得られる。"""
    positions = np.arange(40.0, 49.0)
    angles = np.full(positions.shape, math.pi / 2.0)
    offsets = 5.0 + 0.5 * (positions - positions.mean())
    plane = build_surface_plane(positions, angles, offsets)
    assert plane.reliable
    assert plane.slope_px_per_slice == pytest.approx(0.5, rel=1e-6)
    assert plane.rho_at_reference_px == pytest.approx(5.0, abs=1e-6)


def test_build_surface_plane_falls_back_to_vertical() -> None:
    """傾きが雑音に埋もれる面は垂直平面 (`k=0`) として返す。"""
    rng = np.random.default_rng(0)
    positions = np.arange(40.0, 49.0)
    angles = np.full(positions.shape, math.pi / 2.0)
    offsets = 5.0 + rng.normal(0.0, 0.3, positions.shape)
    plane = build_surface_plane(positions, angles, offsets)
    assert not plane.reliable
    assert plane.slope_px_per_slice == 0.0


def test_build_surface_plane_needs_enough_slices() -> None:
    """スライス数が足りない面は垂直平面として返す。"""
    positions = np.array([40.0, 41.0, 42.0])
    angles = np.full(positions.shape, math.pi / 2.0)
    offsets = 5.0 + 2.0 * (positions - positions.mean())
    plane = build_surface_plane(positions, angles, offsets)
    assert not plane.reliable
    assert plane.slope_px_per_slice == 0.0


def test_gt_plane_from_slices_matches_input() -> None:
    """スライス教師と傾きからGT平面を組み直せる。"""
    positions = centered_positions(SLAB_SIZE, torch.device("cpu"), torch.float32)
    slope = 0.4
    diagonal = math.sqrt(2.0) * IMAGE_SIZE
    line_params = torch.full((1, SLAB_SIZE, 4, 2), float("nan"))
    label_mask = torch.zeros(1, SLAB_SIZE, 4, dtype=torch.bool)
    for index in range(4, 11):
        line_params[0, index, 0, 0] = math.pi / 2.0
        line_params[0, index, 0, 1] = (3.0 + slope * positions[index]) / diagonal
        label_mask[0, index, 0] = True

    plane = gt_plane_from_slices(
        line_params,
        label_mask,
        torch.tensor([[slope, 0.0, 0.0, 0.0]]),
        positions,
        IMAGE_SIZE,
    )
    assert plane.rho_0[0, 0].item() == pytest.approx(3.0, abs=1e-3)
    assert plane.normal[0, 0, 1].item() == pytest.approx(1.0, abs=1e-5)
    assert plane.slope[0, 0].item() == pytest.approx(slope)


def test_extract_gt_line_params_normalizes_to_upper_half_plane() -> None:
    """GTポリラインの法線は上半平面へ正規化される。"""
    phi, rho = extract_gt_line_params([[10.0, 20.0], [50.0, 20.0]], IMAGE_SIZE)
    assert math.sin(phi) >= 0.0
    # 画像行20は中心32より上、数学座標では +12
    assert rho * math.sqrt(2.0) * IMAGE_SIZE == pytest.approx(12.0, abs=1e-6)
