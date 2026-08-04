"""厳密平面の幾何。GT平面の構築と、heatmapからの微分可能な平面射影。

1つの境界面は3Dの平面1枚として表す。パラメータは3つだけ:

- `phi`   : 面内法線の角度。z方向に**一定**。
- `rho_0` : 基準スライスでの符号付きオフセット [px]。
- `k`     : zあたりのオフセット変化 [px/slice]。これが平面の傾き。

面内の線は `n(phi) . x = rho_0 + k * (z - z_ref)` である。

重要な設計:

- 法線はスライスごとにcanonicalizeしない。符号反転が起きると `k` が壊れる。
  スラブ全体でdoubled-angleを平均し、**一度だけ**上半平面へ正規化する。
- `rho` は必ずその共有法線から作る。重心をx,y独立に回帰すると、
  線に沿った重心移動（教師の描画長の違いによる雑音）が傾きへ混入する。
- `valid` なスライスだけを回帰に使う。全ゼロheatmapを等重みで含めると傾きが減衰する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

MIN_MASS = 1e-6
MIN_WEIGHT_SUM = 1e-6
RIDGE_EPSILON = 1e-6
RIDGE_SCALE = 0.5
MIN_UNSHRUNK_SLICES = 3.0
MIN_EFFECTIVE_SLICES = 1.5
MIN_ORIENTATION_RESULTANT = 0.2

# GT平面のreliable判定
MIN_GT_SLICES = 5
MIN_GT_SPAN_SLICES = 4
MAX_GT_POINT_RESIDUAL_PX = 2.0
MAX_GT_ANGLE_RESIDUAL_DEG = 5.0
MIN_GT_MOVEMENT_PX = 1.0
STRONG_GT_MOVEMENT_PX = 2.0
MIN_GT_LOO_SIGN_AGREEMENT = 0.8


@dataclass(frozen=True)
class PlaneFit:
    """スラブ1窓・1面あたりの平面パラメータ。

    形状はいずれも `(B, 4)`。`rho_0` はスラブ中心での値。
    """

    doubled_normal: torch.Tensor
    normal: torch.Tensor
    rho_0: torch.Tensor
    slope: torch.Tensor
    weight_sum: torch.Tensor
    valid: torch.Tensor

    def tilt_vector(self) -> torch.Tensor:
        """符号表現の反転に不変な傾きベクトル `v = k * n` を返す。"""
        return self.slope.unsqueeze(-1) * self.normal

    def rho_at(self, positions: torch.Tensor) -> torch.Tensor:
        """スラブ中心基準のz位置での `rho` を返す。形状 `(B, N, 4)`。"""
        return self.rho_0.unsqueeze(1) + self.slope.unsqueeze(1) * positions.reshape(
            1, -1, 1
        )


@dataclass(frozen=True)
class SurfacePlane:
    """手動アノテーション全体から作った1面のGT平面。

    `angle_rad` と `rho_at_reference_px` は**増補前**の座標系での値。
    増補ありの学習では使わず、増補後の教師線から組み直すこと。
    """

    slope_px_per_slice: float
    reliable: bool
    reference_slice: float
    movement_px: float
    slice_count: int
    angle_rad: float = 0.0
    rho_at_reference_px: float = 0.0


def centered_positions(
    slab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """スラブ中心を0とするz座標を返す。"""
    return torch.arange(slab_size, device=device, dtype=dtype) - (slab_size - 1) / 2.0


def doubled_angle(
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """法線を180度周期のdoubled-angleへ変換する。"""
    cosine = normal_x.square() - normal_y.square()
    sine = 2.0 * normal_x * normal_y
    norm = torch.hypot(cosine, sine).clamp(min=1e-8)
    return cosine / norm, sine / norm


def canonical_normal(doubled: torch.Tensor) -> torch.Tensor:
    """doubled-angleから上半平面 (`n_y >= 0`) の単位法線を復元する。"""
    angle = 0.5 * torch.atan2(doubled[..., 1], doubled[..., 0])
    normal_x = torch.cos(angle)
    normal_y = torch.sin(angle)
    flip = (normal_y < 0) | ((normal_y == 0) & (normal_x < 0))
    sign = torch.where(flip, -torch.ones_like(normal_x), torch.ones_like(normal_x))
    return torch.stack([normal_x * sign, normal_y * sign], dim=-1)


@dataclass(frozen=True)
class SliceMoments:
    """スラブ内の各heatmapから抽出した重心・向き・信頼度。"""

    centroid: torch.Tensor
    doubled: torch.Tensor
    confidence: torch.Tensor
    major_variance: torch.Tensor
    valid: torch.Tensor


def compute_slice_moments(
    heatmaps: torch.Tensor,
    min_mass: float = MIN_MASS,
) -> SliceMoments:
    """`(B,N,4,H,W)` heatmapから重心と主軸向きを取り出す。"""
    if heatmaps.ndim != 5:
        raise ValueError(f"heatmapsは5次元が必要です: {heatmaps.shape}")
    working = heatmaps.float()
    batch_size, slab_size, line_count, height, width = working.shape
    flat_count = batch_size * slab_size * line_count
    values = working.reshape(flat_count, height * width)
    device = working.device
    dtype = working.dtype

    y_axis = -(torch.arange(height, device=device, dtype=dtype) - height / 2.0)
    x_axis = torch.arange(width, device=device, dtype=dtype) - width / 2.0
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    mass = values.sum(dim=-1)
    safe_mass = mass.clamp(min=min_mass)
    centroid_x = (values * x_flat).sum(dim=-1) / safe_mass
    centroid_y = (values * y_flat).sum(dim=-1) / safe_mass

    delta_x = x_flat[None, :] - centroid_x[:, None]
    delta_y = y_flat[None, :] - centroid_y[:, None]
    covariance_xx = (values * delta_x.square()).sum(dim=-1) / safe_mass + 1e-6
    covariance_yy = (values * delta_y.square()).sum(dim=-1) / safe_mass + 1e-6
    covariance_xy = (values * delta_x * delta_y).sum(dim=-1) / safe_mass

    discriminant = torch.hypot(covariance_xx - covariance_yy, 2.0 * covariance_xy)
    trace = covariance_xx + covariance_yy
    major_variance = (trace + discriminant) / 2.0
    valid = (mass >= min_mass) & (major_variance > 1e-8)
    confidence = torch.where(
        valid,
        discriminant / (trace + 1e-8),
        torch.zeros_like(discriminant),
    )

    degenerate = discriminant < 1e-8
    numerator = torch.where(
        degenerate, torch.zeros_like(covariance_xy), 2.0 * covariance_xy
    )
    denominator = torch.where(
        degenerate,
        torch.ones_like(covariance_xx),
        covariance_xx - covariance_yy,
    )
    direction_angle = 0.5 * torch.atan2(numerator, denominator)
    # 主軸が線方向、その直交が法線
    normal_x = -torch.sin(direction_angle)
    normal_y = torch.cos(direction_angle)
    doubled_cosine, doubled_sine = doubled_angle(normal_x, normal_y)

    shape = (batch_size, slab_size, line_count)
    return SliceMoments(
        centroid=torch.stack(
            [centroid_x.reshape(shape), centroid_y.reshape(shape)], dim=-1
        ),
        doubled=torch.stack(
            [doubled_cosine.reshape(shape), doubled_sine.reshape(shape)], dim=-1
        ),
        confidence=confidence.reshape(shape),
        major_variance=major_variance.reshape(shape),
        valid=valid.reshape(shape),
    )


def _weighted_line_fit(
    values: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`values` を z方向へridge付き重み付き1次回帰する。

    引数:
        values: `(B, N, 4)`
        weights: `(B, N, 4)` 非負。勾配は流さない想定。
        positions: `(N,)` スラブ中心基準

    戻り値:
        (intercept, slope, effective_slice_count) いずれも `(B, 4)`
    """
    half_span = float(positions.abs().max().clamp(min=1.0))
    unit = (positions / half_span).reshape(1, -1, 1)

    weight_sum = weights.sum(dim=1).clamp(min=MIN_WEIGHT_SUM)
    normalized = weights / weight_sum.unsqueeze(1)
    mean_unit = (normalized * unit).sum(dim=1)
    mean_value = (normalized * values).sum(dim=1)
    delta_unit = unit - mean_unit.unsqueeze(1)
    sum_uu = (normalized * delta_unit.square()).sum(dim=1)
    sum_uv = (normalized * delta_unit * (values - mean_value.unsqueeze(1))).sum(dim=1)

    # ridgeは有効スライス数が足りないときだけ効かせる。定数を足すと
    # 教師帯が狭い面で傾きが一律に減衰してしまう
    effective = 1.0 / normalized.square().sum(dim=1).clamp(min=1e-8)
    reference_uu = float((unit.reshape(-1) ** 2).mean())
    ridge = RIDGE_EPSILON + RIDGE_SCALE * reference_uu * torch.clamp(
        MIN_UNSHRUNK_SLICES / effective.clamp(min=1e-3) - 1.0, min=0.0
    )
    unit_slope = sum_uv / (sum_uu + ridge)
    slope = unit_slope / half_span
    intercept = mean_value - unit_slope * mean_unit
    return intercept, slope, effective


def fit_plane(
    heatmaps: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> PlaneFit:
    """heatmapスラブから1面につき1枚の厳密平面を微分可能に取り出す。"""
    moments = compute_slice_moments(heatmaps)
    slab_size = heatmaps.shape[1]
    if positions is None:
        positions = centered_positions(slab_size, heatmaps.device, torch.float32)

    # confidenceに勾配を通すと、heatmapを平坦化して重みを下げる逃げ道ができる
    weights = (moments.confidence * moments.valid).detach()

    weight_sum = weights.sum(dim=1)
    safe_sum = weight_sum.clamp(min=MIN_WEIGHT_SUM).unsqueeze(-1)
    doubled_mean = (moments.doubled * weights.unsqueeze(-1)).sum(dim=1) / safe_sum
    resultant = torch.linalg.vector_norm(doubled_mean, dim=-1)
    doubled_normal = doubled_mean / resultant.clamp(min=1e-4).unsqueeze(-1)
    normal = canonical_normal(doubled_normal)

    # 共有法線から符号付きoffsetを作る。線方向の重心移動はここで落ちる
    rho = (moments.centroid * normal.unsqueeze(1)).sum(dim=-1)
    rho_0, slope, effective = _weighted_line_fit(rho, weights, positions)

    tilt_valid = (
        (effective >= MIN_EFFECTIVE_SLICES)
        & (resultant >= MIN_ORIENTATION_RESULTANT)
        & (weight_sum > MIN_WEIGHT_SUM)
    )
    slope = torch.where(tilt_valid, slope, torch.zeros_like(slope))
    return PlaneFit(
        doubled_normal=doubled_normal,
        normal=normal,
        rho_0=rho_0,
        slope=slope,
        weight_sum=weight_sum,
        valid=tilt_valid,
    )


def gt_plane_from_slices(
    line_params: torch.Tensor,
    label_mask: torch.Tensor,
    slope: torch.Tensor,
    positions: torch.Tensor,
    image_size: int,
) -> PlaneFit:
    """窓内の教師線と事前計算した傾きからGT平面を組み立てる。

    角度と `rho_0` は増補後の教師線から作るので、回転・拡大の影響を自動的に受ける。
    `slope` は面内回転に不変なため、事前計算値をそのまま使ってよい。

    引数:
        line_params: `(B,N,4,2)` の `(phi, rho_normalized)`。未教師はNaN。
        label_mask: `(B,N,4)`
        slope: `(B,4)` [px/slice]
        positions: `(N,)` スラブ中心基準
        image_size: 正規化rhoをpxへ戻すための画像サイズ
    """
    weights = label_mask.to(dtype=slope.dtype)
    zero = torch.zeros_like(weights)
    phi = torch.where(label_mask, line_params[..., 0], zero)
    diagonal = math.sqrt(2.0) * image_size
    rho_px = torch.where(label_mask, line_params[..., 1], zero) * diagonal

    slice_doubled = torch.stack(
        [torch.cos(2.0 * phi), torch.sin(2.0 * phi)], dim=-1
    ) * weights.unsqueeze(-1)
    weight_sum = weights.sum(dim=1)
    safe_sum = weight_sum.clamp(min=MIN_WEIGHT_SUM).unsqueeze(-1)
    doubled_mean = slice_doubled.sum(dim=1) / safe_sum
    resultant = torch.linalg.vector_norm(doubled_mean, dim=-1)
    doubled_normal = doubled_mean / resultant.clamp(min=1e-4).unsqueeze(-1)
    normal = canonical_normal(doubled_normal)

    # 各スライスの法線符号を共有法線へ揃えてからoffsetを比較可能にする
    slice_normal = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    alignment = torch.sign((slice_normal * normal.unsqueeze(1)).sum(dim=-1))
    alignment = torch.where(alignment == 0, torch.ones_like(alignment), alignment)
    aligned_rho = rho_px * alignment

    detrended = aligned_rho - slope.unsqueeze(1) * positions.reshape(1, -1, 1)
    rho_0 = (detrended * weights).sum(dim=1) / weight_sum.clamp(min=MIN_WEIGHT_SUM)
    return PlaneFit(
        doubled_normal=doubled_normal,
        normal=normal,
        rho_0=rho_0,
        slope=slope,
        weight_sum=weight_sum,
        valid=weight_sum > 0,
    )


def aligned_rho_targets(
    line_params: torch.Tensor,
    label_mask: torch.Tensor,
    normal: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """教師線の符号付きoffsetを、指定した共有法線の符号へ揃えて返す。"""
    zero = torch.zeros_like(label_mask, dtype=normal.dtype)
    phi = torch.where(label_mask, line_params[..., 0], zero)
    diagonal = math.sqrt(2.0) * image_size
    rho_px = torch.where(label_mask, line_params[..., 1], zero) * diagonal
    slice_normal = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    alignment = torch.sign((slice_normal * normal.unsqueeze(1)).sum(dim=-1))
    alignment = torch.where(alignment == 0, torch.ones_like(alignment), alignment)
    return rho_px * alignment


def extract_gt_line_params(
    polyline_points: list[list[float]] | None,
    image_size: int,
) -> tuple[float, float]:
    """GTポリラインからPCAで `(phi, rho_normalized)` を抽出する。

    法線は上半平面 (`n_y >= 0`) へ正規化する。`rho` は画像対角長で割った値。
    """
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
    return float(phi), float(rho / (math.sqrt(2.0) * image_size))


def _fit_slope(positions: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """通常最小二乗で傾きと切片を返す。"""
    centered = positions - positions.mean()
    denominator = float((centered**2).sum())
    if denominator < 1e-9:
        return 0.0, float(offsets.mean())
    slope = float((centered * (offsets - offsets.mean())).sum() / denominator)
    return slope, float(offsets.mean() - slope * positions.mean())


def build_surface_plane(
    slice_positions: np.ndarray,
    angles_rad: np.ndarray,
    offsets_px: np.ndarray,
) -> SurfacePlane:
    """1面の手動アノテーション全体からGT平面を作る。

    傾きが確からしいと言えない面は垂直平面 (`k = 0`) として返す。

    引数:
        slice_positions: `(N,)` スライスindex
        angles_rad: `(N,)` 各スライスの法線角度（上半平面正規化済み）
        offsets_px: `(N,)` 各スライスの符号付きoffset [px]
    """
    count = int(slice_positions.size)
    reference = float(slice_positions.mean()) if count else 0.0
    if count < 2:
        return SurfacePlane(0.0, False, reference, 0.0, count)

    # 共有法線をdoubled-angle平均で決め、各スライスのoffsetをその符号へ揃える
    shared = 0.5 * math.atan2(
        float(np.sin(2.0 * angles_rad).mean()),
        float(np.cos(2.0 * angles_rad).mean()),
    )
    alignment = np.sign(np.cos(shared - angles_rad))
    alignment[alignment == 0] = 1.0
    aligned = offsets_px * alignment

    slope, intercept = _fit_slope(slice_positions, aligned)
    span = float(slice_positions.max() - slice_positions.min())
    movement = abs(slope) * span
    residuals = aligned - (intercept + slope * slice_positions)
    point_residual = float(np.sqrt((residuals**2).mean()))
    angle_residual = float(
        np.degrees(
            np.sqrt(
                (
                    (
                        np.arctan2(
                            np.sin(angles_rad - shared), np.cos(angles_rad - shared)
                        )
                        + np.pi / 2.0
                    )
                    % np.pi
                    - np.pi / 2.0
                )
                ** 2
            ).mean()
        )
    )

    reliable = _is_reliable_tilt(
        slice_positions,
        aligned,
        slope,
        span,
        movement,
        point_residual,
        angle_residual,
    )
    # 傾きが不確かな面は垂直平面として扱う
    final_slope = slope if reliable else 0.0
    rho_at_reference = (
        intercept + slope * reference if reliable else float(aligned.mean())
    )
    return SurfacePlane(
        slope_px_per_slice=final_slope,
        reliable=reliable,
        reference_slice=reference,
        movement_px=movement,
        slice_count=count,
        angle_rad=shared,
        rho_at_reference_px=rho_at_reference,
    )


def _is_reliable_tilt(
    slice_positions: np.ndarray,
    aligned: np.ndarray,
    slope: float,
    span: float,
    movement: float,
    point_residual: float,
    angle_residual: float,
) -> bool:
    """符号付き傾きを信頼してよいかを判定する。"""
    count = int(slice_positions.size)
    if count < MIN_GT_SLICES or span < MIN_GT_SPAN_SLICES:
        return False
    if point_residual > MAX_GT_POINT_RESIDUAL_PX:
        return False
    if angle_residual > MAX_GT_ANGLE_RESIDUAL_DEG:
        return False
    if movement < MIN_GT_MOVEMENT_PX:
        return False

    sign = np.sign(slope)
    agreements = 0
    for index in range(count):
        keep = np.ones(count, dtype=bool)
        keep[index] = False
        left_out, _ = _fit_slope(slice_positions[keep], aligned[keep])
        agreements += int(np.sign(left_out) == sign)
    if agreements / count < MIN_GT_LOO_SIGN_AGREEMENT:
        return False

    if movement >= STRONG_GT_MOVEMENT_PX:
        return True
    odd_slope, _ = _fit_slope(slice_positions[1::2], aligned[1::2])
    even_slope, _ = _fit_slope(slice_positions[0::2], aligned[0::2])
    return bool(np.sign(odd_slope) == sign and np.sign(even_slope) == sign)
