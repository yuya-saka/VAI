from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

NEAR_DUPLICATE_DISTANCE = 1.0
PARALLEL_EPS = 1e-6
MIN_CANAL_AREA = 20.0
MAX_LOCAL_CORRECTION_FRACTION = 0.07

_LAST_DEBUG_INFO: dict[str, object] = {}


@dataclass(frozen=True)
class FittedLine:
    """TLS/PCAで推定した2次元直線を保持する。"""

    direction: np.ndarray
    centroid: np.ndarray


def preprocess_polyline(points: list) -> list:
    """近接した重複点を除去し、有効な折れ線点列を返す。"""
    if not points:
        raise ValueError("ポリラインが空です")

    deduped: list[list[float]] = []
    for point in points:
        current = np.asarray(point, dtype=np.float64)
        if current.shape != (2,):
            raise ValueError("ポリライン点は2次元座標である必要があります")
        if not deduped:
            deduped.append([float(current[0]), float(current[1])])
            continue
        previous = np.asarray(deduped[-1], dtype=np.float64)
        if float(np.linalg.norm(current - previous)) >= NEAR_DUPLICATE_DISTANCE:
            deduped.append([float(current[0]), float(current[1])])

    if len(deduped) < 2:
        raise ValueError("重複除去後の点数が2未満です")
    return deduped


def fit_tls_line(points: list) -> FittedLine:
    """点群に対してTLS/PCA直線フィットを行う。"""
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("TLS入力は (N,2) 形式である必要があります")
    if point_array.shape[0] < 2:
        raise ValueError("TLSには2点以上が必要です")

    centroid = point_array.mean(axis=0)
    centered = point_array - centroid
    if np.allclose(centered, 0.0):
        raise ValueError("すべて同一点のため直線を推定できません")

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0].astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < PARALLEL_EPS:
        raise ValueError("直線方向ベクトルのノルムが0です")
    return FittedLine(direction=direction / norm, centroid=centroid)


def _nearest_endpoint_junction(points_a: list, points_b: list) -> np.ndarray:
    if len(points_a) < 2 or len(points_b) < 2:
        raise ValueError("端点推定には各ポリライン2点以上が必要です")
    endpoints_a = [
        np.asarray(points_a[0], dtype=np.float64),
        np.asarray(points_a[-1], dtype=np.float64),
    ]
    endpoints_b = [
        np.asarray(points_b[0], dtype=np.float64),
        np.asarray(points_b[-1], dtype=np.float64),
    ]
    best_midpoint: np.ndarray | None = None
    best_distance = float("inf")
    for endpoint_a in endpoints_a:
        for endpoint_b in endpoints_b:
            distance = float(np.linalg.norm(endpoint_a - endpoint_b))
            if distance < best_distance:
                best_distance = distance
                best_midpoint = (endpoint_a + endpoint_b) * 0.5
    if best_midpoint is None:
        raise ValueError("端点接合点を推定できません")
    return best_midpoint


def _body_side_normal(line: FittedLine, body_ref: np.ndarray) -> np.ndarray:
    normal = np.array([-line.direction[1], line.direction[0]], dtype=np.float64)
    offset = body_ref.astype(np.float64) - line.centroid.astype(np.float64)
    if float(np.dot(normal, offset)) < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm < PARALLEL_EPS:
        raise ValueError("法線ベクトルを正規化できません")
    return normal / norm


def _signed_distance_grid(
    line: FittedLine,
    normal: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    return (xs - float(line.centroid[0])) * float(normal[0]) + (
        ys - float(line.centroid[1])
    ) * float(normal[1])


def _classify_by_half_planes(
    vertebra_mask: np.ndarray,
    fitted_lines: dict[str, FittedLine],
    j_r: np.ndarray,
    j_l: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mask = np.asarray(vertebra_mask) > 0
    ys, xs = np.indices(mask.shape, dtype=np.float64)
    l1 = fitted_lines["line_1"]
    l2 = fitted_lines["line_2"]
    l3 = fitted_lines["line_3"]
    l4 = fitted_lines["line_4"]
    n1 = _body_side_normal(l1, np.asarray(j_l, dtype=np.float64))
    n2 = _body_side_normal(
        l2,
        l2.centroid + np.array([0.0, -1.0], dtype=np.float64),
    )
    n3 = _body_side_normal(l3, np.asarray(j_r, dtype=np.float64))
    n4 = _body_side_normal(
        l4,
        l4.centroid + np.array([0.0, -1.0], dtype=np.float64),
    )
    s1 = _signed_distance_grid(l1, n1, xs, ys)
    s2 = _signed_distance_grid(l2, n2, xs, ys)
    s3 = _signed_distance_grid(l3, n3, xs, ys)
    s4 = _signed_distance_grid(l4, n4, xs, ys)
    x_mid = (float(j_r[0]) + float(j_l[0])) / 2.0
    right_side = xs >= x_mid
    body = (
        mask
        & (s1 >= 0.0)
        & (s3 >= 0.0)
        & (~right_side | (s2 >= 0.0))
        & (right_side | (s4 >= 0.0))
    )
    right_foramen = mask & (s1 < 0.0) & (s2 >= 0.0)
    left_foramen = mask & (s3 < 0.0) & (s4 >= 0.0) & (~right_foramen)
    posterior = mask & (~body) & (~right_foramen) & (~left_foramen)
    regions = {
        "body": body,
        "right_foramen": right_foramen,
        "left_foramen": left_foramen,
        "posterior": posterior,
    }
    geometry = {"xs": xs, "ys": ys, "s1": s1, "s3": s3}
    return regions, geometry


def _find_canal_anterior_boundary(
    vertebra_mask: np.ndarray,
    j_l: np.ndarray,
    j_r: np.ndarray,
) -> tuple[np.ndarray, str]:
    mask = (np.asarray(vertebra_mask) > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None:
        return np.stack([j_l, j_r]), "straight_bridge"

    midpoint = (j_l + j_r) * 0.5
    candidates: list[tuple[float, np.ndarray]] = []
    for index, contour in enumerate(contours):
        if int(hierarchy[0][index][3]) < 0:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < MIN_CANAL_AREA:
            continue
        points = contour[:, 0, :].astype(np.float64)
        centroid = points.mean(axis=0)
        score = float(np.linalg.norm(centroid - midpoint)) - 0.02 * area
        candidates.append((score, points))
    if not candidates:
        return np.stack([j_l, j_r]), "straight_bridge"

    contour_points = min(candidates, key=lambda candidate: candidate[0])[1]
    left_index = int(np.argmin(np.linalg.norm(contour_points - j_l, axis=1)))
    right_index = int(np.argmin(np.linalg.norm(contour_points - j_r, axis=1)))
    start_index, end_index = sorted((left_index, right_index))
    direct_arc = contour_points[start_index : end_index + 1]
    wrapped_arc = np.concatenate(
        [contour_points[end_index:], contour_points[: start_index + 1]]
    )[::-1]
    boundary = min(
        (direct_arc, wrapped_arc),
        key=lambda arc: float(arc[:, 1].mean()),
    )
    return boundary, "canal_anterior_arc"


def _boundary_y_by_column(
    boundary_points: np.ndarray,
    width: int,
) -> np.ndarray:
    boundary_y = np.full(width, np.nan, dtype=np.float64)
    rounded_x = np.rint(boundary_points[:, 0]).astype(np.int32)
    for x_coord in np.unique(rounded_x):
        if not 0 <= x_coord < width:
            continue
        y_values = boundary_points[rounded_x == x_coord, 1]
        boundary_y[x_coord] = float(np.min(y_values))

    valid_x = np.flatnonzero(np.isfinite(boundary_y))
    if valid_x.size < 2:
        return boundary_y
    all_x = np.arange(width)
    boundary_y = np.interp(all_x, valid_x, boundary_y[valid_x])
    return boundary_y


def _correct_central_posterior_wedge(
    regions: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    vertebra_mask: np.ndarray,
    j_l: np.ndarray,
    j_r: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    boundary_points, boundary_type = _find_canal_anterior_boundary(
        vertebra_mask,
        j_l,
        j_r,
    )

    mask = np.asarray(vertebra_mask) > 0
    boundary_y = _boundary_y_by_column(boundary_points, mask.shape[1])
    xs = geometry["xs"]
    ys = geometry["ys"]
    x_min = min(float(j_l[0]), float(j_r[0]))
    x_max = max(float(j_l[0]), float(j_r[0]))
    boundary_grid = boundary_y[np.clip(xs.astype(np.int32), 0, mask.shape[1] - 1)]
    candidate = (
        regions["posterior"]
        & (xs >= x_min)
        & (xs <= x_max)
        & (geometry["s1"] >= 0.0)
        & (geometry["s3"] >= 0.0)
        & np.isfinite(boundary_grid)
        & (ys <= boundary_grid)
    )
    corrected_pixels = int(candidate.sum())
    max_pixels = int(np.ceil(mask.sum() * MAX_LOCAL_CORRECTION_FRACTION))
    if corrected_pixels == 0:
        return regions, {
            "corrected_pixels": 0,
            "correction_status": "not_needed",
            "correction_boundary": boundary_type,
        }
    if corrected_pixels > max_pixels:
        return regions, {
            "corrected_pixels": 0,
            "rejected_pixels": corrected_pixels,
            "correction_status": "area_limit_exceeded",
            "correction_boundary": boundary_type,
        }

    corrected_regions = {
        **regions,
        "body": regions["body"] | candidate,
        "posterior": regions["posterior"] & (~candidate),
    }
    return corrected_regions, {
        "corrected_pixels": corrected_pixels,
        "correction_status": "applied",
        "correction_boundary": boundary_type,
    }


def _build_seg_mask_from_regions(
    regions: dict[str, np.ndarray],
    vertebra_mask: np.ndarray,
) -> np.ndarray:
    inside = np.asarray(vertebra_mask) > 0
    seg = np.zeros((5, *inside.shape), dtype=np.uint8)
    body = np.asarray(regions["body"], dtype=bool) & inside
    right = np.asarray(regions["right_foramen"], dtype=bool) & inside & (~body)
    left = np.asarray(regions["left_foramen"], dtype=bool) & inside & (~body) & (~right)
    posterior = (
        np.asarray(regions["posterior"], dtype=bool)
        & inside
        & (~body)
        & (~right)
        & (~left)
    )
    background = inside & (~body) & (~right) & (~left) & (~posterior)
    seg[0, background] = 1
    seg[1, body] = 1
    seg[2, right] = 1
    seg[3, left] = 1
    seg[4, posterior] = 1
    return seg


def generate_region_mask(
    line_1: list,
    line_2: list,
    line_3: list,
    line_4: list,
    vertebra_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    global _LAST_DEBUG_INFO
    processed_line_1 = preprocess_polyline(line_1)
    processed_line_2 = preprocess_polyline(line_2)
    processed_line_3 = preprocess_polyline(line_3)
    processed_line_4 = preprocess_polyline(line_4)
    fitted_lines = {
        "line_1": fit_tls_line(processed_line_1),
        "line_2": fit_tls_line(processed_line_2),
        "line_3": fit_tls_line(processed_line_3),
        "line_4": fit_tls_line(processed_line_4),
    }
    j_r = _nearest_endpoint_junction(processed_line_1, processed_line_2)
    j_l = _nearest_endpoint_junction(processed_line_3, processed_line_4)
    regions, geometry = _classify_by_half_planes(
        np.asarray(vertebra_mask),
        fitted_lines,
        j_r,
        j_l,
    )
    corrected_regions, correction_debug = _correct_central_posterior_wedge(
        regions,
        geometry,
        np.asarray(vertebra_mask),
        j_l,
        j_r,
    )
    seg_mask = _build_seg_mask_from_regions(
        corrected_regions,
        np.asarray(vertebra_mask),
    )
    region_areas = {
        "body": int(seg_mask[1].sum()),
        "right_foramen": int(seg_mask[2].sum()),
        "left_foramen": int(seg_mask[3].sum()),
        "posterior": int(seg_mask[4].sum()),
    }
    region_union = ((seg_mask[1] + seg_mask[2] + seg_mask[3] + seg_mask[4]) > 0).astype(
        np.uint8
    )
    debug_info: dict[str, Any] = {
        "J_R": j_r.astype(np.float64),
        "J_L": j_l.astype(np.float64),
        "component_count": int(
            cv2.connectedComponents(region_union, connectivity=8)[0] - 1
        ),
        "region_areas": region_areas,
        "fallback_type": (
            "half_plane_degenerate"
            if any(area == 0 for area in region_areas.values())
            else ""
        ),
        "method": "half_plane_with_local_posterior_correction",
        **correction_debug,
    }
    if float(j_r[0]) <= float(j_l[0]):
        debug_info["swap_detected"] = True
    _LAST_DEBUG_INFO = debug_info
    return seg_mask.astype(np.uint8), debug_info


def validate_region_mask(
    seg_mask: np.ndarray,
    vertebra_mask: np.ndarray,
) -> dict:
    """生成マスクの整合性を検証し、ハード失敗と警告を返す。"""
    warnings: list[str] = []
    hard_fail = False
    mask = np.asarray(vertebra_mask) > 0
    seg = np.asarray(seg_mask)

    if seg.ndim != 3 or seg.shape[0] != 5 or seg.shape[1:] != mask.shape:
        return {"hard_fail": True, "warnings": ["seg_maskの形状が不正です"]}

    summed = seg.sum(axis=0)
    if bool(np.any(summed[mask] != 1)):
        hard_fail = True
        warnings.append("椎骨内one-hot条件に違反しています")
    if bool(np.any(summed[mask] == 0)):
        hard_fail = True
        warnings.append("椎骨内に未被覆画素があります")
    if bool(np.any(seg[:, ~mask] > 0)):
        hard_fail = True
        warnings.append("背景画素に前景チャネル値があります")

    region_names = {
        1: "body",
        2: "right_foramen",
        3: "left_foramen",
        4: "posterior",
    }
    for channel, name in region_names.items():
        if int(seg[channel].sum()) == 0:
            warnings.append(f"領域面積が0です: {name}")

    debug = _LAST_DEBUG_INFO if isinstance(_LAST_DEBUG_INFO, dict) else {}
    j_r = debug.get("J_R")
    j_l = debug.get("J_L")
    if isinstance(j_r, np.ndarray) and isinstance(j_l, np.ndarray):
        if j_r.shape == (2,) and j_l.shape == (2,) and float(j_r[0]) <= float(j_l[0]):
            warnings.append("J_RとJ_Lのx順序が逆転しています")
    if "swap_detected" in debug:
        warnings.append("左右入れ替えが検出されています")
    if debug.get("correction_status") == "area_limit_exceeded":
        warnings.append("中央posterior補正が面積上限を超えたため適用されませんでした")
    return {"hard_fail": hard_fail, "warnings": warnings}


__all__ = [
    "generate_region_mask",
    "validate_region_mask",
    "FittedLine",
    "fit_tls_line",
    "preprocess_polyline",
]
