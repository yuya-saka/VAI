from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

NEAR_DUPLICATE_DISTANCE = 1.0
PARALLEL_EPS = 1e-6

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
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (2,):
            raise ValueError("ポリライン点は2次元座標である必要があります")
        if not deduped:
            deduped.append([float(p[0]), float(p[1])])
            continue
        prev = np.asarray(deduped[-1], dtype=np.float64)
        if float(np.linalg.norm(p - prev)) >= NEAR_DUPLICATE_DISTANCE:
            deduped.append([float(p[0]), float(p[1])])

    if len(deduped) < 2:
        raise ValueError("重複除去後の点数が2未満です")
    return deduped


def fit_tls_line(points: list) -> FittedLine:
    """点群に対してTLS/PCA直線フィットを行う。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("TLS入力は (N,2) 形式である必要があります")
    if arr.shape[0] < 2:
        raise ValueError("TLSには2点以上が必要です")

    centroid = arr.mean(axis=0)
    centered = arr - centroid

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
    endpoints_a = [np.asarray(points_a[0], dtype=np.float64), np.asarray(points_a[-1], dtype=np.float64)]
    endpoints_b = [np.asarray(points_b[0], dtype=np.float64), np.asarray(points_b[-1], dtype=np.float64)]
    best_midpoint, best_distance = None, float("inf")
    for endpoint_a in endpoints_a:
        for endpoint_b in endpoints_b:
            distance = float(np.linalg.norm(endpoint_a - endpoint_b))
            if distance < best_distance:
                best_distance, best_midpoint = distance, (endpoint_a + endpoint_b) * 0.5
    if best_midpoint is None:
        raise ValueError("端点接合点を推定できません")
    return best_midpoint


def _body_side_normal(line: FittedLine, body_ref: np.ndarray) -> np.ndarray:
    normal = np.array([-line.direction[1], line.direction[0]], dtype=np.float64)
    if float(np.dot(normal, body_ref.astype(np.float64) - line.centroid.astype(np.float64))) < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm < PARALLEL_EPS:
        raise ValueError("法線ベクトルを正規化できません")
    return normal / norm


def _classify_by_half_planes(
    vertebra_mask: np.ndarray,
    fitted_lines: dict[str, FittedLine],
    j_r: np.ndarray,
    j_l: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.asarray(vertebra_mask) > 0
    ys, xs = np.indices(mask.shape, dtype=np.float64)
    l1, l2, l3, l4 = (fitted_lines["line_1"], fitted_lines["line_2"], fitted_lines["line_3"], fitted_lines["line_4"])
    n1 = _body_side_normal(l1, np.asarray(j_l, dtype=np.float64))
    n2 = _body_side_normal(l2, l2.centroid + np.array([0.0, -1.0], dtype=np.float64))
    n3 = _body_side_normal(l3, np.asarray(j_r, dtype=np.float64))
    n4 = _body_side_normal(l4, l4.centroid + np.array([0.0, -1.0], dtype=np.float64))
    s1 = (xs - float(l1.centroid[0])) * float(n1[0]) + (ys - float(l1.centroid[1])) * float(n1[1])
    s2 = (xs - float(l2.centroid[0])) * float(n2[0]) + (ys - float(l2.centroid[1])) * float(n2[1])
    s3 = (xs - float(l3.centroid[0])) * float(n3[0]) + (ys - float(l3.centroid[1])) * float(n3[1])
    s4 = (xs - float(l4.centroid[0])) * float(n4[0]) + (ys - float(l4.centroid[1])) * float(n4[1])
    x_mid = (float(j_r[0]) + float(j_l[0])) / 2.0
    right_side = xs >= x_mid
    body = mask & (s1 >= 0.0) & (s3 >= 0.0) & (~right_side | (s2 >= 0.0)) & (right_side | (s4 >= 0.0))
    right_foramen = mask & (s1 < 0.0) & (s2 >= 0.0)
    left_foramen = mask & (s3 < 0.0) & (s4 >= 0.0) & (~right_foramen)
    posterior = mask & (~body) & (~right_foramen) & (~left_foramen)
    return {"body": body, "right_foramen": right_foramen, "left_foramen": left_foramen, "posterior": posterior}


def _build_seg_mask_from_regions(regions: dict[str, np.ndarray], vertebra_mask: np.ndarray) -> np.ndarray:
    inside = np.asarray(vertebra_mask) > 0
    seg = np.zeros((5, *inside.shape), dtype=np.uint8)
    body = np.asarray(regions["body"], dtype=bool) & inside
    right = np.asarray(regions["right_foramen"], dtype=bool) & inside & (~body)
    left = np.asarray(regions["left_foramen"], dtype=bool) & inside & (~body) & (~right)
    posterior = np.asarray(regions["posterior"], dtype=bool) & inside & (~body) & (~right) & (~left)
    background = inside & (~body) & (~right) & (~left) & (~posterior)
    seg[0, background], seg[1, body], seg[2, right], seg[3, left], seg[4, posterior] = 1, 1, 1, 1, 1
    return seg


def generate_region_mask(
    line_1: list,
    line_2: list,
    line_3: list,
    line_4: list,
    vertebra_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    global _LAST_DEBUG_INFO
    processed_line_1, processed_line_2 = preprocess_polyline(line_1), preprocess_polyline(line_2)
    processed_line_3, processed_line_4 = preprocess_polyline(line_3), preprocess_polyline(line_4)
    fitted_lines = {
        "line_1": fit_tls_line(processed_line_1),
        "line_2": fit_tls_line(processed_line_2),
        "line_3": fit_tls_line(processed_line_3),
        "line_4": fit_tls_line(processed_line_4),
    }
    j_r = _nearest_endpoint_junction(processed_line_1, processed_line_2)
    j_l = _nearest_endpoint_junction(processed_line_3, processed_line_4)
    regions = _classify_by_half_planes(np.asarray(vertebra_mask), fitted_lines, j_r, j_l)
    seg_mask = _build_seg_mask_from_regions(regions, np.asarray(vertebra_mask))
    region_areas = {
        "body": int(seg_mask[1].sum()),
        "right_foramen": int(seg_mask[2].sum()),
        "left_foramen": int(seg_mask[3].sum()),
        "posterior": int(seg_mask[4].sum()),
    }
    region_union = ((seg_mask[1] + seg_mask[2] + seg_mask[3] + seg_mask[4]) > 0).astype(np.uint8)
    debug_info: dict[str, Any] = {
        "J_R": j_r.astype(np.float64),
        "J_L": j_l.astype(np.float64),
        "component_count": int(cv2.connectedComponents(region_union, connectivity=8)[0] - 1),
        "region_areas": region_areas,
        "fallback_type": "half_plane_degenerate" if any(area == 0 for area in region_areas.values()) else "",
    }
    if float(j_r[0]) <= float(j_l[0]):
        debug_info["swap_detected"] = True
    _LAST_DEBUG_INFO = debug_info
    return seg_mask.astype(np.uint8), debug_info


def validate_region_mask(seg_mask: np.ndarray, vertebra_mask: np.ndarray) -> dict:
    """生成マスクの整合性を検証し、ハード失敗と警告を返す。"""
    warnings: list[str] = []
    hard_fail = False

    mask = (np.asarray(vertebra_mask) > 0)
    seg = np.asarray(seg_mask)

    if seg.ndim != 3 or seg.shape[0] != 5 or seg.shape[1:] != mask.shape:
        return {
            "hard_fail": True,
            "warnings": ["seg_maskの形状が不正です"],
        }

    summed = seg.sum(axis=0)
    inside = mask
    outside = ~inside

    one_hot_violation = bool(np.any(summed[inside] != 1))
    if one_hot_violation:
        hard_fail = True
        warnings.append("椎骨内one-hot条件に違反しています")

    coverage_violation = bool(np.any(summed[inside] == 0))
    if coverage_violation:
        hard_fail = True
        warnings.append("椎骨内に未被覆画素があります")

    # 椎骨外のすべてのチャンネル（ch0含む）が0であることを確認
    background_violation = bool(np.any(seg[:, outside] > 0))
    if background_violation:
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

    return {
        "hard_fail": hard_fail,
        "warnings": warnings,
    }


__all__ = [
    "generate_region_mask",
    "validate_region_mask",
    "FittedLine",
    "fit_tls_line",
    "preprocess_polyline",
]
