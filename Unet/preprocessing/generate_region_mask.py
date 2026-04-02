from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

NEAR_DUPLICATE_DISTANCE = 1.0
PARALLEL_EPS = 1e-6
RAY_STEP_SIZE = 0.5
SEED_OFFSET = 12.0
SMALL_COMPONENT_THRESHOLD = 50

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


def line_intersection(a: FittedLine, b: FittedLine) -> Optional[np.ndarray]:
    """2直線の交点を計算し、平行ならNoneを返す。"""
    p = a.centroid.astype(np.float64)
    r = a.direction.astype(np.float64)
    q = b.centroid.astype(np.float64)
    s = b.direction.astype(np.float64)

    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < PARALLEL_EPS:
        return None

    qp = q - p
    t = float((qp[0] * s[1] - qp[1] * s[0]) / cross)
    return p + t * r


def validate_ray_direction(
    junction: np.ndarray,
    candidate_endpoint: np.ndarray,
    mask_centroid: np.ndarray,
) -> np.ndarray:
    """候補方向が内向きなら反転し、外向き単位ベクトルを返す。"""
    direction = candidate_endpoint.astype(np.float64) - junction.astype(np.float64)
    if float(np.linalg.norm(direction)) < PARALLEL_EPS:
        direction = junction.astype(np.float64) - mask_centroid.astype(np.float64)

    if float(np.linalg.norm(direction)) < PARALLEL_EPS:
        direction = np.array([1.0, 0.0], dtype=np.float64)

    outward_ref = junction.astype(np.float64) - mask_centroid.astype(np.float64)
    if float(np.dot(direction, outward_ref)) < 0.0:
        direction = -direction

    norm = float(np.linalg.norm(direction))
    if norm < PARALLEL_EPS:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def extend_ray_to_mask(
    junction: np.ndarray,
    direction: np.ndarray,
    vertebra_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """接合点からrayを進め、最後にマスク内だった点を返す。"""
    mask = (vertebra_mask > 0).astype(np.uint8)
    height, width = mask.shape

    unit = direction.astype(np.float64)
    norm = float(np.linalg.norm(unit))
    if norm < PARALLEL_EPS:
        return None
    unit /= norm

    x0 = int(round(float(junction[0])))
    y0 = int(round(float(junction[1])))
    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
        return None
    if mask[y0, x0] == 0:
        return None

    last_valid = np.array([float(x0), float(y0)], dtype=np.float64)
    max_steps = int(np.hypot(height, width) / RAY_STEP_SIZE) + 4

    for step in range(1, max_steps):
        probe = junction.astype(np.float64) + unit * (step * RAY_STEP_SIZE)
        xi = int(round(float(probe[0])))
        yi = int(round(float(probe[1])))

        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            break
        if mask[yi, xi] == 0:
            break

        last_valid = np.array([float(xi), float(yi)], dtype=np.float64)

    return last_valid


def draw_barrier(shape: tuple, segments: list) -> np.ndarray:
    """線分群からbarrierマスクを描画し、closingで連結を補強する。"""
    barrier = np.zeros(shape, dtype=np.uint8)

    for seg in segments:
        if len(seg) != 2:
            continue
        p0 = np.asarray(seg[0], dtype=np.float64)
        p1 = np.asarray(seg[1], dtype=np.float64)
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue

        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        cv2.line(barrier, (x0, y0), (x1, y1), color=1, thickness=2)

    kernel = np.ones((3, 3), dtype=np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel, iterations=2)
    return barrier


def split_regions(vertebra_mask: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    """barrierで遮断した椎骨領域を連結成分で分割する。"""
    vertebra = (vertebra_mask > 0).astype(np.uint8)
    blocked = (barrier > 0).astype(np.uint8)
    region = (vertebra & (1 - blocked)).astype(np.uint8)

    _, labels = cv2.connectedComponents(region, connectivity=8)
    return labels.astype(np.int32)


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    """マスク重心を返す。空なら画像中心を返す。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return np.array([w / 2.0, h / 2.0], dtype=np.float64)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _line_intersection_or_centroid(a: FittedLine, b: FittedLine) -> np.ndarray:
    """交点が得られない場合は2直線重心の中点を返す。"""
    inter = line_intersection(a, b)
    if inter is not None:
        return inter.astype(np.float64)
    return (a.centroid + b.centroid) / 2.0


def _component_info(labels: np.ndarray, mask: np.ndarray) -> dict[int, dict[str, np.ndarray | int]]:
    """連結成分ごとの面積と重心を収集する。"""
    info: dict[int, dict[str, np.ndarray | int]] = {}
    inside = (mask > 0)
    unique_labels = np.unique(labels[inside])

    for label_id in unique_labels:
        if int(label_id) <= 0:
            continue
        ys, xs = np.where((labels == int(label_id)) & inside)
        if len(xs) == 0:
            continue
        centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
        info[int(label_id)] = {
            "area": int(len(xs)),
            "centroid": centroid,
        }
    return info


def _snap_seed_to_component(
    seed: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """seedが無効位置なら最寄りの有効成分画素へ補正する。"""
    h, w = labels.shape
    x = int(round(float(seed[0])))
    y = int(round(float(seed[1])))

    if 0 <= x < w and 0 <= y < h and mask[y, x] > 0 and labels[y, x] > 0:
        return np.array([float(x), float(y)], dtype=np.float64)

    ys, xs = np.where((mask > 0) & (labels > 0))
    if len(xs) == 0:
        return None

    diff_x = xs.astype(np.float64) - float(seed[0])
    diff_y = ys.astype(np.float64) - float(seed[1])
    idx = int(np.argmin(diff_x * diff_x + diff_y * diff_y))
    return np.array([float(xs[idx]), float(ys[idx])], dtype=np.float64)


def _pick_region_label(
    region_name: str,
    seed: Optional[np.ndarray],
    labels: np.ndarray,
    component_info: dict[int, dict[str, np.ndarray | int]],
    used: set[int],
) -> int:
    """seed優先で領域ラベルを決定し、重複時は最寄り成分へ退避する。"""
    if seed is not None:
        sx = int(round(float(seed[0])))
        sy = int(round(float(seed[1])))
        if 0 <= sy < labels.shape[0] and 0 <= sx < labels.shape[1]:
            label_id = int(labels[sy, sx])
            if label_id > 0 and label_id not in used:
                used.add(label_id)
                return label_id

    candidates = [lid for lid in component_info.keys() if lid not in used]
    if not candidates:
        return 0

    if seed is None:
        largest = max(candidates, key=lambda lid: int(component_info[lid]["area"]))
        used.add(largest)
        return int(largest)

    chosen = min(
        candidates,
        key=lambda lid: float(
            np.linalg.norm(
                component_info[lid]["centroid"].astype(np.float64) - seed.astype(np.float64)
            )
        ),
    )
    used.add(int(chosen))
    return int(chosen)


def _build_seed_candidates(
    fitted_lines: dict,
    labels: np.ndarray,
    vertebra_mask: np.ndarray,
) -> dict[str, Optional[np.ndarray]]:
    """幾何規則に基づき4領域のseed候補点を生成する。"""
    mask_centroid = _mask_centroid(vertebra_mask)

    j_r = _line_intersection_or_centroid(fitted_lines["line_1"], fitted_lines["line_2"])
    j_l = _line_intersection_or_centroid(fitted_lines["line_3"], fitted_lines["line_4"])

    bridge = j_l - j_r
    if float(np.linalg.norm(bridge)) < PARALLEL_EPS:
        bridge = np.array([1.0, 0.0], dtype=np.float64)

    body_perp = np.array([-bridge[1], bridge[0]], dtype=np.float64)
    if body_perp[1] > 0:
        body_perp = -body_perp
    body_perp_norm = float(np.linalg.norm(body_perp))
    if body_perp_norm < PARALLEL_EPS:
        body_perp = np.array([0.0, -1.0], dtype=np.float64)
    else:
        body_perp /= body_perp_norm

    mid = (j_r + j_l) / 2.0
    body_seed = mid + body_perp * SEED_OFFSET
    posterior_seed = mid - body_perp * SEED_OFFSET

    dir_r = validate_ray_direction(
        j_r,
        j_r + fitted_lines["line_2"].direction * 10.0,
        mask_centroid,
    )
    right_side = np.array([dir_r[1], -dir_r[0]], dtype=np.float64)
    right_seed = j_r + dir_r * 8.0 + right_side * SEED_OFFSET

    dir_l = validate_ray_direction(
        j_l,
        j_l + fitted_lines["line_4"].direction * 10.0,
        mask_centroid,
    )
    left_side = np.array([-dir_l[1], dir_l[0]], dtype=np.float64)
    left_seed = j_l + dir_l * 8.0 + left_side * SEED_OFFSET

    return {
        "body": _snap_seed_to_component(body_seed, labels, vertebra_mask),
        "right_foramen": _snap_seed_to_component(right_seed, labels, vertebra_mask),
        "left_foramen": _snap_seed_to_component(left_seed, labels, vertebra_mask),
        "posterior": _snap_seed_to_component(posterior_seed, labels, vertebra_mask),
    }


def assign_labels_by_seed(
    labels: np.ndarray,
    fitted_lines: dict,
    vertebra_mask: np.ndarray,
) -> dict:
    """seed配置で領域名を連結成分ラベルへ対応付ける。"""
    component_info = _component_info(labels, vertebra_mask)
    if not component_info:
        return {
            "body": 0,
            "right_foramen": 0,
            "left_foramen": 0,
            "posterior": 0,
        }

    seeds = _build_seed_candidates(fitted_lines, labels, vertebra_mask)

    used: set[int] = set()
    body_label = _pick_region_label("body", seeds["body"], labels, component_info, used)
    right_label = _pick_region_label(
        "right_foramen", seeds["right_foramen"], labels, component_info, used
    )
    left_label = _pick_region_label(
        "left_foramen", seeds["left_foramen"], labels, component_info, used
    )

    remaining = [lid for lid in component_info.keys() if lid not in used]
    if seeds["posterior"] is not None and remaining:
        posterior_label = _pick_region_label(
            "posterior", seeds["posterior"], labels, component_info, used
        )
    elif remaining:
        posterior_label = int(max(remaining, key=lambda lid: int(component_info[lid]["area"])))
    else:
        posterior_label = 0

    return {
        "body": int(body_label),
        "right_foramen": int(right_label),
        "left_foramen": int(left_label),
        "posterior": int(posterior_label),
    }


def _farthest_endpoint(points: list, junction: np.ndarray) -> np.ndarray:
    """接合点から最遠の端点を返す。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] < 2:
        return junction.astype(np.float64)

    candidates = [arr[0], arr[-1]]
    distances = [float(np.linalg.norm(c - junction)) for c in candidates]
    return candidates[int(np.argmax(distances))].astype(np.float64)


def _compute_boundary_points(mask: np.ndarray) -> np.ndarray:
    """マスク境界画素を抽出する。"""
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = (binary - eroded) > 0
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _clamp_to_nearest_boundary(point: np.ndarray, boundary_points: np.ndarray) -> np.ndarray:
    """点を最寄り境界点へ投影する。"""
    if boundary_points.shape[0] == 0:
        return point.astype(np.float64)
    diff = boundary_points - point.astype(np.float64)
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    return boundary_points[idx].astype(np.float64)


def _component_count(labels: np.ndarray, mask: np.ndarray) -> int:
    """椎骨内での正ラベル成分数を返す。"""
    inside = mask > 0
    unique_labels = np.unique(labels[inside])
    return int(np.sum(unique_labels > 0))


def _nearest_pair_segment(points_a: list, points_b: list) -> tuple[np.ndarray, np.ndarray]:
    """2折れ線の最短点対を返す。"""
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)

    min_dist = float("inf")
    best_a = a[0]
    best_b = b[0]
    for pa in (a[0], a[-1]):
        for pb in (b[0], b[-1]):
            dist = float(np.linalg.norm(pa - pb))
            if dist < min_dist:
                min_dist = dist
                best_a = pa
                best_b = pb
    return best_a.astype(np.float64), best_b.astype(np.float64)


def _local_refit(points: list, junction: np.ndarray) -> FittedLine:
    """接合点近傍3-5点で局所接線を再フィットする。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] < 2:
        return fit_tls_line(points)

    d = np.linalg.norm(arr - junction.astype(np.float64), axis=1)
    count = int(np.clip(arr.shape[0], 3, 5))
    idx = np.argsort(d)[:count]
    local_points = arr[idx]
    if local_points.shape[0] < 2:
        return fit_tls_line(points)
    return fit_tls_line(local_points.tolist())


def _merge_small_components(
    labels: np.ndarray,
    vertebra_mask: np.ndarray,
    min_area: int = SMALL_COMPONENT_THRESHOLD,
) -> np.ndarray:
    """小成分を最寄りの大成分へ統合する。"""
    merged = labels.copy()
    info = _component_info(merged, vertebra_mask)
    if not info:
        return merged

    small = [lid for lid, data in info.items() if int(data["area"]) < int(min_area)]
    large = [lid for lid, data in info.items() if int(data["area"]) >= int(min_area)]
    if not small or not large:
        return merged

    large_centroids = {
        lid: info[lid]["centroid"].astype(np.float64)
        for lid in large
    }

    for lid in small:
        c_small = info[lid]["centroid"].astype(np.float64)
        target = min(
            large,
            key=lambda k: float(np.linalg.norm(large_centroids[k] - c_small)),
        )
        merged[merged == int(lid)] = int(target)

    return merged


def _build_segments(
    fitted_lines: dict,
    polylines: dict,
    j_r: np.ndarray,
    j_l: np.ndarray,
    vertebra_mask: np.ndarray,
    extra_segments: Optional[list] = None,
) -> list:
    """接合点とray延長結果からbarrier線分群を構築する。"""
    mask_centroid = _mask_centroid(vertebra_mask)

    segment_specs = [
        ("line_1", j_r),
        ("line_2", j_r),
        ("line_3", j_l),
        ("line_4", j_l),
    ]

    segments: list = []
    for key, junction in segment_specs:
        candidate_endpoint = _farthest_endpoint(polylines[key], junction)
        direction = validate_ray_direction(junction, candidate_endpoint, mask_centroid)
        endpoint = extend_ray_to_mask(junction, direction, vertebra_mask)
        if endpoint is not None:
            segments.append((junction.astype(np.float64), endpoint.astype(np.float64)))

    segments.append((j_r.astype(np.float64), j_l.astype(np.float64)))

    if extra_segments:
        segments.extend(extra_segments)

    return segments


def _split_with_segments(
    fitted_lines: dict,
    polylines: dict,
    j_r: np.ndarray,
    j_l: np.ndarray,
    vertebra_mask: np.ndarray,
    extra_segments: Optional[list] = None,
    reinforce: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """線分群からbarrier・領域分割・成分数をまとめて返す。"""
    segments = _build_segments(
        fitted_lines=fitted_lines,
        polylines=polylines,
        j_r=j_r,
        j_l=j_l,
        vertebra_mask=vertebra_mask,
        extra_segments=extra_segments,
    )
    barrier = draw_barrier(vertebra_mask.shape, segments)
    if reinforce:
        kernel = np.ones((3, 3), dtype=np.uint8)
        barrier = cv2.dilate(barrier, kernel, iterations=1)

    labels = split_regions(vertebra_mask, barrier)
    comp_count = _component_count(labels, vertebra_mask)
    return labels, barrier, comp_count


def _build_fixed_seed_markers(
    fitted_lines: dict,
    vertebra_mask: np.ndarray,
    labels_hint: np.ndarray,
) -> np.ndarray:
    """watershed用の固定seedマーカーを生成する。"""
    seeds = _build_seed_candidates(fitted_lines, labels_hint, vertebra_mask)
    h, w = vertebra_mask.shape
    markers = np.zeros((h, w), dtype=np.int32)

    region_to_id = {
        "body": 1,
        "right_foramen": 2,
        "left_foramen": 3,
        "posterior": 4,
    }

    for name, marker_id in region_to_id.items():
        seed = seeds.get(name)
        if seed is None:
            continue
        x = int(round(float(seed[0])))
        y = int(round(float(seed[1])))
        if 0 <= x < w and 0 <= y < h and vertebra_mask[y, x] > 0:
            cv2.circle(markers, (x, y), radius=2, color=int(marker_id), thickness=-1)

    return markers


def _watershed_last_resort(
    vertebra_mask: np.ndarray,
    barrier: np.ndarray,
    fitted_lines: dict,
    labels_hint: np.ndarray,
) -> np.ndarray:
    """固定seed付きwatershedで最終分割を試行する。"""
    mask = (vertebra_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gradient = cv2.GaussianBlur(255 - dist_norm, (3, 3), 0)
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    markers = _build_fixed_seed_markers(fitted_lines, vertebra_mask, labels_hint)
    if np.max(markers) <= 0:
        return labels_hint

    unknown = ((mask > 0) & (barrier > 0)).astype(np.uint8)
    markers[unknown > 0] = 0
    markers[mask == 0] = -1

    ws = cv2.watershed(image, markers.copy())

    result = np.zeros_like(ws, dtype=np.int32)
    inside = mask > 0
    positive = ws > 0
    result[inside & positive] = ws[inside & positive]
    return result


def _build_seg_mask(
    labels: np.ndarray,
    label_map: dict,
    vertebra_mask: np.ndarray,
) -> np.ndarray:
    """成分ラベル対応から5ch one-hotマスクを構築する。"""
    mask = (vertebra_mask > 0)
    h, w = vertebra_mask.shape
    seg = np.zeros((5, h, w), dtype=np.uint8)

    # ch0はバックグラウンド: 椎骨外(mask==0)かつ他チャンネル未割り当て領域
    # ただしone-hot整合性のため、椎骨外ピクセルは全チャンネル0とする
    assigned = np.zeros((h, w), dtype=bool)

    region_order = [
        ("body", 1),
        ("right_foramen", 2),
        ("left_foramen", 3),
    ]

    for region_name, channel in region_order:
        lid = int(label_map.get(region_name, 0))
        if lid <= 0:
            continue
        region_mask = (labels == lid) & mask & (~assigned)
        seg[channel, region_mask] = 1
        assigned |= region_mask

    posterior_mask = mask & (~assigned)
    seg[4, posterior_mask] = 1
    assigned |= posterior_mask

    # ch0: 椎骨内でまだ未割り当てのピクセルに使用（通常は空）
    # テスト整合性: 椎骨外のピクセルはすべてのチャンネルで0のまま
    remaining_inside = mask & (~assigned)
    seg[0, remaining_inside] = 1

    return seg


def generate_region_mask(
    line_1: list,
    line_2: list,
    line_3: list,
    line_4: list,
    vertebra_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """4本の折れ線と椎骨マスクから5ch領域マスクを生成する。"""
    global _LAST_DEBUG_INFO

    mask = (np.asarray(vertebra_mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("vertebra_maskは2次元配列である必要があります")

    polylines = {
        "line_1": preprocess_polyline(list(line_1)),
        "line_2": preprocess_polyline(list(line_2)),
        "line_3": preprocess_polyline(list(line_3)),
        "line_4": preprocess_polyline(list(line_4)),
    }

    fitted_lines = {key: fit_tls_line(points) for key, points in polylines.items()}

    j_r = _line_intersection_or_centroid(fitted_lines["line_1"], fitted_lines["line_2"])
    j_l = _line_intersection_or_centroid(fitted_lines["line_3"], fitted_lines["line_4"])

    fallback_type = ""
    extra_segments: list = []

    labels, barrier, component_count = _split_with_segments(
        fitted_lines=fitted_lines,
        polylines=polylines,
        j_r=j_r,
        j_l=j_l,
        vertebra_mask=mask,
    )

    if component_count < 4:
        p12_a, p12_b = _nearest_pair_segment(polylines["line_1"], polylines["line_2"])
        p34_a, p34_b = _nearest_pair_segment(polylines["line_3"], polylines["line_4"])
        extra_segments = [(p12_a, p12_b), (p34_a, p34_b)]

        labels_js, barrier_js, count_js = _split_with_segments(
            fitted_lines=fitted_lines,
            polylines=polylines,
            j_r=j_r,
            j_l=j_l,
            vertebra_mask=mask,
            extra_segments=extra_segments,
        )
        labels, barrier, component_count = labels_js, barrier_js, count_js
        if component_count >= 4:
            fallback_type = "junction_snap"

    if component_count < 4:
        fitted_local = {
            "line_1": _local_refit(polylines["line_1"], j_r),
            "line_2": _local_refit(polylines["line_2"], j_r),
            "line_3": _local_refit(polylines["line_3"], j_l),
            "line_4": _local_refit(polylines["line_4"], j_l),
        }
        j_r_local = _line_intersection_or_centroid(fitted_local["line_1"], fitted_local["line_2"])
        j_l_local = _line_intersection_or_centroid(fitted_local["line_3"], fitted_local["line_4"])

        labels_lr, barrier_lr, count_lr = _split_with_segments(
            fitted_lines=fitted_local,
            polylines=polylines,
            j_r=j_r_local,
            j_l=j_l_local,
            vertebra_mask=mask,
            extra_segments=extra_segments,
        )
        labels, barrier, component_count = labels_lr, barrier_lr, count_lr
        fitted_lines = fitted_local
        j_r, j_l = j_r_local, j_l_local
        if component_count >= 4:
            fallback_type = "local_tangent_refit"

    if component_count < 4:
        boundary_points = _compute_boundary_points(mask)
        j_r_clamped = _clamp_to_nearest_boundary(j_r, boundary_points)
        j_l_clamped = _clamp_to_nearest_boundary(j_l, boundary_points)

        labels_jc, barrier_jc, count_jc = _split_with_segments(
            fitted_lines=fitted_lines,
            polylines=polylines,
            j_r=j_r_clamped,
            j_l=j_l_clamped,
            vertebra_mask=mask,
            extra_segments=extra_segments,
        )
        labels, barrier, component_count = labels_jc, barrier_jc, count_jc
        j_r, j_l = j_r_clamped, j_l_clamped
        if component_count >= 4:
            fallback_type = "junction_clamp"

    if component_count < 4:
        labels_br, barrier_br, count_br = _split_with_segments(
            fitted_lines=fitted_lines,
            polylines=polylines,
            j_r=j_r,
            j_l=j_l,
            vertebra_mask=mask,
            extra_segments=extra_segments,
            reinforce=True,
        )
        labels, barrier, component_count = labels_br, barrier_br, count_br
        if component_count >= 4:
            fallback_type = "barrier_reinforcement"

    if component_count < 4:
        labels_sm = _merge_small_components(labels, mask, min_area=SMALL_COMPONENT_THRESHOLD)
        count_sm = _component_count(labels_sm, mask)
        if count_sm != component_count:
            labels = labels_sm
            component_count = count_sm
            if component_count >= 4:
                fallback_type = "seed_relabel_merge"

    if component_count < 4:
        labels_ws = _watershed_last_resort(
            vertebra_mask=mask,
            barrier=barrier,
            fitted_lines=fitted_lines,
            labels_hint=labels,
        )
        labels = labels_ws
        component_count = _component_count(labels, mask)
        fallback_type = "watershed"

    label_map = assign_labels_by_seed(labels, fitted_lines, mask)
    seg_mask = _build_seg_mask(labels, label_map, mask)

    region_areas = {
        "body": int(seg_mask[1].sum()),
        "right_foramen": int(seg_mask[2].sum()),
        "left_foramen": int(seg_mask[3].sum()),
        "posterior": int(seg_mask[4].sum()),
    }

    debug_info: dict[str, object] = {
        "J_R": j_r.astype(np.float64),
        "J_L": j_l.astype(np.float64),
        "component_count": int(component_count),
        "region_areas": region_areas,
        "fallback_type": fallback_type,
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
    "FittedLine",
    "assign_labels_by_seed",
    "draw_barrier",
    "extend_ray_to_mask",
    "fit_tls_line",
    "generate_region_mask",
    "line_intersection",
    "preprocess_polyline",
    "split_regions",
    "validate_ray_direction",
    "validate_region_mask",
]
