from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import nibabel.processing
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing.segmentation_dataset.convert_to_png import (  # noqa: E402
    apply_window,
    center_crop,
    load_vertebra_data,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (  # noqa: E402
    FittedLine,
    fit_tls_line,
    generate_region_mask,
    preprocess_polyline,
    validate_region_mask,
)
from data_preprocessing.segmentation_dataset.preprocess_all import build_overlay_image  # noqa: E402

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
TARGET_SIZE = 224
MIN_MASK_AREA = 50
DEFAULT_TREND_ANCHORS = 4
DEFAULT_CONFIDENCE_DECAY_MM = 4.0
DEFAULT_Z_SPACING_MM = 0.4
DELTA_MAX_JUNCTION = 0.25       # 正規化座標でのjunction最大変位
DELTA_MAX_ANGLE_DOUBLED = 0.5   # doubled-angle成分の最大変位 (≈15°相当)
MIN_JUNCTION_X_SEP = 0.05       # 正規化座標でのJ_R.x - J_L.x 最小値


@dataclass(frozen=True)
class MaskGeometry:
    """椎骨マスクの正規化に使う重心と大きさを保持する。"""

    centroid: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class PropagatedLines:
    """1スライス分の伝播線と由来情報を保持する。"""

    lines: dict[str, list[list[float]]]
    provenance: str
    nearest_anchor_distance: int
    confidence: float


@dataclass(frozen=True)
class SliceState:
    """1スライスのjunction + doubled-angle状態を保持する。"""

    j_r: np.ndarray     # (2,) 正規化座標 line_1∩line_2
    j_l: np.ndarray     # (2,) 正規化座標 line_3∩line_4
    doubled: np.ndarray  # (4, 2) [cos 2α, sin 2α] per line


# ─── ローダーとユーティリティ ───────────────────────────────────────────


def is_valid_line_entry(entry: object) -> bool:
    """4本の有効な折れ線を持つか判定する。"""
    if not isinstance(entry, dict):
        return False
    for line_key in LINE_KEYS:
        points = entry.get(line_key)
        if not isinstance(points, list) or len(points) < 2:
            return False
        if any(not isinstance(p, list | tuple) or len(p) != 2 for p in points):
            return False
    return True


def load_manual_lines(lines_path: Path) -> dict[int, dict[str, list[list[float]]]]:
    """手動アノテーション済みの4本線を読み込む。"""
    raw_data = json.loads(lines_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("lines.jsonの形式が不正です")
    result: dict[int, dict[str, list[list[float]]]] = {}
    for raw_slice_idx, entry in raw_data.items():
        if not is_valid_line_entry(entry):
            continue
        slice_idx = int(raw_slice_idx)
        result[slice_idx] = {
            line_key: [[float(p[0]), float(p[1])] for p in entry[line_key]]
            for line_key in LINE_KEYS
        }
    if len(result) < 2:
        raise ValueError("線の伝播には2枚以上の手動アノテーションが必要です")
    return result


def compute_mask_geometry(mask: np.ndarray) -> MaskGeometry:
    """マスクの重心とbboxサイズを計算する。"""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) < MIN_MASK_AREA:
        raise ValueError("椎骨マスク面積が小さすぎます")
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    width = max(float(xs.max() - xs.min() + 1), 1.0)
    height = max(float(ys.max() - ys.min() + 1), 1.0)
    return MaskGeometry(
        centroid=centroid,
        scale=np.array([width, height], dtype=np.float64),
    )


def normalize_points(points: np.ndarray, geometry: MaskGeometry) -> np.ndarray:
    """画像座標を椎骨マスク基準の正規化座標へ変換する。"""
    return (np.asarray(points, dtype=np.float64) - geometry.centroid) / geometry.scale


def denormalize_points(
    normalized_points: np.ndarray,
    geometry: MaskGeometry,
    image_size: int = TARGET_SIZE,
) -> np.ndarray:
    """正規化座標を画像座標へ戻す。"""
    points = np.asarray(normalized_points, dtype=np.float64) * geometry.scale + geometry.centroid
    return np.clip(points, 0.0, float(image_size - 1))


def confidence_from_provenance(
    provenance: str,
    nearest_anchor_distance: int,
    z_spacing_mm: float,
    decay_mm: float,
) -> float:
    """由来とアンカー距離から学習用信頼度を計算する。"""
    if provenance == "manual":
        return 1.0
    if provenance == "interpolated":
        return 0.9
    distance_mm = nearest_anchor_distance * z_spacing_mm
    return max(0.1, float(math.exp(-distance_mm / decay_mm)))


# ─── 状態抽出とフィット ─────────────────────────────────────────────────


def _line_intersection_2d(line1: FittedLine, line2: FittedLine) -> np.ndarray | None:
    """2直線の交点を計算する。平行な場合はNoneを返す。"""
    d1, d2 = line1.direction, line2.direction
    c1, c2 = line1.centroid, line2.centroid
    a00, a01 = float(d1[0]), -float(d2[0])
    a10, a11 = float(d1[1]), -float(d2[1])
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-6:
        return None
    bx, by = float(c2[0] - c1[0]), float(c2[1] - c1[1])
    t = (bx * a11 - by * a01) / det
    return c1 + t * d1


def _nearest_endpoint_midpoint(pts_a: list, pts_b: list) -> np.ndarray:
    """2折れ線の最近接端点ペアの中点を返す。"""
    ends_a = [np.asarray(pts_a[0], np.float64), np.asarray(pts_a[-1], np.float64)]
    ends_b = [np.asarray(pts_b[0], np.float64), np.asarray(pts_b[-1], np.float64)]
    best, best_d = None, float("inf")
    for ea in ends_a:
        for eb in ends_b:
            d = float(np.linalg.norm(ea - eb))
            if d < best_d:
                best_d, best = d, (ea + eb) * 0.5
    if best is None:
        raise ValueError("端点中点を計算できません")
    return best


def extract_slice_state(lines_slice: dict, geometry: MaskGeometry) -> SliceState:
    """手動アノテーション1スライスから(J_R, J_L, doubled-angle)を抽出する。"""
    pts = [preprocess_polyline(lines_slice[k]) for k in LINE_KEYS]
    fitted = [fit_tls_line(p) for p in pts]

    j_r_img = _line_intersection_2d(fitted[0], fitted[1])
    if j_r_img is None:
        j_r_img = _nearest_endpoint_midpoint(pts[0], pts[1])

    j_l_img = _line_intersection_2d(fitted[2], fitted[3])
    if j_l_img is None:
        j_l_img = _nearest_endpoint_midpoint(pts[2], pts[3])

    j_r = normalize_points(j_r_img[np.newaxis], geometry)[0]
    j_l = normalize_points(j_l_img[np.newaxis], geometry)[0]

    doubled = np.zeros((4, 2), dtype=np.float64)
    for i, f in enumerate(fitted):
        alpha = np.arctan2(float(f.direction[1]), float(f.direction[0]))
        doubled[i] = [np.cos(2.0 * alpha), np.sin(2.0 * alpha)]

    return SliceState(j_r=j_r, j_l=j_l, doubled=doubled)


def _states_to_matrix(
    states: dict[int, SliceState],
) -> tuple[np.ndarray, np.ndarray]:
    """SliceState dict を z配列 (n,) と値行列 (n, 12) に変換する。"""
    z_sorted = sorted(states)
    rows = [
        np.concatenate([states[z].j_r, states[z].j_l, states[z].doubled.ravel()])
        for z in z_sorted
    ]
    return np.array(z_sorted, dtype=np.float64), np.array(rows, dtype=np.float64)


def build_smooth_trajectory(
    anchor_states: dict[int, SliceState],
    trend_anchor_count: int = DEFAULT_TREND_ANCHORS,
) -> tuple[CubicSpline, dict[str, Any]]:
    """アンカーから平滑スプライン軌跡と外挿パラメータを構築する。"""
    z_anchors, packed = _states_to_matrix(anchor_states)
    n = len(z_anchors)
    if n < 2:
        raise ValueError("アンカーが2枚以上必要です")

    spline = CubicSpline(z_anchors, packed)

    k = max(1, min(trend_anchor_count, n))
    dz_lo = max(float(z_anchors[k - 1] - z_anchors[0]), 1.0)
    dz_hi = max(float(z_anchors[-1] - z_anchors[-k]), 1.0)
    slope_lo = (packed[k - 1] - packed[0]) / dz_lo
    slope_hi = (packed[-1] - packed[-k]) / dz_hi

    delta_max = np.concatenate([
        np.full(2, DELTA_MAX_JUNCTION),
        np.full(2, DELTA_MAX_JUNCTION),
        np.full(8, DELTA_MAX_ANGLE_DOUBLED),
    ])

    return spline, {
        "z_lo": float(z_anchors[0]),
        "z_hi": float(z_anchors[-1]),
        "val_lo": packed[0].copy(),
        "val_hi": packed[-1].copy(),
        "slope_lo": slope_lo,
        "slope_hi": slope_hi,
        "delta_max": delta_max,
        "z_anchors": [int(z) for z in z_anchors],
    }


def evaluate_trajectory(
    spline: CubicSpline,
    extrap_params: dict[str, Any],
    z: int,
    mask_area: float,
    area_at_lo: float,
    area_at_hi: float,
) -> tuple[np.ndarray, str, int]:
    """指定zの状態ベクトルを評価する。内挿域はスプライン、外側はtanh外挿。"""
    z_lo = extrap_params["z_lo"]
    z_hi = extrap_params["z_hi"]
    delta_max = extrap_params["delta_max"]

    if z < z_lo:
        taper = float(np.clip(np.sqrt(mask_area / max(area_at_lo, 1.0)), 0.4, 1.0))
        dz = float(z) - z_lo
        raw = extrap_params["slope_lo"] * dz
        capped = delta_max * np.tanh(raw / np.maximum(np.abs(delta_max), 1e-12))
        return extrap_params["val_lo"] + capped * taper, "extrapolated", int(z_lo) - z

    if z > z_hi:
        taper = float(np.clip(np.sqrt(mask_area / max(area_at_hi, 1.0)), 0.4, 1.0))
        dz = float(z) - z_hi
        raw = extrap_params["slope_hi"] * dz
        capped = delta_max * np.tanh(raw / np.maximum(np.abs(delta_max), 1e-12))
        return extrap_params["val_hi"] + capped * taper, "extrapolated", z - int(z_hi)

    state_vec = spline(float(z))
    nearest_dist = min(abs(z - za) for za in extrap_params["z_anchors"])
    provenance = "interpolated" if nearest_dist > 0 else "manual"
    return state_vec, provenance, nearest_dist


def reconstruct_lines_from_state(
    state_vec: np.ndarray,
    geometry: MaskGeometry,
    image_size: int = TARGET_SIZE,
) -> dict[str, list[list[float]]]:
    """状態ベクトルから4本の線点列を復元する。"""
    j_r_norm = state_vec[0:2].copy()
    j_l_norm = state_vec[2:4].copy()
    doubled = state_vec[4:12].reshape(4, 2).copy()

    for i in range(4):
        norm = float(np.linalg.norm(doubled[i]))
        if norm > 1e-8:
            doubled[i] /= norm
        else:
            doubled[i] = np.array([1.0, 0.0])  # ノルム消失時は水平線にフォールバック

    j_r = denormalize_points(j_r_norm[np.newaxis], geometry, image_size)[0]
    j_l = denormalize_points(j_l_norm[np.newaxis], geometry, image_size)[0]

    # J_R は J_L より右側を保証
    if j_r[0] <= j_l[0]:
        margin = MIN_JUNCTION_X_SEP * float(geometry.scale[0]) * 0.5 + 1.0
        mid_x = (j_r[0] + j_l[0]) / 2.0
        j_r = np.array([mid_x + margin, j_r[1]])
        j_l = np.array([mid_x - margin, j_l[1]])

    junctions = [j_r, j_r, j_l, j_l]
    full_len = image_size * 1.2
    lines: dict[str, list[list[float]]] = {}

    for i, (line_key, junction) in enumerate(zip(LINE_KEYS, junctions)):
        alpha = 0.5 * np.arctan2(doubled[i, 1], doubled[i, 0])
        direction = np.array([np.cos(alpha), np.sin(alpha)])
        # junction を始点にする: _nearest_endpoint_junction が (junction, junction) = 距離0 を選び
        # 正しい junction を返す。中心に置くと clipping 後の端点が junction から離れて誤検出する。
        p_start = np.clip(junction, 0.0, float(image_size - 1))
        p_far = np.clip(junction + direction * full_len, 0.0, float(image_size - 1))
        # clipping で2点が一致する場合は逆方向へ延ばして最低限の長さを確保する
        if float(np.linalg.norm(p_far - p_start)) < 1.0:
            p_far = np.clip(junction - direction * full_len, 0.0, float(image_size - 1))
        lines[line_key] = [p_start.tolist(), p_far.tolist()]

    return lines


# ─── 内挿（旧方式: 正規化8点線形補間）────────────────────────────────────


def _align_polyline_direction(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """前スライスと対応する向きへ折れ線の点順を揃える。"""
    forward = float(np.mean(np.linalg.norm(points - reference, axis=1)))
    reversed_pts = points[::-1].copy()
    reverse = float(np.mean(np.linalg.norm(reversed_pts - reference, axis=1)))
    return reversed_pts if reverse < forward else points


def propagate_interior_slice(
    z_target: int,
    manual_lines: dict[int, dict],
    geometries: dict[int, MaskGeometry],
    point_count: int = 8,
) -> dict[str, list[list[float]]]:
    """アンカー間スライスを旧方式の線形補間で復元する。"""
    anchor_indices = sorted(manual_lines)
    lower_idx = max(z for z in anchor_indices if z <= z_target)
    upper_idx = min(z for z in anchor_indices if z >= z_target)
    if lower_idx == upper_idx:
        return manual_lines[lower_idx]
    ratio = (z_target - lower_idx) / (upper_idx - lower_idx)

    lines: dict[str, list[list[float]]] = {}
    for line_key in LINE_KEYS:
        sampled_lo = resample_polyline(manual_lines[lower_idx][line_key], point_count)
        sampled_hi = resample_polyline(manual_lines[upper_idx][line_key], point_count)
        norm_lo = normalize_points(sampled_lo, geometries[lower_idx])
        norm_hi = normalize_points(sampled_hi, geometries[upper_idx])
        norm_hi = _align_polyline_direction(norm_hi, norm_lo)
        interpolated = norm_lo * (1.0 - ratio) + norm_hi * ratio
        denormed = denormalize_points(interpolated, geometries[z_target])
        lines[line_key] = denormed.tolist()
    return lines


# ─── 評価 ─────────────────────────────────────────────────────────────


def resample_polyline(points: list, point_count: int) -> np.ndarray:
    """折れ線を弧長等間隔で再サンプリングする。"""
    pts = np.asarray(points, dtype=np.float64)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumul = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total = float(cumul[-1])
    if total <= 1e-6:
        raise ValueError("長さ0の折れ線は再サンプリングできません")
    s = np.linspace(0.0, total, point_count)
    result = np.empty((point_count, 2))
    result[:, 0] = np.interp(s, cumul, pts[:, 0])
    result[:, 1] = np.interp(s, cumul, pts[:, 1])
    return result


def line_angle_deg(points: list) -> float:
    """折れ線の主軸角度を0-180度で返す。"""
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)


def acute_angle_error_deg(first_angle: float, second_angle: float) -> float:
    """180度周期の2直線間の鋭角誤差を返す。"""
    difference = abs(first_angle - second_angle) % 180.0
    return min(difference, 180.0 - difference)


def matched_point_error(
    predicted: list,
    target: list,
    point_count: int,
) -> float:
    """点順を考慮した再サンプリング線間の平均距離を返す。"""
    p = resample_polyline(predicted, point_count)
    t = resample_polyline(target, point_count)
    forward = float(np.mean(np.linalg.norm(p - t, axis=1)))
    reverse = float(np.mean(np.linalg.norm(p - t[::-1], axis=1)))
    return min(forward, reverse)


def class_dice(predicted: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """指定領域クラスのDiceを計算する。"""
    pm = predicted == class_id
    tm = target == class_id
    denom = int(pm.sum() + tm.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pm, tm).sum()) / denom


def summarize_values(values: list[float]) -> dict[str, float]:
    """評価値リストを平均・中央値・95パーセンタイルへ要約する。"""
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
    }


def evaluate_leave_one_out(
    manual_lines: dict[int, dict],
    geometries: dict[int, MaskGeometry],
    masks: dict[int, np.ndarray],
    mask_areas: dict[int, float],
    trend_anchor_count: int,
    point_count: int = 8,
) -> dict[str, Any]:
    """内部アンカーを1枚ずつ隠して補間精度を評価する。"""
    anchor_indices = sorted(manual_lines)
    point_errors: list[float] = []
    angle_errors: list[float] = []
    dice_scores: dict[int, list[float]] = {c: [] for c in range(1, 5)}
    evaluated_slices: list[int] = []
    failures: list[dict[str, Any]] = []

    for held_out in anchor_indices[1:-1]:
        reduced = {z: l for z, l in manual_lines.items() if z != held_out}
        if not any(z < held_out for z in reduced) or not any(z > held_out for z in reduced):
            continue

        try:
            pred_lines = propagate_interior_slice(held_out, reduced, geometries, point_count)

            for line_key in LINE_KEYS:
                point_errors.append(
                    matched_point_error(pred_lines[line_key], manual_lines[held_out][line_key], point_count)
                )
                angle_errors.append(
                    acute_angle_error_deg(
                        line_angle_deg(pred_lines[line_key]),
                        line_angle_deg(manual_lines[held_out][line_key]),
                    )
                )

            pred_seg, _ = generate_region_mask(
                line_1=pred_lines["line_1"], line_2=pred_lines["line_2"],
                line_3=pred_lines["line_3"], line_4=pred_lines["line_4"],
                vertebra_mask=masks[held_out],
            )
            tgt_seg, _ = generate_region_mask(
                line_1=manual_lines[held_out]["line_1"], line_2=manual_lines[held_out]["line_2"],
                line_3=manual_lines[held_out]["line_3"], line_4=manual_lines[held_out]["line_4"],
                vertebra_mask=masks[held_out],
            )
            pred_lbl = np.argmax(pred_seg, axis=0)
            tgt_lbl = np.argmax(tgt_seg, axis=0)
            for c in range(1, 5):
                dice_scores[c].append(class_dice(pred_lbl, tgt_lbl, c))
            evaluated_slices.append(held_out)
        except Exception as error:
            failures.append({"slice": held_out, "reason": str(error)})

    all_dice = [s for cs in dice_scores.values() for s in cs]
    return {
        "evaluated_slices": evaluated_slices,
        "failure_count": len(failures),
        "failures": failures,
        "point_error_px": summarize_values(point_errors),
        "angle_error_deg": summarize_values(angle_errors),
        "region_dice": {
            "all_classes": summarize_values(all_dice),
            **{f"class_{c}": summarize_values(cs) for c, cs in dice_scores.items()},
        },
    }


# ─── ボリューム処理 ──────────────────────────────────────────────────────


def contiguous_valid_slices(mask_data: np.ndarray, min_mask_area: int) -> list[int]:
    """最大マスク面積スライスを含む連続した有効z範囲を返す。"""
    areas = np.sum(mask_data > 0, axis=(0, 1))
    peak_idx = int(np.argmax(areas))
    valid = areas >= min_mask_area
    if not valid[peak_idx]:
        return []
    lower = peak_idx
    while lower > 0 and valid[lower - 1]:
        lower -= 1
    upper = peak_idx
    while upper + 1 < len(valid) and valid[upper + 1]:
        upper += 1
    return list(range(lower, upper + 1))


def extract_png_slice(volume: np.ndarray, slice_idx: int, is_mask: bool) -> np.ndarray:
    """既存datasetと同じ変換で軸位断面PNG配列を作る。"""
    source_slice = volume[:, :, slice_idx].T
    cropped, _ = center_crop(source_slice, TARGET_SIZE)
    flipped = np.flipud(cropped)
    if is_mask:
        return (flipped > 0).astype(np.uint8) * 255
    return (apply_window(flipped) * 255).astype(np.uint8)


def parse_nrrd_vector(value: str) -> np.ndarray:
    """NRRDヘッダの3次元ベクトルを読み取る。"""
    match = re.fullmatch(r"\(([^)]+)\)", value.strip())
    if match is None:
        raise ValueError(f"NRRDベクトル形式が不正です: {value}")
    return np.asarray(
        [float(c) for c in match.group(1).split(",")],
        dtype=np.float64,
    )


def load_nrrd_as_nifti(path: Path) -> nib.Nifti1Image:
    """gzip/rawエンコードの3次元NRRDをNIfTI画像として読み込む。"""
    file_bytes = path.read_bytes()
    separator = b"\n\n"
    header_end = file_bytes.find(separator)
    if header_end < 0:
        separator = b"\r\n\r\n"
        header_end = file_bytes.find(separator)
    if header_end < 0:
        raise ValueError(f"NRRDヘッダ終端が見つかりません: {path}")

    header_text = file_bytes[:header_end].decode("ascii")
    payload = file_bytes[header_end + len(separator):]
    fields: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    sizes = tuple(int(v) for v in fields["sizes"].split())
    if len(sizes) != 3:
        raise ValueError(f"3次元以外のNRRDは未対応です: {path}")

    dtype_map = {
        "uchar": np.uint8, "unsigned char": np.uint8,
        "short": np.int16, "int16": np.int16,
        "ushort": np.uint16, "unsigned short": np.uint16,
        "float": np.float32, "double": np.float64,
    }
    dtype = np.dtype(dtype_map[fields["type"]])
    dtype = dtype.newbyteorder(">" if fields.get("endian", "little") == "big" else "<")

    encoding = fields.get("encoding", "raw").lower()
    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise ValueError(f"未対応のNRRD encodingです: {encoding}")

    data = np.frombuffer(payload, dtype=dtype).reshape(sizes, order="F")
    directions = re.findall(r"\([^)]+\)", fields["space directions"])
    if len(directions) != 3:
        raise ValueError(f"NRRD space directionsが不正です: {path}")

    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4, dtype=np.float64)
    for axis, raw_dir in enumerate(directions):
        affine[:3, axis] = lps_to_ras @ parse_nrrd_vector(raw_dir)
    affine[:3, 3] = lps_to_ras @ parse_nrrd_vector(fields["space origin"])
    return nib.Nifti1Image(data, affine)


def load_source_volumes(
    source_dir: Path,
    fallback_dir: Path | None,
) -> tuple[nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage, str]:
    """NIfTI、NRRD、予備NIfTIの順にCTとマスクを読み込む。"""
    try:
        ct_nii, mask_nii, _ = load_vertebra_data(source_dir)
        return ct_nii, mask_nii, "annotation_data_nifti"
    except FileNotFoundError:
        ct_nrrd = next(iter(sorted(source_dir.glob("ct*.nrrd"))), None)
        mask_nrrd = next(iter(sorted(source_dir.glob("mask*.nrrd"))), None)
        if ct_nrrd is not None and mask_nrrd is not None:
            return load_nrrd_as_nifti(ct_nrrd), load_nrrd_as_nifti(mask_nrrd), "annotation_data_nrrd"
        if fallback_dir is not None:
            ct_path = fallback_dir / "ct.nii.gz"
            mask_path = fallback_dir / "mask.nii.gz"
            if ct_path.exists() and mask_path.exists():
                return nib.load(ct_path), nib.load(mask_path), "predata_simple_fallback"
        raise


def save_json(path: Path, payload: object) -> None:
    """JSONをUTF-8で保存する。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_vertebra(
    source_dir: Path,
    fallback_source_dir: Path | None,
    anchor_dir: Path,
    output_dir: Path,
    trend_anchor_count: int,
    confidence_decay_mm: float,
    min_mask_area: int,
) -> dict[str, Any]:
    """1椎骨について全z線伝播と4領域マスク生成を行う。"""
    ct_nii, mask_nii, source_mode = load_source_volumes(source_dir, fallback_source_dir)
    if not np.allclose(ct_nii.affine, mask_nii.affine) or ct_nii.shape != mask_nii.shape:
        mask_nii = nibabel.processing.resample_from_to(mask_nii, ct_nii, order=0)

    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    valid_slices = contiguous_valid_slices(mask_data, min_mask_area=min_mask_area)
    if not valid_slices:
        raise ValueError("有効な椎骨マスクスライスがありません")

    manual_lines = load_manual_lines(anchor_dir / "lines.json")
    missing_anchors = sorted(set(manual_lines) - set(valid_slices))
    if missing_anchors:
        raise ValueError(f"有効z範囲外の手動アンカーがあります: {missing_anchors}")

    images: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    geometries: dict[int, MaskGeometry] = {}
    mask_areas: dict[int, float] = {}
    for slice_idx in valid_slices:
        images[slice_idx] = extract_png_slice(ct_data, slice_idx, is_mask=False)
        masks[slice_idx] = extract_png_slice(mask_data, slice_idx, is_mask=True)
        geometries[slice_idx] = compute_mask_geometry(masks[slice_idx])
        mask_areas[slice_idx] = float(np.sum(masks[slice_idx] > 0))

    anchor_states = {
        z: extract_slice_state(manual_lines[z], geometries[z])
        for z in manual_lines
        if z in geometries
    }
    spline, extrap_params = build_smooth_trajectory(anchor_states, trend_anchor_count)
    area_at_lo = mask_areas.get(int(extrap_params["z_lo"]), 1.0)
    area_at_hi = mask_areas.get(int(extrap_params["z_hi"]), 1.0)

    z_spacing_mm = float(ct_nii.header.get_zooms()[2])
    if z_spacing_mm <= 0:
        z_spacing_mm = DEFAULT_Z_SPACING_MM

    for dirname in ("images", "masks", "gt_masks", "gt_overlays"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    output_lines: dict[str, dict] = {}
    provenance_log: dict[str, dict] = {}
    failures: list[dict[str, Any]] = []
    counts = {"manual": 0, "interpolated": 0, "extrapolated": 0}

    z_lo_anchor = min(manual_lines)
    z_hi_anchor = max(manual_lines)

    for slice_idx in valid_slices:
        if slice_idx in manual_lines:
            lines_out = manual_lines[slice_idx]
            prov, nearest_dist = "manual", 0
        elif z_lo_anchor < slice_idx < z_hi_anchor:
            lines_out = propagate_interior_slice(slice_idx, manual_lines, geometries)
            lower = max(z for z in manual_lines if z < slice_idx)
            upper = min(z for z in manual_lines if z > slice_idx)
            nearest_dist = min(slice_idx - lower, upper - slice_idx)
            prov = "interpolated"
        else:
            state_vec, prov, nearest_dist = evaluate_trajectory(
                spline, extrap_params, slice_idx,
                mask_areas.get(slice_idx, 1.0), area_at_lo, area_at_hi,
            )
            lines_out = reconstruct_lines_from_state(state_vec, geometries[slice_idx])

        confidence = confidence_from_provenance(prov, nearest_dist, z_spacing_mm, confidence_decay_mm)
        output_lines[str(slice_idx)] = lines_out
        counts[prov] += 1

        Image.fromarray(images[slice_idx]).save(output_dir / "images" / f"slice_{slice_idx:03d}.png")
        Image.fromarray(masks[slice_idx]).save(output_dir / "masks" / f"slice_{slice_idx:03d}.png")

        warnings: list[str] = []
        fallback_type = ""
        mask_generated = False
        try:
            seg_mask, debug_info = generate_region_mask(
                line_1=lines_out["line_1"], line_2=lines_out["line_2"],
                line_3=lines_out["line_3"], line_4=lines_out["line_4"],
                vertebra_mask=masks[slice_idx],
            )
            validation = validate_region_mask(seg_mask, masks[slice_idx])
            warnings = [str(w) for w in validation.get("warnings", [])]
            fallback_type = str(debug_info.get("fallback_type", ""))
            if validation.get("hard_fail"):
                failures.append({"slice": slice_idx, "reason": f"hard_fail: {warnings}"})
            else:
                label_image = np.argmax(seg_mask, axis=0).astype(np.uint8)
                cv2.imwrite(str(output_dir / "gt_masks" / f"slice_{slice_idx:03d}.png"), label_image)
                overlay = build_overlay_image(images[slice_idx], label_image)
                cv2.imwrite(str(output_dir / "gt_overlays" / f"slice_{slice_idx:03d}.png"), overlay)
                mask_generated = True
        except Exception as error:
            warnings = [str(error)]
            failures.append({"slice": slice_idx, "reason": str(error)})

        provenance_log[str(slice_idx)] = {
            "source": prov,
            "nearest_anchor_distance_slices": nearest_dist,
            "nearest_anchor_distance_mm": round(nearest_dist * z_spacing_mm, 4),
            "confidence": round(confidence, 6),
            "mask_generated": mask_generated,
            "fallback_type": fallback_type,
            "qc_warnings": warnings,
        }

    save_json(output_dir / "lines.json", output_lines)
    save_json(output_dir / "line_provenance.json", provenance_log)

    leave_one_out = evaluate_leave_one_out(
        manual_lines=manual_lines,
        geometries=geometries,
        masks=masks,
        mask_areas=mask_areas,
        trend_anchor_count=trend_anchor_count,
    )

    report = {
        "source_dir": str(source_dir),
        "source_mode": source_mode,
        "anchor_dir": str(anchor_dir),
        "valid_z_range": [valid_slices[0], valid_slices[-1]],
        "slice_count": len(valid_slices),
        "anchor_range": [min(manual_lines), max(manual_lines)],
        "anchor_count": len(manual_lines),
        "counts": counts,
        "mask_generation_failures": failures,
        "leave_one_out": leave_one_out,
        "parameters": {
            "trend_anchor_count": trend_anchor_count,
            "confidence_decay_mm": confidence_decay_mm,
            "min_mask_area": min_mask_area,
            "z_spacing_mm": z_spacing_mm,
            "delta_max_junction": DELTA_MAX_JUNCTION,
            "delta_max_angle_doubled": DELTA_MAX_ANGLE_DOUBLED,
        },
    }
    save_json(output_dir / "generation_report.json", report)
    return report


def collect_vertebra_pairs(
    source_root: Path,
    anchor_root: Path,
) -> list[tuple[str, str]]:
    """元ボリュームと手動線の両方が存在する症例・椎骨を列挙する。"""
    pairs: list[tuple[str, str]] = []
    for lines_path in sorted(anchor_root.glob("sample*/C*/lines.json")):
        sample = lines_path.parent.parent.name
        vertebra = lines_path.parent.name
        if (source_root / sample / vertebra).is_dir():
            pairs.append((sample, vertebra))
    return pairs


def summarize_batch_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """椎骨単位レポートを全体集計へまとめる。"""
    total_slices = sum(int(r.get("slice_count", 0)) for r in reports)
    total_anchors = sum(int(r.get("anchor_count", 0)) for r in reports)
    total_mask_failures = sum(len(r.get("mask_generation_failures", [])) for r in reports)
    counts = {"manual": 0, "interpolated": 0, "extrapolated": 0}
    angle_means: list[float] = []
    dice_means: list[float] = []

    for report in reports:
        for key in counts:
            counts[key] += int(report.get("counts", {}).get(key, 0))
        loo = report.get("leave_one_out", {})
        angle_mean = loo.get("angle_error_deg", {}).get("mean")
        dice_mean = loo.get("region_dice", {}).get("all_classes", {}).get("mean")
        if isinstance(angle_mean, int | float) and math.isfinite(float(angle_mean)):
            angle_means.append(float(angle_mean))
        if isinstance(dice_mean, int | float) and math.isfinite(float(dice_mean)):
            dice_means.append(float(dice_mean))

    return {
        "success_count": len(reports),
        "total_slices": total_slices,
        "total_anchors": total_anchors,
        "counts": counts,
        "mask_generation_failure_count": total_mask_failures,
        "vertebra_mean_leave_one_out_angle_deg": summarize_values(angle_means),
        "vertebra_mean_leave_one_out_region_dice": summarize_values(dice_means),
    }


def process_all_vertebrae(
    source_root: Path,
    fallback_source_root: Path,
    anchor_root: Path,
    output_root: Path,
    trend_anchor_count: int,
    confidence_decay_mm: float,
    min_mask_area: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """全症例・全椎骨を処理し、中断再開可能な集計レポートを保存する。"""
    pairs = collect_vertebra_pairs(source_root, anchor_root)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skipped = 0
    batch_report_path = output_root / "batch_report.json"
    output_root.mkdir(parents=True, exist_ok=True)

    for index, (sample, vertebra) in enumerate(pairs, start=1):
        output_dir = output_root / sample / vertebra
        report_path = output_dir / "generation_report.json"
        print(f"[{index}/{len(pairs)}] {sample}/{vertebra}", flush=True)

        try:
            if skip_existing and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                skipped += 1
            else:
                report = process_vertebra(
                    source_dir=source_root / sample / vertebra,
                    fallback_source_dir=fallback_source_root / sample / vertebra,
                    anchor_dir=anchor_root / sample / vertebra,
                    output_dir=output_dir,
                    trend_anchor_count=trend_anchor_count,
                    confidence_decay_mm=confidence_decay_mm,
                    min_mask_area=min_mask_area,
                )
            reports.append({"sample": sample, "vertebra": vertebra, **report})
        except Exception as error:
            failures.append({"sample": sample, "vertebra": vertebra, "reason": str(error)})

        batch_report = {
            "target_count": len(pairs),
            "processed_count": index,
            "skipped_existing_count": skipped,
            "summary": summarize_batch_reports(reports),
            "failures": failures,
            "reports": reports,
        }
        save_json(batch_report_path, batch_report)

    return batch_report


def main() -> None:
    """指定椎骨または全症例の全z線伝播を実行する。"""
    parser = argparse.ArgumentParser(
        description="少数スライスの4本線を全zへ平滑補間・外挿し、4領域マスクを生成する"
    )
    parser.add_argument("--sample", help="例: sample1")
    parser.add_argument("--vertebra", help="例: C3")
    parser.add_argument("--all", action="store_true", help="全症例・全椎骨を処理する")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--source_root", type=Path, default=ROOT_DIR / "data" / "annotation_data")
    parser.add_argument("--anchor_root", type=Path, default=ROOT_DIR / "data" / "dataset")
    parser.add_argument("--fallback_source_root", type=Path, default=ROOT_DIR / "data" / "predata_simple")
    parser.add_argument("--output_root", type=Path, default=ROOT_DIR / "data" / "dataset_zprop")
    parser.add_argument("--trend_anchors", type=int, default=DEFAULT_TREND_ANCHORS)
    parser.add_argument("--confidence_decay_mm", type=float, default=DEFAULT_CONFIDENCE_DECAY_MM)
    parser.add_argument("--min_mask_area", type=int, default=MIN_MASK_AREA)
    args = parser.parse_args()

    if args.all:
        report = process_all_vertebrae(
            source_root=args.source_root,
            fallback_source_root=args.fallback_source_root,
            anchor_root=args.anchor_root,
            output_root=args.output_root,
            trend_anchor_count=args.trend_anchors,
            confidence_decay_mm=args.confidence_decay_mm,
            min_mask_area=args.min_mask_area,
            skip_existing=args.skip_existing,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    if not args.sample or not args.vertebra:
        parser.error("--all または --sample と --vertebra を指定してください")

    report = process_vertebra(
        source_dir=args.source_root / args.sample / args.vertebra,
        fallback_source_dir=args.fallback_source_root / args.sample / args.vertebra,
        anchor_dir=args.anchor_root / args.sample / args.vertebra,
        output_dir=args.output_root / args.sample / args.vertebra,
        trend_anchor_count=args.trend_anchors,
        confidence_decay_mm=args.confidence_decay_mm,
        min_mask_area=args.min_mask_area,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
