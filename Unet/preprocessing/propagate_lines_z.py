from __future__ import annotations

import argparse
import bisect
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Unet.preprocessing.convert_to_png import (  # noqa: E402
    apply_window,
    center_crop,
    load_vertebra_data,
)
from Unet.preprocessing.generate_region_mask import (  # noqa: E402
    generate_region_mask,
    validate_region_mask,
)
from Unet.preprocessing.preprocess_all import build_overlay_image  # noqa: E402

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
TARGET_SIZE = 224
MIN_MASK_AREA = 50
DEFAULT_RESAMPLE_POINTS = 8
DEFAULT_TREND_ANCHORS = 4
DEFAULT_MAX_EXTRAPOLATION = 0.35
DEFAULT_CONFIDENCE_DECAY_MM = 4.0
DEFAULT_Z_SPACING_MM = 0.4
MIN_OUTPUT_LINE_LENGTH = 4.0


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


def is_valid_line_entry(entry: object) -> bool:
    """4本の有効な折れ線を持つか判定する。"""
    if not isinstance(entry, dict):
        return False
    for line_key in LINE_KEYS:
        points = entry.get(line_key)
        if not isinstance(points, list) or len(points) < 2:
            return False
        if any(not isinstance(point, list | tuple) or len(point) != 2 for point in points):
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
            line_key: [[float(point[0]), float(point[1])] for point in entry[line_key]]
            for line_key in LINE_KEYS
        }

    if len(result) < 2:
        raise ValueError("線の伝播には2枚以上の手動アノテーションが必要です")
    return result


def resample_polyline(points: list[list[float]], point_count: int) -> np.ndarray:
    """折れ線を正規化弧長上の固定点数へ再サンプリングする。"""
    points_array = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        raise ValueError("長さ0の折れ線は再サンプリングできません")

    sample_positions = np.linspace(0.0, total_length, point_count)
    result = np.empty((point_count, 2), dtype=np.float64)
    result[:, 0] = np.interp(sample_positions, cumulative, points_array[:, 0])
    result[:, 1] = np.interp(sample_positions, cumulative, points_array[:, 1])
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
    """正規化座標を対象スライスの画像座標へ戻す。"""
    points = np.asarray(normalized_points, dtype=np.float64) * geometry.scale + geometry.centroid
    return np.clip(points, 0.0, float(image_size - 1))


def stabilize_polyline(
    points: np.ndarray,
    geometry: MaskGeometry,
    image_size: int = TARGET_SIZE,
) -> np.ndarray:
    """画像端で縮退した外挿線に最小長を与える。"""
    points_array = np.asarray(points, dtype=np.float64)
    if float(np.linalg.norm(points_array[-1] - points_array[0])) >= MIN_OUTPUT_LINE_LENGTH:
        return points_array

    centered = points_array - points_array.mean(axis=0)
    if np.allclose(centered, 0.0):
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

    half_length = MIN_OUTPUT_LINE_LENGTH / 2.0
    margin = half_length + 1.0
    center = np.clip(
        points_array.mean(axis=0),
        margin,
        float(image_size - 1) - margin,
    )
    if not np.all(np.isfinite(center)):
        center = geometry.centroid
    endpoints = np.stack(
        [center - direction * half_length, center + direction * half_length],
        axis=0,
    )
    return np.linspace(endpoints[0], endpoints[1], len(points_array))


def align_polyline_direction(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """前スライスと対応する向きへ折れ線の点順を揃える。"""
    forward_error = float(np.mean(np.linalg.norm(points - reference, axis=1)))
    reversed_points = points[::-1].copy()
    reverse_error = float(np.mean(np.linalg.norm(reversed_points - reference, axis=1)))
    return reversed_points if reverse_error < forward_error else points


def prepare_anchor_lines(
    manual_lines: dict[int, dict[str, list[list[float]]]],
    geometries: dict[int, MaskGeometry],
    point_count: int,
) -> dict[str, dict[int, np.ndarray]]:
    """手動線を正規化・再サンプリングし、z方向の点対応を揃える。"""
    prepared = {line_key: {} for line_key in LINE_KEYS}
    anchor_indices = sorted(manual_lines)

    for line_key in LINE_KEYS:
        previous: np.ndarray | None = None
        for slice_idx in anchor_indices:
            sampled = resample_polyline(manual_lines[slice_idx][line_key], point_count)
            normalized = normalize_points(sampled, geometries[slice_idx])
            if previous is not None:
                normalized = align_polyline_direction(normalized, previous)
            prepared[line_key][slice_idx] = normalized
            previous = normalized
    return prepared


def estimate_tail_slope(
    anchor_values: dict[int, np.ndarray],
    use_lower_tail: bool,
    trend_anchor_count: int,
) -> np.ndarray:
    """端側の複数アンカーから正規化線のz方向傾きを推定する。"""
    anchor_indices = sorted(anchor_values)
    selected = (
        anchor_indices[:trend_anchor_count]
        if use_lower_tail
        else anchor_indices[-trend_anchor_count:]
    )
    z_values = np.asarray(selected, dtype=np.float64)
    centered_z = z_values - z_values.mean()
    denominator = float(np.sum(centered_z**2))
    if denominator <= 1e-6:
        return np.zeros_like(anchor_values[selected[0]])

    stacked = np.stack([anchor_values[slice_idx] for slice_idx in selected], axis=0)
    centered_values = stacked - stacked.mean(axis=0, keepdims=True)
    return np.sum(centered_z[:, None, None] * centered_values, axis=0) / denominator


def constrain_extrapolation(
    displacement: np.ndarray,
    max_normalized_displacement: float,
) -> np.ndarray:
    """各制御点の外挿変位を上限内へ収める。"""
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    scale = np.minimum(
        1.0,
        max_normalized_displacement / np.maximum(norms, 1e-12),
    )
    return displacement * scale


def propagate_normalized_line(
    anchor_values: dict[int, np.ndarray],
    target_slice_idx: int,
    trend_anchor_count: int = DEFAULT_TREND_ANCHORS,
    max_extrapolation: float = DEFAULT_MAX_EXTRAPOLATION,
) -> tuple[np.ndarray, str, int]:
    """アンカー線から対象zの正規化線を補間または制約付き外挿する。"""
    anchor_indices = sorted(anchor_values)
    if target_slice_idx in anchor_values:
        return anchor_values[target_slice_idx].copy(), "manual", 0

    insertion_idx = bisect.bisect_left(anchor_indices, target_slice_idx)
    if 0 < insertion_idx < len(anchor_indices):
        lower_idx = anchor_indices[insertion_idx - 1]
        upper_idx = anchor_indices[insertion_idx]
        ratio = (target_slice_idx - lower_idx) / (upper_idx - lower_idx)
        interpolated = (
            anchor_values[lower_idx] * (1.0 - ratio)
            + anchor_values[upper_idx] * ratio
        )
        nearest_distance = min(target_slice_idx - lower_idx, upper_idx - target_slice_idx)
        return interpolated, "interpolated", nearest_distance

    use_lower_tail = insertion_idx == 0
    outer_idx = anchor_indices[0] if use_lower_tail else anchor_indices[-1]
    slope = estimate_tail_slope(
        anchor_values,
        use_lower_tail=use_lower_tail,
        trend_anchor_count=trend_anchor_count,
    )
    raw_displacement = slope * float(target_slice_idx - outer_idx)
    displacement = constrain_extrapolation(raw_displacement, max_extrapolation)
    extrapolated = anchor_values[outer_idx] + displacement
    return extrapolated, "extrapolated", abs(target_slice_idx - outer_idx)


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


def propagate_all_lines(
    target_slice_idx: int,
    anchor_lines: dict[str, dict[int, np.ndarray]],
    target_geometry: MaskGeometry,
    trend_anchor_count: int,
    max_extrapolation: float,
    z_spacing_mm: float,
    confidence_decay_mm: float,
) -> PropagatedLines:
    """4本の線を対象スライスへ伝播する。"""
    output_lines: dict[str, list[list[float]]] = {}
    provenance = "manual"
    nearest_anchor_distance = 0

    for line_key in LINE_KEYS:
        normalized, line_provenance, line_distance = propagate_normalized_line(
            anchor_lines[line_key],
            target_slice_idx=target_slice_idx,
            trend_anchor_count=trend_anchor_count,
            max_extrapolation=max_extrapolation,
        )
        denormalized = denormalize_points(normalized, target_geometry)
        output_lines[line_key] = stabilize_polyline(
            denormalized,
            target_geometry,
        ).tolist()
        provenance = line_provenance
        nearest_anchor_distance = line_distance

    confidence = confidence_from_provenance(
        provenance=provenance,
        nearest_anchor_distance=nearest_anchor_distance,
        z_spacing_mm=z_spacing_mm,
        decay_mm=confidence_decay_mm,
    )
    return PropagatedLines(
        lines=output_lines,
        provenance=provenance,
        nearest_anchor_distance=nearest_anchor_distance,
        confidence=confidence,
    )


def line_angle_deg(points: list[list[float]]) -> float:
    """折れ線の主軸角度を0-180度で返す。"""
    points_array = np.asarray(points, dtype=np.float64)
    centered = points_array - points_array.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)


def acute_angle_error_deg(first_angle: float, second_angle: float) -> float:
    """180度周期の2直線間の鋭角誤差を返す。"""
    difference = abs(first_angle - second_angle) % 180.0
    return min(difference, 180.0 - difference)


def matched_point_error(
    predicted: list[list[float]],
    target: list[list[float]],
    point_count: int,
) -> float:
    """点順を考慮した再サンプリング線間の平均距離を返す。"""
    predicted_points = resample_polyline(predicted, point_count)
    target_points = resample_polyline(target, point_count)
    forward = float(np.mean(np.linalg.norm(predicted_points - target_points, axis=1)))
    reverse = float(
        np.mean(np.linalg.norm(predicted_points - target_points[::-1], axis=1))
    )
    return min(forward, reverse)


def class_dice(predicted: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """指定領域クラスのDiceを計算する。"""
    predicted_mask = predicted == class_id
    target_mask = target == class_id
    denominator = int(predicted_mask.sum() + target_mask.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(predicted_mask, target_mask).sum())
    return 2.0 * intersection / denominator


def summarize_values(values: list[float]) -> dict[str, float]:
    """評価値リストを平均・中央値・95パーセンタイルへ要約する。"""
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(values_array.mean()), 6),
        "median": round(float(np.median(values_array)), 6),
        "p95": round(float(np.percentile(values_array, 95)), 6),
    }


def evaluate_leave_one_out(
    manual_lines: dict[int, dict[str, list[list[float]]]],
    geometries: dict[int, MaskGeometry],
    masks: dict[int, np.ndarray],
    point_count: int,
    trend_anchor_count: int,
    max_extrapolation: float,
) -> dict[str, Any]:
    """内部アンカーを1枚ずつ隠して補間精度を評価する。"""
    anchor_indices = sorted(manual_lines)
    point_errors: list[float] = []
    angle_errors: list[float] = []
    dice_scores: dict[int, list[float]] = {class_id: [] for class_id in range(1, 5)}
    evaluated_slices: list[int] = []
    failures: list[dict[str, Any]] = []

    for held_out_idx in anchor_indices[1:-1]:
        reduced_lines = {
            slice_idx: lines
            for slice_idx, lines in manual_lines.items()
            if slice_idx != held_out_idx
        }
        lower_exists = any(slice_idx < held_out_idx for slice_idx in reduced_lines)
        upper_exists = any(slice_idx > held_out_idx for slice_idx in reduced_lines)
        if not lower_exists or not upper_exists:
            continue

        try:
            reduced_anchors = prepare_anchor_lines(
                manual_lines=reduced_lines,
                geometries=geometries,
                point_count=point_count,
            )
            predicted = propagate_all_lines(
                target_slice_idx=held_out_idx,
                anchor_lines=reduced_anchors,
                target_geometry=geometries[held_out_idx],
                trend_anchor_count=trend_anchor_count,
                max_extrapolation=max_extrapolation,
                z_spacing_mm=DEFAULT_Z_SPACING_MM,
                confidence_decay_mm=DEFAULT_CONFIDENCE_DECAY_MM,
            )

            for line_key in LINE_KEYS:
                point_errors.append(
                    matched_point_error(
                        predicted.lines[line_key],
                        manual_lines[held_out_idx][line_key],
                        point_count=point_count,
                    )
                )
                angle_errors.append(
                    acute_angle_error_deg(
                        line_angle_deg(predicted.lines[line_key]),
                        line_angle_deg(manual_lines[held_out_idx][line_key]),
                    )
                )

            predicted_seg, _ = generate_region_mask(
                line_1=predicted.lines["line_1"],
                line_2=predicted.lines["line_2"],
                line_3=predicted.lines["line_3"],
                line_4=predicted.lines["line_4"],
                vertebra_mask=masks[held_out_idx],
            )
            target_seg, _ = generate_region_mask(
                line_1=manual_lines[held_out_idx]["line_1"],
                line_2=manual_lines[held_out_idx]["line_2"],
                line_3=manual_lines[held_out_idx]["line_3"],
                line_4=manual_lines[held_out_idx]["line_4"],
                vertebra_mask=masks[held_out_idx],
            )
            predicted_labels = np.argmax(predicted_seg, axis=0)
            target_labels = np.argmax(target_seg, axis=0)
            for class_id in range(1, 5):
                dice_scores[class_id].append(
                    class_dice(predicted_labels, target_labels, class_id)
                )
            evaluated_slices.append(held_out_idx)
        except Exception as error:
            failures.append({"slice": held_out_idx, "reason": str(error)})

    all_dice = [
        score
        for class_scores in dice_scores.values()
        for score in class_scores
    ]
    return {
        "evaluated_slices": evaluated_slices,
        "failure_count": len(failures),
        "failures": failures,
        "point_error_px": summarize_values(point_errors),
        "angle_error_deg": summarize_values(angle_errors),
        "region_dice": {
            "all_classes": summarize_values(all_dice),
            **{
                f"class_{class_id}": summarize_values(class_scores)
                for class_id, class_scores in dice_scores.items()
            },
        },
    }


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


def extract_png_slice(
    volume: np.ndarray,
    slice_idx: int,
    is_mask: bool,
) -> np.ndarray:
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
        [float(component) for component in match.group(1).split(",")],
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

    sizes = tuple(int(value) for value in fields["sizes"].split())
    if len(sizes) != 3:
        raise ValueError(f"3次元以外のNRRDは未対応です: {path}")

    dtype_map = {
        "uchar": np.uint8,
        "unsigned char": np.uint8,
        "short": np.int16,
        "int16": np.int16,
        "ushort": np.uint16,
        "unsigned short": np.uint16,
        "float": np.float32,
        "double": np.float64,
    }
    dtype = np.dtype(dtype_map[fields["type"]])
    if fields.get("endian", "little") == "big":
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

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
    for axis, raw_direction in enumerate(directions):
        affine[:3, axis] = lps_to_ras @ parse_nrrd_vector(raw_direction)
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
            return (
                load_nrrd_as_nifti(ct_nrrd),
                load_nrrd_as_nifti(mask_nrrd),
                "annotation_data_nrrd",
            )
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
    point_count: int,
    trend_anchor_count: int,
    max_extrapolation: float,
    confidence_decay_mm: float,
    min_mask_area: int,
) -> dict[str, Any]:
    """1椎骨について全z線伝播と4領域マスク生成を行う。"""
    ct_nii, mask_nii, source_mode = load_source_volumes(
        source_dir,
        fallback_source_dir,
    )
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
    for slice_idx in valid_slices:
        images[slice_idx] = extract_png_slice(ct_data, slice_idx, is_mask=False)
        masks[slice_idx] = extract_png_slice(mask_data, slice_idx, is_mask=True)
        geometries[slice_idx] = compute_mask_geometry(masks[slice_idx])

    anchor_lines = prepare_anchor_lines(
        manual_lines=manual_lines,
        geometries=geometries,
        point_count=point_count,
    )
    z_spacing_mm = float(ct_nii.header.get_zooms()[2])
    if z_spacing_mm <= 0:
        z_spacing_mm = DEFAULT_Z_SPACING_MM

    for dirname in ("images", "masks", "gt_masks", "gt_overlays"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    output_lines: dict[str, dict[str, list[list[float]]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    counts = {"manual": 0, "interpolated": 0, "extrapolated": 0}

    for slice_idx in valid_slices:
        if slice_idx in manual_lines:
            propagated = PropagatedLines(
                lines=manual_lines[slice_idx],
                provenance="manual",
                nearest_anchor_distance=0,
                confidence=1.0,
            )
        else:
            propagated = propagate_all_lines(
                target_slice_idx=slice_idx,
                anchor_lines=anchor_lines,
                target_geometry=geometries[slice_idx],
                trend_anchor_count=trend_anchor_count,
                max_extrapolation=max_extrapolation,
                z_spacing_mm=z_spacing_mm,
                confidence_decay_mm=confidence_decay_mm,
            )
        slice_key = str(slice_idx)
        output_lines[slice_key] = propagated.lines
        counts[propagated.provenance] += 1

        image_path = output_dir / "images" / f"slice_{slice_idx:03d}.png"
        mask_path = output_dir / "masks" / f"slice_{slice_idx:03d}.png"
        Image.fromarray(images[slice_idx]).save(image_path)
        Image.fromarray(masks[slice_idx]).save(mask_path)

        warnings: list[str] = []
        fallback_type = ""
        mask_generated = False
        try:
            seg_mask, debug_info = generate_region_mask(
                line_1=propagated.lines["line_1"],
                line_2=propagated.lines["line_2"],
                line_3=propagated.lines["line_3"],
                line_4=propagated.lines["line_4"],
                vertebra_mask=masks[slice_idx],
            )
            validation = validate_region_mask(seg_mask, masks[slice_idx])
            warnings = [str(item) for item in validation.get("warnings", [])]
            fallback_type = str(debug_info.get("fallback_type", ""))
            label_image = np.argmax(seg_mask, axis=0).astype(np.uint8)
            gt_mask_path = output_dir / "gt_masks" / f"slice_{slice_idx:03d}.png"
            cv2.imwrite(str(gt_mask_path), label_image)
            overlay = build_overlay_image(images[slice_idx], label_image)
            cv2.imwrite(
                str(output_dir / "gt_overlays" / f"slice_{slice_idx:03d}.png"),
                overlay,
            )
            mask_generated = True
        except Exception as error:
            warnings = [str(error)]
            failures.append({"slice": slice_idx, "reason": str(error)})

        provenance[slice_key] = {
            "source": propagated.provenance,
            "nearest_anchor_distance_slices": propagated.nearest_anchor_distance,
            "nearest_anchor_distance_mm": round(
                propagated.nearest_anchor_distance * z_spacing_mm,
                4,
            ),
            "confidence": round(propagated.confidence, 6),
            "mask_generated": mask_generated,
            "fallback_type": fallback_type,
            "qc_warnings": warnings,
        }

    save_json(output_dir / "lines.json", output_lines)
    save_json(output_dir / "line_provenance.json", provenance)
    leave_one_out = evaluate_leave_one_out(
        manual_lines=manual_lines,
        geometries=geometries,
        masks=masks,
        point_count=point_count,
        trend_anchor_count=trend_anchor_count,
        max_extrapolation=max_extrapolation,
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
            "point_count": point_count,
            "trend_anchor_count": trend_anchor_count,
            "max_extrapolation": max_extrapolation,
            "confidence_decay_mm": confidence_decay_mm,
            "min_mask_area": min_mask_area,
            "z_spacing_mm": z_spacing_mm,
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
    total_slices = sum(int(report.get("slice_count", 0)) for report in reports)
    total_anchors = sum(int(report.get("anchor_count", 0)) for report in reports)
    total_mask_failures = sum(
        len(report.get("mask_generation_failures", []))
        for report in reports
    )
    counts = {"manual": 0, "interpolated": 0, "extrapolated": 0}
    angle_means: list[float] = []
    dice_means: list[float] = []

    for report in reports:
        report_counts = report.get("counts", {})
        for key in counts:
            counts[key] += int(report_counts.get(key, 0))

        leave_one_out = report.get("leave_one_out", {})
        angle_mean = leave_one_out.get("angle_error_deg", {}).get("mean")
        dice_mean = (
            leave_one_out.get("region_dice", {})
            .get("all_classes", {})
            .get("mean")
        )
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
    point_count: int,
    trend_anchor_count: int,
    max_extrapolation: float,
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
                    point_count=point_count,
                    trend_anchor_count=trend_anchor_count,
                    max_extrapolation=max_extrapolation,
                    confidence_decay_mm=confidence_decay_mm,
                    min_mask_area=min_mask_area,
                )
            reports.append({"sample": sample, "vertebra": vertebra, **report})
        except Exception as error:
            failures.append(
                {
                    "sample": sample,
                    "vertebra": vertebra,
                    "reason": str(error),
                }
            )

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
        description="少数スライスの4本線を全zへ補間・外挿し、4領域マスクを生成する"
    )
    parser.add_argument("--sample", help="例: sample1")
    parser.add_argument("--vertebra", help="例: C3")
    parser.add_argument("--all", action="store_true", help="全症例・全椎骨を処理する")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="generation_report.jsonがある椎骨を再利用する",
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=ROOT_DIR / "annotation_data",
    )
    parser.add_argument(
        "--anchor_root",
        type=Path,
        default=ROOT_DIR / "dataset",
    )
    parser.add_argument(
        "--fallback_source_root",
        type=Path,
        default=ROOT_DIR / "predata_simple",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT_DIR / "dataset_zprop",
    )
    parser.add_argument("--point_count", type=int, default=DEFAULT_RESAMPLE_POINTS)
    parser.add_argument("--trend_anchors", type=int, default=DEFAULT_TREND_ANCHORS)
    parser.add_argument(
        "--max_extrapolation",
        type=float,
        default=DEFAULT_MAX_EXTRAPOLATION,
    )
    parser.add_argument(
        "--confidence_decay_mm",
        type=float,
        default=DEFAULT_CONFIDENCE_DECAY_MM,
    )
    parser.add_argument("--min_mask_area", type=int, default=MIN_MASK_AREA)
    args = parser.parse_args()

    if args.all:
        report = process_all_vertebrae(
            source_root=args.source_root,
            fallback_source_root=args.fallback_source_root,
            anchor_root=args.anchor_root,
            output_root=args.output_root,
            point_count=args.point_count,
            trend_anchor_count=args.trend_anchors,
            max_extrapolation=args.max_extrapolation,
            confidence_decay_mm=args.confidence_decay_mm,
            min_mask_area=args.min_mask_area,
            skip_existing=args.skip_existing,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    if not args.sample or not args.vertebra:
        parser.error("--all または --sample と --vertebra を指定してください")

    source_dir = args.source_root / args.sample / args.vertebra
    anchor_dir = args.anchor_root / args.sample / args.vertebra
    output_dir = args.output_root / args.sample / args.vertebra
    report = process_vertebra(
        source_dir=source_dir,
        fallback_source_dir=args.fallback_source_root / args.sample / args.vertebra,
        anchor_dir=anchor_dir,
        output_dir=output_dir,
        point_count=args.point_count,
        trend_anchor_count=args.trend_anchors,
        max_extrapolation=args.max_extrapolation,
        confidence_decay_mm=args.confidence_decay_mm,
        min_mask_area=args.min_mask_area,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
