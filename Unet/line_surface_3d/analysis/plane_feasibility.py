"""手動線が単一3D平面として表現できるか検証する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_SPACING_MM = 0.4
MIN_SLICES = 5


@dataclass(frozen=True)
class LineObservation:
    """1スライスに描かれた1本の線を表す。"""

    z_index: int
    points_xy: np.ndarray
    centroid_xy: np.ndarray
    normal_xy: np.ndarray


@dataclass(frozen=True)
class SurfaceRecord:
    """同一症例・椎体・境界線の観測列を表す。"""

    sample: str
    vertebra: str
    line_key: str
    observations: tuple[LineObservation, ...]


@dataclass(frozen=True)
class PlaneParameters:
    """`normal_xy・p = rho_center + slope・(z-z_center)` を表す。"""

    normal_xy: np.ndarray
    rho_center: float
    slope_px_per_slice: float
    z_center: float


@dataclass(frozen=True)
class PlaneFitResult:
    """1枚の平面fitと傾斜方向の安定性指標。"""

    sample: str
    vertebra: str
    line_key: str
    slice_count: int
    z_span_slices: float
    normal_x: float
    normal_y: float
    rho_center_px: float
    slope_px_per_slice: float
    movement_x_px_per_slice: float
    movement_y_px_per_slice: float
    signed_tilt_deg: float
    net_shift_px: float
    angle_residual_rms_deg: float
    rho_residual_rms_px: float
    point_residual_rms_px: float
    slope_standard_error: float
    slope_ci95_low: float
    slope_ci95_high: float
    slope_ci_excludes_zero: bool
    loo_sign_agreement: float
    odd_even_sign_agreement: bool | None
    half_split_sign_agreement: bool | None


@dataclass(frozen=True)
class PredictionResult:
    """隠した1スライスに対する線予測誤差。"""

    sample: str
    vertebra: str
    line_key: str
    protocol: str
    model: str
    test_z_index: int
    angle_error_deg: float
    point_error_rms_px: float


def _read_json(path: Path) -> Any:
    """JSONを読み、失敗時は空dictを返す。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _valid_polyline(points: Any) -> bool:
    """2点以上の有限座標ポリラインか判定する。"""
    if not isinstance(points, list) or len(points) < 2:
        return False
    array = np.asarray(points, dtype=np.float64)
    return bool(
        array.ndim == 2 and array.shape[1] >= 2 and np.isfinite(array[:, :2]).all()
    )


def _excluded_slices(annotation_root: Path, vertebra_dir: Path) -> set[int]:
    """全体QCと椎体QCから除外スライスを取得する。"""
    sample = vertebra_dir.parent.name
    vertebra = vertebra_dir.name
    bad_data = _read_json(annotation_root / "bad_slices_all.json")
    bad_entries = (
        bad_data if isinstance(bad_data, list) else bad_data.get("bad_slices", [])
    )
    excluded: set[int] = set()
    for entry in bad_entries:
        if not isinstance(entry, dict):
            continue
        slice_value = entry.get("slice_idx", entry.get("slice"))
        if slice_value is None:
            continue
        if str(entry.get("sample")) != sample:
            continue
        if str(entry.get("vertebra")) != vertebra:
            continue
        excluded.add(int(slice_value))

    qc_data = _read_json(vertebra_dir / "qc_scores.json")
    if isinstance(qc_data, dict):
        excluded.update(
            int(slice_key)
            for slice_key, entry in qc_data.items()
            if isinstance(entry, dict) and entry.get("label") == "exclude"
        )
    elif isinstance(qc_data, list):
        excluded.update(
            int(entry["slice_idx"])
            for entry in qc_data
            if isinstance(entry, dict)
            and entry.get("label") == "exclude"
            and entry.get("slice_idx") is not None
        )
    return excluded


def _canonical_normal(normal_xy: np.ndarray) -> np.ndarray:
    """法線符号をY正、水平時はX正へ統一する。"""
    normal = np.asarray(normal_xy, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("法線を正規化できません")
    normal = normal / norm
    if normal[1] < 0 or (abs(normal[1]) <= 1e-12 and normal[0] < 0):
        normal = -normal
    return normal


def _line_geometry(
    points: Any, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像座標ポリラインから数学座標点、重心、単位法線を返す。"""
    raw_points = np.asarray(points, dtype=np.float64)[:, :2]
    center = image_size / 2.0
    math_points = np.column_stack(
        [raw_points[:, 0] - center, -(raw_points[:, 1] - center)]
    )
    centroid = math_points.mean(axis=0)
    centered = math_points - centroid
    covariance = centered.T @ centered / max(1, len(math_points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = _canonical_normal(np.array([-direction[1], direction[0]]))
    return math_points, centroid, normal


def load_surface_records(
    annotation_root: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    min_slices: int = MIN_SLICES,
) -> list[SurfaceRecord]:
    """有効な手動線を境界面単位へまとめる。"""
    records: list[SurfaceRecord] = []
    for lines_path in sorted(annotation_root.glob("sample*/C*/lines.json")):
        vertebra_dir = lines_path.parent
        lines_data = _read_json(lines_path)
        if not isinstance(lines_data, dict):
            continue
        excluded = _excluded_slices(annotation_root, vertebra_dir)
        observations: dict[str, list[LineObservation]] = {
            line_key: [] for line_key in LINE_KEYS
        }
        for slice_key, entry in sorted(
            lines_data.items(), key=lambda item: int(item[0])
        ):
            slice_index = int(slice_key)
            if slice_index in excluded or not isinstance(entry, dict):
                continue
            if not all(_valid_polyline(entry.get(line_key)) for line_key in LINE_KEYS):
                continue
            for line_key in LINE_KEYS:
                points_xy, centroid_xy, normal_xy = _line_geometry(
                    entry[line_key], image_size
                )
                observations[line_key].append(
                    LineObservation(
                        z_index=slice_index,
                        points_xy=points_xy,
                        centroid_xy=centroid_xy,
                        normal_xy=normal_xy,
                    )
                )
        for line_key, line_observations in observations.items():
            if len(line_observations) < min_slices:
                continue
            records.append(
                SurfaceRecord(
                    sample=vertebra_dir.parent.name,
                    vertebra=vertebra_dir.name,
                    line_key=line_key,
                    observations=tuple(line_observations),
                )
            )
    return records


def _common_normal(observations: tuple[LineObservation, ...]) -> np.ndarray:
    """180度周期の線法線をdoubled-angle平均する。"""
    normals = np.stack([observation.normal_xy for observation in observations])
    doubled_cosine = np.square(normals[:, 0]) - np.square(normals[:, 1])
    doubled_sine = 2.0 * normals[:, 0] * normals[:, 1]
    angle = 0.5 * math.atan2(float(doubled_sine.mean()), float(doubled_cosine.mean()))
    return _canonical_normal(np.array([math.cos(angle), math.sin(angle)]))


def _fit_plane_parameters(
    observations: tuple[LineObservation, ...],
    allow_tilt: bool,
) -> tuple[PlaneParameters, np.ndarray, np.ndarray]:
    """共通角度とzに対する線位置を最小二乗fitする。"""
    normal = _common_normal(observations)
    z_values = np.asarray([item.z_index for item in observations], dtype=np.float64)
    rho_values = np.asarray(
        [float(np.mean(item.points_xy @ normal)) for item in observations],
        dtype=np.float64,
    )
    z_center = float(z_values.mean())
    centered_z = z_values - z_center
    rho_center = float(rho_values.mean())
    denominator = float(centered_z @ centered_z)
    slope = (
        float(centered_z @ (rho_values - rho_center)) / denominator
        if allow_tilt and denominator > 1e-12
        else 0.0
    )
    return (
        PlaneParameters(
            normal_xy=normal,
            rho_center=rho_center,
            slope_px_per_slice=slope,
            z_center=z_center,
        ),
        z_values,
        rho_values,
    )


def _rho_at(parameters: PlaneParameters, z_index: int | float) -> float:
    """指定zでの平面とスライスの交線位置を返す。"""
    return parameters.rho_center + parameters.slope_px_per_slice * (
        float(z_index) - parameters.z_center
    )


def _angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """符号不変な2法線間角度を度で返す。"""
    cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _point_error_rms(
    observation: LineObservation,
    normal_xy: np.ndarray,
    rho: float,
) -> float:
    """観測ポリライン点から予測交線までのRMS距離を返す。"""
    distances = observation.points_xy @ normal_xy - rho
    return float(np.sqrt(np.mean(np.square(distances))))


def _sign(value: float, epsilon: float = 1e-8) -> int:
    """許容幅付き符号を返す。"""
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def fit_surface(record: SurfaceRecord, spacing_mm: float) -> PlaneFitResult:
    """全観測から平面残差と傾斜符号の安定性を求める。"""
    parameters, z_values, rho_values = _fit_plane_parameters(
        record.observations, allow_tilt=True
    )
    predicted_rho = np.asarray(
        [_rho_at(parameters, z_value) for z_value in z_values], dtype=np.float64
    )
    rho_residual = rho_values - predicted_rho
    angle_errors = np.asarray(
        [
            _angle_error_deg(parameters.normal_xy, observation.normal_xy)
            for observation in record.observations
        ]
    )
    point_errors = np.asarray(
        [
            _point_error_rms(
                observation,
                parameters.normal_xy,
                _rho_at(parameters, observation.z_index),
            )
            for observation in record.observations
        ]
    )

    centered_z = z_values - z_values.mean()
    degrees_of_freedom = len(z_values) - 2
    slope_standard_error = float("nan")
    if degrees_of_freedom > 0 and float(centered_z @ centered_z) > 1e-12:
        residual_variance = float(rho_residual @ rho_residual) / degrees_of_freedom
        slope_standard_error = math.sqrt(
            residual_variance / float(centered_z @ centered_z)
        )
    critical_value = float(student_t.ppf(0.975, degrees_of_freedom))
    ci_radius = critical_value * slope_standard_error
    ci_low = parameters.slope_px_per_slice - ci_radius
    ci_high = parameters.slope_px_per_slice + ci_radius

    full_sign = _sign(parameters.slope_px_per_slice)
    loo_signs: list[int] = []
    for index in range(len(record.observations)):
        subset = record.observations[:index] + record.observations[index + 1 :]
        subset_parameters, _, _ = _fit_plane_parameters(subset, allow_tilt=True)
        loo_signs.append(_sign(subset_parameters.slope_px_per_slice))
    loo_sign_agreement = float(np.mean([sign == full_sign for sign in loo_signs]))

    odd_observations = record.observations[::2]
    even_observations = record.observations[1::2]
    odd_even_agreement: bool | None = None
    if len(odd_observations) >= 3 and len(even_observations) >= 3:
        odd_parameters, _, _ = _fit_plane_parameters(odd_observations, allow_tilt=True)
        even_parameters, _, _ = _fit_plane_parameters(
            even_observations, allow_tilt=True
        )
        odd_even_agreement = _sign(odd_parameters.slope_px_per_slice) == _sign(
            even_parameters.slope_px_per_slice
        )

    half_split_agreement: bool | None = None
    split_index = len(record.observations) // 2
    lower_observations = record.observations[:split_index]
    upper_observations = record.observations[split_index:]
    if len(lower_observations) >= 3 and len(upper_observations) >= 3:
        lower_parameters, _, _ = _fit_plane_parameters(
            lower_observations, allow_tilt=True
        )
        upper_parameters, _, _ = _fit_plane_parameters(
            upper_observations, allow_tilt=True
        )
        half_split_agreement = _sign(lower_parameters.slope_px_per_slice) == _sign(
            upper_parameters.slope_px_per_slice
        )

    z_span = float(z_values.max() - z_values.min())
    spacing_ratio = spacing_mm / spacing_mm
    signed_tilt = math.degrees(math.atan(parameters.slope_px_per_slice * spacing_ratio))
    movement = parameters.slope_px_per_slice * parameters.normal_xy
    return PlaneFitResult(
        sample=record.sample,
        vertebra=record.vertebra,
        line_key=record.line_key,
        slice_count=len(record.observations),
        z_span_slices=z_span,
        normal_x=float(parameters.normal_xy[0]),
        normal_y=float(parameters.normal_xy[1]),
        rho_center_px=parameters.rho_center,
        slope_px_per_slice=parameters.slope_px_per_slice,
        movement_x_px_per_slice=float(movement[0]),
        movement_y_px_per_slice=float(movement[1]),
        signed_tilt_deg=signed_tilt,
        net_shift_px=parameters.slope_px_per_slice * z_span,
        angle_residual_rms_deg=float(np.sqrt(np.mean(np.square(angle_errors)))),
        rho_residual_rms_px=float(np.sqrt(np.mean(np.square(rho_residual)))),
        point_residual_rms_px=float(np.sqrt(np.mean(np.square(point_errors)))),
        slope_standard_error=slope_standard_error,
        slope_ci95_low=ci_low,
        slope_ci95_high=ci_high,
        slope_ci_excludes_zero=bool(ci_low > 0 or ci_high < 0),
        loo_sign_agreement=loo_sign_agreement,
        odd_even_sign_agreement=odd_even_agreement,
        half_split_sign_agreement=half_split_agreement,
    )


def _fit_linear_values(
    z_values: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """複数成分をzに対して一次fitし、切片と傾きを返す。"""
    z_center = float(z_values.mean())
    centered_z = z_values - z_center
    denominator = float(centered_z @ centered_z)
    intercept = values.mean(axis=0)
    slope = (
        (centered_z[:, None] * (values - intercept)).sum(axis=0) / denominator
        if denominator > 1e-12
        else np.zeros(values.shape[1], dtype=np.float64)
    )
    return intercept, slope


def _predict_ribbon(
    observations: tuple[LineObservation, ...],
    z_index: int,
) -> tuple[np.ndarray, float]:
    """現行リボン相当の重心・doubled-angle一次fitで線を予測する。"""
    z_values = np.asarray([item.z_index for item in observations], dtype=np.float64)
    centroids = np.stack([item.centroid_xy for item in observations])
    normals = np.stack([item.normal_xy for item in observations])
    doubled = np.column_stack(
        [
            np.square(normals[:, 0]) - np.square(normals[:, 1]),
            2.0 * normals[:, 0] * normals[:, 1],
        ]
    )
    centroid_intercept, centroid_slope = _fit_linear_values(z_values, centroids)
    doubled_intercept, doubled_slope = _fit_linear_values(z_values, doubled)
    centered_z = float(z_index) - float(z_values.mean())
    centroid = centroid_intercept + centroid_slope * centered_z
    doubled_prediction = doubled_intercept + doubled_slope * centered_z
    doubled_norm = float(np.linalg.norm(doubled_prediction))
    if doubled_norm <= 1e-12:
        doubled_prediction = np.array([1.0, 0.0])
    else:
        doubled_prediction = doubled_prediction / doubled_norm
    angle = 0.5 * math.atan2(float(doubled_prediction[1]), float(doubled_prediction[0]))
    normal = _canonical_normal(np.array([math.cos(angle), math.sin(angle)]))
    return normal, float(normal @ centroid)


def _predict_model(
    observations: tuple[LineObservation, ...],
    z_index: int,
    model: str,
) -> tuple[np.ndarray, float]:
    """指定比較モデルでzスライス上の線を予測する。"""
    if model == "ribbon":
        return _predict_ribbon(observations, z_index)
    allow_tilt = model == "plane"
    parameters, _, _ = _fit_plane_parameters(observations, allow_tilt=allow_tilt)
    return parameters.normal_xy, _rho_at(parameters, z_index)


def _evaluate_split(
    record: SurfaceRecord,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    protocol: str,
) -> list[PredictionResult]:
    """指定した観測分割で3種類の表現を比較する。"""
    training = tuple(record.observations[index] for index in train_indices)
    results: list[PredictionResult] = []
    for test_index in test_indices:
        observation = record.observations[test_index]
        for model in ("no_tilt", "plane", "ribbon"):
            normal, rho = _predict_model(training, observation.z_index, model)
            results.append(
                PredictionResult(
                    sample=record.sample,
                    vertebra=record.vertebra,
                    line_key=record.line_key,
                    protocol=protocol,
                    model=model,
                    test_z_index=observation.z_index,
                    angle_error_deg=_angle_error_deg(normal, observation.normal_xy),
                    point_error_rms_px=_point_error_rms(observation, normal, rho),
                )
            )
    return results


def evaluate_record(record: SurfaceRecord) -> list[PredictionResult]:
    """LOO、中央から両端、片半分から反対側の外挿を実行する。"""
    results: list[PredictionResult] = []
    all_indices = tuple(range(len(record.observations)))
    for test_index in all_indices:
        train_indices = tuple(index for index in all_indices if index != test_index)
        results.extend(
            _evaluate_split(
                record,
                train_indices,
                (test_index,),
                protocol="leave_one_slice_out",
            )
        )
    results.extend(
        _evaluate_split(
            record,
            all_indices[1:-1],
            (all_indices[0], all_indices[-1]),
            protocol="central_to_edges",
        )
    )
    split_index = len(all_indices) // 2
    lower_indices = all_indices[:split_index]
    upper_indices = all_indices[split_index:]
    if len(lower_indices) >= 3 and len(upper_indices) >= 3:
        results.extend(
            _evaluate_split(
                record,
                lower_indices,
                upper_indices,
                protocol="half_to_half_extrapolation",
            )
        )
        results.extend(
            _evaluate_split(
                record,
                upper_indices,
                lower_indices,
                protocol="half_to_half_extrapolation",
            )
        )
    return results


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    """数値列の件数、平均、中央値、p90、p95を返す。"""
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "p95": math.nan,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
    }


def summarize_results(
    records: list[SurfaceRecord],
    fits: list[PlaneFitResult],
    predictions: list[PredictionResult],
) -> dict[str, Any]:
    """全体・線別の平面fitと予測結果を集約する。"""
    fit_metrics = {
        "angle_residual_rms_deg": _summary_stats(
            [fit.angle_residual_rms_deg for fit in fits]
        ),
        "rho_residual_rms_px": _summary_stats(
            [fit.rho_residual_rms_px for fit in fits]
        ),
        "point_residual_rms_px": _summary_stats(
            [fit.point_residual_rms_px for fit in fits]
        ),
        "absolute_tilt_deg": _summary_stats([abs(fit.signed_tilt_deg) for fit in fits]),
        "absolute_net_shift_px": _summary_stats(
            [abs(fit.net_shift_px) for fit in fits]
        ),
    }
    direction = {
        "slope_ci_excludes_zero_rate": float(
            np.mean([fit.slope_ci_excludes_zero for fit in fits])
        ),
        "loo_all_signs_agree_rate": float(
            np.mean([fit.loo_sign_agreement == 1.0 for fit in fits])
        ),
        "loo_sign_agreement": _summary_stats([fit.loo_sign_agreement for fit in fits]),
        "odd_even_evaluable_count": sum(
            fit.odd_even_sign_agreement is not None for fit in fits
        ),
        "odd_even_sign_agreement_rate": float(
            np.mean(
                [
                    fit.odd_even_sign_agreement
                    for fit in fits
                    if fit.odd_even_sign_agreement is not None
                ]
            )
        ),
        "half_split_evaluable_count": sum(
            fit.half_split_sign_agreement is not None for fit in fits
        ),
        "half_split_sign_agreement_rate": float(
            np.mean(
                [
                    fit.half_split_sign_agreement
                    for fit in fits
                    if fit.half_split_sign_agreement is not None
                ]
            )
        ),
    }

    prediction_summary: dict[str, Any] = {}
    for protocol in (
        "leave_one_slice_out",
        "central_to_edges",
        "half_to_half_extrapolation",
    ):
        prediction_summary[protocol] = {}
        for model in ("no_tilt", "plane", "ribbon"):
            selected = [
                result
                for result in predictions
                if result.protocol == protocol and result.model == model
            ]
            prediction_summary[protocol][model] = {
                "angle_error_deg": _summary_stats(
                    [result.angle_error_deg for result in selected]
                ),
                "point_error_rms_px": _summary_stats(
                    [result.point_error_rms_px for result in selected]
                ),
            }

    per_line: dict[str, Any] = {}
    for line_key in LINE_KEYS:
        line_fits = [fit for fit in fits if fit.line_key == line_key]
        line_predictions = [
            result for result in predictions if result.line_key == line_key
        ]
        per_line[line_key] = {
            "surface_count": len(line_fits),
            "angle_residual_rms_deg": _summary_stats(
                [fit.angle_residual_rms_deg for fit in line_fits]
            ),
            "point_residual_rms_px": _summary_stats(
                [fit.point_residual_rms_px for fit in line_fits]
            ),
            "slope_ci_excludes_zero_rate": float(
                np.mean([fit.slope_ci_excludes_zero for fit in line_fits])
            ),
            "central_edge_plane_point_error_px": _summary_stats(
                [
                    result.point_error_rms_px
                    for result in line_predictions
                    if result.protocol == "central_to_edges" and result.model == "plane"
                ]
            ),
        }

    plane_edges = {
        (item.sample, item.vertebra, item.line_key, item.test_z_index): item
        for item in predictions
        if item.protocol == "central_to_edges" and item.model == "plane"
    }
    no_tilt_edges = {
        (item.sample, item.vertebra, item.line_key, item.test_z_index): item
        for item in predictions
        if item.protocol == "central_to_edges" and item.model == "no_tilt"
    }
    shared_keys = sorted(plane_edges.keys() & no_tilt_edges.keys())
    edge_improvements = [
        no_tilt_edges[key].point_error_rms_px - plane_edges[key].point_error_rms_px
        for key in shared_keys
    ]
    return {
        "dataset": {
            "vertebra_count": len(
                {(record.sample, record.vertebra) for record in records}
            ),
            "surface_count": len(records),
            "observation_count": sum(len(record.observations) for record in records),
            "minimum_slices": MIN_SLICES,
        },
        "full_plane_fit": fit_metrics,
        "tilt_direction": direction,
        "held_out_prediction": prediction_summary,
        "central_edge_plane_vs_no_tilt": {
            "point_error_improvement_px": _summary_stats(edge_improvements),
            "plane_better_rate": float(
                np.mean([improvement > 0 for improvement in edge_improvements])
            ),
        },
        "per_line": per_line,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """dict列をCSVへ保存する。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    output_dir: Path,
    summary: dict[str, Any],
    fits: list[PlaneFitResult],
    predictions: list[PredictionResult],
) -> None:
    """集約JSONと症例別CSVを保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "surfaces.csv", [asdict(fit) for fit in fits])
    _write_csv(
        output_dir / "held_out_predictions.csv",
        [asdict(result) for result in predictions],
    )


def _format_metric(metric: dict[str, float | int]) -> str:
    """主要な分布統計を1行へ整形する。"""
    return (
        f"median={float(metric['median']):.3f}, "
        f"p90={float(metric['p90']):.3f}, p95={float(metric['p95']):.3f}"
    )


def print_summary(summary: dict[str, Any]) -> None:
    """検証結果の主要値を標準出力へ表示する。"""
    dataset = summary["dataset"]
    fit = summary["full_plane_fit"]
    direction = summary["tilt_direction"]
    print(
        f"対象: {dataset['vertebra_count']}椎体, "
        f"{dataset['surface_count']}面, {dataset['observation_count']}線観測"
    )
    print("\n[全観測に対する単一平面fit]")
    print(f"角度残差[deg]: {_format_metric(fit['angle_residual_rms_deg'])}")
    print(f"点距離残差[px]: {_format_metric(fit['point_residual_rms_px'])}")
    print(f"|z傾斜|[deg]: {_format_metric(fit['absolute_tilt_deg'])}")
    print(f"|全z移動|[px]: {_format_metric(fit['absolute_net_shift_px'])}")
    print("\n[z傾斜方向の安定性]")
    print(
        f"95% CIが0を除外: {direction['slope_ci_excludes_zero_rate']:.1%}, "
        f"全LOOで符号一致: {direction['loo_all_signs_agree_rate']:.1%}, "
        f"奇偶分割で符号一致: {direction['odd_even_sign_agreement_rate']:.1%}, "
        f"前後半で符号一致: {direction['half_split_sign_agreement_rate']:.1%}"
    )
    print("\n[中央スライスから両端を予測]")
    edge = summary["held_out_prediction"]["central_to_edges"]
    for model in ("no_tilt", "plane", "ribbon"):
        print(
            f"{model:8s} angle={_format_metric(edge[model]['angle_error_deg'])}, "
            f"point={_format_metric(edge[model]['point_error_rms_px'])}"
        )
    comparison = summary["central_edge_plane_vs_no_tilt"]
    print(f"平面が傾きなしより点距離で良い割合: {comparison['plane_better_rate']:.1%}")
    print("\n[片半分から反対側全体を外挿]")
    half = summary["held_out_prediction"]["half_to_half_extrapolation"]
    for model in ("no_tilt", "plane", "ribbon"):
        print(
            f"{model:8s} angle={_format_metric(half[model]['angle_error_deg'])}, "
            f"point={_format_metric(half[model]['point_error_rms_px'])}"
        )


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--spacing-mm", type=float, default=DEFAULT_SPACING_MM)
    parser.add_argument("--min-slices", type=int, default=MIN_SLICES)
    return parser.parse_args()


def main() -> None:
    """全手動アノテーションで平面実現可能性を検証する。"""
    args = parse_args()
    records = load_surface_records(
        args.annotation_root,
        image_size=args.image_size,
        min_slices=args.min_slices,
    )
    if not records:
        raise ValueError("検証可能な境界面がありません")
    fits = [fit_surface(record, args.spacing_mm) for record in records]
    predictions = [
        prediction for record in records for prediction in evaluate_record(record)
    ]
    summary = summarize_results(records, fits, predictions)
    summary["dataset"]["minimum_slices"] = args.min_slices
    summary["dataset"]["spacing_mm"] = args.spacing_mm
    save_results(args.output_dir, summary, fits, predictions)
    print_summary(summary)


if __name__ == "__main__":
    main()
