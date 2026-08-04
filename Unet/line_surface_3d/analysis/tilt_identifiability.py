"""z傾斜の識別可能性と評価設計の前提を実測する解析。

`plane_feasibility.py` が「中央手動線から平面GTを作れるか」を検証したのに対し、
本スクリプトは「その傾斜をモデルが学習・評価できる見込みがあるか」を検証する。

実測する4項目:

1. 線方向の重心ドリフト対法線方向移動の比
   現行 `fit_ribbon()` は重心x,yをz方向へ独立に1次fitするため、
   線に沿った重心の移動（教師polylineの描画長の違いによる雑音）が
   傾斜信号に混入する。その混入量を測る。
2. 傾斜kの信号対雑音比
   GTアノテーション由来のSE(k)と、モデルのrho誤差から導かれるSE(k)を比較し、
   後付け平面fitで傾斜が分離できるかを判定する。
3. 事前分布ベースラインの強さ
   符号予測の対照群となる level+line 平均の符号一致率を
   leave-one-sample-out で測る。
4. 信頼性判定ルールの通過数
   Codex提案の reliable 判定を適用し、面数がGo条件を満たすか確認する。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
MIN_POLYLINE_POINTS = 2
MIN_SLICES_FOR_FIT = 5
IMAGE_DIAGONAL_PX = float(np.hypot(224.0, 224.0))


@dataclass(frozen=True)
class DriftResult:
    """線方向ドリフトと法線方向移動の比較結果。"""

    along_px: np.ndarray
    perpendicular_px: np.ndarray

    def summary(self) -> dict[str, float]:
        """中央値と、ドリフトが信号を上回る割合を返す。"""
        return {
            "surfaces": int(self.along_px.size),
            "perpendicular_median_px": float(np.median(self.perpendicular_px)),
            "along_median_px": float(np.median(self.along_px)),
            "drift_ratio": float(
                np.median(self.along_px) / max(np.median(self.perpendicular_px), 1e-9)
            ),
            "drift_exceeds_signal_rate": float(
                np.mean(self.along_px > self.perpendicular_px)
            ),
        }


def _load_polyline_centroids(
    lines_path: Path,
    line_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1面について、スライスz・重心・主方向を返す。"""
    payload = json.loads(lines_path.read_text())
    slice_positions: list[float] = []
    centroids: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for key, slice_lines in payload.items():
        points = slice_lines.get(line_key) or []
        if len(points) < MIN_POLYLINE_POINTS:
            continue
        coordinates = np.asarray(points, dtype=np.float64)
        centroid = coordinates.mean(axis=0)
        _, _, right_vectors = np.linalg.svd(coordinates - centroid)
        slice_positions.append(float(key))
        centroids.append(centroid)
        directions.append(right_vectors[0])
    return (
        np.asarray(slice_positions),
        np.asarray(centroids),
        np.asarray(directions),
    )


def _shared_tangent(directions: np.ndarray) -> np.ndarray:
    """doubled-angle平均から符号に依存しない共通接線を返す。"""
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    mean_angle = 0.5 * np.arctan2(
        np.sin(2.0 * angles).mean(),
        np.cos(2.0 * angles).mean(),
    )
    return np.array([np.cos(mean_angle), np.sin(mean_angle)])


def measure_centroid_drift(annotation_root: Path) -> DriftResult:
    """線方向ドリフトと法線方向移動を全面について測る。"""
    along_values: list[float] = []
    perpendicular_values: list[float] = []
    for lines_path in sorted(annotation_root.glob("sample*/C*/lines.json")):
        for line_key in LINE_KEYS:
            positions, centroids, directions = _load_polyline_centroids(
                lines_path,
                line_key,
            )
            if positions.size < MIN_SLICES_FOR_FIT:
                continue
            tangent = _shared_tangent(directions)
            normal = np.array([-tangent[1], tangent[0]])
            centered = positions - positions.mean()
            sum_squares = float((centered**2).sum())
            if sum_squares < 1e-9:
                continue
            span = float(positions.max() - positions.min())
            along_slope = float((centroids @ tangent * centered).sum() / sum_squares)
            normal_slope = float((centroids @ normal * centered).sum() / sum_squares)
            along_values.append(abs(along_slope) * span)
            perpendicular_values.append(abs(normal_slope) * span)
    return DriftResult(
        along_px=np.asarray(along_values),
        perpendicular_px=np.asarray(perpendicular_values),
    )


def measure_tilt_snr(
    surfaces: pd.DataFrame, model_rho_error_px: float
) -> dict[str, Any]:
    """GT側とモデル側のSE(k)を比較し、傾斜の分離可能性を評価する。"""
    slopes = np.abs(surfaces["slope_px_per_slice"].to_numpy())
    annotation_se = surfaces["slope_standard_error"].to_numpy()
    slice_counts = surfaces["slice_count"].to_numpy(dtype=np.float64)
    spans = surfaces["z_span_slices"].to_numpy(dtype=np.float64)

    # 等間隔配置での Sxx = N * span^2 / 12
    band_sum_squares = slice_counts * spans**2 / 12.0
    band_model_se = model_rho_error_px / np.sqrt(np.maximum(band_sum_squares, 1e-9))
    window_sum_squares = 15.0 * 14.0**2 / 12.0
    window_model_se = model_rho_error_px / np.sqrt(window_sum_squares)

    return {
        "signal_median_px_per_slice": float(np.median(slopes)),
        "annotation_se_median": float(np.median(annotation_se)),
        "annotation_snr_median": float(
            np.median(slopes / np.maximum(annotation_se, 1e-9))
        ),
        "model_se_over_band_median": float(np.median(band_model_se)),
        "model_se_over_window": float(window_model_se),
        "model_snr_over_band_median": float(np.median(slopes / band_model_se)),
        "model_snr_over_window_median": float(np.median(slopes) / window_model_se),
        "required_rho_error_for_snr2_px": float(
            np.median(slopes) * np.sqrt(window_sum_squares) / 2.0
        ),
    }


def measure_prior_baseline(surfaces: pd.DataFrame) -> dict[str, float]:
    """leave-one-sample-outで事前分布ベースラインの符号一致率を測る。"""
    reliable = surfaces[surfaces["net_shift_px"].abs() >= 1.0]
    results: dict[str, float] = {}
    for label, group_keys in (
        ("level_and_line", ["vertebra", "line_key"]),
        ("line_only", ["line_key"]),
        ("global", []),
    ):
        correct = total = 0
        for sample in reliable["sample"].unique():
            train = reliable[reliable["sample"] != sample]
            test = reliable[reliable["sample"] == sample]
            for _, row in test.iterrows():
                matched = train
                for key in group_keys:
                    matched = matched[matched[key] == row[key]]
                if matched.empty:
                    continue
                predicted = np.sign(matched["slope_px_per_slice"].mean())
                total += 1
                correct += int(predicted == np.sign(row["slope_px_per_slice"]))
        results[label] = correct / max(total, 1)
    positive_rate = float((reliable["slope_px_per_slice"] > 0).mean())
    results["majority_class"] = max(positive_rate, 1.0 - positive_rate)
    results["n_surfaces"] = float(len(reliable))
    return results


def select_reliable(surfaces: pd.DataFrame) -> np.ndarray:
    """Codex提案のreliable判定を適用したbool maskを返す。"""
    slice_counts = surfaces["slice_count"].to_numpy()
    spans = surfaces["z_span_slices"].to_numpy()
    movement = surfaces["net_shift_px"].abs().to_numpy()
    slopes = np.abs(surfaces["slope_px_per_slice"].to_numpy())
    standard_errors = surfaces["slope_standard_error"].to_numpy()

    critical = np.asarray(
        [stats.t.ppf(0.975, max(int(count) - 2, 1)) for count in slice_counts]
    )
    t_statistic = slopes / np.maximum(standard_errors, 1e-12)

    base_qc = (
        (slice_counts >= 5)
        & (spans >= 4)
        & (surfaces["point_residual_rms_px"].to_numpy() <= 2.0)
        & (surfaces["angle_residual_rms_deg"].to_numpy() <= 5.0)
        & (movement >= 1.0)
        & (surfaces["loo_sign_agreement"].to_numpy() >= 0.8)
    )
    strong_movement = movement >= 2.0
    marginal_but_significant = (
        (movement >= 1.0)
        & (movement < 2.0)
        & (t_statistic >= critical)
        & (surfaces["odd_even_sign_agreement"].to_numpy() >= 1.0)
    )
    return np.asarray(base_qc & (strong_movement | marginal_but_significant))


def main() -> None:
    """全解析を実行し、結果をJSONと標準出力へ出す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--surfaces-csv",
        type=Path,
        default=Path("Unet/outputs/line_surface_3d/plane_feasibility/surfaces.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Unet/outputs/line_surface_3d/tilt_identifiability"),
    )
    parser.add_argument("--model-rho-error-px", type=float, default=3.116)
    arguments = parser.parse_args()

    surfaces = pd.read_csv(arguments.surfaces_csv)
    reliable_mask = select_reliable(surfaces)
    report = {
        "centroid_drift": measure_centroid_drift(arguments.annotation_root).summary(),
        "tilt_snr": measure_tilt_snr(surfaces, arguments.model_rho_error_px),
        "prior_baseline": measure_prior_baseline(surfaces),
        "reliable_selection": {
            "total_surfaces": int(len(surfaces)),
            "reliable_surfaces": int(reliable_mask.sum()),
            "reliable_rate": float(reliable_mask.mean()),
            "per_line": surfaces[reliable_mask]["line_key"].value_counts().to_dict(),
            "per_vertebra": surfaces[reliable_mask]["vertebra"]
            .value_counts()
            .to_dict(),
            "samples_with_reliable": int(surfaces[reliable_mask]["sample"].nunique()),
        },
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
