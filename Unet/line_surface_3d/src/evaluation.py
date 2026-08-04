"""椎体×面（平面1枚）を単位とした評価。

角度と重心位置（rho）は「予測平面（複数窓を集約した1枚）が、実際に
アノテーションされた各スライスの生の線をどれだけ正しく説明するか」で測る。
面単位で集約したGT同士を比べると、スライスごとのズレが平均で相殺されて隠れる。

重なり窓は先に集約してから誤差を測る。窓ごとの観測をそのまま数えると、
同じスライスが10〜15回重複カウントされ、実運用の集約後出力とも一致しない。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.losses import ANGLE_ARM_RATIO, compute_plane_loss
from ..utils.metrics import (
    aligned_rho_error_px,
    angle_error_deg,
    collect_blob_ious,
    plane_normal_error_deg,
    summarize_errors,
    tilt_sign_correct,
    tilt_vector_error,
)
from ..utils.plane import canonical_normal, centered_positions
from .model import VERTEBRA_TO_INDEX, reshape_slab_heatmaps

SurfaceKey = tuple[str, str, int]
RawObservations = dict[SurfaceKey, dict[int, tuple[float, float]]]


@dataclass
class SurfaceAccumulator:
    """同一面へ寄与する全窓の平面予測を集約する。"""

    doubled_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rho_sum: float = 0.0
    slope_sum: float = 0.0
    weight_sum: float = 0.0
    windows: int = 0

    def add(
        self,
        doubled: np.ndarray,
        rho_at_reference: float,
        slope: float,
        weight: float,
    ) -> None:
        """1窓分の平面を、共通の基準zへ換算したうえで加える。"""
        self.doubled_sum += weight * doubled
        self.rho_sum += weight * rho_at_reference
        self.slope_sum += weight * slope
        self.weight_sum += weight
        self.windows += 1

    def finalize(self) -> tuple[np.ndarray, float, float] | None:
        """集約した (法線, rho, slope) を返す。有効窓がなければNone。"""
        if self.weight_sum <= 1e-8:
            return None
        doubled = self.doubled_sum / self.weight_sum
        norm = float(np.linalg.norm(doubled))
        if norm < 1e-6:
            return None
        doubled = doubled / norm
        normal = canonical_normal(torch.from_numpy(doubled)).numpy()
        return normal, self.rho_sum / self.weight_sum, self.slope_sum / self.weight_sum


def vertebra_indices(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """椎体名をモデル条件付け用indexへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_INDEX[str(name)] for name in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


def _resolve_blob_threshold(evaluation_config: dict[str, Any]) -> float:
    """設定からBlob IoUの閾値を決める。"""
    threshold = evaluation_config.get("blob_iou_threshold")
    if threshold is not None:
        return float(threshold)
    heatmap_threshold = evaluation_config.get("heatmap_threshold", 0.2)
    if isinstance(heatmap_threshold, dict):
        return float(heatmap_threshold.get("min", 0.15))
    if isinstance(heatmap_threshold, int | float):
        return float(heatmap_threshold)
    return 0.15


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """各アノテーション済みスライスに対する線精度と、面単位の傾きを集計する。"""
    model.eval()
    data_config = config["data"]
    evaluation_config = config.get("evaluation", {})
    plane_config = config.get("loss", {}).get("plane", {})
    slab_size = int(data_config["slab_size"])
    image_size = int(data_config["image_size"])
    blob_threshold = _resolve_blob_threshold(evaluation_config)
    positions = centered_positions(slab_size, device, torch.float32)
    diagonal = math.sqrt(2.0) * image_size

    loss_sums = {"total": 0.0, "heatmap": 0.0, "angle": 0.0, "rho": 0.0, "tilt": 0.0}
    batch_count = 0
    blob_ious: list[float] = []
    accumulators: dict[SurfaceKey, SurfaceAccumulator] = {}
    targets_by_surface: dict[SurfaceKey, dict[str, Any]] = {}
    # 面ごと・global zごとに1回だけ保持する生GT。窓の重複ぶんは数えない
    raw_by_surface: RawObservations = {}

    for batch in loader:
        images = batch["image"].to(device).float()
        target_heatmaps = batch["heatmaps"].to(device).float()
        label_mask = batch["label_mask"].to(device).bool()
        line_params_gt = batch["line_params_gt"].to(device).float()
        gt_slope = batch["plane_slope_gt"].to(device).float()
        gt_reliable = batch["plane_reliable"].to(device).bool()

        logits = model(images, vertebra_indices(batch, device))
        predictions = torch.sigmoid(reshape_slab_heatmaps(logits, slab_size)).float()
        loss_output = compute_plane_loss(
            predictions,
            target_heatmaps,
            label_mask,
            line_params_gt,
            gt_slope,
            gt_reliable,
            positions,
            image_size,
            plane_config,
            geometry_weight=1.0,
        )
        loss_sums["total"] += float(loss_output.total)
        loss_sums["heatmap"] += float(loss_output.heatmap)
        loss_sums["angle"] += float(loss_output.angle)
        loss_sums["rho"] += float(loss_output.rho)
        loss_sums["tilt"] += float(loss_output.tilt)
        batch_count += 1

        batch_size, _, line_count, height, width = predictions.shape
        flat_mask = label_mask.reshape(batch_size * slab_size, line_count).cpu().numpy()
        blob_ious.extend(
            collect_blob_ious(
                predictions.reshape(batch_size * slab_size, line_count, height, width)
                .cpu()
                .numpy(),
                target_heatmaps.reshape(
                    batch_size * slab_size, line_count, height, width
                )
                .cpu()
                .numpy(),
                flat_mask,
                blob_threshold,
            )
        )

        plane = loss_output.prediction
        doubled = plane.doubled_normal.cpu().numpy()
        rho_0 = plane.rho_0.cpu().numpy()
        slope = plane.slope.cpu().numpy()
        weights = plane.weight_sum.cpu().numpy()
        valid = plane.valid.cpu().numpy()
        slice_indices = batch["slice_indices"].cpu().numpy()
        reference_z = batch["plane_reference_z"].cpu().numpy()
        gt_angle = batch["plane_angle_gt"].cpu().numpy()
        gt_rho0 = batch["plane_rho0_gt"].cpu().numpy()
        gt_slope_numpy = gt_slope.cpu().numpy()
        gt_reliable_numpy = gt_reliable.cpu().numpy()
        label_mask_numpy = label_mask.cpu().numpy()
        line_params_numpy = line_params_gt.cpu().numpy()
        has_label = label_mask_numpy.any(axis=1)

        for batch_index in range(batch_size):
            sample = str(batch["sample"][batch_index])
            vertebra = str(batch["vertebra"][batch_index])
            window_center = float(np.mean(slice_indices[batch_index]))
            for line_index in range(line_count):
                if not has_label[batch_index, line_index]:
                    continue
                key: SurfaceKey = (sample, vertebra, line_index)
                targets_by_surface.setdefault(
                    key,
                    {
                        "angle": float(gt_angle[batch_index, line_index]),
                        "rho": float(gt_rho0[batch_index, line_index]),
                        "slope": float(gt_slope_numpy[batch_index, line_index]),
                        "reliable": bool(gt_reliable_numpy[batch_index, line_index]),
                        "reference_z": float(reference_z[batch_index, line_index]),
                    },
                )
                surface_raw = raw_by_surface.setdefault(key, {})
                for slab_index in range(slab_size):
                    if not label_mask_numpy[batch_index, slab_index, line_index]:
                        continue
                    global_z = int(slice_indices[batch_index, slab_index])
                    phi = float(
                        line_params_numpy[batch_index, slab_index, line_index, 0]
                    )
                    rho_px = (
                        float(line_params_numpy[batch_index, slab_index, line_index, 1])
                        * diagonal
                    )
                    surface_raw.setdefault(global_z, (phi, rho_px))

                if not valid[batch_index, line_index]:
                    continue
                # 窓ごとのrho_0は窓中心の値。共通の基準zへ換算してから平均する
                offset = float(reference_z[batch_index, line_index]) - window_center
                accumulators.setdefault(key, SurfaceAccumulator()).add(
                    doubled=doubled[batch_index, line_index],
                    rho_at_reference=float(rho_0[batch_index, line_index])
                    + float(slope[batch_index, line_index]) * offset,
                    slope=float(slope[batch_index, line_index]),
                    weight=float(weights[batch_index, line_index]),
                )

    tilt_arm_slices = float(positions.abs().max().clamp(min=1.0))
    angle_arm_px = ANGLE_ARM_RATIO * image_size
    return _summarize(
        accumulators,
        targets_by_surface,
        raw_by_surface,
        loss_sums,
        max(1, batch_count),
        blob_ious,
        blob_threshold,
        angle_arm_px,
        tilt_arm_slices,
    )


def _summarize(
    accumulators: dict[SurfaceKey, SurfaceAccumulator],
    targets: dict[SurfaceKey, dict[str, Any]],
    raw_by_surface: RawObservations,
    loss_sums: dict[str, float],
    batch_count: int,
    blob_ious: list[float],
    blob_threshold: float,
    angle_arm_px: float,
    tilt_arm_slices: float,
) -> dict[str, Any]:
    """予測平面を、集約前GT平面と実アノテーション済み各スライスの両方と比較する。"""
    plane_angle_errors: list[float] = []
    plane_rho_errors: list[float] = []
    line_angle_errors: list[float] = []
    line_rho_errors: list[float] = []
    tilt_errors: list[float] = []
    normal_errors: list[float] = []
    sign_correct: list[float] = []
    per_vertebra: dict[str, list[float]] = {}

    for key, target in sorted(targets.items()):
        aggregate = accumulators.get(key)
        finalized = aggregate.finalize() if aggregate is not None else None
        if finalized is None:
            continue
        predicted_normal, predicted_rho, predicted_slope = finalized
        predicted_doubled = np.array(
            [
                predicted_normal[0] ** 2 - predicted_normal[1] ** 2,
                2.0 * predicted_normal[0] * predicted_normal[1],
            ]
        )

        target_angle = target["angle"]
        target_normal = np.array([math.cos(target_angle), math.sin(target_angle)])
        target_doubled = np.array(
            [math.cos(2.0 * target_angle), math.sin(2.0 * target_angle)]
        )

        # 面単位（集約 vs 集約）: 平面仮定そのものの整合性を見る診断指標
        plane_angle = float(angle_error_deg(predicted_doubled, target_doubled))
        plane_angle_errors.append(plane_angle)
        plane_rho_errors.append(
            float(
                aligned_rho_error_px(
                    predicted_normal,
                    np.asarray(predicted_rho),
                    target_normal,
                    np.asarray(target["rho"]),
                )
            )
        )

        # 各画像（各アノテーション済みスライス）: 予測平面がその生の線をどれだけ説明するか。
        # これが主指標。集約同士の比較と違い、スライスごとのズレが平均で相殺されない
        reference_z = target["reference_z"]
        for global_z, (phi_gt, rho_gt_px) in raw_by_surface.get(key, {}).items():
            predicted_rho_z = predicted_rho + predicted_slope * (global_z - reference_z)
            gt_normal_z = np.array([math.cos(phi_gt), math.sin(phi_gt)])
            gt_doubled_z = np.array([math.cos(2.0 * phi_gt), math.sin(2.0 * phi_gt)])
            line_angle_errors.append(
                float(angle_error_deg(predicted_doubled, gt_doubled_z))
            )
            line_rho_errors.append(
                float(
                    aligned_rho_error_px(
                        predicted_normal,
                        np.asarray(predicted_rho_z),
                        gt_normal_z,
                        np.asarray(rho_gt_px),
                    )
                )
            )

        predicted_tilt = predicted_slope * predicted_normal
        target_tilt = target["slope"] * target_normal
        tilt_errors.append(float(tilt_vector_error(predicted_tilt, target_tilt)))
        normal_errors.append(
            float(
                plane_normal_error_deg(
                    predicted_normal,
                    np.asarray(predicted_slope),
                    target_normal,
                    np.asarray(target["slope"]),
                )
            )
        )
        if target["reliable"]:
            sign_correct.append(float(tilt_sign_correct(predicted_tilt, target_tilt)))
        per_vertebra.setdefault(key[1], []).append(plane_angle)

    plane_angle_summary = summarize_errors(plane_angle_errors)
    plane_rho_summary = summarize_errors(plane_rho_errors)
    line_angle_summary = summarize_errors(line_angle_errors)
    line_rho_summary = summarize_errors(line_rho_errors)
    tilt_summary = summarize_errors(tilt_errors)
    normal_summary = summarize_errors(normal_errors)

    # 各画像の角度・重心位置＋面の傾きを、同じpx尺度（線の位置ずれ）へ揃えて合算する。
    # ここはHuberで潰さない生の平均値を使う。選定用の指標であって学習勾配ではないため
    combined_px = (
        line_rho_summary["mean"]
        + angle_arm_px * math.radians(line_angle_summary["mean"])
        + tilt_arm_slices * tilt_summary["mean"]
        if line_angle_errors
        else float("nan")
    )

    return {
        "val_loss_mse": loss_sums["heatmap"] / batch_count,
        "val_loss": loss_sums["total"] / batch_count,
        "val_angle_loss": loss_sums["angle"] / batch_count,
        "val_rho_loss": loss_sums["rho"] / batch_count,
        "val_tilt_loss": loss_sums["tilt"] / batch_count,
        # 各画像（アノテーション済みスライス）単位。主指標
        "line_angle_error_deg": line_angle_summary["mean"],
        "line_angle_error_deg_median": line_angle_summary["median"],
        "line_angle_error_deg_p90": line_angle_summary["p90"],
        "line_rho_error_px": line_rho_summary["mean"],
        "line_rho_error_px_median": line_rho_summary["median"],
        "line_rho_error_px_p90": line_rho_summary["p90"],
        "line_observation_count": float(len(line_angle_errors)),
        # 面単位（集約 vs 集約）。平面仮定の整合性を見る診断指標
        "plane_angle_error_deg": plane_angle_summary["mean"],
        "plane_angle_error_deg_median": plane_angle_summary["median"],
        "plane_angle_error_deg_p90": plane_angle_summary["p90"],
        "plane_rho_error_px": plane_rho_summary["mean"],
        "plane_rho_error_px_median": plane_rho_summary["median"],
        "plane_rho_error_px_p90": plane_rho_summary["p90"],
        # 面単位。傾き
        "tilt_error_px_per_slice": tilt_summary["mean"],
        "tilt_error_px_per_slice_median": tilt_summary["median"],
        "plane_normal_error_deg": normal_summary["mean"],
        "tilt_sign_accuracy": (
            float(np.mean(sign_correct)) if sign_correct else float("nan")
        ),
        "reliable_surface_count": float(len(sign_correct)),
        "evaluated_surface_count": float(len(plane_angle_errors)),
        # checkpoint選定に使う合成指標: 各画像の角度・重心位置＋面の傾き
        "plane_combined_error_px": combined_px,
        "blob_iou": float(np.mean(blob_ious)) if blob_ious else float("nan"),
        "blob_iou_threshold": blob_threshold,
        "per_vertebra": {
            vertebra: {
                "plane_angle_error_deg": float(np.mean(values)),
                "surfaces": len(values),
            }
            for vertebra, values in sorted(per_vertebra.items())
        },
    }
