"""ラベル付きスラブに対するline_only共通・surface固有評価。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.detection import extract_pred_params_cc_batch
from ..utils.losses import compute_surface_loss
from ..utils.metrics import (
    collect_angle_errors,
    collect_angle_errors_deg,
    collect_blob_ious,
    collect_centroid_errors,
    collect_rho_errors,
    summarize_errors,
)
from ..utils.ribbon import compute_heatmap_moments, fit_ribbon
from .model import VERTEBRA_TO_INDEX, reshape_slab_heatmaps


def peak_dist(prediction: np.ndarray, target: np.ndarray) -> float:
    """line_onlyと同じヒートマップピーク間距離を返す。"""
    target_y, target_x = np.unravel_index(np.argmax(target), target.shape)
    prediction_y, prediction_x = np.unravel_index(
        np.argmax(prediction),
        prediction.shape,
    )
    return math.sqrt((prediction_x - target_x) ** 2 + (prediction_y - target_y) ** 2)


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


def _mean_or_nan(values: list[float]) -> float:
    """空でない値の平均を返す。"""
    return float(np.nanmean(values)) if values else float("nan")


@torch.no_grad()  # type: ignore[untyped-decorator]
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """line_only共通指標へsurface固有のraw/fitted指標を追加する。"""
    model.eval()
    data_config = config["data"]
    evaluation_config = config.get("evaluation", {})
    slab_size = int(data_config["slab_size"])
    image_size = int(data_config["image_size"])
    ribbon_config = config.get("loss", {}).get("ribbon", {})
    heatmap_threshold = evaluation_config.get("heatmap_threshold", 0.2)
    outlier_angle_threshold = float(
        evaluation_config.get("outlier_angle_threshold_deg", 10.0)
    )
    outlier_rho_threshold = float(
        evaluation_config.get("outlier_rho_threshold_px", 8.0)
    )
    loss_sums = {
        "total": 0.0,
        "heatmap": 0.0,
        "angle": 0.0,
        "centroid": 0.0,
        "ribbon": 0.0,
    }
    batch_count = 0
    peak_distances: list[float] = []
    blob_ious: list[float] = []
    angle_errors: list[float] = []
    rho_errors: list[float] = []
    surface_raw_angles: list[float] = []
    surface_fitted_angles: list[float] = []
    surface_raw_centroids: list[float] = []
    surface_fitted_centroids: list[float] = []
    detected_line_count = 0
    labeled_line_count = 0
    per_vertebra: dict[str, dict[str, list[float]]] = {
        vertebra: {"peak_dists": []} for vertebra in VERTEBRA_TO_INDEX
    }

    for batch in loader:
        images = batch["image"].to(device).float()
        targets = batch["heatmaps"].to(device).float()
        label_mask = batch["label_mask"].to(device).bool()
        target_params = batch["line_params_gt"].to(device).float()
        logits = model(images, vertebra_indices(batch, device))
        predictions = torch.sigmoid(reshape_slab_heatmaps(logits, slab_size))
        loss_output = compute_surface_loss(
            predictions,
            targets,
            label_mask,
            ribbon_config,
            geometry_weight=1.0,
        )
        loss_sums["total"] += float(loss_output.total)
        loss_sums["heatmap"] += float(loss_output.heatmap)
        loss_sums["angle"] += float(loss_output.angle)
        loss_sums["centroid"] += float(loss_output.centroid)
        loss_sums["ribbon"] += float(loss_output.ribbon)
        batch_count += 1

        batch_size, _, line_count, height, width = predictions.shape
        flat_predictions = predictions.reshape(
            batch_size * slab_size,
            line_count,
            height,
            width,
        )
        flat_targets = targets.reshape_as(flat_predictions)
        flat_label_mask = label_mask.reshape(batch_size * slab_size, line_count)
        flat_target_params = target_params.reshape(
            batch_size * slab_size,
            line_count,
            2,
        )
        prediction_numpy = flat_predictions.cpu().numpy()
        target_numpy = flat_targets.cpu().numpy()
        label_mask_numpy = flat_label_mask.cpu().numpy()
        blob_ious.extend(
            collect_blob_ious(
                prediction_numpy,
                target_numpy,
                label_mask_numpy,
            )
        )
        for batch_index, vertebra in enumerate(batch["vertebra"]):
            vertebra_name = str(vertebra)
            for slab_index in range(slab_size):
                flat_index = batch_index * slab_size + slab_index
                for line_index in range(line_count):
                    if not label_mask_numpy[flat_index, line_index]:
                        continue
                    distance = peak_dist(
                        prediction_numpy[flat_index, line_index],
                        target_numpy[flat_index, line_index],
                    )
                    peak_distances.append(distance)
                    per_vertebra[vertebra_name]["peak_dists"].append(distance)

        predicted_params_numpy, confidence_numpy = extract_pred_params_cc_batch(
            prediction_numpy,
            image_size,
            threshold=heatmap_threshold,
        )
        predicted_params = torch.from_numpy(predicted_params_numpy).to(device)
        confidence = torch.from_numpy(confidence_numpy).to(device)
        common_valid_mask = (
            flat_label_mask
            & ~torch.isnan(flat_target_params).any(dim=-1)
            & (confidence > 0)
        )
        labeled_line_count += int(flat_label_mask.sum())
        detected_line_count += int(common_valid_mask.sum())
        angle_errors.extend(
            collect_angle_errors(
                predicted_params,
                flat_target_params,
                common_valid_mask,
            )
        )
        rho_errors.extend(
            collect_rho_errors(
                predicted_params,
                flat_target_params,
                image_size,
                common_valid_mask,
            )
        )

        target_fit = fit_ribbon(compute_heatmap_moments(targets))
        surface_valid_mask = label_mask & target_fit.valid & loss_output.fit.valid
        surface_raw_angles.extend(
            collect_angle_errors_deg(
                loss_output.fit.raw_values[..., 2:4],
                target_fit.raw_values[..., 2:4],
                surface_valid_mask,
            )
        )
        surface_fitted_angles.extend(
            collect_angle_errors_deg(
                loss_output.fit.fitted_values[..., 2:4],
                target_fit.raw_values[..., 2:4],
                surface_valid_mask,
            )
        )
        surface_raw_centroids.extend(
            collect_centroid_errors(
                loss_output.fit.raw_values[..., :2],
                target_fit.raw_values[..., :2],
                surface_valid_mask,
            )
        )
        surface_fitted_centroids.extend(
            collect_centroid_errors(
                loss_output.fit.fitted_values[..., :2],
                target_fit.raw_values[..., :2],
                surface_valid_mask,
            )
        )

    denominator = max(1, batch_count)
    common_angle_summary = summarize_errors(angle_errors)
    common_rho_summary = summarize_errors(rho_errors)
    raw_angle_summary = summarize_errors(surface_raw_angles)
    fitted_angle_summary = summarize_errors(surface_fitted_angles)
    raw_centroid_summary = summarize_errors(surface_raw_centroids)
    fitted_centroid_summary = summarize_errors(surface_fitted_centroids)
    per_vertebra_metrics = {
        vertebra: {
            "peak_dist_mean": _mean_or_nan(values["peak_dists"]),
            "n_samples": len(values["peak_dists"]) // 4,
        }
        for vertebra, values in per_vertebra.items()
        if values["peak_dists"]
    }
    heatmap_loss = loss_sums["heatmap"] / denominator
    return {
        "val_loss_mse": heatmap_loss,
        "peak_dist_mean": _mean_or_nan(peak_distances),
        "blob_iou": _mean_or_nan(blob_ious),
        "angle_error_deg": common_angle_summary["mean"],
        "rho_error_px": common_rho_summary["mean"],
        "val_outlier_angle_rate": (
            float(np.mean(np.asarray(angle_errors) > outlier_angle_threshold))
            if angle_errors
            else float("nan")
        ),
        "val_outlier_rho_rate": (
            float(np.mean(np.asarray(rho_errors) > outlier_rho_threshold))
            if rho_errors
            else float("nan")
        ),
        "per_vertebra": per_vertebra_metrics,
        "val_loss": loss_sums["total"] / denominator,
        "val_heatmap_loss": heatmap_loss,
        "val_angle_loss": loss_sums["angle"] / denominator,
        "val_centroid_loss": loss_sums["centroid"] / denominator,
        "val_ribbon_loss": loss_sums["ribbon"] / denominator,
        "surface_raw_angle_error_deg": raw_angle_summary["mean"],
        "surface_raw_angle_error_deg_median": raw_angle_summary["median"],
        "surface_raw_angle_error_deg_p90": raw_angle_summary["p90"],
        "surface_fitted_angle_error_deg": fitted_angle_summary["mean"],
        "surface_fitted_angle_error_deg_median": fitted_angle_summary["median"],
        "surface_fitted_angle_error_deg_p90": fitted_angle_summary["p90"],
        "surface_raw_centroid_error_px": raw_centroid_summary["mean"],
        "surface_raw_centroid_error_px_median": raw_centroid_summary["median"],
        "surface_raw_centroid_error_px_p90": raw_centroid_summary["p90"],
        "surface_fitted_centroid_error_px": fitted_centroid_summary["mean"],
        "surface_fitted_centroid_error_px_median": fitted_centroid_summary["median"],
        "surface_fitted_centroid_error_px_p90": fitted_centroid_summary["p90"],
        "surface_detection_rate": (
            detected_line_count / labeled_line_count
            if labeled_line_count > 0
            else float("nan")
        ),
        "labeled_line_count": float(labeled_line_count),
        "detected_line_count": float(detected_line_count),
    }
