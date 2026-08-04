"""中心画像単位のheatmap・直線評価。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .metrics import detect_line_from_heatmap, dice_score, line_errors
from .model import VERTEBRA_TO_INDEX


def vertebra_indices(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """椎体名をモデル条件付け用indexへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_INDEX.get(name, 0) for name in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """各教師画像を1回だけ評価し、崩壊率を含む指標を返す。"""
    model.eval()
    image_size = int(config["data"]["image_size"])
    evaluation_config = config.get("evaluation", {})
    threshold = evaluation_config.get(
        "heatmap_threshold",
        {"min": 0.1, "peak_ratio": 0.4},
    )
    min_confidence = float(evaluation_config.get("min_line_confidence", 0.05))
    outlier_angle_threshold = float(
        evaluation_config.get("outlier_angle_threshold_deg", 10.0)
    )
    outlier_rho_threshold = float(
        evaluation_config.get("outlier_rho_threshold_px", 8.0)
    )
    angle_errors: list[float] = []
    rho_errors: list[float] = []
    dice_scores: list[float] = []
    squared_error_sum = 0.0
    pixel_count = 0
    invalid_count = 0
    line_count = 0
    per_line_errors = {
        line_index: {"angle": [], "rho": [], "invalid": 0, "count": 0}
        for line_index in range(4)
    }

    for batch in loader:
        images = batch["image"].to(device).float()
        targets = batch["heatmaps"].to(device).float()
        predictions = torch.sigmoid(model(images, vertebra_indices(batch, device)))
        center_predictions = predictions[:, predictions.shape[1] // 2]
        squared_error_sum += float((center_predictions - targets).square().sum())
        pixel_count += targets.numel()
        prediction_numpy = center_predictions.cpu().numpy()
        target_numpy = targets.cpu().numpy()
        gt_params = batch["line_params_gt"].cpu().numpy()

        for batch_index in range(prediction_numpy.shape[0]):
            for line_index in range(4):
                line_count += 1
                per_line_errors[line_index]["count"] += 1
                dice_scores.append(
                    dice_score(
                        prediction_numpy[batch_index, line_index],
                        target_numpy[batch_index, line_index],
                    )
                )
                detected = detect_line_from_heatmap(
                    prediction_numpy[batch_index, line_index],
                    image_size,
                    threshold,
                    min_confidence,
                )
                if detected is None:
                    invalid_count += 1
                    per_line_errors[line_index]["invalid"] += 1
                    continue
                angle_error, rho_error = line_errors(
                    detected,
                    float(gt_params[batch_index, line_index, 0]),
                    float(gt_params[batch_index, line_index, 1]),
                )
                angle_errors.append(angle_error)
                rho_errors.append(rho_error)
                per_line_errors[line_index]["angle"].append(angle_error)
                per_line_errors[line_index]["rho"].append(rho_error)

    angle_mean = float(np.mean(angle_errors)) if angle_errors else float("nan")
    rho_mean = float(np.mean(rho_errors)) if rho_errors else float("nan")
    angle_arm = image_size / 4.0
    combined = (
        rho_mean + angle_arm * math.radians(angle_mean)
        if math.isfinite(angle_mean) and math.isfinite(rho_mean)
        else float("nan")
    )
    return {
        "val_heatmap_mse": squared_error_sum / max(1, pixel_count),
        "val_heatmap_dice": float(np.mean(dice_scores))
        if dice_scores
        else float("nan"),
        "line_angle_error_deg": angle_mean,
        "line_rho_error_px": rho_mean,
        "line_combined_error_px": combined,
        "heatmap_collapse_rate": invalid_count / max(1, line_count),
        "outlier_angle_rate": (
            float(np.mean(np.asarray(angle_errors) > outlier_angle_threshold))
            if angle_errors
            else float("nan")
        ),
        "outlier_rho_rate": (
            float(np.mean(np.asarray(rho_errors) > outlier_rho_threshold))
            if rho_errors
            else float("nan")
        ),
        "evaluated_line_count": len(angle_errors),
        "per_line": {
            f"line_{line_index + 1}": {
                "angle_error_deg": (
                    float(np.mean(values["angle"])) if values["angle"] else float("nan")
                ),
                "rho_error_px": (
                    float(np.mean(values["rho"])) if values["rho"] else float("nan")
                ),
                "collapse_rate": values["invalid"] / max(1, values["count"]),
                "count": values["count"],
            }
            for line_index, values in per_line_errors.items()
        },
    }
