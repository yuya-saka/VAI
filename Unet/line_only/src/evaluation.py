"""検証データに対するヒートマップ・直線評価。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import losses as line_losses
from ..utils import metrics as line_metrics
from ..utils.metrics import collect_blob_ious
from .model import VERTEBRA_TO_IDX

Batch = dict[str, Any]


def peak_dist(pred: np.ndarray, gt: np.ndarray) -> float:
    """ヒートマップピーク間距離を返す。"""
    gt_y, gt_x = np.unravel_index(np.argmax(gt), gt.shape)
    pred_y, pred_x = np.unravel_index(np.argmax(pred), pred.shape)
    return math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)


def _vertebra_indices(batch: Batch, device: torch.device) -> torch.Tensor:
    """バッチの椎体名をモデル入力用インデックスへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0) for vertebra in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    image_size: int = 224,
    heatmap_threshold: line_losses.ThresholdSpec = 0.2,
    outlier_angle_thresh: float | None = None,
    outlier_rho_thresh: float | None = None,
) -> dict[str, Any]:
    """モデルのヒートマップ品質と直線幾何誤差を評価する。"""
    model.eval()
    mse_sum = 0.0
    batch_count = 0
    peak_dists: list[float] = []
    angle_errors: list[float] = []
    rho_errors: list[float] = []
    blob_ious: list[float] = []
    per_vertebra = {
        vertebra: {"peak_dists": []}
        for vertebra in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }

    for batch in loader:
        images = batch["image"].to(device).float()
        vertebra_indices = _vertebra_indices(batch, device)
        gt_heatmaps = batch["heatmaps"].to(device).float()
        gt_params = batch.get("line_params_gt")
        pred_heatmaps = torch.sigmoid(model(images, vertebra_indices))

        mse_sum += F.mse_loss(
            pred_heatmaps,
            gt_heatmaps,
            reduction="mean",
        ).item()
        batch_count += 1

        pred_numpy = pred_heatmaps.cpu().numpy()
        gt_numpy = gt_heatmaps.cpu().numpy()
        blob_ious.extend(collect_blob_ious(pred_numpy, gt_numpy))
        for batch_index, vertebra in enumerate(batch["vertebra"]):
            for channel in range(4):
                distance = peak_dist(
                    pred_numpy[batch_index, channel],
                    gt_numpy[batch_index, channel],
                )
                peak_dists.append(distance)
                if vertebra in per_vertebra:
                    per_vertebra[vertebra]["peak_dists"].append(distance)

        if gt_params is None:
            continue

        gt_params = gt_params.to(device).float()
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            pred_heatmaps,
            image_size,
            threshold=heatmap_threshold,
        )
        valid_mask = ~torch.isnan(gt_params).any(dim=-1) & (confidence > 0)
        angle_errors.extend(
            line_metrics.collect_angle_errors(pred_params, gt_params, valid_mask)
        )
        rho_errors.extend(
            line_metrics.collect_rho_errors(pred_params, gt_params, image_size, valid_mask)
        )

    per_vertebra_metrics = {
        vertebra: {
            "peak_dist_mean": float(np.nanmean(values["peak_dists"])),
            "n_samples": len(values["peak_dists"]) // 4,
        }
        for vertebra, values in per_vertebra.items()
        if values["peak_dists"]
    }
    metrics: dict[str, Any] = {
        "val_loss_mse": mse_sum / max(1, batch_count),
        "peak_dist_mean": float(np.nanmean(peak_dists)),
        "blob_iou": float(np.nanmean(blob_ious)) if blob_ious else float("nan"),
        "per_vertebra": per_vertebra_metrics,
    }
    if angle_errors:
        metrics["angle_error_deg"] = float(np.nanmean(angle_errors))
        metrics["rho_error_px"] = float(np.nanmean(rho_errors))
        if outlier_angle_thresh is not None:
            metrics["val_outlier_angle_rate"] = float(
                np.mean([e > outlier_angle_thresh for e in angle_errors])
            )
        if outlier_rho_thresh is not None:
            metrics["val_outlier_rho_rate"] = float(
                np.mean([e > outlier_rho_thresh for e in rho_errors])
            )
    return metrics
