"""テスト直線推論とline_only互換の画像・JSON保存。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .evaluation import vertebra_indices
from .metrics import (
    DetectedLine,
    detect_line_from_heatmap,
    detected_line_to_dict,
    line_errors,
    line_extent,
)
from .visualization import (
    draw_heatmap_with_lines,
    draw_line_comparison,
    save_heatmap_grid,
    save_heatmap_overlay,
)


def _sample_name(batch: dict[str, Any], batch_index: int) -> str:
    """line_onlyと同じsample識別名を返す。"""
    sample = str(batch["sample"][batch_index])
    vertebra = str(batch["vertebra"][batch_index])
    slice_index = int(batch["slice_idx"][batch_index])
    return f"{sample}_{vertebra}_slice{slice_index:03d}"


def _parse_lines(batch: dict[str, Any], batch_index: int) -> dict[str, Any]:
    """Datasetが渡したGT線JSONを復元する。"""
    value = json.loads(batch["lines_json"][batch_index])
    return value if isinstance(value, dict) else {}


def _perpendicular_distance(
    points_xy: list[list[float]],
    detection: DetectedLine,
    image_size: int,
) -> float:
    """GT点から予測直線までの平均垂直距離を返す。"""
    points = np.asarray(points_xy, dtype=np.float64)
    centered = points - image_size / 2.0
    normal = np.asarray(
        [-math.sin(detection.angle_rad), math.cos(detection.angle_rad)],
        dtype=np.float64,
    )
    return float(np.mean(np.abs(centered @ normal - detection.rho_px)))


@torch.no_grad()
def save_examples(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    n_save: int = 16,
    tag: str = "VAL",
) -> None:
    """GTと予測heatmapのgrid・overlayを指定件数保存する。"""
    if n_save <= 0:
        return
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for batch in loader:
        images = batch["image"].to(device).float()
        targets = batch["heatmaps"].cpu().numpy()
        predictions = torch.sigmoid(model(images, vertebra_indices(batch, device)))
        center_position = predictions.shape[1] // 2
        center_images = images[:, center_position, 0].cpu().numpy()
        center_predictions = predictions[:, center_position].cpu().numpy()
        for batch_index in range(center_predictions.shape[0]):
            name = _sample_name(batch, batch_index)
            save_heatmap_grid(
                center_images[batch_index],
                targets[batch_index],
                output_dir / f"{tag}_{name}_GT_grid.png",
            )
            save_heatmap_grid(
                center_images[batch_index],
                center_predictions[batch_index],
                output_dir / f"{tag}_{name}_PRED_grid.png",
            )
            save_heatmap_overlay(
                center_images[batch_index],
                targets[batch_index],
                output_dir / f"{tag}_{name}_GT_merged.png",
            )
            save_heatmap_overlay(
                center_images[batch_index],
                center_predictions[batch_index],
                output_dir / f"{tag}_{name}_PRED_merged.png",
            )
            saved_count += 1
            if saved_count >= n_save:
                return


@torch.no_grad()
def predict_lines_and_save(
    config: dict[str, Any],
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """テスト全件の比較画像、3列画像、予測JSONを保存する。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = config.get("evaluation", {})
    threshold = evaluation_config.get(
        "heatmap_threshold",
        {"mode": "adaptive", "min": 0.1, "peak_ratio": 0.4},
    )
    min_confidence = float(evaluation_config.get("min_line_confidence", 0.05))
    line_extend_ratio = float(evaluation_config.get("line_extend_ratio", 1.0))
    image_size = int(config["data"]["image_size"])
    saved_count = 0

    for batch in loader:
        images = batch["image"].to(device).float()
        predictions = torch.sigmoid(model(images, vertebra_indices(batch, device)))
        center_position = predictions.shape[1] // 2
        center_images = images[:, center_position, 0].cpu().numpy()
        center_predictions = predictions[:, center_position].cpu().numpy()
        gt_params = batch["line_params_gt"].cpu().numpy()
        for batch_index in range(center_predictions.shape[0]):
            name = _sample_name(batch, batch_index)
            gt_lines = _parse_lines(batch, batch_index)
            pred_lines: dict[str, Any] = {}
            sample_metrics: dict[str, Any] = {}
            for channel in range(4):
                line_name = f"line_{channel + 1}"
                gt_points = gt_lines.get(line_name, [])
                detection = detect_line_from_heatmap(
                    center_predictions[batch_index, channel],
                    image_size,
                    threshold,
                    min_confidence,
                )
                pred_lines[line_name] = detected_line_to_dict(
                    detection,
                    line_extent(gt_points),
                    line_extend_ratio,
                    center_predictions.shape[-2:],
                )
                if detection is None:
                    sample_metrics[line_name] = {
                        "angle_error_deg": None,
                        "rho_error_px": None,
                        "perpendicular_dist_px": None,
                    }
                    continue
                angle_error, rho_error = line_errors(
                    detection,
                    float(gt_params[batch_index, channel, 0]),
                    float(gt_params[batch_index, channel, 1]),
                )
                sample_metrics[line_name] = {
                    "angle_error_deg": angle_error,
                    "rho_error_px": rho_error,
                    "perpendicular_dist_px": _perpendicular_distance(
                        gt_points,
                        detection,
                        image_size,
                    ),
                    "gt_angle_rad": float(gt_params[batch_index, channel, 0]),
                    "gt_rho_px": float(gt_params[batch_index, channel, 1]),
                    "pred_angle_rad": detection.angle_rad,
                    "pred_rho_px": detection.rho_px,
                }

            draw_line_comparison(
                center_images[batch_index],
                pred_lines,
                gt_lines,
                output_dir / f"{name}_comparison.png",
            )
            draw_heatmap_with_lines(
                center_images[batch_index],
                center_predictions[batch_index],
                pred_lines,
                gt_lines,
                output_dir / f"{name}_heatmap_lines.png",
            )
            (output_dir / f"{name}_PRED_lines.json").write_text(
                json.dumps(
                    {
                        "pred_lines": pred_lines,
                        "metrics": sample_metrics,
                        "heatmap_threshold_ref": threshold,
                        "line_extend_ratio": line_extend_ratio,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            saved_count += 1

    return {
        "n_samples": saved_count,
        "line_extend_ratio": line_extend_ratio,
        "heatmap_threshold_ref": threshold,
        "out_dir": str(output_dir),
    }
