"""テストデータに対する直線推論・集計・成果物保存。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils import losses as line_losses
from ..utils import metrics as line_metrics
from ..utils.detection import LinesJsonCache, detect_line_moments, line_extent
from ..utils.visualization import draw_heatmap_with_lines, draw_line_comparison
from .model import VERTEBRA_TO_IDX

Batch = dict[str, Any]
MetricLists = dict[str, list[float]]


def _mean_valid(values: list[float]) -> float:
    """None と NaN を除外した平均値を返す。"""
    valid_values = [
        value
        for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    ]
    if not valid_values:
        return float("nan")
    return float(np.mean(valid_values))


def _empty_metric_groups() -> dict[int, MetricLists]:
    """4本の直線別メトリクス格納先を作成する。"""
    return {channel: {"angle": [], "rho": [], "perp": []} for channel in range(1, 5)}


def _empty_vertebra_metrics() -> dict[str, MetricLists]:
    """椎体別メトリクス格納先を作成する。"""
    return {
        vertebra: {"angle": [], "rho": [], "perp": []}
        for vertebra in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }


def _vertebra_indices(batch: Batch, device: torch.device) -> torch.Tensor:
    """バッチの椎体名をモデル入力用インデックスへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0) for vertebra in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


def _line_metrics(
    gt_points: list[list[float]],
    gt_phi: float,
    gt_rho: float,
    pred_phi: float,
    pred_rho: float,
    image_size: int,
) -> tuple[float, float, float]:
    """単一直線の角度・rho・垂直距離誤差を計算する。"""
    gt_params = torch.tensor([[gt_phi, gt_rho]], dtype=torch.float32)
    pred_params = torch.tensor([[pred_phi, pred_rho]], dtype=torch.float32)
    valid_mask = torch.tensor([True])
    angle_error = line_metrics.compute_angle_error(
        pred_params,
        gt_params,
        valid_mask,
    )
    rho_error = line_metrics.compute_rho_error(
        pred_params,
        gt_params,
        image_size,
        valid_mask,
    )
    perpendicular_distance = line_metrics.compute_perpendicular_distance(
        gt_points,
        pred_phi,
        pred_rho,
        image_size,
    )
    return angle_error, rho_error, perpendicular_distance


def _append_metrics(
    values: MetricLists,
    angle_error: float,
    rho_error: float,
    perpendicular_distance: float,
) -> None:
    """メトリクス値を集計用リストへ追加する。"""
    values["angle"].append(angle_error)
    values["rho"].append(rho_error)
    values["perp"].append(perpendicular_distance)


def _summarize(
    saved_count: int,
    angle_errors: list[float],
    rho_errors: list[float],
    perpendicular_distances: list[float],
    per_channel: dict[int, MetricLists],
    per_vertebra: dict[str, MetricLists],
    line_extend_ratio: float,
    heatmap_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    """推論結果を全体・直線別・椎体別に集約する。"""
    return {
        "n_samples": int(saved_count),
        "angle_error_deg_mean": _mean_valid(angle_errors),
        "rho_error_px_mean": _mean_valid(rho_errors),
        "perpendicular_dist_px_mean": _mean_valid(perpendicular_distances),
        "per_channel": {
            f"line_{channel}": {
                "angle_error_deg_mean": _mean_valid(values["angle"]),
                "rho_error_px_mean": _mean_valid(values["rho"]),
                "perpendicular_dist_px_mean": _mean_valid(values["perp"]),
                "n": int(len(values["angle"])),
            }
            for channel, values in per_channel.items()
        },
        "per_vertebra": {
            vertebra: {
                "angle_error_deg_mean": (
                    _mean_valid(values["angle"]) if values["angle"] else None
                ),
                "rho_error_px_mean": (
                    _mean_valid(values["rho"]) if values["rho"] else None
                ),
                "perpendicular_dist_px_mean": (
                    _mean_valid(values["perp"]) if values["perp"] else None
                ),
                "n": int(len(values["angle"])),
            }
            for vertebra, values in per_vertebra.items()
        },
        "line_extend_ratio": float(line_extend_ratio),
        "heatmap_threshold_ref": float(heatmap_threshold),
        "out_dir": str(output_dir),
    }


@torch.no_grad()
def predict_lines_and_eval_test(
    cfg: dict[str, Any],
    model: nn.Module,
    test_loader: Iterable[Batch],
    device: torch.device,
    dataset_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """テストデータの直線推論、評価、可視化保存を実行する。"""
    model.eval()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = cfg.get("evaluation", {})
    line_extend_ratio = float(evaluation_config.get("line_extend_ratio", 1.10))
    heatmap_threshold = float(evaluation_config.get("heatmap_threshold", 0.15))
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    cache = LinesJsonCache(Path(dataset_root))

    angle_errors: list[float] = []
    rho_errors: list[float] = []
    perpendicular_distances: list[float] = []
    per_channel = _empty_metric_groups()
    per_vertebra = _empty_vertebra_metrics()
    saved_count = 0

    for batch in test_loader:
        images = batch["image"].to(device).float()
        vertebra_indices = _vertebra_indices(batch, device)
        pred_heatmaps = torch.sigmoid(model(images, vertebra_indices))
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            pred_heatmaps,
            image_size,
            threshold=heatmap_threshold,
        )

        images_numpy = images.cpu().numpy()
        heatmaps_numpy = pred_heatmaps.cpu().numpy()
        params_numpy = pred_params.cpu().numpy()
        confidence_numpy = confidence.cpu().numpy()

        for batch_index in range(heatmaps_numpy.shape[0]):
            sample = batch["sample"][batch_index]
            vertebra = batch["vertebra"][batch_index]
            slice_index = int(batch["slice_idx"][batch_index])
            ct_image = images_numpy[batch_index, 0]
            name = f"{sample}_{vertebra}_slice{slice_index:03d}"
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_index) or {}
            pred_lines: dict[str, Any] = {}
            sample_metrics: dict[str, Any] = {}

            for channel in range(4):
                line_name = f"line_{channel + 1}"
                gt_points = gt_lines.get(line_name)
                gt_phi, gt_rho = line_losses.extract_gt_line_params(
                    gt_points,
                    image_size,
                )
                pred_phi = float(params_numpy[batch_index, channel, 0])
                pred_rho = float(params_numpy[batch_index, channel, 1])
                pred_confidence = float(confidence_numpy[batch_index, channel])
                gt_length = line_extent(gt_points)
                length = gt_length if gt_length > 1e-6 else None
                pred_lines[line_name] = detect_line_moments(
                    heatmaps_numpy[batch_index, channel],
                    length_px=length,
                    extend_ratio=line_extend_ratio,
                )

                if np.isnan(gt_phi) or pred_confidence <= 0:
                    sample_metrics[line_name] = {
                        "angle_error_deg": None,
                        "rho_error_px": None,
                        "perpendicular_dist_px": None,
                    }
                    continue

                angle_error, rho_error, perpendicular_distance = _line_metrics(
                    gt_points,
                    gt_phi,
                    gt_rho,
                    pred_phi,
                    pred_rho,
                    image_size,
                )
                angle_errors.append(angle_error)
                rho_errors.append(rho_error)
                perpendicular_distances.append(perpendicular_distance)
                _append_metrics(
                    per_channel[channel + 1],
                    angle_error,
                    rho_error,
                    perpendicular_distance,
                )
                if vertebra in per_vertebra:
                    _append_metrics(
                        per_vertebra[vertebra],
                        angle_error,
                        rho_error,
                        perpendicular_distance,
                    )
                sample_metrics[line_name] = {
                    "angle_error_deg": float(angle_error),
                    "rho_error_px": float(rho_error),
                    "perpendicular_dist_px": float(perpendicular_distance),
                    "gt_phi": float(gt_phi),
                    "gt_rho": float(gt_rho),
                    "pred_phi": pred_phi,
                    "pred_rho": pred_rho,
                }

            draw_line_comparison(
                ct_image,
                pred_lines,
                gt_lines,
                output_dir / f"{name}_comparison.png",
            )
            draw_heatmap_with_lines(
                ct_image,
                heatmaps_numpy[batch_index],
                pred_lines,
                gt_lines,
                output_dir / f"{name}_heatmap_lines.png",
            )
            with (output_dir / f"{name}_PRED_lines.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "pred_lines": pred_lines,
                        "metrics": sample_metrics,
                        "heatmap_threshold_ref": heatmap_threshold,
                    },
                    file,
                    indent=2,
                )
            saved_count += 1

    return _summarize(
        saved_count,
        angle_errors,
        rho_errors,
        perpendicular_distances,
        per_channel,
        per_vertebra,
        line_extend_ratio,
        heatmap_threshold,
        output_dir,
    )
