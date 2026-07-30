"""line_only共通指標とsurface固有指標の評価テスト。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from line_surface_3d.src.evaluation import evaluate
from line_surface_3d.utils.detection import extract_gt_line_params


class FixedHeatmapModel(nn.Module):
    """固定ヒートマップをlogitとして返すテストモデル。"""

    def __init__(self, heatmaps: torch.Tensor) -> None:
        super().__init__()
        clipped = heatmaps.clamp(1e-6, 1.0 - 1e-6)
        self.register_buffer(
            "fixed_logits",
            torch.logit(clipped).flatten(1, 2),
        )

    def forward(
        self,
        images: torch.Tensor,
        vertebra_indices: torch.Tensor,
    ) -> torch.Tensor:
        """固定logitを返す。"""
        del images, vertebra_indices
        return self.fixed_logits


def _horizontal_heatmaps(
    slab_size: int,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """水平線heatmapとline_only形式GT parameterを作る。"""
    y_grid = np.arange(image_size, dtype=np.float32)[:, None]
    line_positions = (4.0, 6.0, 9.0, 11.0)
    slices: list[np.ndarray] = []
    parameters = np.empty((slab_size, 4, 2), dtype=np.float32)
    for slab_index in range(slab_size):
        channels = []
        for line_index, line_y in enumerate(line_positions):
            heatmap = np.exp(-((y_grid - line_y) ** 2) / (2.0 * 0.7**2))
            heatmap = np.repeat(heatmap, image_size, axis=1)
            channels.append(heatmap.astype(np.float32))
            parameters[slab_index, line_index] = extract_gt_line_params(
                [[2.0, line_y], [13.0, line_y]],
                image_size,
            )
        slices.append(np.stack(channels))
    return (
        torch.from_numpy(np.stack(slices)).unsqueeze(0),
        torch.from_numpy(parameters).unsqueeze(0),
    )


def test_evaluate_returns_line_only_common_and_surface_metrics() -> None:
    """共通指標を同名で出し、surface固有指標を追加する。"""
    slab_size = 3
    image_size = 16
    heatmaps, line_params_gt = _horizontal_heatmaps(slab_size, image_size)
    batch: dict[str, Any] = {
        "image": torch.zeros(1, 2 * slab_size, image_size, image_size),
        "heatmaps": heatmaps,
        "label_mask": torch.ones(1, slab_size, 4, dtype=torch.bool),
        "line_params_gt": line_params_gt,
        "vertebra": ["C1"],
    }
    config = {
        "data": {
            "slab_size": slab_size,
            "image_size": image_size,
        },
        "evaluation": {
            "heatmap_threshold": {
                "mode": "adaptive",
                "min": 0.1,
                "peak_ratio": 0.4,
            },
            "outlier_angle_threshold_deg": 10.0,
            "outlier_rho_threshold_px": 8.0,
        },
        "loss": {"ribbon": {"enabled": False}},
    }

    metrics = evaluate(
        FixedHeatmapModel(heatmaps),
        [batch],
        torch.device("cpu"),
        config,
    )

    common_keys = {
        "val_loss_mse",
        "peak_dist_mean",
        "blob_iou",
        "angle_error_deg",
        "rho_error_px",
        "val_outlier_angle_rate",
        "val_outlier_rho_rate",
        "per_vertebra",
    }
    surface_keys = {
        "surface_raw_angle_error_deg",
        "surface_fitted_angle_error_deg",
        "surface_raw_centroid_error_px",
        "surface_fitted_centroid_error_px",
        "surface_detection_rate",
    }
    assert common_keys <= metrics.keys()
    assert surface_keys <= metrics.keys()
    assert metrics["val_loss_mse"] < 1e-8
    assert metrics["peak_dist_mean"] == 0.0
    assert metrics["blob_iou"] == 1.0
    assert metrics["angle_error_deg"] < 1e-3
    assert metrics["rho_error_px"] < 0.1
    assert metrics["surface_detection_rate"] == 1.0
    assert metrics["per_vertebra"]["C1"]["n_samples"] == slab_size
