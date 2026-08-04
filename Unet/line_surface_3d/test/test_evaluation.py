"""椎体×面を単位とした評価のテスト。"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from line_surface_3d.src.evaluation import evaluate

SLAB_SIZE = 15
IMAGE_SIZE = 64
SIGMA = 4.0
DIAGONAL = math.sqrt(2.0) * IMAGE_SIZE


class FixedHeatmapModel(nn.Module):
    """固定ヒートマップをlogitとして返すテストモデル。"""

    def __init__(self, heatmaps: torch.Tensor) -> None:
        super().__init__()
        clipped = heatmaps.clamp(1e-6, 1.0 - 1e-6)
        self.register_buffer("fixed_logits", torch.logit(clipped).flatten(1, 2))

    def forward(
        self,
        images: torch.Tensor,
        vertebra_indices: torch.Tensor,
    ) -> torch.Tensor:
        """固定logitを返す。"""
        del images, vertebra_indices
        return self.fixed_logits


def _config() -> dict[str, Any]:
    """テスト用の最小設定を返す。"""
    return {
        "data": {"slab_size": SLAB_SIZE, "image_size": IMAGE_SIZE},
        "loss": {"plane": {"enabled": True}},
        "evaluation": {"blob_iou_threshold": 0.1},
    }


def _batch(slope_image: float, window_start: int) -> dict[str, Any]:
    """既知の平面から1窓分のバッチを作る。"""
    rows = torch.arange(IMAGE_SIZE).float()[:, None]
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    line_params = torch.full((1, SLAB_SIZE, 4, 2), float("nan"))
    label_mask = torch.zeros(1, SLAB_SIZE, 4, dtype=torch.bool)
    slice_indices = torch.arange(window_start, window_start + SLAB_SIZE)
    reference_z = float(window_start + (SLAB_SIZE - 1) / 2.0)

    for index in range(SLAB_SIZE):
        global_z = float(slice_indices[index])
        center_y = 32.0 + slope_image * (global_z - reference_z)
        ridge = torch.exp(-((rows - center_y) ** 2) / (2.0 * SIGMA**2))
        heatmaps[0, index, 0] = ridge.expand(IMAGE_SIZE, IMAGE_SIZE)
        if 4 <= index <= 10:
            line_params[0, index, 0, 0] = math.pi / 2.0
            line_params[0, index, 0, 1] = (32.0 - center_y) / DIAGONAL
            label_mask[0, index, 0] = True

    return {
        "image": torch.zeros(1, 2 * SLAB_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        "heatmaps": heatmaps,
        "label_mask": label_mask,
        "line_params_gt": line_params,
        "slice_indices": slice_indices[None, :],
        "plane_slope_gt": torch.tensor([[-slope_image, 0.0, 0.0, 0.0]]),
        "plane_reliable": torch.tensor([[True, False, False, False]]),
        "plane_angle_gt": torch.tensor([[math.pi / 2.0, 0.0, 0.0, 0.0]]),
        "plane_rho0_gt": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        "plane_reference_z": torch.tensor([[reference_z] * 4]),
        "sample": ["sample1"],
        "vertebra": ["C4"],
    }


def test_evaluate_recovers_plane_parameters() -> None:
    """予測が教師と一致すれば平面3パラメータの誤差はほぼ0になる。"""
    batch = _batch(0.5, window_start=20)
    model = FixedHeatmapModel(batch["heatmaps"])
    metrics = evaluate(model, [batch], torch.device("cpu"), _config())

    assert metrics["plane_angle_error_deg"] < 0.5
    assert metrics["plane_rho_error_px"] < 0.5
    assert metrics["line_angle_error_deg"] < 0.5
    assert metrics["line_rho_error_px"] < 0.5
    assert metrics["tilt_error_px_per_slice"] < 0.05
    assert metrics["tilt_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["plane_combined_error_px"] < 1.0


def test_each_surface_is_counted_once() -> None:
    """重なり窓は集約され、面は1回だけ数えられる。

    以前は窓ごとの観測をそのまま数え、同じスライスを10〜15回重複計上していた。
    """
    batches = [_batch(0.5, window_start=start) for start in (20, 21, 22)]
    model = FixedHeatmapModel(batches[0]["heatmaps"])
    metrics = evaluate(model, batches, torch.device("cpu"), _config())

    # 3窓すべて同じ椎体・同じ1面なので、評価対象は1面
    assert metrics["evaluated_surface_count"] == 1.0
    assert metrics["reliable_surface_count"] == 1.0
    # global z 20..32 のうちアノテーション範囲(相対4..10)が重なる分は
    # 重複カウントせず、ユニークなzの数だけ観測が残る
    unique_annotated_z = len(
        {z for start in (20, 21, 22) for z in range(start + 4, start + 11)}
    )
    assert metrics["line_observation_count"] == float(unique_annotated_z)


def test_line_metrics_detect_per_slice_angle_noise_that_aggregate_hides() -> None:
    """各画像単位の指標は、面単位の集約比較では隠れるズレを検出する。

    モデルの予測平面は常に水平(角度一定)。GTは各スライスで
    ±10度ずつ振動させ、平均するとほぼ元の角度へ戻る(アノテーション雑音を模す)。
    面単位の集約同士を比べると打ち消し合ってほぼ0度になるが、
    各画像単位で見れば実際の10度ズレが検出できるべきである。
    """
    rows = torch.arange(IMAGE_SIZE).float()[:, None]
    heatmaps = torch.zeros(1, SLAB_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    line_params = torch.full((1, SLAB_SIZE, 4, 2), float("nan"))
    label_mask = torch.zeros(1, SLAB_SIZE, 4, dtype=torch.bool)
    slice_indices = torch.arange(20, 20 + SLAB_SIZE)

    for index in range(SLAB_SIZE):
        ridge = torch.exp(-((rows - 32.0) ** 2) / (2.0 * SIGMA**2))
        heatmaps[0, index, 0] = ridge.expand(IMAGE_SIZE, IMAGE_SIZE)

    noise_deg = 10.0
    for offset, index in enumerate(range(4, 11)):
        sign = 1.0 if offset % 2 == 0 else -1.0
        line_params[0, index, 0, 0] = math.pi / 2.0 + sign * math.radians(noise_deg)
        line_params[0, index, 0, 1] = 0.0
        label_mask[0, index, 0] = True

    batch = {
        "image": torch.zeros(1, 2 * SLAB_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        "heatmaps": heatmaps,
        "label_mask": label_mask,
        "line_params_gt": line_params,
        "slice_indices": slice_indices[None, :],
        "plane_slope_gt": torch.zeros(1, 4),
        "plane_reliable": torch.zeros(1, 4, dtype=torch.bool),
        "plane_angle_gt": torch.tensor([[math.pi / 2.0, 0.0, 0.0, 0.0]]),
        "plane_rho0_gt": torch.zeros(1, 4),
        "plane_reference_z": torch.tensor([[27.0, 0.0, 0.0, 0.0]]),
        "sample": ["sample1"],
        "vertebra": ["C4"],
    }
    model = FixedHeatmapModel(heatmaps)
    metrics = evaluate(model, [batch], torch.device("cpu"), _config())

    assert metrics["plane_angle_error_deg"] < 1.0
    assert metrics["line_angle_error_deg"] > 8.0


def test_metric_keys_are_present() -> None:
    """後段が参照する指標キーが揃っている。"""
    batch = _batch(0.3, window_start=20)
    model = FixedHeatmapModel(batch["heatmaps"])
    metrics = evaluate(model, [batch], torch.device("cpu"), _config())

    for key in (
        "val_loss_mse",
        "line_angle_error_deg",
        "line_rho_error_px",
        "plane_angle_error_deg",
        "plane_rho_error_px",
        "tilt_error_px_per_slice",
        "tilt_sign_accuracy",
        "plane_normal_error_deg",
        "plane_combined_error_px",
        "blob_iou",
        "per_vertebra",
    ):
        assert key in metrics
    assert "peak_dist_mean" not in metrics
