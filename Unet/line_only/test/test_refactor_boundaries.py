"""リファクタリング後の公開API境界を検証する。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from line_only.src.evaluation import evaluate, peak_dist
from line_only.src.example_writer import save_examples
from line_only.src.inference import predict_lines_and_eval_test
from line_only.src.trainer import (
    evaluate as trainer_evaluate,
)
from line_only.src.trainer import (
    predict_lines_and_eval_test as trainer_predict_lines,
)
from line_only.src.trainer import (
    run_training_loop,
)
from line_only.src.trainer import (
    save_examples as trainer_save_examples,
)


class _ConstantHeatmapModel(nn.Module):
    """学習ループ検証用の最小モデル。"""

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        images: torch.Tensor,
        vertebra_indices: torch.Tensor,
    ) -> torch.Tensor:
        """中央行を強調した4チャンネルヒートマップを返す（異方性→confidence > 0）。"""
        del vertebra_indices
        B, _, H, W = images.shape
        stripe = torch.zeros(B, 4, H, W, device=images.device)
        stripe[:, :, H // 2, :] = 3.0
        return self.logit + stripe


def test_trainer_reexports_public_functions() -> None:
    """既存 import パスが新モジュール実装を再公開する。"""
    assert trainer_evaluate is evaluate
    assert trainer_predict_lines is predict_lines_and_eval_test
    assert trainer_save_examples is save_examples


def test_peak_dist_returns_euclidean_distance() -> None:
    """ピーク距離がユークリッド距離と一致する。"""
    prediction = np.zeros((8, 8), dtype=np.float32)
    ground_truth = np.zeros((8, 8), dtype=np.float32)
    prediction[5, 6] = 1.0
    ground_truth[1, 3] = 1.0

    assert peak_dist(prediction, ground_truth) == math.sqrt(25)


def test_run_training_loop_saves_best_checkpoint(tmp_path: Path) -> None:
    """1 epoch の学習でベストチェックポイントを保存する。"""
    model = _ConstantHeatmapModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    batch = {
        "image": torch.zeros(1, 2, 8, 8),
        "heatmaps": torch.zeros(1, 4, 8, 8),
        "vertebra": ["C1"],
        "line_params_gt": torch.zeros(1, 4, 2),
    }
    checkpoint_path = tmp_path / "best.pt"

    run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=[batch],
        val_loader=[batch],
        device=torch.device("cpu"),
        cfg={
            "data": {"image_size": 8},
            "training": {"epochs": 1, "early_stopping_patience": 1},
            "evaluation": {"heatmap_threshold": 0.2},
            "loss": {"use_line_loss": False},
        },
        best_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
