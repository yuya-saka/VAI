from __future__ import annotations

import math

import pytest
from torch import nn

from fracture_detection.baseline0.training.optimization import (
    LearningRateController,
    create_cosine_scheduler,
    create_optimizer,
    optimizer_learning_rates,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.head = nn.Linear(4, 1)
        self.trainable_history: list[bool] = []

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.trainable_history.append(trainable)


def test_optimizer_applies_weight_decay_to_all_parameters() -> None:
    """RSNA Stage1と同じく、bias・BatchNormもweight decayの対象にする。"""
    model = _TinyModel()

    optimizer = create_optimizer(
        model,
        weight_decay=1e-4,
        backbone_learning_rate=1e-4,
        head_learning_rate=3e-4,
    )

    assert {group["weight_decay"] for group in optimizer.param_groups} == {1e-4}
    assert {group["category"] for group in optimizer.param_groups} == {
        "backbone",
        "head",
    }
    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(grouped) == len(list(model.parameters()))
    assert any(parameter.ndim <= 1 for parameter in grouped)


def test_learning_rate_controller_can_warm_all_layers() -> None:
    model = _TinyModel()
    optimizer = create_optimizer(
        model,
        weight_decay=1e-4,
        backbone_learning_rate=1e-4,
        head_learning_rate=3e-4,
    )
    controller = LearningRateController(
        steps_per_epoch=10,
        freeze_backbone_epochs=0,
        warmup_epochs=2,
        warmup_start_factor=0.1,
        backbone_learning_rate=1e-4,
        head_learning_rate=3e-4,
    )

    controller.set_epoch_state(model, 0)
    initial = controller.apply(optimizer, global_step=0)
    warmup_end = controller.apply(optimizer, global_step=19)

    assert model.trainable_history == [True]
    assert initial == pytest.approx((1e-5, 3e-5))
    assert warmup_end == pytest.approx((1e-4, 3e-4))


def test_cosine_scheduler_matches_rsna_single_cycle() -> None:
    model = _TinyModel()
    optimizer = create_optimizer(
        model,
        weight_decay=1e-4,
        backbone_learning_rate=2.3e-4,
        head_learning_rate=2.3e-4,
    )
    scheduler = create_cosine_scheduler(
        optimizer,
        max_epochs=4,
        backbone_min_learning_rate=2.3e-5,
        head_min_learning_rate=2.3e-5,
    )

    epoch_learning_rates = [optimizer_learning_rates(optimizer)]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        epoch_learning_rates.append(optimizer_learning_rates(optimizer))

    for epoch_index, learning_rates in enumerate(epoch_learning_rates):
        expected = (
            2.3e-5
            + (2.3e-4 - 2.3e-5) * (1.0 + math.cos(math.pi * epoch_index / 4)) / 2.0
        )
        assert learning_rates == pytest.approx((expected, expected))

    optimizer.step()
    scheduler.step()

    assert optimizer_learning_rates(optimizer) == pytest.approx((2.3e-5, 2.3e-5))
