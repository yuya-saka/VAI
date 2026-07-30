from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from train_models.stage4 import train as train_module
from train_models.stage4.scripts.run_stage4_confirmatory import (
    validate_confirmatory_config_difference,
)
from train_models.stage4.src import trainer
from train_models.stage4.src.data_utils import load_config
from train_models.stage4.src.model import Stage4Output
from train_models.stage4.src.trainer import _amp_settings, _stage4_loss, train_epoch
from train_models.stage4.utils.diagnostics import DiagnosticHistory


class _TinyStage4Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, images: Tensor, region_masks: Tensor) -> Stage4Output:
        del region_masks
        batch, slices = images.shape[:2]
        base = images.mean(dim=(1, 2, 3, 4)) * self.scale
        instances = base[:, None, None].expand(batch, slices, 4)
        regions = base[:, None].expand(batch, 4)
        valid_instances = torch.ones_like(instances, dtype=torch.bool)
        valid_regions = torch.ones_like(regions, dtype=torch.bool)
        attention = torch.full_like(instances, 1.0 / slices)
        weights = torch.softmax(regions, dim=1)
        plane_valid = torch.ones(batch, slices, dtype=torch.bool)
        return Stage4Output(
            base,
            instances,
            regions,
            attention,
            instances.mean(dim=2),
            weights,
            valid_instances,
            valid_regions,
            plane_valid,
            torch.ones(batch, dtype=torch.bool),
        )


def test_default_configs_fix_epochs_and_differ_only_by_region_lambda() -> None:
    mixed = load_config(Path("train_models/stage4/config/stage4_mixed.yaml"))
    weak = load_config(Path("train_models/stage4/config/stage4_weak_only.yaml"))

    assert mixed["training"]["fixed_epochs"] == 75
    assert mixed["training"]["validation_interval_epochs"] == 1
    assert "early_stopping_patience" not in mixed["training"]
    assert mixed["training"]["lambda_neg"] == 0.05
    assert mixed["augmentation"]["vertical_flip_p"] == 0.0
    assert mixed["augmentation"]["transpose_p"] == 0.0
    assert mixed["training"]["lambda_region_scale"] == 1.0
    assert weak["training"]["lambda_region_scale"] == 0.0

    mixed_comparable = {
        **mixed,
        "experiment": {**mixed["experiment"], "name": "arm"},
        "training": {
            **mixed["training"],
            "lambda_region_scale": "arm-specific",
        },
    }
    weak_comparable = {
        **weak,
        "experiment": {**weak["experiment"], "name": "arm"},
        "training": {
            **weak["training"],
            "lambda_region_scale": "arm-specific",
        },
    }
    assert mixed_comparable == weak_comparable
    validate_confirmatory_config_difference()


def test_seed_override_keeps_fixed_negative_sampler_seed() -> None:
    config = load_config(Path("train_models/stage4/config/stage4_mixed.yaml"))
    arguments = argparse.Namespace(
        config=None,
        start_fold=2,
        end_fold=2,
        seed=46,
        resume=False,
    )

    updated = train_module.apply_overrides(config, arguments)

    assert updated["data"]["random_seed"] == 46
    assert updated["model"]["scramble_seed"] == 46
    assert updated["training"]["negative_sampler_seed"] == 42
    assert updated["training"]["fixed_epochs"] == 75


def test_amp_settings_disables_amp_on_cpu() -> None:
    enabled, dtype = _amp_settings(
        {"training": {"use_amp": True, "amp_dtype": "bfloat16"}},
        torch.device("cpu"),
    )

    assert enabled is False
    assert dtype == torch.bfloat16


def test_weak_only_region_loss_has_zero_gradient() -> None:
    model = _TinyStage4Model()
    images = torch.ones(4, 2, 1, 4, 4)
    regions = torch.ones(4, 2, 4, 4, dtype=torch.uint8)
    output = model(images, regions)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    region_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    supervision = torch.tensor([True, False, True, False])

    _, parts = _stage4_loss(
        output,
        targets,
        region_targets,
        supervision,
        torch.ones(4),
        {
            "training": {
                "positive_weight": 2.0,
                "lambda_neg": 0.05,
                "lambda_region_scale": 0.0,
            }
        },
        epoch=4,
        population_counts=(1, 1, 2, 1),
    )
    gradient = torch.autograd.grad(
        parts["weighted_region_loss"],
        output.region_evidence_logits,
    )[0]

    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_train_epoch_logs_three_losses_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = torch.randint(0, 255, (8, 2, 1, 4, 4), dtype=torch.uint8)
    regions = torch.ones(8, 2, 4, 4, dtype=torch.uint8)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0] * 2)
    region_targets = torch.tensor([[1.0, 0.0, 0.0, 1.0]] * 2 + [[0.0] * 4] * 2).repeat(
        (2, 1)
    )
    supervision = torch.tensor([True, False, True, False] * 2)
    loader = DataLoader(
        TensorDataset(
            images,
            regions,
            targets,
            region_targets,
            supervision,
        ),
        batch_size=4,
    )
    model = _TinyStage4Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    config = {
        "training": {
            "p_mixup": 0.0,
            "positive_weight": 2.0,
            "lambda_neg": 0.05,
            "lambda_region_scale": 1.0,
            "diagnostic_interval_steps": 1,
            "gradient_clip_norm": 1.0,
            "use_amp": False,
            "amp_dtype": "bfloat16",
        }
    }
    monkeypatch.setattr(
        trainer,
        "_shared_encoder_parameters",
        lambda current_model: list(current_model.parameters()),
    )

    stats, diagnostics = train_epoch(
        model,
        loader,
        optimizer,
        scaler,
        torch.device("cpu"),
        config,
        epoch=0,
        population_counts=(100, 300, 900, 100),
        pos_weight=torch.ones(4),
        diagnostic_history=DiagnosticHistory(),
        is_main=True,
    )

    assert stats["vertebra_loss"] > 0
    assert stats["region_loss"] > 0
    assert stats["negative_instance_loss"] > 0
    assert stats["weighted_region_loss"] > 0
    assert stats["weighted_negative_loss"] > 0
    assert diagnostics is not None
    assert "warn_gradient_conflict" in diagnostics
