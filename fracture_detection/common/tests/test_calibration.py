from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from fracture_detection.common.calibration import calibrate_gradient_weight


def test_calibration_uses_median_log_ratio_without_mutating_model() -> None:
    model = nn.Linear(2, 1, bias=False)
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])]

    def loss_pair(inputs: Tensor) -> tuple[Tensor, Tensor]:
        prediction = model(inputs).sum()
        return prediction.square(), (prediction * 2.0).square()

    result = calibrate_gradient_weight(
        model,
        batches,
        tuple(model.parameters()),
        loss_pair,
        expected_batches=2,
    )

    assert result.coefficient == result.unclipped_coefficient
    assert result.coefficient == pytest.approx(0.125)
    assert all(
        torch.equal(initial[name], value) for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_calibration_keeps_recurrent_modules_training_and_restores_rng() -> None:
    class CalibrationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_norm = nn.BatchNorm1d(2)
            self.lstm = nn.LSTM(2, 2, num_layers=2, dropout=0.5, batch_first=True)
            self.head = nn.Linear(2, 1)

        def forward(self, inputs: Tensor) -> Tensor:
            normalized = self.batch_norm(inputs.transpose(1, 2)).transpose(1, 2)
            features, _ = self.lstm(normalized)
            return self.head(features[:, -1]).sum()

    model = CalibrationModel()
    batches = [torch.randn(2, 3, 2), torch.randn(2, 3, 2)]
    observed_modes: list[tuple[bool, bool, bool]] = []
    rng_before = torch.get_rng_state().clone()

    def loss_pair(inputs: Tensor) -> tuple[Tensor, Tensor]:
        observed_modes.append(
            (model.training, model.batch_norm.training, model.lstm.training)
        )
        prediction = model(inputs)
        return prediction.square(), (prediction * 2.0).square()

    calibrate_gradient_weight(
        model,
        batches,
        tuple(model.parameters()),
        loss_pair,
        expected_batches=2,
    )

    assert observed_modes == [(True, False, True), (True, False, True)]
    assert torch.equal(torch.get_rng_state(), rng_before)
