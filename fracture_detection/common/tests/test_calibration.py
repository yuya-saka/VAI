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
