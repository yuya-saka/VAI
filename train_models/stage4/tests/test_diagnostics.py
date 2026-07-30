import pytest
import torch

from train_models.stage4.utils.diagnostics import (
    DiagnosticHistory,
    gradient_alignment,
    pooling_diagnostics,
)


def test_gradient_alignment_reports_conflict_and_weighted_ratio() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    vertebra_loss = parameter.sum()
    region_loss = -2.0 * parameter.sum()

    result = gradient_alignment(
        vertebra_loss,
        region_loss,
        [parameter],
        lambda_region=0.5,
    )

    assert result["gradient_cosine"] == pytest.approx(-1.0)
    assert result["weighted_gradient_norm_ratio"] == pytest.approx(1.0)


def test_pooling_diagnostics_flags_single_region_collapse() -> None:
    logits = torch.tensor([[4.0, 1.0, 0.0, -1.0]] * 4)
    weights = torch.tensor([[0.99, 0.01, 0.0, 0.0]] * 4)
    targets = torch.ones(4)
    valid = torch.ones(4, 4, dtype=torch.bool)

    result = pooling_diagnostics(logits, weights, targets, valid)

    assert result["winner_share_r1"] == 1.0
    assert result["max_winner_share"] == 1.0
    assert result["pool_weight_above_095_fraction"] == 1.0


def test_diagnostic_history_requires_two_epochs_for_gradient_warnings() -> None:
    history = DiagnosticHistory()
    record = {
        "gradient_cosine": -0.5,
        "weighted_gradient_norm_ratio": 4.0,
        "max_winner_share": 0.8,
        "pool_weight_above_095_fraction": 0.9,
    }

    history.add_step(record)
    first = history.summarize_epoch()
    history.add_step(record)
    second = history.summarize_epoch()

    assert first["warn_gradient_conflict"] is False
    assert first["warn_gradient_ratio"] is False
    assert second["warn_gradient_conflict"] is True
    assert second["warn_gradient_ratio"] is True
    assert second["warn_winner_collapse"] is True
    assert second["warn_pool_concentration"] is True
