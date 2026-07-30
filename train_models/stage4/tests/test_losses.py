import numpy as np
import pytest
import torch

from train_models.stage3.utils.losses import weighted_bce
from train_models.stage4.utils.losses import (
    compute_region_pos_weight,
    lambda_region_schedule,
    region_loss,
    stratified_negative_instance_loss,
    stratified_vertebra_loss,
)


def test_compute_region_pos_weight_includes_sampled_negatives_and_clips() -> None:
    labels = np.asarray(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.int8,
    )

    weights = compute_region_pos_weight(labels, n_negative_sampled=3)

    torch.testing.assert_close(
        weights,
        torch.tensor([2.0, 5.0, 5.0, 1.0]),
    )


def test_region_loss_uses_strong_and_negative_but_not_weak() -> None:
    logits = torch.tensor(
        [
            [0.2, -0.3, 0.4, -0.5],
            [20.0, 20.0, 20.0, 20.0],
            [-0.2, -0.4, -0.6, -0.8],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    supervision = torch.tensor([True, False, True])
    weights = torch.ones(4)

    loss = region_loss(logits, targets, supervision, weights)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[[0, 2]],
        targets[[0, 2]],
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), expected.detach())
    assert torch.count_nonzero(logits.grad[1]).item() == 0
    assert torch.count_nonzero(logits.grad[[0, 2]]).item() > 0


def test_region_loss_without_supervision_is_differentiable_zero() -> None:
    logits = torch.randn(2, 4, requires_grad=True)

    loss = region_loss(
        logits,
        torch.zeros_like(logits),
        torch.zeros(2, dtype=torch.bool),
        torch.ones(4),
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.25), (1, 0.4375), (2, 0.625), (3, 0.8125), (4, 1.0), (10, 1.0)],
)
def test_lambda_region_schedule(epoch: int, expected: float) -> None:
    assert lambda_region_schedule(epoch) == expected


def test_stratified_vertebra_loss_matches_direct_population_risk() -> None:
    logits = torch.tensor([0.2, 0.2, -0.4, -0.4, 0.5, 0.5, -1.0, -1.0])
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    supervision = torch.tensor([True, True, False, False, True, True, False, False])
    n_strong, n_weak, n_negative, n_sampled_negative = 3, 7, 20, 3

    loss, parts = stratified_vertebra_loss(
        logits,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        supervision,
        n_strong=n_strong,
        n_weak=n_weak,
        n_negative=n_negative,
        n_sampled_negative=n_sampled_negative,
        positive_weight=2.0,
    )
    population_logits = torch.tensor(
        [0.2] * n_strong
        + [-0.4] * n_weak
        + [0.5] * n_sampled_negative
        + [-1.0] * (n_negative - n_sampled_negative)
    )
    population_targets = torch.tensor([1.0] * (n_strong + n_weak) + [0.0] * n_negative)
    expected = weighted_bce(population_logits, population_targets, 2.0)

    torch.testing.assert_close(loss, expected)
    assert set(parts) == {
        "strong_bag_loss",
        "weak_bag_loss",
        "negative_bag_loss",
        "sampled_negative_bag_loss",
        "other_negative_bag_loss",
    }


def test_stratified_vertebra_loss_requires_all_batch_strata() -> None:
    logits = torch.zeros(4)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    supervision = torch.tensor([True, True, True, False])

    with pytest.raises(ValueError, match="all four strata"):
        stratified_vertebra_loss(
            logits,
            targets,
            torch.ones(4, dtype=torch.bool),
            supervision,
            n_strong=1,
            n_weak=1,
            n_negative=2,
            n_sampled_negative=1,
        )


def test_stratified_negative_instance_loss_restores_negative_population() -> None:
    logits = torch.tensor([0.5, 0.5, -1.0, -1.0]).view(4, 1, 1)
    valid = torch.ones_like(logits, dtype=torch.bool)
    targets = torch.zeros(4)
    supervision = torch.tensor([True, True, False, False])

    loss = stratified_negative_instance_loss(
        logits,
        valid,
        targets,
        torch.ones(4, dtype=torch.bool),
        supervision,
        n_negative=10,
        n_sampled_negative=2,
    )
    population_logits = torch.tensor([0.5] * 2 + [-1.0] * 8)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        population_logits,
        torch.zeros_like(population_logits),
    )

    torch.testing.assert_close(loss, expected)
