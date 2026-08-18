from __future__ import annotations

import pytest
import torch

from fracture_detection.baseline0.modeling.losses import (
    bag_probabilities,
    broadcast_bce_loss,
    broadcast_targets,
)


def test_broadcast_bce_matches_manual_target_expansion_with_pos_weight() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0.0, 1.0])

    loss = broadcast_bce_loss(logits, targets, pos_weight=2.0)
    expanded_targets = targets.unsqueeze(1).expand(-1, 2)
    element_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        expanded_targets,
        reduction="none",
    )
    weights = torch.where(expanded_targets > 0, 2.0, 1.0)
    expected = (element_losses * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert broadcast_targets(targets, 2).tolist() == [[0.0, 0.0], [1.0, 1.0]]


def test_broadcast_bce_rejects_non_positive_pos_weight() -> None:
    logits = torch.zeros((2, 2))
    targets = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="pos_weight"):
        broadcast_bce_loss(logits, targets, pos_weight=0.0)


def test_bag_probabilities_are_mean_sigmoid() -> None:
    logits = torch.tensor([[0.0, 0.0], [-10.0, 10.0]])

    probabilities = bag_probabilities(logits)

    assert torch.allclose(probabilities, torch.tensor([0.5, 0.5]), atol=1e-4)
