from __future__ import annotations

import torch

from fracture_detection.common.losses import region_bce


def test_region_bce_uses_only_explicitly_valid_targets() -> None:
    region_logits = torch.zeros((3, 4), requires_grad=True)
    region_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    region_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )

    loss = region_bce(region_logits, region_targets, region_valid)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-6)
    assert region_logits.grad is not None
    assert torch.any(region_logits.grad[:2] != 0)
    assert torch.all(region_logits.grad[2] == 0)


def test_region_bce_returns_differentiable_zero_without_annotations() -> None:
    logits = torch.ones((2, 4), requires_grad=True)

    loss = region_bce(logits, torch.zeros_like(logits), torch.zeros_like(logits).bool())
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.all(logits.grad == 0)
