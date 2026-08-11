from __future__ import annotations

import torch

from fracture_detection.common.losses import region_bce


def test_region_bce_uses_entailed_negatives() -> None:
    region_logits = torch.zeros((4, 4), requires_grad=True)
    region_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    region_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    vertebra_targets = torch.tensor([1.0, 1.0, 0.0, 1.0])

    loss = region_bce(region_logits, region_targets, region_valid, vertebra_targets)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-6)
    assert region_logits.grad is not None
    assert torch.any(region_logits.grad[2] != 0)
    assert torch.all(region_logits.grad[3] == 0)
