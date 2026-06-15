"""Tests for shared training utilities."""

from __future__ import annotations

import pytest
import torch

from learning.utils.training import (
    collate_bags,
    compute_ranking_metrics,
    lr_scale,
)


def test_lr_scale_warms_up_then_stays_constant() -> None:
    assert lr_scale(0, 3) == pytest.approx(0.2)
    assert lr_scale(1, 3) == pytest.approx(0.6)
    assert lr_scale(2, 3) == pytest.approx(1.0)
    assert lr_scale(3, 3) == pytest.approx(1.0)


def test_compute_ranking_metrics_returns_zero_for_single_class() -> None:
    assert compute_ranking_metrics([1, 1], [0.7, 0.8]) == (0.0, 0.0)


def test_collate_bags_preserves_variable_length_tensors() -> None:
    first = torch.zeros(2, 3, 8, 8)
    second = torch.zeros(4, 3, 8, 8)

    stacks, labels, samples, vertebrae = collate_bags(
        [
            (first, 0, "sample1", "C1"),
            (second, 1, "sample2", "C2"),
        ]
    )

    assert stacks[0] is first
    assert stacks[1] is second
    assert labels == [0, 1]
    assert samples == ["sample1", "sample2"]
    assert vertebrae == ["C1", "C2"]
