"""スラブ用TinyUNetのテスト。"""

from __future__ import annotations

import pytest
import torch
from line_surface_3d.src.model import TinyUNet, reshape_slab_heatmaps


def test_tiny_unet_outputs_all_slab_lines() -> None:
    """`2N -> 4N` のshape契約を満たす。"""
    slab_size = 3
    model = TinyUNet(
        in_channels=2 * slab_size,
        out_channels=4 * slab_size,
        features=(4, 8, 16, 32),
        num_vertebra=7,
    )
    inputs = torch.randn(2, 2 * slab_size, 32, 32)
    vertebra_indices = torch.tensor([0, 6])
    logits = model(inputs, vertebra_indices)
    heatmaps = reshape_slab_heatmaps(logits, slab_size)
    assert logits.shape == (2, 12, 32, 32)
    assert heatmaps.shape == (2, 3, 4, 32, 32)


def test_reshape_rejects_channel_mismatch() -> None:
    """誤ったchannel数を黙ってreshapeしない。"""
    with pytest.raises(ValueError):
        reshape_slab_heatmaps(torch.zeros(1, 11, 8, 8), slab_size=3)
