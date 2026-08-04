"""共有2D+z方向融合モデルのテスト。"""

from __future__ import annotations

import torch

from Unet.line_2p5d.src.model import SliceSharedUNet, TemporalResidualBlock


def test_temporal_block_starts_as_identity() -> None:
    """z融合追加時に初期予測を壊さない。"""
    block = TemporalResidualBlock(channels=8)
    inputs = torch.randn(2, 8, 5, 4, 4)
    assert torch.equal(block(inputs), inputs)


def test_model_outputs_four_lines_for_all_five_slices() -> None:
    """`(B,5,2,H,W) -> (B,5,4,H,W)` 契約を満たす。"""
    model = SliceSharedUNet(
        features=(4, 8, 16, 32),
        temporal_blocks=1,
        num_vertebra=7,
    )
    inputs = torch.randn(2, 5, 2, 32, 32)
    vertebra_indices = torch.tensor([0, 6])
    outputs = model(inputs, vertebra_indices)
    assert outputs.shape == (2, 5, 4, 32, 32)
