"""line_only TinyUNet を基準にしたスラブ用2D U-Net。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VERTEBRA_TO_INDEX = {
    "C1": 0,
    "C2": 1,
    "C3": 2,
    "C4": 3,
    "C5": 4,
    "C6": 5,
    "C7": 6,
}


class DoubleConv(nn.Module):
    """2回の畳み込み、BatchNorm、ReLUを適用する。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """特徴マップを変換する。"""
        return self.net(inputs)


class TinyUNet(nn.Module):
    """4段の軽量U-Netに椎体条件付けを追加したモデル。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: tuple[int, int, int, int] = (16, 32, 64, 128),
        dropout: float = 0.0,
        num_vertebra: int = 0,
    ):
        super().__init__()
        feature_1, feature_2, feature_3, feature_4 = features
        self.num_vertebra = num_vertebra
        self.down_1 = DoubleConv(in_channels, feature_1, dropout)
        self.pool_1 = nn.MaxPool2d(2)
        self.down_2 = DoubleConv(feature_1, feature_2, dropout)
        self.pool_2 = nn.MaxPool2d(2)
        self.down_3 = DoubleConv(feature_2, feature_3, dropout)
        self.pool_3 = nn.MaxPool2d(2)
        self.down_4 = DoubleConv(feature_3, feature_4, dropout)

        if num_vertebra > 0:
            self.condition_projection = nn.Conv2d(
                feature_4 + num_vertebra,
                feature_4,
                kernel_size=1,
                bias=True,
            )
            self._initialize_condition_projection(feature_4)

        self.up_sample_3 = nn.ConvTranspose2d(feature_4, feature_3, 2, stride=2)
        self.up_3 = DoubleConv(feature_3 * 2, feature_3, dropout)
        self.up_sample_2 = nn.ConvTranspose2d(feature_3, feature_2, 2, stride=2)
        self.up_2 = DoubleConv(feature_2 * 2, feature_2, dropout)
        self.up_sample_1 = nn.ConvTranspose2d(feature_2, feature_1, 2, stride=2)
        self.up_1 = DoubleConv(feature_1 * 2, feature_1, dropout)
        self.output = nn.Conv2d(feature_1, out_channels, 1)

    def _initialize_condition_projection(self, feature_count: int) -> None:
        """条件結合層を元特徴の恒等写像として初期化する。"""
        with torch.no_grad():
            self.condition_projection.weight.zero_()
            self.condition_projection.bias.zero_()
            identity = torch.eye(feature_count)
            self.condition_projection.weight[:, :feature_count, 0, 0] = identity

    def _condition_map(
        self,
        vertebra_indices: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """椎体indexを空間方向へ展開したone-hot mapへ変換する。"""
        one_hot = F.one_hot(
            vertebra_indices,
            num_classes=self.num_vertebra,
        ).to(dtype=dtype, device=device)
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def forward(
        self,
        inputs: torch.Tensor,
        vertebra_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """スラブ入力から全スライスの4線logitを返す。"""
        feature_1 = self.down_1(inputs)
        feature_2 = self.down_2(self.pool_1(feature_1))
        feature_3 = self.down_3(self.pool_2(feature_2))
        feature_4 = self.down_4(self.pool_3(feature_3))

        if vertebra_indices is not None and self.num_vertebra > 0:
            condition = self._condition_map(
                vertebra_indices,
                feature_4.shape[-2],
                feature_4.shape[-1],
                feature_4.dtype,
                feature_4.device,
            )
            feature_4 = self.condition_projection(
                torch.cat([feature_4, condition], dim=1)
            )

        decoded = self.up_sample_3(feature_4)
        decoded = self.up_3(torch.cat([decoded, feature_3], dim=1))
        decoded = self.up_sample_2(decoded)
        decoded = self.up_2(torch.cat([decoded, feature_2], dim=1))
        decoded = self.up_sample_1(decoded)
        decoded = self.up_1(torch.cat([decoded, feature_1], dim=1))
        return self.output(decoded)


def reshape_slab_heatmaps(logits: torch.Tensor, slab_size: int) -> torch.Tensor:
    """slice-majorの出力を `(B,N,4,H,W)` へ復元する。"""
    batch_size, channel_count, height, width = logits.shape
    expected_channels = slab_size * 4
    if channel_count != expected_channels:
        raise ValueError(
            f"出力channel数が不正です: {channel_count} != {expected_channels}"
        )
    return logits.reshape(batch_size, slab_size, 4, height, width)
