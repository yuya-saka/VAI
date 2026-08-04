"""共有2D U-Netと軽量z方向Residual Convによる2.5Dモデル。"""

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


class TemporalResidualBlock(nn.Module):
    """空間解像度を変えず、隣接スライス特徴だけを混合する。"""

    def __init__(self, channels: int):
        super().__init__()
        group_count = min(16, channels)
        while channels % group_count != 0:
            group_count -= 1
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(group_count, channels)
        self.activation = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """`(B,C,N,H,W)` の特徴へz方向残差を加える。"""
        residual = self.depthwise(inputs)
        residual = self.activation(self.norm(residual))
        return inputs + self.pointwise(residual)


class SliceSharedUNet(nn.Module):
    """全スライスで重みを共有し、各画像の4線heatmapを返す。"""

    def __init__(
        self,
        in_channels_per_slice: int = 2,
        out_channels_per_slice: int = 4,
        features: tuple[int, int, int, int] = (16, 32, 64, 128),
        dropout: float = 0.0,
        temporal_blocks: int = 1,
        num_vertebra: int = 0,
    ):
        super().__init__()
        feature_1, feature_2, feature_3, feature_4 = features
        self.num_vertebra = num_vertebra
        self.down_1 = DoubleConv(in_channels_per_slice, feature_1, dropout)
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

        self.temporal_fusion = nn.Sequential(
            *(TemporalResidualBlock(feature_4) for _ in range(temporal_blocks))
        )
        self.up_sample_3 = nn.ConvTranspose2d(feature_4, feature_3, 2, stride=2)
        self.up_3 = DoubleConv(feature_3 * 2, feature_3, dropout)
        self.up_sample_2 = nn.ConvTranspose2d(feature_3, feature_2, 2, stride=2)
        self.up_2 = DoubleConv(feature_2 * 2, feature_2, dropout)
        self.up_sample_1 = nn.ConvTranspose2d(feature_2, feature_1, 2, stride=2)
        self.up_1 = DoubleConv(feature_1 * 2, feature_1, dropout)
        self.output = nn.Conv2d(feature_1, out_channels_per_slice, 1)

    def _initialize_condition_projection(self, feature_count: int) -> None:
        """条件結合層を元特徴の恒等写像として初期化する。"""
        with torch.no_grad():
            self.condition_projection.weight.zero_()
            self.condition_projection.bias.zero_()
            self.condition_projection.weight[:, :feature_count, 0, 0] = torch.eye(
                feature_count
            )

    def _apply_conditioning(
        self,
        features: torch.Tensor,
        vertebra_indices: torch.Tensor | None,
        slice_count: int,
    ) -> torch.Tensor:
        """各スライスのbottleneckへ同じ椎体one-hotを追加する。"""
        if vertebra_indices is None or self.num_vertebra <= 0:
            return features
        repeated = vertebra_indices.repeat_interleave(slice_count)
        one_hot = F.one_hot(repeated, num_classes=self.num_vertebra).to(
            dtype=features.dtype,
            device=features.device,
        )
        condition = one_hot[:, :, None, None].expand(
            -1,
            -1,
            features.shape[-2],
            features.shape[-1],
        )
        return self.condition_projection(torch.cat([features, condition], dim=1))

    def forward(
        self,
        inputs: torch.Tensor,
        vertebra_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B,N,2,H,W)` から `(B,N,4,H,W)` logitsを返す。"""
        if inputs.ndim != 5:
            raise ValueError(f"入力shapeが不正です: {tuple(inputs.shape)}")
        batch_size, slice_count, channel_count, height, width = inputs.shape
        flattened = inputs.reshape(
            batch_size * slice_count,
            channel_count,
            height,
            width,
        )
        feature_1 = self.down_1(flattened)
        feature_2 = self.down_2(self.pool_1(feature_1))
        feature_3 = self.down_3(self.pool_2(feature_2))
        feature_4 = self.down_4(self.pool_3(feature_3))
        feature_4 = self._apply_conditioning(
            feature_4,
            vertebra_indices,
            slice_count,
        )
        bottleneck_height, bottleneck_width = feature_4.shape[-2:]
        temporal = feature_4.reshape(
            batch_size,
            slice_count,
            feature_4.shape[1],
            bottleneck_height,
            bottleneck_width,
        ).permute(0, 2, 1, 3, 4)
        temporal = self.temporal_fusion(temporal)
        feature_4 = temporal.permute(0, 2, 1, 3, 4).reshape(
            batch_size * slice_count,
            -1,
            bottleneck_height,
            bottleneck_width,
        )

        decoded = self.up_sample_3(feature_4)
        decoded = self.up_3(torch.cat([decoded, feature_3], dim=1))
        decoded = self.up_sample_2(decoded)
        decoded = self.up_2(torch.cat([decoded, feature_2], dim=1))
        decoded = self.up_sample_1(decoded)
        decoded = self.up_1(torch.cat([decoded, feature_1], dim=1))
        logits = self.output(decoded)
        return logits.reshape(
            batch_size,
            slice_count,
            logits.shape[1],
            height,
            width,
        )
