"""stride 16で4領域へ分岐するPMGAN式モデル。"""

from __future__ import annotations

import copy
from typing import cast

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fracture_detection.common.constants import N_PLANES, N_REGIONS
from fracture_detection.core.contracts import ArmOutput
from fracture_detection.core.losses import plane_max_whole_logits


class MaskGuidedAttention(nn.Module):
    """spatial/channel attentionを融合し、残差再重み付けする。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.spatial_down = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.spatial_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.channel_down = nn.Conv2d(channels, hidden, kernel_size=1)
        self.channel_up = nn.Conv2d(hidden, channels, kernel_size=1)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """再重み付け特徴とsigmoid済みspatial mapを返す。"""
        spatial_source = features.mean(dim=1, keepdim=True)
        spatial = self.spatial_up(F.relu(self.spatial_down(spatial_source)))
        if spatial.shape[-2:] != features.shape[-2:]:
            spatial = F.interpolate(
                spatial,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        spatial = spatial.sigmoid()
        channel = F.adaptive_avg_pool2d(features, output_size=1)
        channel = self.channel_up(F.relu(self.channel_down(channel))).sigmoid()
        fused = self.fusion(spatial * channel).sigmoid()
        return features * (1.0 + fused), spatial


class LateBranch(nn.Module):
    """独立late CNN・BiLSTM・面head。"""

    def __init__(
        self,
        late_block: nn.Module,
        conv_head: nn.Module,
        bn2: nn.Module,
        global_pool: nn.Module,
        feature_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        drop_rate: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.late_block = copy.deepcopy(late_block)
        self.conv_head = copy.deepcopy(conv_head)
        self.bn2 = copy.deepcopy(bn2)
        self.global_pool = copy.deepcopy(global_pool)
        self.lstm = nn.LSTM(
            feature_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=drop_rate if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.Dropout(head_dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, features: Tensor, batch_size: int, plane_count: int) -> Tensor:
        """shared featureから面logit [B,N]を返す。"""
        values = self.late_block(features)
        values = self.bn2(self.conv_head(values))
        values = cast(Tensor, self.global_pool(values)).flatten(1)
        sequence = values.reshape(batch_size, plane_count, -1)
        contextual, _ = self.lstm(sequence)
        logits = cast(
            Tensor, self.head(contextual.reshape(batch_size * plane_count, -1))
        )
        return logits.reshape(batch_size, plane_count)

    def cnn_parameters(self) -> list[nn.Parameter]:
        """late CNN側のparameterを返す。"""
        return [
            *self.late_block.parameters(),
            *self.conv_head.parameters(),
            *self.bn2.parameters(),
        ]

    def temporal_parameters(self) -> list[nn.Parameter]:
        """BiLSTMと分類head parameterを返す。"""
        return [*self.lstm.parameters(), *self.head.parameters()]


class ProposedModel(nn.Module):
    """4領域MA branchと任意の独立whole branchを持つ。"""

    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s",
        *,
        pretrained: bool,
        whole_method: str,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_planes: int = N_PLANES,
    ) -> None:
        super().__init__()
        if whole_method not in {"independent", "max"}:
            raise ValueError("whole_methodはindependentまたはmaxが必要です")
        self.whole_method = whole_method
        self.n_planes = n_planes
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=10,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        blocks = getattr(backbone, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) != 6:
            raise ValueError("EfficientNetV2-Sの6 blocksが必要です")
        conv_stem = getattr(backbone, "conv_stem", None)
        bn1 = getattr(backbone, "bn1", None)
        conv_head = getattr(backbone, "conv_head", None)
        bn2 = getattr(backbone, "bn2", None)
        global_pool = getattr(backbone, "global_pool", None)
        if not all(
            isinstance(module, nn.Module)
            for module in (conv_stem, bn1, conv_head, bn2, global_pool)
        ):
            raise ValueError("timm backboneのstem/head moduleを取得できません")
        self.conv_stem: nn.Module = cast(nn.Module, conv_stem)
        self.bn1: nn.Module = cast(nn.Module, bn1)
        self.shared_blocks = nn.Sequential(*list(blocks.children())[:5])
        feature_dim = getattr(backbone, "num_features", None)
        if not isinstance(feature_dim, int):
            raise ValueError("timm backboneから特徴次元を取得できません")
        late_block = blocks[5]

        def new_branch() -> LateBranch:
            return LateBranch(
                late_block=late_block,
                conv_head=cast(nn.Module, conv_head),
                bn2=cast(nn.Module, bn2),
                global_pool=cast(nn.Module, global_pool),
                feature_dim=feature_dim,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                drop_rate=drop_rate,
                head_dropout=head_dropout,
            )

        shared_channels = 160
        self.attention_modules = nn.ModuleList(
            [MaskGuidedAttention(shared_channels) for _ in range(N_REGIONS)]
        )
        self.region_branches = nn.ModuleList([new_branch() for _ in range(N_REGIONS)])
        self.whole_branch: LateBranch | None = (
            new_branch() if whole_method == "independent" else None
        )
        self._backbone_frozen = False

    def forward(self, inputs: Tensor) -> ArmOutput:
        """面whole/region logitと14×14 spatial attentionを返す。"""
        if inputs.ndim != 5:
            raise ValueError("入力は[B,N,10,H,W]である必要があります")
        batch_size, plane_count, channels, height, width = inputs.shape
        if plane_count != self.n_planes or channels != 10:
            raise ValueError(
                f"入力shapeが不正です: expected [B,{self.n_planes},10,H,W], got {inputs.shape}"
            )
        flattened = inputs.reshape(batch_size * plane_count, channels, height, width)
        shared = self.shared_blocks(self.bn1(self.conv_stem(flattened)))
        region_logits: list[Tensor] = []
        spatial_maps: list[Tensor] = []
        for attention_module, branch_module in zip(
            self.attention_modules, self.region_branches, strict=True
        ):
            attention = cast(MaskGuidedAttention, attention_module)
            branch = cast(LateBranch, branch_module)
            weighted, spatial = attention(shared)
            region_logits.append(branch(weighted, batch_size, plane_count))
            spatial_maps.append(
                spatial.reshape(batch_size, plane_count, *spatial.shape[1:])
            )
        regions = torch.stack(region_logits, dim=-1)
        attention_maps = torch.cat(spatial_maps, dim=2)
        if self.whole_branch is None:
            whole = plane_max_whole_logits(regions)
        else:
            whole = self.whole_branch(shared, batch_size, plane_count)
        return ArmOutput(
            whole_logits=whole,
            region_logits=regions,
            spatial_attention=attention_maps,
        )

    def shared_parameters(self) -> list[nn.Parameter]:
        """校正対象の共有`blocks[4]` parameterを返す。"""
        return list(self.shared_blocks[4].parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        """shared/MA/late CNN parameterを返す。"""
        parameters = [
            *self.conv_stem.parameters(),
            *self.bn1.parameters(),
            *self.shared_blocks.parameters(),
            *self.attention_modules.parameters(),
        ]
        for branch_module in self.region_branches:
            branch = cast(LateBranch, branch_module)
            parameters.extend(branch.cnn_parameters())
        if self.whole_branch is not None:
            parameters.extend(self.whole_branch.cnn_parameters())
        return parameters

    def head_parameters(self) -> list[nn.Parameter]:
        """全branchのtemporal/head parameterを返す。"""
        parameters: list[nn.Parameter] = []
        for branch_module in self.region_branches:
            branch = cast(LateBranch, branch_module)
            parameters.extend(branch.temporal_parameters())
        if self.whole_branch is not None:
            parameters.extend(self.whole_branch.temporal_parameters())
        return parameters

    def set_backbone_trainable(self, trainable: bool) -> None:
        """CNN/MA parameterと凍結BN状態を切り替える。"""
        self._backbone_frozen = not trainable
        for parameter in self.backbone_parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self._set_backbone_batch_norm_eval()

    def train(self, mode: bool = True) -> ProposedModel:
        """凍結CNNのBNをtrainへ戻さない。"""
        super().train(mode)
        if mode and self._backbone_frozen:
            self._set_backbone_batch_norm_eval()
        return self

    def _set_backbone_batch_norm_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
