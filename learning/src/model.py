"""
椎体骨折分類モデル（ResNet18 + MIL head）。

torchvision の ResNet18 (ImageNet pretrained) を backbone に使う。
conv1 は 3ch のまま流用（2.5D スタックは RGB に対応）。
最終 FC を 1 出力（logit）に置換し、dropout を追加する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

NUM_VERTEBRAE = 7


class FractureResNet18(nn.Module):
    """
    per-slice logit を出力する椎体部位条件付きResNet18モデル。

    MIL の集約は trainer 側で行うため、本モデルは
    バッチ内の全sliceに対してlogitを返す。C1-C7のone-hotを
    ResNet特徴量へ結合して分類する。

    Args:
        dropout: FC 直前の dropout 確率
        pretrained: ImageNet 事前学習済み重みを使うかどうか
    """

    def __init__(
        self,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # conv1は3chのまま（RGB=2.5Dスタック）
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features + NUM_VERTEBRAE, 1),
        )
        self.backbone = backbone
        self.freeze_batch_norm = freeze_batch_norm

        if self.freeze_batch_norm:
            self._freeze_batch_norm()

    def train(self, mode: bool = True) -> FractureResNet18:
        """
        モデルの train/eval を切り替える。

        bag 単位 forward では BatchNorm の統計が患者ごとに偏るため、
        freeze_batch_norm=True の場合は学習中も ImageNet の移動統計を固定する。
        """
        super().train(mode)
        if mode and self.freeze_batch_norm:
            self._freeze_batch_norm()
        return self

    def _freeze_batch_norm(self) -> None:
        """全 BatchNorm 層を eval に固定し、affine パラメータも凍結する。"""
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        vertebra_index: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: shape [B, 3, H, W]（B は bag 内の全 slice 数）
            vertebra_index: C1=0〜C7=6の部位index。単一値またはshape [B]

        Returns:
            logits: shape [B]（per-slice logit）
        """
        indices = torch.as_tensor(
            vertebra_index,
            device=x.device,
            dtype=torch.long,
        ).reshape(-1)
        if indices.numel() == 1:
            indices = indices.expand(x.shape[0])
        if indices.numel() != x.shape[0]:
            raise ValueError("vertebra_indexの要素数は1またはbatch sizeと一致させてください")
        if torch.any((indices < 0) | (indices >= NUM_VERTEBRAE)):
            raise ValueError("vertebra_indexは0〜6で指定してください")

        features = self.backbone(x)  # [B, 512]
        vertebra_one_hot = F.one_hot(
            indices,
            num_classes=NUM_VERTEBRAE,
        ).to(dtype=features.dtype)
        conditioned_features = torch.cat([features, vertebra_one_hot], dim=1)
        return self.classifier(conditioned_features).squeeze(1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        """ResNet backboneのパラメータ。differential LR用。"""
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """部位条件付き分類headのパラメータ。differential LR用。"""
        return list(self.classifier.parameters())
