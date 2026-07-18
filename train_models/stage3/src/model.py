"""Hierarchical weak-label Stage3 model."""

from __future__ import annotations

import math
from typing import NamedTuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_models.stage1.src.model import TimmModel
from train_models.stage3.utils.losses import (
    max_pool,
    noisy_or_pool,
    normalized_lse_pool,
    normalized_smoothmax,
    tied_attention_pool,
)

STAGE3_ARCHITECTURE_VERSION = 1


class Stage3Output(NamedTuple):
    vertebra_logit: Tensor
    instance_evidence_logits: Tensor
    region_evidence_logits: Tensor
    slice_attention: Tensor
    slice_evidence_logits: Tensor
    region_pool_weights: Tensor
    region_plane_valid: Tensor
    region_valid: Tensor
    plane_valid: Tensor
    vertebra_valid: Tensor


class HierarchicalRegionHead(nn.Module):
    """A shared FP32 BiLSTM that produces one contextual evidence logit."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, layers: int, bias: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers=layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.0 if layers == 1 else 0.3,
        )
        self.evidence = nn.Linear(hidden_dim * 2, 1)
        nn.init.normal_(self.evidence.weight, std=1e-3)
        nn.init.constant_(self.evidence.bias, bias)

    def forward(self, features: Tensor) -> Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            sequence, _ = self.lstm(features.float())
            return self.evidence(sequence).squeeze(-1)


class Stage3Model(nn.Module):
    """Mask-pooled, no-direct-head hierarchical evidence model."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        in_chans: int = 6,
        n_slices: int = 15,
        n_regions: int = 4,
        fpn_channels: int = 256,
        region_hidden: int = 256,
        region_layers: int = 2,
        pretrained: bool = True,
        region_mode: str = "masked",
        slice_agg: str = "tied_attention",
        region_agg: str = "normalized_smoothmax",
        attention_temperature: float = 1.0,
        slice_lse_temperature: float = 1.0,
        region_temperature: float = 1.0,
        class_prior: float = 0.5,
    ) -> None:
        super().__init__()
        if region_mode not in {"masked", "global", "scramble"}:
            raise ValueError(f"unsupported region_mode: {region_mode!r}")
        if slice_agg not in {"tied_attention", "normalized_lse"}:
            raise ValueError(f"unsupported slice_agg: {slice_agg!r}")
        if region_agg not in {"normalized_smoothmax", "max", "noisy_or"}:
            raise ValueError(f"unsupported region_agg: {region_agg!r}")
        self.n_slices, self.n_regions, self.in_chans = n_slices, n_regions, in_chans
        self.region_mode, self.slice_agg, self.region_agg = (
            region_mode,
            slice_agg,
            region_agg,
        )
        (
            self.attention_temperature,
            self.slice_lse_temperature,
            self.region_temperature,
        ) = attention_temperature, slice_lse_temperature, region_temperature
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=1,
            features_only=False,
            pretrained=pretrained,
        )
        self.encoder.classifier = nn.Identity()
        self.encoder.conv_head = nn.Identity()
        self.encoder.bn2 = nn.Identity()
        channels = [
            self.encoder.feature_info[index]["num_chs"] for index in (1, 2, 3, 4)
        ]
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channel, fpn_channels, 1) for channel in channels
        )
        self.fpn_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(channels), fpn_channels, 1),
            nn.GroupNorm(32, fpn_channels),
            nn.SiLU(),
        )
        prior = min(max(class_prior, 1e-4), 1.0 - 1e-4)
        self.region_head = HierarchicalRegionHead(
            fpn_channels, region_hidden, region_layers, math.log(prior / (1.0 - prior))
        )
        self.feature_dim = TimmModel._get_feature_dim(backbone)

    def forward(self, images: Tensor, region_masks: Tensor) -> Stage3Output:
        batch, slices = images.shape[:2]
        if slices != self.n_slices:
            raise ValueError(f"expected {self.n_slices} slices, got {slices}")
        flat = images.reshape(batch * slices, self.in_chans, *images.shape[-2:]).to(
            memory_format=torch.channels_last
        )
        intermediates = self.encoder.forward_intermediates(
            flat, indices=(1, 2, 3, 4), intermediates_only=True
        )
        target_size = intermediates[0].shape[-2:]
        projected = [
            conv(feature)
            if feature.shape[-2:] == target_size
            else F.interpolate(
                conv(feature), size=target_size, mode="bilinear", align_corners=False
            )
            for conv, feature in zip(self.lateral_convs, intermediates, strict=True)
        ]
        fused = self.fpn_fusion(torch.cat(projected, dim=1))
        features, region_plane_valid = self._region_features(
            fused, region_masks, batch, slices
        )
        sequence = features.permute(0, 2, 1, 3).reshape(
            batch * self.n_regions, slices, -1
        )
        instances = (
            self.region_head(sequence)
            .view(batch, self.n_regions, slices)
            .permute(0, 2, 1)
        )
        if self.slice_agg == "tied_attention":
            regions, attention, region_valid = tied_attention_pool(
                instances, region_plane_valid, self.attention_temperature
            )
        else:
            regions, attention, region_valid = normalized_lse_pool(
                instances, region_plane_valid, self.slice_lse_temperature
            )
        if self.region_agg == "normalized_smoothmax":
            vertebra, weights, vertebra_valid = normalized_smoothmax(
                regions, region_valid, dim=1, tau=self.region_temperature
            )
        elif self.region_agg == "max":
            vertebra, weights, vertebra_valid = max_pool(regions, region_valid)
        else:
            vertebra, weights, vertebra_valid = noisy_or_pool(regions, region_valid)
        slice_evidence, _, plane_valid = normalized_smoothmax(
            instances, region_plane_valid, dim=2, tau=self.region_temperature
        )
        return Stage3Output(
            vertebra,
            instances,
            regions,
            attention,
            slice_evidence,
            weights,
            region_plane_valid,
            region_valid,
            plane_valid,
            vertebra_valid,
        )

    def _region_features(
        self, feature_map: Tensor, region_masks: Tensor, batch: int, slices: int
    ) -> tuple[Tensor, Tensor]:
        soft_masks = self._soft_masks(region_masks, feature_map.dtype)
        valid = soft_masks.sum(dim=(-2, -1)) > 0
        if self.region_mode == "global":
            pooled = (
                feature_map.mean(dim=(-2, -1))
                .view(batch, slices, 1, -1)
                .expand(-1, -1, self.n_regions, -1)
            )
            return torch.where(
                valid.unsqueeze(-1), pooled, torch.zeros_like(pooled)
            ), valid
        masks = F.adaptive_avg_pool2d(
            soft_masks.reshape(batch * slices, self.n_regions, *soft_masks.shape[-2:]),
            feature_map.shape[-2:],
        )
        numerator = torch.einsum("bchw,brhw->brc", feature_map, masks)
        denominator = masks.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1e-6)
        return (numerator / denominator).view(batch, slices, self.n_regions, -1), valid

    def _soft_masks(self, region_masks: Tensor, dtype: torch.dtype) -> Tensor:
        if region_masks.ndim == 4:
            return (
                F.one_hot(
                    region_masks.long().clamp(0, self.n_regions), self.n_regions + 1
                )[..., 1:]
                .permute(0, 1, 4, 2, 3)
                .to(dtype)
            )
        if region_masks.ndim == 5:
            if region_masks.shape[2] != self.n_regions:
                raise ValueError("unexpected region count")
            return region_masks.to(dtype)
        raise ValueError("region_masks must have shape [B,S,H,W] or [B,S,R,H,W]")

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        backbone_ids = {id(parameter) for parameter in self.encoder.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_ids
        ]
