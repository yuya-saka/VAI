"""Stage1-compatible primary classifier plus a region-pooling auxiliary branch."""

from __future__ import annotations

from typing import NamedTuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_models.stage1.src.model import TimmModel

STAGE2_ARCHITECTURE_VERSION = 2


class Stage2Output(NamedTuple):
    """Named model outputs used by training and diagnostics.

    ``region_logits``, ``region_plane_valid``, and ``plane_valid`` are ``None``
    when the model is called without ``region_masks``, so the primary path can
    be exercised in isolation (e.g. for the Stage1 parity test).
    """

    slice_logits: Tensor
    region_logits: Tensor | None
    region_plane_valid: Tensor | None
    plane_valid: Tensor | None


class SharedRegionHead(nn.Module):
    """Weight-shared temporal head applied identically to all anatomical regions."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return per-plane region logits, shape (B*n_regions, n_slices)."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            sequence, _ = self.lstm(features.float())
            plane_logits = self.head(sequence).squeeze(-1)
        return plane_logits


class Stage2Model(nn.Module):
    """Shared-encoder model with a Stage1-identical primary path.

    The ``encoder``/``lstm``/``head`` modules are constructed exactly like
    ``train_models.stage1.src.model.TimmModel`` so that Stage1 weights can be
    copied in directly and reproduce identical per-plane logits. A single
    encoder forward pass also yields four intermediate feature maps used by
    an auxiliary region-evidence branch; the region masks only pool that
    feature map and are never concatenated into the CNN input.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        in_chans: int = 6,
        n_slices: int = 15,
        n_regions: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_rate_last: float = 0.3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        out_dim: int = 1,
        fpn_channels: int = 256,
        region_hidden: int = 256,
        region_layers: int = 2,
        region_dropout: float = 0.3,
        pretrained: bool = True,
        force_primary_fp32: bool = True,
        region_mode: str = "masked",
    ) -> None:
        super().__init__()
        if region_mode not in ("masked", "global"):
            raise ValueError(f"unsupported region_mode: {region_mode!r}")
        self.n_slices = n_slices
        self.n_regions = n_regions
        self.in_chans = in_chans
        self.force_primary_fp32 = force_primary_fp32
        self.region_mode = region_mode

        # Primary path: identical construction to Stage1's TimmModel so that
        # state_dicts are interchangeable between the two encoder/lstm/head triples.
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=out_dim,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        hdim = TimmModel._get_feature_dim(backbone)
        self.encoder.classifier = nn.Identity()

        lstm_drop = drop_rate if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            hdim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_drop,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_hidden, out_dim),
        )

        # Region auxiliary path: pools intermediate FPN features, never seen
        # by the primary path.
        feature_channels = [
            self.encoder.feature_info[index]["num_chs"] for index in (1, 2, 3, 4)
        ]
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, fpn_channels, kernel_size=1)
            for channels in feature_channels
        )
        self.fpn_fusion = nn.Sequential(
            nn.Conv2d(
                fpn_channels * len(feature_channels), fpn_channels, kernel_size=1
            ),
            nn.GroupNorm(32, fpn_channels),
            nn.SiLU(),
        )
        self.region_head = SharedRegionHead(
            fpn_channels, region_hidden, region_layers, region_dropout
        )

    def forward(
        self, images: Tensor, region_masks: Tensor | None = None
    ) -> Stage2Output:
        """Run one encoder pass and return the primary logits plus optional region output.

        Args:
            images: shape (bs, n_slices, in_chans, H, W)
            region_masks: shape (bs, n_slices, H, W) integer IDs or
                (bs, n_slices, n_regions, H, W) soft masks. When ``None``,
                the region branch is skipped entirely.
        """
        batch_size, n_slices = images.shape[:2]
        if n_slices != self.n_slices:
            raise ValueError(f"expected {self.n_slices} slices, got {n_slices}")
        flattened = images.reshape(
            batch_size * n_slices, self.in_chans, images.shape[-2], images.shape[-1]
        ).to(memory_format=torch.channels_last)
        final_map, intermediates = self.encoder.forward_intermediates(
            flattened, indices=(1, 2, 3, 4), intermediates_only=False
        )
        feat = self.encoder.forward_head(final_map, pre_logits=True)
        feat = feat.view(batch_size, n_slices, -1)
        if self.force_primary_fp32:
            with torch.autocast(device_type=feat.device.type, enabled=False):
                slice_logits = self._primary_head(feat.float(), batch_size, n_slices)
        else:
            slice_logits = self._primary_head(feat, batch_size, n_slices)

        if region_masks is None:
            return Stage2Output(slice_logits, None, None, None)

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
        if self.region_mode == "global":
            region_logits, region_plane_valid, plane_valid = self._global_region_head(
                fused, batch_size, n_slices
            )
        else:
            region_features, region_plane_valid = self._pool_regions(
                fused, region_masks, batch_size, n_slices
            )
            flattened_regions = region_features.permute(0, 2, 1, 3).reshape(
                batch_size * self.n_regions, n_slices, -1
            )
            region_logits = (
                self.region_head(flattened_regions)
                .view(batch_size, self.n_regions, n_slices)
                .permute(0, 2, 1)
            )
            plane_valid = region_plane_valid.any(dim=-1)
        return Stage2Output(
            slice_logits, region_logits, region_plane_valid, plane_valid
        )

    def _primary_head(self, feat: Tensor, batch_size: int, n_slices: int) -> Tensor:
        """Run the Stage1-identical BiLSTM/head on per-plane encoder features."""
        sequence, _ = self.lstm(feat)
        sequence = sequence.contiguous().view(batch_size * n_slices, -1)
        return self.head(sequence).view(batch_size, n_slices).contiguous()

    def _pool_regions(
        self,
        feature_map: Tensor,
        region_masks: Tensor,
        batch_size: int,
        n_slices: int,
    ) -> tuple[Tensor, Tensor]:
        """Mask-normalized average pooling per plane and per anatomical region."""
        if region_masks.ndim == 4:
            one_hot = F.one_hot(
                region_masks.long().clamp(0, self.n_regions),
                num_classes=self.n_regions + 1,
            )[..., 1:].permute(0, 1, 4, 2, 3)
            soft_masks = one_hot.to(feature_map.dtype)
        elif region_masks.ndim == 5:
            soft_masks = region_masks.to(feature_map.dtype)
        else:
            raise ValueError("region_masks must have shape [B,S,H,W] or [B,S,R,H,W]")

        region_plane_valid = soft_masks.sum(dim=(-2, -1)) > 0
        masks = soft_masks.reshape(
            batch_size * n_slices, self.n_regions, *soft_masks.shape[-2:]
        )
        masks = F.adaptive_avg_pool2d(masks, feature_map.shape[-2:])
        numerator = torch.einsum("bchw,brhw->brc", feature_map, masks)
        denominator = masks.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1e-6)
        pooled = numerator / denominator
        return pooled.view(batch_size, n_slices, self.n_regions, -1), region_plane_valid

    def _global_region_head(
        self, feature_map: Tensor, batch_size: int, n_slices: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unmasked global-average pooling ablation: same FPN feature, no region split.

        Isolates whether the FPN neck alone (multi-scale features vs. the
        primary path's single deepest-layer feature) explains a region-head
        improvement, independent of anatomical mask pooling. Every plane is
        treated as valid, matching how the primary path uses all planes
        unconditionally rather than gating on region-mask coverage.
        """
        pooled = feature_map.mean(dim=(-2, -1)).view(batch_size, n_slices, -1)
        region_logits = self.region_head(pooled).unsqueeze(-1)
        region_plane_valid = torch.ones(
            batch_size, n_slices, 1, dtype=torch.bool, device=feature_map.device
        )
        plane_valid = region_plane_valid.any(dim=-1)
        return region_logits, region_plane_valid, plane_valid

    def backbone_parameters(self) -> list[nn.Parameter]:
        """encoder (timm backbone) parameters, for differential LR."""
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """All non-backbone parameters: primary lstm/head plus the region branch."""
        backbone_ids = {id(parameter) for parameter in self.encoder.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_ids
        ]
