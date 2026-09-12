"""Stride-4 FPN fusion and mask-normalized region pooling.

Self-contained (no coupling to any specific model or dataset), so it is
implemented directly in this package rather than imported from
``region_branch`` -- keeping the weak-label model's rollback unit limited to
``fracture_detection/weak/``.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

POOLING_EPSILON = 1e-6


def _group_norm_groups(channels: int, preferred: int = 32) -> int:
    """Largest divisor of ``channels`` at or below ``preferred``."""
    for candidate in range(min(preferred, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


class RegionFpn(nn.Module):
    """Fuse stage1-4 features to the highest-resolution (stride 4) map."""

    def __init__(self, in_channels: tuple[int, ...], fpn_channels: int) -> None:
        super().__init__()
        if len(in_channels) < 1:
            raise ValueError("in_channels must have at least one element")
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, fpn_channels, 1) for channels in in_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(in_channels), fpn_channels, 1),
            nn.GroupNorm(_group_norm_groups(fpn_channels), fpn_channels),
            nn.SiLU(),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """Fuse [stage1..stage4] features into one stride-4 feature map."""
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f"feature count does not match lateral_convs: {len(features)}"
            )
        target_size = features[0].shape[-2:]
        projected = [
            conv(feature)
            if feature.shape[-2:] == target_size
            else F.interpolate(
                conv(feature), size=target_size, mode="bilinear", align_corners=False
            )
            for conv, feature in zip(self.lateral_convs, features, strict=True)
        ]
        return cast(Tensor, self.fusion(torch.cat(projected, dim=1)))


def mask_normalized_pool(
    feature_map: Tensor, region_mask: Tensor, n_regions: int
) -> tuple[Tensor, Tensor]:
    """Area-pool features per anatomical region using a one-hot label map.

    Args:
        feature_map: [N, C, H, W] fused FPN feature (N = batch * plane_count).
        region_mask: [N, H0, W0] integer label map, values 0..n_regions, at
            the ORIGINAL image resolution (not the feature resolution).
        n_regions: number of foreground regions (label 0 is background).

    Returns:
        pooled: [N, n_regions, C].
        plane_valid: [N, n_regions] bool, whether that (plane, region) has
            any foreground pixel, evaluated at the mask's native resolution
            BEFORE downsampling -- a small region must not be reported
            invalid merely because it vanishes under area pooling.
    """
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must be [N,C,H,W]: {feature_map.shape}")
    if region_mask.ndim != 3:
        raise ValueError(f"region_mask must be [N,H,W]: {region_mask.shape}")
    if feature_map.shape[0] != region_mask.shape[0]:
        raise ValueError("feature_map and region_mask batch dimensions differ")

    one_hot = F.one_hot(region_mask.long().clamp(0, n_regions), n_regions + 1)[..., 1:]
    one_hot = one_hot.permute(0, 3, 1, 2).to(feature_map.dtype)  # [N, R, H0, W0]
    plane_valid = one_hot.sum(dim=(-2, -1)) > 0  # [N, R], native resolution

    target_height, target_width = feature_map.shape[-2], feature_map.shape[-1]
    resized_mask = F.adaptive_avg_pool2d(
        one_hot, (int(target_height), int(target_width))
    )  # [N, R, H, W]
    numerator = torch.einsum("nchw,nrhw->nrc", feature_map, resized_mask)
    denominator = (
        resized_mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(POOLING_EPSILON)
    )
    return numerator / denominator, plane_valid
