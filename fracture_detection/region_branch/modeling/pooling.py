"""stage1-4のFPN融合とmask-normalized pooling。

`train_models/stage3/src/model.py:80-213`のFPN・masked pooling実装を踏襲する。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_norm_groups(channels: int, preferred: int = 32) -> int:
    """`channels`を割り切る最大のgroup数を、`preferred`以下から選ぶ。

    本番configは`fpn_channels=256`で常に32を返すが、testで小さいchannel数を
    使えるよう頑健にする。
    """
    for candidate in range(min(preferred, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


class RegionFpn(nn.Module):
    """stage1-4の中間特徴を最も高解像度(stride 4)へ1x1 conv+bilinearで統合する。"""

    def __init__(self, in_channels: tuple[int, ...], fpn_channels: int) -> None:
        super().__init__()
        if len(in_channels) < 1:
            raise ValueError("in_channelsは1要素以上である必要があります")
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, fpn_channels, 1) for channels in in_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(in_channels), fpn_channels, 1),
            nn.GroupNorm(_group_norm_groups(fpn_channels), fpn_channels),
            nn.SiLU(),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """[stage1..stage4]特徴を融合したstride 4特徴を返す。"""
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"feature数がlateral_convsと一致しません: {len(features)}")
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
    """4領域maskで特徴を面積正規化poolingする。

    Args:
        feature_map: [N, C, H, W] FPN融合後の特徴（N = batch*plane）。
        region_mask: [N, H0, W0] 値0..n_regionsの整数mask（元解像度）。
        n_regions: 領域数。

    Returns:
        pooled: [N, n_regions, C]。
        plane_valid: [N, n_regions] bool、その面のmaskが非空かどうか。
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_mapは[N,C,H,W]である必要があります: {feature_map.shape}"
        )
    if region_mask.ndim != 3:
        raise ValueError(
            f"region_maskは[N,H,W]である必要があります: {region_mask.shape}"
        )
    if feature_map.shape[0] != region_mask.shape[0]:
        raise ValueError("feature_mapとregion_maskのバッチ次元が一致しません")

    one_hot = F.one_hot(region_mask.long().clamp(0, n_regions), n_regions + 1)[..., 1:]
    one_hot = one_hot.permute(0, 3, 1, 2).to(feature_map.dtype)  # [N, R, H0, W0]
    plane_valid = one_hot.sum(dim=(-2, -1)) > 0  # [N, R]

    target_height, target_width = feature_map.shape[-2], feature_map.shape[-1]
    resized_mask = F.adaptive_avg_pool2d(
        one_hot, (int(target_height), int(target_width))
    )  # [N, R, H, W]
    numerator = torch.einsum("nchw,nrhw->nrc", feature_map, resized_mask)
    denominator = resized_mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1e-6)
    return numerator / denominator, plane_valid
