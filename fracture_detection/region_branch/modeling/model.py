"""Shared trunk + whole path (Baseline 0互換) + region pathのモデル。

whole pathは`Baseline0Model`とbit-exactに一致する（`encoder(x)`と
`encoder.forward_head(encoder.forward_intermediates(x, intermediates_only=False)[0])`
が数値的に完全一致することをtest_model.pyで検証する）。共有されるのはCNN trunkのみで、
conv_head/bn2/whole BiLSTM/whole headはwhole loss、FPN/region BiLSTM/region head群は
region lossからのみ勾配を受ける。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import timm
import torch
from torch import Tensor, nn

from fracture_detection.region_branch.data_pipeline.constants import N_PLANES, N_REGIONS
from fracture_detection.region_branch.modeling.pooling import (
    RegionFpn,
    mask_normalized_pool,
)

FPN_STAGE_INDICES = (1, 2, 3, 4)


@dataclass(frozen=True)
class RegionBranchOutput:
    """1 forwardの出力。不要な経路はNoneのままにする。"""

    whole_plane_logits: Tensor | None
    region_plane_logits: Tensor | None
    region_plane_valid: Tensor | None


def build_model(config: dict[str, Any]) -> RegionBranchModel:
    """Configのmodel/region sectionからRegionBranchModelを構築する。"""
    model = config["model"]
    region = config["region"]
    active_regions = tuple(int(value) for value in region["active_regions"])
    return RegionBranchModel(
        backbone_name=str(model["backbone"]),
        pretrained=bool(model["pretrained"]),
        drop_rate=float(model["drop_rate"]),
        drop_path_rate=float(model["drop_path_rate"]),
        head_dropout=float(model["head_dropout"]),
        lstm_hidden=int(model["lstm_hidden"]),
        lstm_layers=int(model["lstm_layers"]),
        n_planes=int(model["n_planes"]),
        active_regions=active_regions,
        fpn_channels=int(region["fpn_channels"]),
        region_lstm_hidden=int(region["region_lstm_hidden"]),
        region_lstm_layers=int(region["region_lstm_layers"]),
        region_head_dropout=float(region["region_head_dropout"]),
    )


class RegionBranchModel(nn.Module):
    """whole plane logitとactive regionのplane logitを返す2経路モデル。"""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float,
        lstm_hidden: int,
        lstm_layers: int,
        active_regions: tuple[int, ...],
        fpn_channels: int,
        region_lstm_hidden: int,
        region_lstm_layers: int,
        region_head_dropout: float,
        n_planes: int = N_PLANES,
        n_regions: int = N_REGIONS,
    ) -> None:
        super().__init__()
        if not active_regions or not set(active_regions).issubset(range(n_regions)):
            raise ValueError(f"active_regionsが不正です: {active_regions}")
        self.n_planes = n_planes
        self.n_regions = n_regions
        self.active_regions = tuple(sorted(set(active_regions)))
        self.n_active_regions = len(self.active_regions)

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=6,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        num_features = getattr(self.encoder, "num_features", None)
        if not isinstance(num_features, int):
            raise ValueError("timm backboneから特徴次元を取得できません")

        # whole path: Baseline 0とhidden/構造を揃える。
        self.whole_lstm = nn.LSTM(
            num_features,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=drop_rate if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.whole_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.Dropout(head_dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_hidden, 1),
        )

        # region path: FPN -> mask pooling -> shared region BiLSTM -> 領域別head。
        feature_info = getattr(self.encoder, "feature_info", None)
        if feature_info is None:
            raise ValueError("timm backboneからfeature_infoを取得できません")
        stage_channels = tuple(
            int(feature_info[index]["num_chs"]) for index in FPN_STAGE_INDICES
        )
        self.fpn = RegionFpn(stage_channels, fpn_channels)
        self.region_lstm = nn.LSTM(
            fpn_channels,
            region_lstm_hidden,
            num_layers=region_lstm_layers,
            dropout=drop_rate if region_lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.region_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(region_lstm_hidden * 2, region_lstm_hidden),
                nn.BatchNorm1d(region_lstm_hidden),
                nn.Dropout(region_head_dropout),
                nn.LeakyReLU(0.1),
                nn.Linear(region_lstm_hidden, 1),
            )
            for _ in self.active_regions
        )
        self._backbone_frozen = False

    def forward(
        self,
        inputs: Tensor,
        region_mask: Tensor,
        need_whole: bool = True,
        need_region: bool = True,
        region_sample_indices: Tensor | None = None,
    ) -> RegionBranchOutput:
        """[B,15,6,H,W]と[B,15,H,W]から、要求された経路のlogitを返す。"""
        if not need_whole and not need_region:
            raise ValueError(
                "need_whole/need_regionのどちらかはTrueである必要があります"
            )
        if inputs.ndim != 5:
            raise ValueError(f"入力は5次元である必要があります: {inputs.shape}")
        batch_size, plane_count, channels, height, width = inputs.shape
        if plane_count != self.n_planes or channels != 6:
            raise ValueError(
                f"入力shapeが不正です: expected [B,{self.n_planes},6,H,W], got {inputs.shape}"
            )
        if region_mask.shape != (batch_size, plane_count, height, width):
            raise ValueError(f"region_maskのshapeが不正です: {region_mask.shape}")
        if region_sample_indices is not None:
            if not need_region:
                raise ValueError(
                    "region_sample_indicesはneed_region=Trueの場合のみ指定できます"
                )
            if region_sample_indices.dtype != torch.long:
                raise ValueError(
                    "region_sample_indicesはlong tensorである必要があります"
                )
            if region_sample_indices.ndim != 1 or region_sample_indices.shape[0] < 1:
                raise ValueError(
                    "region_sample_indicesは1件以上の1次元tensorが必要です"
                )
        flattened = inputs.reshape(batch_size * plane_count, channels, height, width)

        whole_plane_logits: Tensor | None = None
        region_plane_logits: Tensor | None = None
        region_plane_valid: Tensor | None = None

        if need_region:
            # trunk forwardを1回だけ実行し、region経路のintermediatesを取る。
            # need_wholeも真なら同じ結果からforward_headでwhole特徴も取得し、
            # trunk計算の重複を避ける。
            # forward_intermediatesはtimmが動的に付与するmethodで、nn.Moduleの
            # 静的な型情報には現れないためmypyには見えない。
            final, intermediates = cast(
                tuple[Tensor, list[Tensor]],
                self.encoder.forward_intermediates(  # type: ignore[operator]
                    flattened,
                    indices=FPN_STAGE_INDICES,
                    intermediates_only=False,
                ),
            )
            region_batch_size = batch_size
            region_intermediates = intermediates
            selected_region_mask = region_mask
            if region_sample_indices is not None:
                selected_region_mask = region_mask.index_select(
                    0, region_sample_indices
                )
                region_batch_size = selected_region_mask.shape[0]
                region_intermediates = [
                    feature.reshape(batch_size, plane_count, *feature.shape[1:])
                    .index_select(0, region_sample_indices)
                    .reshape(region_batch_size * plane_count, *feature.shape[1:])
                    for feature in intermediates
                ]
            fused = self.fpn(region_intermediates)
            flattened_mask = selected_region_mask.reshape(
                region_batch_size * plane_count, height, width
            )
            pooled, plane_valid = mask_normalized_pool(
                fused, flattened_mask, self.n_regions
            )
            region_plane_logits, region_plane_valid = self._region_forward(
                pooled, plane_valid, region_batch_size, plane_count
            )
            if need_whole:
                whole_features = cast(
                    Tensor,
                    self.encoder.forward_head(final),  # type: ignore[operator]
                )
                whole_plane_logits = self._whole_forward(
                    whole_features, batch_size, plane_count
                )
        elif need_whole:
            whole_features = cast(Tensor, self.encoder(flattened))
            whole_plane_logits = self._whole_forward(
                whole_features, batch_size, plane_count
            )

        return RegionBranchOutput(
            whole_plane_logits, region_plane_logits, region_plane_valid
        )

    def _whole_forward(
        self, features: Tensor, batch_size: int, plane_count: int
    ) -> Tensor:
        """whole特徴をBiLSTM+headへ通し、[B,S]のlogitを返す。"""
        sequence = features.reshape(batch_size, plane_count, -1)
        contextual, _ = self.whole_lstm(sequence)
        logits = cast(
            Tensor, self.whole_head(contextual.reshape(batch_size * plane_count, -1))
        )
        return logits.reshape(batch_size, plane_count)

    def _region_forward(
        self,
        pooled: Tensor,
        plane_valid: Tensor,
        batch_size: int,
        plane_count: int,
    ) -> tuple[Tensor, Tensor]:
        """poolされた領域特徴を共有BiLSTM+領域別headへ通す。"""
        active = pooled[:, self.active_regions, :]  # [B*S, |R|, C]
        active_valid = plane_valid[:, self.active_regions]  # [B*S, |R|]
        sequence = active.reshape(
            batch_size, plane_count, self.n_active_regions, -1
        ).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size * self.n_active_regions, plane_count, -1)
        contextual, _ = self.region_lstm(sequence)
        contextual = contextual.reshape(
            batch_size, self.n_active_regions, plane_count, -1
        )
        region_logits = []
        for position, head in enumerate(self.region_heads):
            region_features = contextual[:, position].reshape(
                batch_size * plane_count, -1
            )
            region_logits.append(
                cast(Tensor, head(region_features)).reshape(batch_size, plane_count)
            )
        plane_logits = torch.stack(region_logits, dim=2)  # [B, S, |R|]
        plane_valid_out = active_valid.reshape(
            batch_size, plane_count, self.n_active_regions
        )
        return plane_logits, plane_valid_out

    def set_backbone_trainable(self, trainable: bool) -> None:
        """バックボーンの勾配計算と、凍結中のBatchNorm状態を設定する。"""
        self._backbone_frozen = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self._set_backbone_batch_norm_eval()

    def _set_backbone_batch_norm_eval(self) -> None:
        """凍結したバックボーンのBatchNorm統計更新を止める。"""
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> RegionBranchModel:
        """学習状態への復帰後も、凍結バックボーンのBatchNormを評価状態に保つ。"""
        super().train(mode)
        if mode and self._backbone_frozen:
            self._set_backbone_batch_norm_eval()
        return self

    def backbone_parameters(self) -> list[nn.Parameter]:
        """異なる学習率の設定に使うバックボーンのパラメータを返す。"""
        return list(self.encoder.parameters())

    def shared_parameters(self) -> list[nn.Parameter]:
        """whole/region両経路が通る共有trunk終端`blocks[4]`を返す。"""
        blocks = getattr(self.encoder, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) <= 4:
            raise ValueError("backboneにblocks[4]がありません")
        return list(blocks[4].parameters())

    def region_lstm_parameters(self) -> list[nn.Parameter]:
        """alpha校正の対象とするregion BiLSTMパラメータを返す。"""
        return list(self.region_lstm.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """異なる学習率の設定に使うbackbone以外全パラメータを返す。"""
        return [
            *self.whole_lstm.parameters(),
            *self.whole_head.parameters(),
            *self.fpn.parameters(),
            *self.region_lstm.parameters(),
            *self.region_heads.parameters(),
        ]

    def pretrained_parameters(self) -> list[nn.Parameter]:
        """Baseline 0 checkpointから初期化される全パラメータを返す。"""
        return [
            *self.encoder.parameters(),
            *self.whole_lstm.parameters(),
            *self.whole_head.parameters(),
        ]

    def region_parameters(self) -> list[nn.Parameter]:
        """ランダム初期化されるregion pathの全パラメータを返す。"""
        return [
            *self.fpn.parameters(),
            *self.region_lstm.parameters(),
            *self.region_heads.parameters(),
        ]
