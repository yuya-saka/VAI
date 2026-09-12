"""Serial region-to-whole MIL model.

Reference design: ``fracture_detection/REGION_MIL_DESIGN.md`` sections 2-3.

    15 planes x (5ch 2.5D CT + 1ch vertebra mask)
        -> fine-tuned EfficientNetV2-S trunk
        -> stride-4 FPN (256ch)
        -> mask-normalized pooling per (plane, region)
        -> valid-planes-only, original-order sequence per region
        -> ONE shared bidirectional BiLSTM (weights shared across regions,
           inputs kept separate -- NOT a single sequence mixing all regions)
        -> masked mean of BiLSTM features over valid planes
        -> ONE shared Linear(1) applied to each region's pooled feature
        -> 4 region logits z_1..z_4

There is no whole-specific BiLSTM/head and no global-feature bypass. Whole
probability is derived from the 4 region logits by the configured fixed,
parameter-free aggregation (see ``modeling/losses.py``). It is not computed
by this module, so there is no learnable path to "whole" besides the 4 region
logits.

Note: forward() calls ``encoder.forward_intermediates(..., indices=(1,2,3,4),
intermediates_only=True)``, which runs the stem and every block stage but
never reaches the encoder's final ``bn2``/``conv_head`` (those exist only to
produce Baseline 0's globally-pooled whole-path feature). Those two modules
are transferred from a Baseline 0 checkpoint along with the rest of
``encoder.*`` but never receive a gradient here -- confirmed dead weight, not
a bug (see test_model.py::test_bn_running_stats_unchanged_while_affine_grads_exist).

A (plane, region) pair with zero valid pixels is NOT silently filled with a
q=0 prediction: ``region_observed`` reports it as unobserved so the loss can
exclude it and count it, per the design doc's "do not fill empty masks with
q=0" requirement.

Memory: the stride-4 FPN maps cost ~0.57 GiB per bag (15 planes) to keep
for backward. region_branch only runs the FPN on whole-positive bags, but
every bag here needs region logits, so keeping them for all 16 bags measured
26.7 GiB peak versus 19.0 GiB for region_branch. FPN + mask pooling therefore
runs per chunk of ``fpn_chunk_planes`` planes under activation checkpointing:
only the pooled [chunk, 4, C] features are kept and the FPN (not the encoder)
is recomputed chunk by chunk during backward, which measured 18.4 GiB peak at
the cost of ~20% longer steps. Every op in that span is per-sample, so the
result is identical to the unchunked computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import timm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

from fracture_detection.weak.data_pipeline.constants import N_PLANES, N_REGIONS
from fracture_detection.weak.modeling.pooling import RegionFpn, mask_normalized_pool

FPN_STAGE_INDICES = (1, 2, 3, 4)
INPUT_CHANNELS = 6


@dataclass(frozen=True)
class WeakModelOutput:
    """1 forward() の出力。

    region_observed が False の (bag, region) は損失から除外する対象で、
    その region_logits の値を q=0 として扱ってはならない。
    """

    region_logits: Tensor  # [B, N_REGIONS]
    region_observed: Tensor  # [B, N_REGIONS] bool


def build_model(config: dict[str, Any]) -> WeakRegionMilModel:
    """Build a WeakRegionMilModel from ``config["model"]``."""
    model_config = config["model"]
    return WeakRegionMilModel(
        backbone_name=str(model_config["backbone"]),
        pretrained=bool(model_config["pretrained"]),
        drop_rate=float(model_config["drop_rate"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
        fpn_channels=int(model_config["fpn_channels"]),
        region_lstm_hidden=int(model_config["region_lstm_hidden"]),
        region_lstm_layers=int(model_config["region_lstm_layers"]),
        region_head_dropout=float(model_config["region_head_dropout"]),
        freeze_encoder_bn_stats=bool(model_config["freeze_encoder_bn_stats"]),
        n_planes=int(model_config["n_planes"]),
    )


class WeakRegionMilModel(nn.Module):
    """Derive four region logits used by a fixed whole aggregation."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        drop_rate: float,
        drop_path_rate: float,
        fpn_channels: int,
        region_lstm_hidden: int,
        region_lstm_layers: int,
        region_head_dropout: float,
        freeze_encoder_bn_stats: bool,
        n_planes: int = N_PLANES,
        n_regions: int = N_REGIONS,
        fpn_chunk_planes: int | None = None,
    ) -> None:
        super().__init__()
        self.n_planes = n_planes
        self.n_regions = n_regions
        self._freeze_encoder_bn_stats = freeze_encoder_bn_stats
        # One bag's planes per chunk by default: the measured step time was the
        # same for 15/30/60-plane chunks, so the smallest chunk costs nothing.
        self.fpn_chunk_planes = (
            n_planes if fpn_chunk_planes is None else fpn_chunk_planes
        )
        if self.fpn_chunk_planes < 1:
            raise ValueError("fpn_chunk_planesは1以上である必要があります")

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=INPUT_CHANNELS,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
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
        self.region_dropout = nn.Dropout(region_head_dropout)
        # Single shared Linear applied to every region's pooled feature --
        # NOT a per-region ModuleList. No BatchNorm / region-id embedding /
        # global-feature concatenation in the head (design doc section 2).
        self.region_head = nn.Linear(region_lstm_hidden * 2, 1)

    def forward(self, inputs: Tensor, region_mask: Tensor) -> WeakModelOutput:
        """[B,15,6,H,W] と [B,15,H,W] から4領域logitを1回のencoder forwardで返す。"""
        if inputs.ndim != 5:
            raise ValueError(f"入力は5次元である必要があります: {inputs.shape}")
        batch_size, plane_count, channels, height, width = inputs.shape
        if plane_count != self.n_planes or channels != INPUT_CHANNELS:
            raise ValueError(
                f"入力shapeが不正です: expected [B,{self.n_planes},"
                f"{INPUT_CHANNELS},H,W], got {inputs.shape}"
            )
        if region_mask.shape != (batch_size, plane_count, height, width):
            raise ValueError(f"region_maskのshapeが不正です: {region_mask.shape}")

        flattened = inputs.reshape(batch_size * plane_count, channels, height, width)
        intermediates = cast(
            list[Tensor],
            self.encoder.forward_intermediates(  # type: ignore[operator]
                flattened, indices=FPN_STAGE_INDICES, intermediates_only=True
            ),
        )
        flattened_mask = region_mask.reshape(batch_size * plane_count, height, width)
        pooled, plane_valid = self._pool_regions(intermediates, flattened_mask)

        feature_dim = pooled.shape[-1]
        pooled = pooled.reshape(batch_size, plane_count, self.n_regions, feature_dim)
        plane_valid = plane_valid.reshape(batch_size, plane_count, self.n_regions)
        region_logits, region_observed = self._region_forward(pooled, plane_valid)
        return WeakModelOutput(
            region_logits=region_logits, region_observed=region_observed
        )

    def _pool_regions(
        self, intermediates: list[Tensor], flattened_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run FPN + mask pooling per plane chunk without keeping FPN maps.

        With grad enabled each chunk is checkpointed, so backward recomputes
        only that chunk's FPN from the (already stored) encoder features.
        Under no_grad the chunking alone bounds the transient FPN memory.
        """
        pooled_parts: list[Tensor] = []
        valid_parts: list[Tensor] = []
        total_planes = flattened_mask.shape[0]
        for start in range(0, total_planes, self.fpn_chunk_planes):
            end = min(start + self.fpn_chunk_planes, total_planes)
            chunk_features = [feature[start:end] for feature in intermediates]
            chunk_mask = flattened_mask[start:end]
            if torch.is_grad_enabled():
                pooled, valid = cast(
                    tuple[Tensor, Tensor],
                    checkpoint(
                        self._fpn_pool, chunk_mask, *chunk_features, use_reentrant=False
                    ),
                )
            else:
                pooled, valid = self._fpn_pool(chunk_mask, *chunk_features)
            pooled_parts.append(pooled)
            valid_parts.append(valid)
        return torch.cat(pooled_parts), torch.cat(valid_parts)

    def _fpn_pool(
        self, region_mask: Tensor, *features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """FPN fusion followed by mask-normalized pooling for one plane chunk."""
        fused = self.fpn(list(features))
        return mask_normalized_pool(fused, region_mask, self.n_regions)

    def _region_forward(
        self, pooled: Tensor, plane_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        """有効面だけを元の順序で共有BiLSTMへ通し、共有headへ渡す。"""
        batch_size, plane_count, n_regions, feature_dim = pooled.shape
        # [B, S, R, C] -> [B, R, S, C] -> [B*R, S, C] (region moves to batch axis).
        sequence = pooled.permute(0, 2, 1, 3).reshape(
            batch_size * n_regions, plane_count, feature_dim
        )
        valid = plane_valid.permute(0, 2, 1).reshape(
            batch_size * n_regions, plane_count
        )

        lengths = valid.sum(dim=1)
        region_observed = lengths > 0
        # A (bag, region) with zero valid planes is dropped from the loss via
        # region_observed; clamp only to keep pack_padded_sequence well-defined.
        lengths_for_pack = lengths.clamp(min=1)

        # Stable sort puts valid planes first while preserving their original
        # relative order -- required because a bidirectional LSTM's backward
        # pass would otherwise let invalid trailing planes contaminate every
        # valid position's hidden state if they were left in place.
        order = torch.argsort((~valid).long(), dim=1, stable=True)
        gathered = torch.gather(
            sequence, 1, order.unsqueeze(-1).expand(-1, -1, feature_dim)
        )

        packed = pack_padded_sequence(
            gathered, lengths_for_pack.to("cpu"), batch_first=True, enforce_sorted=False
        )
        lstm_out_packed, _ = self.region_lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out_packed, batch_first=True, total_length=plane_count
        )

        position_index = torch.arange(plane_count, device=pooled.device).unsqueeze(0)
        valid_position = position_index < lengths_for_pack.unsqueeze(1)
        masked = lstm_out * valid_position.unsqueeze(-1).to(lstm_out.dtype)
        counts = valid_position.sum(dim=1).clamp_min(1).unsqueeze(-1).to(lstm_out.dtype)
        mean_feature = masked.sum(dim=1) / counts

        mean_feature = self.region_dropout(mean_feature)
        logits = self.region_head(mean_feature).reshape(batch_size, n_regions)
        region_observed = region_observed.reshape(batch_size, n_regions)
        return logits, region_observed

    def train(self, mode: bool = True) -> WeakRegionMilModel:
        """学習状態への復帰後も、保持設定ならBatchNormを評価状態に保つ。

        set_backbone_trainableに相当するメソッドは実装しない。この保持は
        BN統計だけを固定するもので、encoder parameterのrequires_gradは
        一切変更しないため、backbone凍結と混同できない。
        """
        super().train(mode)
        if mode and self._freeze_encoder_bn_stats:
            self._set_encoder_bn_eval()
        return self

    def _set_encoder_bn_eval(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def transferred_parameters(self) -> list[nn.Parameter]:
        """Baseline 0 checkpointから転送されるencoder parameterを返す。"""
        return list(self.encoder.parameters())

    def new_parameters(self) -> list[nn.Parameter]:
        """ランダム初期化される新規module(FPN/region BiLSTM/head)を返す。"""
        return [
            *self.fpn.parameters(),
            *self.region_lstm.parameters(),
            *self.region_head.parameters(),
        ]
