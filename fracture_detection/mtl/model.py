"""Control–B / Baseline 1–B共通のhard-sharing MTLモデル。"""

from __future__ import annotations

from typing import cast

import timm
from torch import Tensor, nn

from fracture_detection.common.constants import N_PLANES, N_REGIONS
from fracture_detection.core.contracts import ArmOutput


class EarlyFusionMtlModel(nn.Module):
    """CNN+BiLSTM特徴からwholeと4領域の面logitを返す。"""

    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s",
        *,
        in_chans: int,
        pretrained: bool,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_planes: int = N_PLANES,
    ) -> None:
        super().__init__()
        if in_chans not in {6, 10}:
            raise ValueError("Controlは6ch、Baseline 1は10chが必要です")
        self.in_chans = in_chans
        self.n_planes = n_planes
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feature_dim = getattr(self.encoder, "num_features", None)
        if not isinstance(feature_dim, int):
            raise ValueError("timm backboneから特徴次元を取得できません")
        self.lstm = nn.LSTM(
            feature_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=drop_rate if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.whole_head = _PlaneHead(lstm_hidden * 2, lstm_hidden, head_dropout, 1)
        self.region_head = _PlaneHead(
            lstm_hidden * 2, lstm_hidden, head_dropout, N_REGIONS
        )
        self._backbone_frozen = False

    def forward(self, inputs: Tensor) -> ArmOutput:
        """[B,N,C,H,W]からwhole [B,N]とregion [B,N,4]を返す。"""
        batch_size, plane_count, flattened = self._flatten_inputs(inputs)
        features = cast(Tensor, self.encoder(flattened))
        if features.ndim != 2:
            raise ValueError(f"backbone特徴shapeが不正です: {features.shape}")
        sequence = features.reshape(batch_size, plane_count, -1)
        contextual, _ = self.lstm(sequence)
        flat_context = contextual.reshape(batch_size * plane_count, -1)
        whole = self.whole_head(flat_context).reshape(batch_size, plane_count)
        regions = self.region_head(flat_context).reshape(
            batch_size, plane_count, N_REGIONS
        )
        return ArmOutput(whole_logits=whole, region_logits=regions)

    def shared_parameters(self) -> list[nn.Parameter]:
        """λ校正対象の`blocks[4]` parameterを返す。"""
        blocks = getattr(self.encoder, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) <= 4:
            raise ValueError("backboneにblocks[4]がありません")
        return list(blocks[4].parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        """CNN parameterを返す。"""
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """BiLSTMと分類head parameterを返す。"""
        return [
            *self.lstm.parameters(),
            *self.whole_head.parameters(),
            *self.region_head.parameters(),
        ]

    def set_backbone_trainable(self, trainable: bool) -> None:
        """CNNのgradientと凍結中BN状態を切り替える。"""
        self._backbone_frozen = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self._set_backbone_batch_norm_eval()

    def train(self, mode: bool = True) -> EarlyFusionMtlModel:
        """凍結CNNのBNをtrain復帰させない。"""
        super().train(mode)
        if mode and self._backbone_frozen:
            self._set_backbone_batch_norm_eval()
        return self

    def _flatten_inputs(self, inputs: Tensor) -> tuple[int, int, Tensor]:
        if inputs.ndim != 5:
            raise ValueError("入力は[B,N,C,H,W]である必要があります")
        batch_size, plane_count, channels, height, width = inputs.shape
        if plane_count != self.n_planes or channels != self.in_chans:
            raise ValueError(
                f"入力shapeが不正です: expected [B,{self.n_planes},{self.in_chans},H,W], "
                f"got {inputs.shape}"
            )
        return (
            batch_size,
            plane_count,
            inputs.reshape(batch_size * plane_count, channels, height, width),
        )

    def _set_backbone_batch_norm_eval(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


class _PlaneHead(nn.Sequential):
    """全アームで同じ面単位分類head。"""

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float, output_dim: int
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
