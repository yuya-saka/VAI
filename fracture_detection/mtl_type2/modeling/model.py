"""RSNA Type2型のbranch分離MTLモデル。

CNNだけを共有し、whole用とregion用のBiLSTM・headを独立に持つ。
RSNA 1st place solutionのstage2 type2が `lstm`/`head` と `lstm2`/`head2` を
分けているのと同じ構成で、patient branchをwhole、C1-C7 branchをregionへ対応させる。
"""

from __future__ import annotations

from typing import cast

import timm
from torch import Tensor, nn

from fracture_detection.common.constants import N_PLANES, N_REGIONS
from fracture_detection.core.contracts import ArmOutput

ALLOWED_INPUT_CHANNELS = (6, 10)


class BranchedMtlModel(nn.Module):
    """共有CNN + 独立したwhole/region BiLSTMから面logitを返す。"""

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
        if in_chans not in ALLOWED_INPUT_CHANNELS:
            raise ValueError(
                f"in_chansは{ALLOWED_INPUT_CHANNELS}のいずれかが必要です: {in_chans}"
            )
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
        lstm_dropout = drop_rate if lstm_layers > 1 else 0.0
        self.whole_lstm = nn.LSTM(
            feature_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.region_lstm = nn.LSTM(
            feature_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
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
        whole_context, _ = self.whole_lstm(sequence)
        whole = self.whole_head(
            whole_context.reshape(batch_size * plane_count, -1)
        ).reshape(batch_size, plane_count)
        region_context, _ = self.region_lstm(sequence)
        regions = self.region_head(
            region_context.reshape(batch_size * plane_count, -1)
        ).reshape(batch_size, plane_count, N_REGIONS)
        return ArmOutput(whole_logits=whole, region_logits=regions)

    def shared_parameters(self) -> list[nn.Parameter]:
        """勾配監査対象の`blocks[4]` parameterを返す。"""
        blocks = getattr(self.encoder, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) <= 4:
            raise ValueError("backboneにblocks[4]がありません")
        return list(blocks[4].parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        """共有CNN parameterを返す。"""
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """whole/region両branchのBiLSTMとhead parameterを返す。"""
        return [
            *self.whole_lstm.parameters(),
            *self.whole_head.parameters(),
            *self.region_lstm.parameters(),
            *self.region_head.parameters(),
        ]

    def region_branch_parameters(self) -> list[nn.Parameter]:
        """region branch専用parameterを返す（学習診断用）。"""
        return [*self.region_lstm.parameters(), *self.region_head.parameters()]

    def set_backbone_trainable(self, trainable: bool) -> None:
        """CNNのgradientと凍結中BN状態を切り替える。"""
        self._backbone_frozen = not trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self._set_backbone_batch_norm_eval()

    def train(self, mode: bool = True) -> BranchedMtlModel:
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
    """面単位の分類head。既存armと同一構成。"""

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
