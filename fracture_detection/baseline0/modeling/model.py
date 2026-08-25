"""Baseline 0のtimmバックボーンとBiLSTMモデル。"""

from __future__ import annotations

from typing import Any, cast

import timm
from torch import Tensor, nn

from fracture_detection.baseline0.data.constants import N_PLANES


def build_model(config: dict[str, Any]) -> Baseline0Model:
    """Configのmodel sectionからBaseline 0を構築する。"""
    model = config["model"]
    return Baseline0Model(
        backbone_name=str(model["backbone"]),
        pretrained=bool(model["pretrained"]),
        drop_rate=float(model["drop_rate"]),
        drop_path_rate=float(model["drop_path_rate"]),
        head_dropout=float(model["head_dropout"]),
        lstm_hidden=int(model["lstm_hidden"]),
        lstm_layers=int(model["lstm_layers"]),
        n_planes=int(model["n_planes"]),
    )


class Baseline0Model(nn.Module):
    """各面の特徴をBiLSTMで集約し、面ごとの骨折logitを返す。"""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_planes: int = N_PLANES,
    ) -> None:
        super().__init__()
        self.n_planes = n_planes
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
        feature_dim = num_features
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
        self._backbone_frozen = False

    def forward(self, inputs: Tensor) -> Tensor:
        """[B, 15, 6, H, W]を受け取り、面logit [B, 15]を返す。"""
        if inputs.ndim != 5:
            raise ValueError(f"入力は5次元である必要があります: {inputs.shape}")
        batch_size, plane_count, channels, height, width = inputs.shape
        if plane_count != self.n_planes or channels != 6:
            raise ValueError(
                f"入力shapeが不正です: expected [B,{self.n_planes},6,H,W], got {inputs.shape}"
            )
        flattened = inputs.reshape(batch_size * plane_count, channels, height, width)
        features = cast(Tensor, self.encoder(flattened))
        if features.ndim != 2:
            raise ValueError(f"backbone特徴shapeが不正です: {features.shape}")
        sequence = features.reshape(batch_size, plane_count, -1)
        contextual_features, _ = self.lstm(sequence)
        logits = cast(
            Tensor, self.head(contextual_features.reshape(batch_size * plane_count, -1))
        )
        return logits.reshape(batch_size, plane_count)

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

    def train(self, mode: bool = True) -> Baseline0Model:
        """学習状態への復帰後も、凍結バックボーンのBatchNormを評価状態に保つ。"""
        super().train(mode)
        if mode and self._backbone_frozen:
            self._set_backbone_batch_norm_eval()
        return self

    def backbone_parameters(self) -> list[nn.Parameter]:
        """異なる学習率の設定に使うバックボーンのパラメータを返す。"""
        return list(self.encoder.parameters())

    def shared_parameters(self) -> list[nn.Parameter]:
        """共有trainerの勾配監査対象`blocks[4]`を返す。"""
        blocks = getattr(self.encoder, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) <= 4:
            raise ValueError("backboneにblocks[4]がありません")
        return list(blocks[4].parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """異なる学習率の設定に使うLSTMと分類部のパラメータを返す。"""
        return [*self.lstm.parameters(), *self.head.parameters()]
