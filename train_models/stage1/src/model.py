"""
RSNA頸椎骨折分類モデル。

RSNA 2022 1位解法 Type1 アーキテクチャ:
timm backbone (EfficientNetV2-S) + BiLSTM + per-slice head。
"""

from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor


class TimmModel(nn.Module):
    """
    RSNA 2022 1位解法 Type1 アーキテクチャ。

    timm backboneで各スライスの特徴量を抽出し、
    BiLSTMでスライス間の時系列文脈をモデリングし、
    per-sliceのfracture logitを出力する。

    Args:
        backbone: timm モデル名（例: "tf_efficientnetv2_s"）
        in_chans: 入力チャンネル数（5ch CT + 1ch mask = 6）
        n_slices: 椎体あたりのスライス数（15）
        drop_rate: backbone dropout率
        drop_path_rate: backbone drop path率
        drop_rate_last: head直前のdropout率
        lstm_hidden: LSTM隠れ層サイズ
        lstm_layers: LSTM積み重ね層数
        out_dim: 出力次元（1 = binary fracture logit）
        use_patient_head: patient-level 補助headを有効化するか
        pretrained: ImageNet事前学習重みを使用するか
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        in_chans: int = 6,
        n_slices: int = 15,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_rate_last: float = 0.3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        out_dim: int = 1,
        use_patient_head: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n_slices = n_slices
        self.in_chans = in_chans
        self.use_patient_head = use_patient_head

        # timm backbone: 最終分類層をIdentityに置換して特徴量を取得
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=out_dim,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )

        # EfficientNetV2系の特徴量次元を取得
        hdim = self._get_feature_dim(backbone)
        self.encoder.classifier = nn.Identity()

        # スライス間の時系列モデリング
        lstm_drop = drop_rate if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            hdim,
            lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_drop,
            bidirectional=True,
            batch_first=True,
        )

        # per-slice分類head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_hidden, out_dim),
        )
        if use_patient_head:
            self.patient_lstm = nn.LSTM(
                hdim,
                lstm_hidden,
                num_layers=lstm_layers,
                dropout=lstm_drop,
                bidirectional=True,
                batch_first=True,
            )
            self.patient_head = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.BatchNorm1d(lstm_hidden),
                nn.Dropout(drop_rate_last),
                nn.LeakyReLU(0.1),
                nn.Linear(lstm_hidden, 1),
            )

    @staticmethod
    def _get_feature_dim(backbone: str) -> int:
        """backboneの特徴量次元を返す。"""
        # EfficientNetV2-S: 1280
        # 他のモデルを追加する場合はここに追記
        dim_map = {
            "tf_efficientnetv2_s": 1280,
            "tf_efficientnetv2_s_in21ft1k": 1280,
            "efficientnetv2_s": 1280,
            "convnext_nano": 640,
            "convnext_tiny": 768,
        }
        for key, dim in dim_map.items():
            if key in backbone:
                return dim
        # 未知モデルはダミーforwardで取得
        return 1280

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x: shape (bs, n_slices, in_chans, H, W)

        Returns:
            per-slice logits: shape (bs, n_slices)
            use_patient_head=True の場合は (per-slice logits, patient logits)
        """
        bs = x.shape[0]
        # 全スライスをバッチ次元に展開して一括処理
        x = x.view(bs * self.n_slices, self.in_chans, x.shape[-2], x.shape[-1])
        feat = self.encoder(x)                          # (bs*n_slices, hdim)
        feat = feat.view(bs, self.n_slices, -1)         # (bs, n_slices, hdim)
        slice_feat, _ = self.lstm(feat)                 # (bs, n_slices, lstm_hidden*2)
        slice_feat = slice_feat.contiguous().view(bs * self.n_slices, -1)
        slice_logits = self.head(slice_feat)            # (bs*n_slices, 1)
        slice_logits = slice_logits.view(bs, self.n_slices).contiguous()

        if not self.use_patient_head:
            return slice_logits

        patient_feat, _ = self.patient_lstm(feat)
        patient_logits = self.patient_head(patient_feat[:, 0]).view(bs)
        return slice_logits, patient_logits

    def backbone_parameters(self) -> list[nn.Parameter]:
        """encoder (timm backbone) のパラメータ。differential LR用。"""
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """LSTM + 分類headのパラメータ。differential LR用。"""
        params = list(self.lstm.parameters()) + list(self.head.parameters())
        if self.use_patient_head:
            params += list(self.patient_lstm.parameters())
            params += list(self.patient_head.parameters())
        return params
