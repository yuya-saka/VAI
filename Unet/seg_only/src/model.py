"""SegOnlyUNet モデル定義 - Encoder + Seg Decoder のみ"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VERTEBRA_TO_IDX = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'C6': 5, 'C7': 6}


class ResBlock(nn.Module):
    """残差ブロック - GroupNorm + SiLU + Dropout付き"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, norm_groups: int = 8):
        super().__init__()
        # グループ数がチャンネル数を超えないよう調整
        g2 = min(norm_groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g2, out_ch)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.act2 = nn.SiLU()
        # チャンネル数が異なる場合はショートカット変換
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差接続付きフォワードパス"""
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act2(h + self.skip(x))


class Encoder(nn.Module):
    """共有エンコーダ - 4段階のダウンサンプリング"""

    def __init__(self, in_ch: int, features: tuple, dropout: float = 0.0, norm_groups: int = 8):
        super().__init__()
        f = features
        self.enc1 = ResBlock(in_ch, f[0], dropout=dropout, norm_groups=norm_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(f[0], f[1], dropout=dropout, norm_groups=norm_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(f[1], f[2], dropout=dropout, norm_groups=norm_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(f[2], f[3], dropout=dropout, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor) -> tuple:
        """エンコーダのフォワードパス - スキップ接続用の特徴マップを返す"""
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    """セグメンテーションデコーダ - アップサンプリングとスキップ接続"""

    def __init__(self, features: tuple, out_ch: int, dropout: float = 0.0, norm_groups: int = 8):
        super().__init__()
        f = features
        self.up4 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec4 = ResBlock(f[2] + f[2], f[2], dropout=dropout, norm_groups=norm_groups)
        self.up3 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec3 = ResBlock(f[1] + f[1], f[1], dropout=dropout, norm_groups=norm_groups)
        self.up2 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec2 = ResBlock(f[0] + f[0], f[0], dropout=dropout, norm_groups=norm_groups)
        self.head = nn.Conv2d(f[0], out_ch, kernel_size=1)

    def forward(self, x4: torch.Tensor, x3: torch.Tensor, x2: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """デコーダのフォワードパス"""
        y = self.up4(x4)
        y = self.dec4(torch.cat([y, x3], dim=1))
        y = self.up3(y)
        y = self.dec3(torch.cat([y, x2], dim=1))
        y = self.up2(y)
        y = self.dec2(torch.cat([y, x1], dim=1))
        return self.head(y)


class SegOnlyUNet(nn.Module):
    """セグメンテーション専用 UNet - Shared Encoder + Seg Decoder のみ

    multitask の ResUNet から line_decoder を除いたモデル。
    Encoder と Decoder クラスを再利用する。

    forward の戻り値:
        {
            'seg_logits': (B, seg_classes, H, W),
        }
    """

    def __init__(
        self,
        in_channels: int = 2,
        seg_classes: int = 5,
        features: tuple = (24, 48, 96, 192),
        dropout: float = 0.05,
        norm_groups: int = 8,
        num_vertebra: int = 0,
    ):
        super().__init__()
        self.num_vertebra = num_vertebra
        self.encoder = Encoder(in_channels, features, dropout=dropout, norm_groups=norm_groups)
        self.seg_decoder = Decoder(features, seg_classes, dropout=dropout, norm_groups=norm_groups)

        if self.num_vertebra > 0:
            self.cond_proj = nn.Conv2d(features[3] + self.num_vertebra, features[3], 1, bias=True)
            self._init_cond_proj_identity(features[3])

    def _init_cond_proj_identity(self, bottleneck_ch: int) -> None:
        """条件結合 1x1 Conv を恒等写像初期化する。"""
        with torch.no_grad():
            self.cond_proj.weight.zero_()
            self.cond_proj.bias.zero_()
            self.cond_proj.weight[:, :bottleneck_ch, 0, 0] = torch.eye(bottleneck_ch)

    def _onehot_map(
        self,
        vertebra_idx: torch.Tensor,
        h: int,
        w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """椎体インデックスを空間展開済み one-hot マップへ変換する。"""
        one_hot = F.one_hot(vertebra_idx, num_classes=self.num_vertebra).to(dtype=dtype, device=device)
        return one_hot[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x: torch.Tensor, vertebra_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """フォワードパス - セグメンテーションロジットを返す"""
        x1, x2, x3, x4 = self.encoder(x)

        if vertebra_idx is not None and self.num_vertebra > 0:
            cond = self._onehot_map(vertebra_idx, x4.shape[-2], x4.shape[-1], x4.dtype, x4.device)
            x4 = self.cond_proj(torch.cat([x4, cond], dim=1))

        seg_logits = self.seg_decoder(x4, x3, x2, x1)
        return {'seg_logits': seg_logits}
