'''ResUNet モデル定義 - Shared Encoder + Dual Decoder'''

import torch
import torch.nn as nn
import torch.nn.functional as F


VERTEBRA_TO_IDX = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'C6': 5, 'C7': 6}


class ResBlock(nn.Module):
    '''Pre-activation 残差ブロック

    構造: GN -> ReLU -> Conv3x3 -> GN -> ReLU -> Dropout2d -> Conv3x3 + shortcut
    shortcut: チャンネル数が変わる場合は 1x1 Conv、同じ場合は identity
    Conv の bias=False（GN の直後は不要）
    '''

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, norm_groups: int = 8):
        super().__init__()
        # num_channels が norm_groups より小さい場合に備えてクランプ
        g1 = min(norm_groups, in_ch)
        g2 = min(norm_groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class Encoder(nn.Module):
    '''共有エンコーダ - 4段、MaxPool2d downsampling

    戻り値（forward）: (x1, x2, x3, x4) - skip connection 用
    x1: Stage1出力 (B, f[0], H, W)
    x2: Stage2出力 (B, f[1], H/2, W/2)
    x3: Stage3出力 (B, f[2], H/4, W/4)
    x4: Stage4出力（ボトルネック）(B, f[3], H/8, W/8)
    '''

    def __init__(self, in_ch: int, features: tuple, dropout: float = 0.0, norm_groups: int = 8):
        super().__init__()
        f = features
        self.stage1 = ResBlock(in_ch, f[0], dropout=dropout, norm_groups=norm_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ResBlock(f[0], f[1], dropout=dropout, norm_groups=norm_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ResBlock(f[1], f[2], dropout=dropout, norm_groups=norm_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.stage4 = ResBlock(f[2], f[3], dropout=dropout, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor) -> tuple:
        x1 = self.stage1(x)
        x2 = self.stage2(self.pool1(x1))
        x3 = self.stage3(self.pool2(x2))
        x4 = self.stage4(self.pool3(x3))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    '''デコーダ - ConvTranspose2d upsampling + skip concat

    Seg / Line それぞれに独立したインスタンスを作成する。
    パラメータは共有しない。
    '''

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
        y = self.up4(x4)
        y = self.dec4(torch.cat([y, x3], dim=1))
        y = self.up3(y)
        y = self.dec3(torch.cat([y, x2], dim=1))
        y = self.up2(y)
        y = self.dec2(torch.cat([y, x1], dim=1))
        return self.head(y)


class ResUNet(nn.Module):
    '''Multitask ResUNet - Shared Encoder + Dual Decoder

    共有エンコーダから seg デコーダと line デコーダに分岐する。
    2つのデコーダは独立したパラメータを持つ。

    forward の戻り値:
        {
            'seg_logits': (B, seg_classes, H, W),
            'line_heatmaps': (B, line_channels, H, W),
        }
    '''

    def __init__(
        self,
        in_channels: int = 2,
        seg_classes: int = 5,
        line_channels: int = 4,
        features: tuple = (24, 48, 96, 192),
        dropout: float = 0.05,
        norm_groups: int = 8,
        num_vertebra: int = 0,
    ):
        super().__init__()
        self.num_vertebra = num_vertebra
        self.encoder = Encoder(in_channels, features, dropout=dropout, norm_groups=norm_groups)
        self.seg_decoder = Decoder(features, seg_classes, dropout=dropout, norm_groups=norm_groups)
        self.line_decoder = Decoder(features, line_channels, dropout=dropout, norm_groups=norm_groups)

        if self.num_vertebra > 0:
            self.cond_proj = nn.Conv2d(features[3] + self.num_vertebra, features[3], 1, bias=True)
            self._init_cond_proj_identity(features[3])

    def _init_cond_proj_identity(self, bottleneck_ch: int) -> None:
        '''条件結合 1x1 Conv を恒等写像初期化する。'''
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
        '''椎体インデックスを空間展開済み one-hot マップへ変換する。'''
        one_hot = F.one_hot(vertebra_idx, num_classes=self.num_vertebra).to(dtype=dtype, device=device)
        return one_hot[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x: torch.Tensor, vertebra_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x1, x2, x3, x4 = self.encoder(x)

        if vertebra_idx is not None and self.num_vertebra > 0:
            cond = self._onehot_map(vertebra_idx, x4.shape[-2], x4.shape[-1], x4.dtype, x4.device)
            x4 = self.cond_proj(torch.cat([x4, cond], dim=1))

        seg_logits = self.seg_decoder(x4, x3, x2, x1)
        line_heatmaps = self.line_decoder(x4, x3, x2, x1)
        return {'seg_logits': seg_logits, 'line_heatmaps': line_heatmaps}
