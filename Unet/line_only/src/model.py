"""UNet モデル定義"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VERTEBRA_TO_IDX = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5, "C7": 6}


class DoubleConv(nn.Module):
    """2回の畳み込み + BatchNorm + ReLU（オプションでDropout付き）"""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """軽量UNet: 4段エンコーダ・デコーダ + スキップ接続"""

    def __init__(
        self, in_ch=2, out_ch=4, feats=(16, 32, 64, 128), dropout=0.0, num_vertebra: int = 0
    ):
        super().__init__()
        f1, f2, f3, f4 = feats
        self.num_vertebra = num_vertebra
        self.d1 = DoubleConv(in_ch, f1, dropout=dropout)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(f1, f2, dropout=dropout)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(f2, f3, dropout=dropout)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(f3, f4, dropout=dropout)

        if self.num_vertebra > 0:
            self.cond_proj = nn.Conv2d(f4 + self.num_vertebra, f4, kernel_size=1, bias=True)
            self._init_cond_proj_identity(f4)

        self.u3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.up3 = DoubleConv(f3 + f3, f3, dropout=dropout)
        self.u2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.up2 = DoubleConv(f2 + f2, f2, dropout=dropout)
        self.u1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.up1 = DoubleConv(f1 + f1, f1, dropout=dropout)

        self.out = nn.Conv2d(f1, out_ch, 1)

    def _init_cond_proj_identity(self, f4: int) -> None:
        with torch.no_grad():
            self.cond_proj.weight.zero_()
            self.cond_proj.bias.zero_()
            self.cond_proj.weight[:, :f4, 0, 0] = torch.eye(f4)

    def _onehot_map(self, vertebra_idx, h, w, dtype, device):
        oh = F.one_hot(vertebra_idx, num_classes=self.num_vertebra).to(
            dtype=dtype, device=device
        )
        return oh[:, :, None, None].expand(-1, -1, h, w)

    def forward(
        self, x: torch.Tensor, vertebra_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        if vertebra_idx is not None and self.num_vertebra > 0:
            cond = self._onehot_map(
                vertebra_idx, x4.shape[-2], x4.shape[-1], x4.dtype, x4.device
            )
            x4 = self.cond_proj(torch.cat([x4, cond], dim=1))

        y = self.u3(x4)
        y = self.up3(torch.cat([y, x3], 1))
        y = self.u2(y)
        y = self.up2(torch.cat([y, x2], 1))
        y = self.u1(y)
        y = self.up1(torch.cat([y, x1], 1))
        return self.out(y)
