"""UNet モデル定義"""

import torch
import torch.nn as nn


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

    def __init__(self, in_ch=2, out_ch=4, feats=(16, 32, 64, 128), dropout=0.0):
        super().__init__()
        f1, f2, f3, f4 = feats
        self.d1 = DoubleConv(in_ch, f1, dropout=dropout)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(f1, f2, dropout=dropout)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(f2, f3, dropout=dropout)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(f3, f4, dropout=dropout)

        self.u3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.up3 = DoubleConv(f3 + f3, f3, dropout=dropout)
        self.u2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.up2 = DoubleConv(f2 + f2, f2, dropout=dropout)
        self.u1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.up1 = DoubleConv(f1 + f1, f1, dropout=dropout)

        self.out = nn.Conv2d(f1, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))

        y = self.u3(x4)
        y = self.up3(torch.cat([y, x3], 1))
        y = self.u2(y)
        y = self.up2(torch.cat([y, x2], 1))
        y = self.u1(y)
        y = self.up1(torch.cat([y, x1], 1))
        return self.out(y)
