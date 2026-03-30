# Codex Analysis: Vertebra-Conditional UNet Implementation
Date: 2026-03-30

## Question

We have a TinyUNet in PyTorch for cervical spine CT analysis. The model takes 2-channel input (CT image + vertebra mask) and outputs 4-channel heatmaps for boundary line detection.

Architecture (model.py):
- 4-stage encoder: d1(2->16), d2(16->32), d3(32->64), d4(64->128)
- MaxPool2d between stages
- 3-stage decoder with skip connections and ConvTranspose2d upsampling
- Output: Conv2d(16->4)
- Bottleneck: d4 output is shape (B, 128, H/8, W/8) for 224x224 input = (B, 128, 28, 28)

Dataset returns dict with keys: 'image'(2ch tensor), 'heatmaps'(4ch), 'line_params_gt'(4,2), 'sample', 'vertebra'(string: C1-C7), 'slice_idx'

Goal: Add vertebra label conditioning (C1-C7) to help the model distinguish vertebra-specific boundary patterns.

Plan from memo:
- Convert vertebra string to one-hot vector (7-dim)
- Expand one-hot spatially to (B, 7, Hb, Wb)
- Concat with bottleneck: (B, 128+7, 28, 28)
- 1x1 conv to restore channel count back to 128
- Then proceed to decoder as normal

Questions:
1. Is this bottleneck injection approach appropriate given the small dataset size (cervical spine CT, likely <100 patients)?
2. Are there better alternatives? Compare: (a) bottleneck concat (proposed), (b) learned embedding instead of one-hot, (c) FiLM conditioning (scale+shift), (d) injecting at multiple decoder levels
3. What are the risks/pitfalls of each approach?
4. Specific implementation recommendation with code sketch
5. Any special considerations for the 1x1 conv initialization to avoid disrupting pretrained weights?

## Codex Response

**結論**
小規模データ（<100患者想定）なら、あなたの案（`bottleneck` へのラベル注入）は妥当です。
ただし安定性のために「`bottleneck-only` + 低自由度 + 既存重みを壊さない初期化」を強く推奨します。

**1) 提案手法の妥当性**
妥当です。理由は以下です。
1. 変更範囲が小さく、ベースライン比較が明確。
2. 条件情報を深層のみで使うので過学習リスクが比較的低い。
3. 7クラスの離散ラベルには one-hot が素直。

**2) 代替案比較（推奨順）**
1. `bottleneck concat (one-hot)`
最初の実験として最適。実装簡単・解釈しやすい。
2. `FiLM (γ,β)`
パラメータ効率が良く、pretrained移行が特に安全。次点で有力。
3. `learned embedding`
7クラスでは one-hot に対する優位が小さい。小データでは過学習しやすい。
4. `multi-level injection`
表現力は上がるが、低Nでは過学習・ラベル依存のリスクが最も高い。

**3) 各手法の主なリスク**
1. bottleneck concat
ラベルに過依存して画像情報を無視する可能性（特にクラス不均衡時）。
2. embedding
埋め込み次元を上げるとすぐ過学習、解釈も難化。
3. FiLM
強すぎる modulation で特徴を潰す可能性（`gamma` 初期値設計が重要）。
4. multi-level
「効くが壊れやすい」典型。学習安定性と汎化を落としやすい。

**4) 実装推奨（具体）**
まずは `bottleneck-only` を以下で実施:
1. `batch["vertebra"]` を `idx (0..6)` に変換。
2. `one-hot -> (B,7,Hb,Wb)` に展開。
3. `cat([x4, cond], dim=1)`。
4. `1x1 conv` で `128` に戻す。
5. そのまま decoder。

最小スケッチ:

```python
# model.py
VERTEBRA_TO_IDX = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5, "C7": 6}

class TinyUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=4, feats=(16,32,64,128), dropout=0.0, num_vertebra=7):
        ...
        f1, f2, f3, f4 = feats
        self.num_vertebra = num_vertebra
        self.cond_proj = nn.Conv2d(f4 + num_vertebra, f4, kernel_size=1, bias=True)
        self._init_cond_proj_identity(f4)  # 下記

    def _init_cond_proj_identity(self, f4: int) -> None:
        with torch.no_grad():
            self.cond_proj.weight.zero_()
            self.cond_proj.bias.zero_()
            self.cond_proj.weight[:, :f4, 0, 0] = torch.eye(f4)

    def _onehot_map(self, vertebra_idx, h, w, dtype, device):
        oh = F.one_hot(vertebra_idx, num_classes=self.num_vertebra).to(dtype=dtype, device=device)
        return oh[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, x, vertebra_idx=None):
        ...
        x4 = self.d4(self.p3(x3))
        if vertebra_idx is not None:
            cond = self._onehot_map(vertebra_idx, x4.shape[-2], x4.shape[-1], x4.dtype, x4.device)
            x4 = self.cond_proj(torch.cat([x4, cond], dim=1))
        ...
```

```python
# trainer.py (train/eval 両方)
v_idx = torch.as_tensor([VERTEBRA_TO_IDX[v] for v in batch["vertebra"]], device=device, dtype=torch.long)
pred = torch.sigmoid(model(x, v_idx))
```

**5) 1x1 conv 初期化（pretrained破壊回避）**
最重要です。
既存重みを使うなら、上の「identity初期化」を使って `t=0` で旧モデルと同じ出力にしてください。
追加で安全にするなら:
1. 既存層は低LR、`cond_proj` は高LR（例: 既存 `5e-5`、新規 `2e-4`）。
2. 最初の数epochは `label dropout`（例 0.1）でラベル過依存を抑制。
