# Codex Analysis: ResUNet Multitask Architecture Design
Date: 2026-04-08

## Question
Architecture design review for 2D ResUNet-medium multitask (seg+line) system

## Codex Response

**全体推奨**: 初回実験は `ResUNet-basic + GN(8) + shared encoder + 完全独立 dual decoder + チャネル幅 [24,48,96,192]`

---

### 1. 残差ブロック設計

- **Pre-activation basic block** を推奨: `GN → ReLU → Conv3x3 → GN → ReLU → Conv3x3 + shortcut`
- チャネルが変わる箇所は必ず **learnable 1x1 projection shortcut** を使用（zero-padding は不可）
- **GroupNorm: 8 groups** を推奨（チャネル [32,64,128,256] でそれぞれ 4/8/16/32 ch/group）

---

### 2. デコーダ共有戦略

- 初回実験は **shared encoder + fully independent decoders** で完結させる
- `seg → line` のクロスタスク融合は partial seg label 下では seg 側不安定時に line 側を巻き込むリスクがある
- 融合は baseline 確立後に、最後の 1-2 stage への `1x1 gate` または FiLM 程度の軽い手段で試す

---

### 3. チャネル次元（重要な修正点）

パラメータ数の推定:

| チャネル幅 | 推定パラメータ数 |
|-----------|----------------|
| [16,32,64,128] | 約 0.70M |
| **[24,48,96,192]** | **約 1.58M ← 推奨** |
| [32,64,128,256] | 約 2.81M（大きすぎ） |

提案の `[32,64,128,256]` は「1-2M 狙い」から外れ、500スライス程度のデータには過大。**[24,48,96,192] が初回本命**。

---

### 4. 正規化

- `batch_size=8` の医用画像なら BN より **GroupNorm** が安定
- 固定 **8 groups**、conv bias は **False**
- residual 接続との相性問題はほぼなし

---

### 5. Loss バランシング（重要な修正点）

- `alpha=0.1` は **強すぎる可能性あり**
  - heatmap MSE は `1e-3` オーダー、CE は `1e-1 ~ 1e0` オーダー
  - `0.1 * CE` が line loss を大きく上回る
- **開始値は alpha=0.02 ~ 0.05（推奨: 0.03）**
- 目安: 「学習初期の weighted seg loss が line loss の 0.3x ~ 1.0x 程度」

---

### 6. Partial Supervision の扱い（重要な修正点）

- 「loss を zero にして batch 全体平均」は避ける
- 正しい方法: **`seg_loss = CE(labeled_samples).mean()`**（ラベルなしサンプルを分母から除外）
- batch 構成: **batch_size=8 のうち 3-4 サンプルは seg GT あり** に調整
  - `WeightedRandomSampler` か 2-stream sampler を使用

---

### 7. パラメータ数の推定（[32,64,128,256] dual decoder の場合）

- Encoder: 約 1,216,000
- Decoder 1本: 約 796,544
- Decoder 2本合計: 約 1,593,088
- Output heads: 297
- **合計: 約 2,809,385（現行 TinyUNet の約 5.8倍）**

---

### 初回実装の推奨設定まとめ

| 項目 | 推奨値 |
|------|--------|
| ブロック | Pre-act basic ResBlock |
| Shortcut | 1x1 projection conv（チャネル変化時） |
| チャネル幅 | **[24,48,96,192]** |
| 正規化 | GroupNorm(8 groups)、bias=False |
| デコーダ | 完全独立（クロス融合は後回し） |
| alpha | **0.03**（0.1は強すぎ） |
| Seg loss | ラベルありサンプルのみで mean |
| Batch構成 | 3-4/8 を seg GT ありに固定 |
