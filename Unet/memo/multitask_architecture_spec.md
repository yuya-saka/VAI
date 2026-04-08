# Multitask ResUNet アーキテクチャ仕様書（ドラフト）

**ステータス: ドラフト — 損失バランス等は後続議論で確定予定**

基づく設計案: `4領域抽出学習設計案.md`

---

## 1. プロジェクト構成

`Unet/multitask/` を `line_only/` と完全独立に作成。共通コードはコピーして独立させる。

```
Unet/
├── line_only/          ← 既存（変更なし）
├── multitask/          ← 新規
│   ├── train.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── model.py        ← ResUNet + dual decoder
│   │   ├── dataset.py      ← line_only からコピー・独立
│   │   ├── data_utils.py   ← モデル生成を multitask 用に変更
│   │   └── trainer.py      ← multitask 対応の訓練ループ
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── losses.py       ← seg CE + line MSE + 部分教師
│   │   ├── metrics.py      ← seg mIoU 追加 + line metrics コピー
│   │   ├── detection.py    ← line_only からコピー
│   │   └── visualization.py ← seg overlay 追加
│   ├── test/
│   │   └── __init__.py
│   └── config/
│       └── config.yaml     ← multitask 用設定
├── config/             ← 既存（line_only 用、変更なし）
└── preprocessing/      ← 共通（変更なし）
```

---

## 2. モデルアーキテクチャ

### 2.1 全体構造: Shared Encoder + Dual Decoder

```
Input (2ch: CT + vertebra mask)
         │
    ┌────▼────┐
    │ Encoder  │  ← Residual blocks, 4段
    │ (shared) │
    └─┬──┬──┬─┘
      │  │  │   skip connections (各段の出力)
  ┌───▼──▼──▼───┐   ┌───▼──▼──▼───┐
  │ Seg Decoder  │   │ Line Decoder │
  │  → 5ch logits│   │  → 4ch hmap  │
  └──────────────┘   └──────────────┘
```

- 初期実験では decoder 間の cross-task 接続は**入れない**（完全独立）
- 発展案（seg→line フィード）は成立性確認後に検討

### 2.2 Encoder

| 段 | 入力ch | 出力ch | 解像度 (224入力時) |
|----|--------|--------|--------------------|
| Stage 1 | 2 | 24 | 224×224 |
| Stage 2 | 24 | 48 | 112×112 |
| Stage 3 | 48 | 96 | 56×56 |
| Stage 4 (bottleneck) | 96 | 192 | 28×28 |

- **Downsampling**: MaxPool2d(2) （既存と同じ）
- 各段は Residual Block × 1

### 2.3 Residual Block

**Pre-activation 方式** を採用（Codex 推奨）:

```
x ─→ GN → ReLU → Conv3x3 → GN → ReLU → Conv3x3 → (+) → out
│                                                    ↑
└──────────────── shortcut (identity or 1x1) ────────┘
```

- チャンネル数が変わる場合: **1×1 Conv projection** で合わせる（zero-padding は不可）
- チャンネル数が同じ場合: identity shortcut
- Conv の bias: **False**（GN の直後は bias 不要）
- Dropout: 2つ目の Conv の前に Dropout2d

### 2.4 GroupNorm 設定

| チャンネル数 | グループ数 | ch/group |
|-------------|-----------|----------|
| 24 | 8 | 3 |
| 48 | 8 | 6 |
| 96 | 8 | 12 |
| 192 | 8 | 24 |

- **グループ数 = 8 で統一**（[24,48,96,192] すべてで割り切れる）
- batch_size=8 の医用画像で BatchNorm より安定

### 2.5 Decoder (Seg / Line 共通構造)

各 decoder は同じ構造だが、パラメータは完全に独立:

| 段 | 入力ch | skip ch | concat後 | 出力ch |
|----|--------|---------|----------|--------|
| Up4 | 192 | 96 | 288 | 96 |
| Up3 | 96 | 48 | 144 | 48 |
| Up2 | 48 | 24 | 72 | 24 |

- **Upsampling**: ConvTranspose2d(kernel=2, stride=2)
- 各段は Residual Block × 1（skip concat 後）

### 2.6 出力ヘッド

| ヘッド | 構造 | 出力ch | 活性化 |
|--------|------|--------|--------|
| Seg | Conv2d(24, 5, 1) | 5 | なし（CE loss に logits を渡す） |
| Line | Conv2d(24, 4, 1) | 4 | なし（trainer 側で sigmoid） |

### 2.7 forward の返り値

```python
def forward(self, x) -> dict[str, torch.Tensor]:
    return {
        "seg_logits": seg_out,    # (B, 5, H, W)
        "line_heatmaps": line_out, # (B, 4, H, W)
    }
```

---

## 3. 損失設計

**⚠️ 暫定値 — 損失バランスは後続議論で確定予定**

### 3.1 基本構成

```
L = L_line + α · L_seg
```

| 損失 | 関数 | 備考 |
|------|------|------|
| L_line | MSE(sigmoid(pred), gt_heatmap) | 既存 line_only と同じ |
| L_seg | CrossEntropyLoss(logits, gt_mask) | 5クラス、ignore不要 |

### 3.2 α の初期値（暫定）

- **α = 0.03** から開始（Codex 推奨）
- MSE ≈ 1e-3 オーダー、CE ≈ 1e-1〜1e0 オーダーのため 0.1 は強すぎる
- 目安: 学習初期の `α·CE` が `MSE` の 0.3〜1.0倍 程度に収まること
- ⚠️ **損失バランスは後続議論で確定予定**

### 3.3 部分教師対応

```python
# ラベルありサンプルのみで CE を計算（分母から除外する）
labeled = has_gt_region_mask  # (B,) bool
if labeled.any():
    seg_loss = CE(seg_logits[labeled], gt_mask[labeled]).mean()
    loss = line_loss + alpha * seg_loss
else:
    loss = line_loss
```

- 「loss をゼロにして全体平均」は **避ける**（分母が膨らんで勾配が希薄化する）
- バッチ構成: **batch_size=8 のうち 3〜4 サンプルは seg GT あり**に調整
  - `WeightedRandomSampler` で実現する

### 3.4 後回しにするもの（設計案 §6.3 に準拠）

以下は初期実験では使わない:
- Dice loss / Focal loss
- angle loss / rho loss / moment loss
- seg-line consistency loss
- 動的 loss weighting
- seg loss の warmup

---

## 4. 評価指標

### Line（既存と同じ）
- angle_error_deg
- rho_error_px
- perpendicular_dist_px
- peak_dist

### Seg（新規追加）
- **mIoU** (mean Intersection over Union, 5クラス)
- **per-class IoU** (背景, 左横突孔, 椎体中心, 右横突孔, 後方要素)
- **Dice** (参考値)

---

## 5. config.yaml 構造（暫定）

```yaml
experiment:
  phase: "multitask_v1"
  name: "baseline"

data:
  # line_only と同じ
  use_png: true
  root_dir: "/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset"
  group: "ALL"
  image_size: 224
  sigma: 3.5
  n_folds: 5
  test_fold: 0
  random_seed: 42

model:
  type: "ResUNet"         # ← 新規: モデル種別
  in_channels: 2
  seg_classes: 5          # ← 新規
  line_channels: 4
  features: [24, 48, 96, 192]  # Codex 推奨: ~1.58M params
  dropout: 0.05
  norm: "group"           # ← 新規: "group" or "batch"
  norm_groups: 8          # ← 新規

training:
  gpu_id: 2
  batch_size: 8
  learning_rate: 2e-4
  weight_decay: 2e-4
  epochs: 230
  early_stopping_patience: 15
  lr_patience: 8
  lr_factor: 0.5
  grad_clip: 1.0
  num_workers: 4

loss:
  # ⚠️ 暫定値（損失バランスは後続議論で確定予定）
  alpha_seg: 0.03         # seg CE の重み（Codex 推奨: 0.02〜0.05）
  use_line_loss: false    # 幾何損失（angle/rho）は初期では無効

augmentation:
  # line_only と同じ設定をコピー
  horizontal_flip: false
  rotation: true
  rotation_limit: 15
  scale: true
  scale_limit: 0.05
  brightness_contrast: true
  brightness_limit: 0.1
  contrast_limit: 0.1

evaluation:
  heatmap_threshold: 0.5
  line_extend_ratio: 1.0
  metrics_frequency: 1

wandb:
  enabled: true
  project: null
  run_name: null
```

---

## 6. パラメータ数見積もり（Codex 実測値）

| チャネル幅 | 合計パラメータ数 |
|-----------|----------------|
| [16,32,64,128] TinyUNet | ~0.70M |
| **[24,48,96,192] 採用** | **~1.58M** |
| [32,64,128,256] 没案 | ~2.81M（過大） |

- TinyUNet 比: 約 2.3 倍
- データ量 (~500 slices) に対して妥当な範囲
- overfitting は validation mIoU / val_loss の乖離で監視する

---

## 7. 比較実験計画（設計案 §6.2 準拠）

| 条件 | モデル | 出力 | 損失 |
|------|--------|------|------|
| A: line-only baseline | TinyUNet [16,32,64,128] | 4ch heatmap | MSE |
| B: multitask | ResUNet [24,48,96,192] dual decoder | 5ch seg + 4ch line | MSE + α·CE |

確認項目:
1. seg を足すことで line が良くなるか（少なくとも壊れないか）
2. seg 自体が成立するか（mIoU）
3. overfitting しないか

---

## 8. 未確定事項（後続議論）

- [ ] **損失バランス α の最適値**（学習初期の loss 比実測に基づいて決定、暫定 0.03）
- [ ] 動的重み付け（uncertainty weighting）の要否
- [ ] seg loss の warmup 要否
- [ ] seg decoder → line decoder の cross-task 接続（発展案、baseline 確立後）
- [ ] Dice loss 追加の要否
- [ ] WeightedRandomSampler の具体的な重み設定（seg GT あり率に依存）
