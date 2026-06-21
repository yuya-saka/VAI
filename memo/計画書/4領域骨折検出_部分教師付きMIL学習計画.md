# 4領域骨折検出：部分教師付きMIL学習計画

## 目的

椎体ごとの4領域（左横突孔, 右横突孔, 椎体中心部, 後方要素）について、領域別の骨折検出を行う。
RSNAデータのbbox付き症例を活用し、Partially Supervised MIL（部分教師付きMIL）で学習する。

---

## 背景

### RSNAデータのラベル構造

| データ | 件数 | ラベル粒度 |
|-------|------|----------|
| 全training | 2019 scans (961 fracture+) | 椎体レベル（C1-C7 binary） |
| bbox付きsubset | 235 scans | axial画像上のbbox |
| 残り | ~1784 scans | 椎体レベルのみ |

4領域ラベルを直接作れるのはbbox付き235例のみ。
残りは「この椎体に骨折がある」としか分からない。

### 先行研究の知見

Hu et al.（RSNAモデル外部検証）で判明した false-negative の最多部位：

```
vertebral body end plate: 36/135
transverse process:       35/135
spinous process:          17/135
```

RSNA上位モデルでも椎体レベル分類では横突起・棘突起の骨折を見逃す。
4領域分割による領域別検出でこの問題に対処する。

---

## 学習手法：Partially Supervised MIL

### 基本構造

```
bag      = 1椎体
instances = 4領域（左横突孔, 右横突孔, 椎体中心, 後方要素）
bag-level label     = 椎体骨折 yes/no（全症例にある）
instance-level label = bbox ∩ 4領域mask → 領域別骨折 yes/no（bbox付き症例のみ）
```

完全に教師なし（MILのみ）ではなく、bbox付き症例の instance-level annotation を明示的に教師として使う。
「少量だけアノテーションがある」という現実的な状況に対応する枠組みである。

### 参考文献

- Hibi et al. "Automated screening of CT using weakly supervised anomaly detection"
  - scan-level label → slice-level prediction（DMIL損失 + center損失）
  - annotation 97.1% 削減で AUC 0.89
- Zhang et al. "Weakly Supervised Object Localization and Detection: A Survey"
  - auxiliary training data with instance-level annotation の活用
  - initialization + refinement の2段階フレームワーク
- Salehinejad et al. "Deep Sequential Learning for Cervical Spine Fracture Detection"
  - ResNet-50 + BLSTM、case-level classification

---

## 3パターンの損失設計

### (A) Negative vertebra（全データ）

```
椎体に骨折なし → 4領域すべて non-fracture
→ 通常の教師ありBCE（instance-level）
```

negative vertebra は clean な negative label を全量提供する。

### (B) Positive vertebra + bbox あり

```
bbox × 4領域mask → overlap計算 → 領域別ラベル確定
→ 通常の教師ありBCE（instance-level）
```

bbox と overlap がある領域に fracture label を付与する。

### (C) Positive vertebra + bbox なし

```
「4領域のうち少なくとも1つが positive」
→ MIL損失（DMIL）
```

椎体レベルの label のみで、どの領域かは不明。
MIL制約で学習する。

### 統合損失関数

```
L = L_supervised(A, B) + λ * L_MIL(C)

L_supervised = Σ BCE(region_pred, region_label)
               ← negative全例 + bbox付きpositive例

L_MIL = DMIL損失
        ← bbox無しのpositive例のみ
```

---

## DMIL損失の詳細

Hibi et al.に基づく。

```
positive vertebra（bbox無し）:
  4領域のスコアを降順にソート
  top-k 領域（k = ceil(4/α), α=4 → k=1）を選択
  選択された領域に対して BCE(pred, 1) を計算

negative vertebra:
  全4領域に対して BCE(pred, 0) を計算（L_supervisedで処理）
```

4 instances と少ないため、k=1（最もスコアの高い1領域のみ positive と仮定）が自然。

---

## instance-level annotation が効く理由

1. **正例の anchor**: MIL がどの instance を positive と判定すべきかの基準ができる
2. **特徴空間の構造化**: 教師あり instance が特徴空間に領域ごとの分離を作る
3. **MIL の退化防止**: pure MIL は全 instance を同じスコアにする退化解がありうるが、explicit label がそれを防ぐ

Hibi et al.の結果（Fig. 6）では、同等アノテーション量の教師ありが AUC 0.63 に対し、WSAD が AUC 0.89。
本研究では教師ありデータが追加されるため、さらに有利な条件。

---

## bbox → 4領域ラベルの割り当てルール

```
fracture bbox
↓
4領域maskとoverlap（IoU）計算
↓
overlap がある全領域に fracture label を付与
```

1つのbboxが複数領域にまたがる場合は、overlap がある全領域を positive とする。

---

## モデルアーキテクチャ

```
Input: 15 slices × 5ch × 224 × 224
       (CT + 4領域mask)

┌─────────────────────┐
│  2D CNN (per slice)   │  ← ResNet-18/50 backbone
└──────────┬──────────┘
           │ slice features: 15 × D
           │
┌──────────┴──────────┐
│  LSTM / GRU           │  ← スライス間の時系列集約
└──────────┬──────────┘
           │ vertebra feature: D
           │
      ┌────┴────┐
      │  4-head FC  │  ← 各領域ごとに1出力
      └────┬────┘
           │
      4 region scores: [p_left, p_right, p_body, p_posterior]
```

### Bag-level prediction の導出

```
p_vertebra = 1 - Π(1 - p_region_i)   ← noisy-or pooling
or
p_vertebra = max(p_region_i)           ← max pooling
```

bag-level prediction は MIL 損失の計算に使用する。

---

## 実験計画

### Phase 1: 椎体レベル分類 baseline

```
入力: 15 × 1(or 5) × 224 × 224
出力: 椎体単位 fracture probability（1出力）
損失: BCE
目的: 椎体単位分類が成立するか確認
```

RSNA 1位解法に近い設計でまず baseline を確立する。

### Phase 2: 4領域出力 + Partially Supervised MIL

```
入力: 15 × 5 × 224 × 224（CT + 4領域mask）
出力: 4領域 fracture probability
損失: L_supervised + λ * L_MIL
データ:
  - negative vertebra → 4領域すべて non-fracture（教師あり）
  - positive + bbox → region-level label（教師あり）
  - positive + bbox無し → DMIL制約
目的: 領域別骨折検出の実現
```

この Phase が本研究の本丸。

### Phase 3: Pseudo-label による精緻化

```
Phase 2 の学習済みモデルで bbox無し症例の region-level を pseudo-label
→ 高確信度（p > threshold）のもので再学習
→ self-training / curriculum learning
目的: 全データの教師あり化による精度向上
```

---

## 評価方法

### Region-level 評価（bbox付きテストデータ）

```
各領域ごとに:
  AUC, Sensitivity, Specificity, F1
```

### Vertebra-level 評価（全テストデータ）

```
4領域スコアから椎体レベル予測を導出:
  p_vertebra = max(p_region_i)
AUC, Sensitivity, Specificity, F1
```

### RSNA baseline との比較

```
Phase 1（椎体レベルのみ）vs Phase 2（4領域）
→ 4領域情報が椎体レベル検出精度にも寄与するか確認
```

---

## 本計画と前処理計画との関係

前処理計画（RSNA設計に寄せた頸椎骨折検出前処理計画.md）の Step 4 に対応する。

```
前処理計画:
  Step 1: RSNA風baseline（CTのみ）
  Step 2: CT + vertebra mask
  Step 3: CT + 4領域mask
  Step 4: 領域別分類          ← 本計画

本計画:
  Phase 1 = 前処理計画 Step 1〜3 の分類部分
  Phase 2 = 前処理計画 Step 4 の具体化
  Phase 3 = 自己学習による拡張
```

---

## まとめ

本計画では、RSNAデータの「椎体レベルラベル（全症例）」と「bboxラベル（一部症例）」を組み合わせ、
Partially Supervised MIL により 4領域ごとの骨折検出を実現する。

```
全症例の椎体ラベル + 235例のbboxラベル
↓
Partially Supervised MIL
↓
4領域ごとの骨折確率
↓
VAIリスク評価に接続
```

少量の instance-level annotation が MIL の学習を大幅に安定させるため、
pure MIL や pure 教師ありのいずれよりも有利な条件で学習できる。
