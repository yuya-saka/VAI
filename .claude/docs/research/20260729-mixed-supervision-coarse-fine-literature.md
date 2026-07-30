# 粗ラベル大量 + 詳細ラベル少量で両方を出力する学習法（文献調査 v2）

Date: 2026-07-29
調査者: Claude (WebSearch/WebFetch による直接調査)

> **v1 からの訂正**: 初回調査は MIL/WSI 系（MS-CLAM 等）を中心に据えたが、
> ユーザー指摘のとおり**問題設定が違う**。WSI 系の「詳細ラベル」は
> ギガピクセル画像内の**空間的タイル注釈**であり、bag を空間分割する話。
> 本プロジェクトの4領域ラベルは **bag 全体に付く属性ラベル**で、
> 粗ラベルと**同じ粒度のまま細分化**されたもの。対応する文献系統は別。
> v2 では正しい系統（階層ラベル半教師 / 部分ラベル / 単一陽性多ラベル）に差し替える。

---

## 0. 問題の正しい定式化

```
モデル出力: 1 bag あたり 4領域スコア  q = (q_body, q_right, q_left, q_post)
粗ラベル出力: 学習された別ヘッドではなく、q から「導出」される
             p_coarse = 1 - Π_r (1 - q_r)      ← noisy-OR = 周辺化
制約:        y_coarse = OR(r_1..r_4)           （構造的に常に成立）
```

これは文献用語で **hierarchical label marginalization を伴う半教師あり細分類**。
「粗ラベル = 詳細ラベルの周辺（marginal）」という関係が全ての鍵。

### 実データの内訳（07-28 実測値より再集計）

| グループ | bag数 | 詳細ラベルの既知度 | 教師として使える情報 |
|---|---:|---|---|
| 陰性 bag | 12,100 | **完全既知**（y=0 ⟹ 4領域すべて0） | 4領域すべてに BCE(0) |
| 陽性・注釈済 | 268 | **完全既知** | 4領域すべてに BCE(0/1) |
| 陽性・注釈なし | **1,064** | **未知**（「少なくとも1領域が1」だけ） | 周辺 p_coarse に BCE(1) のみ |

**重要な再認識**: 弱ラベル扱いが必要なのは 1,064 bag = **全体の 7.9% だけ**。
残り 12,368 bag (92.1%) は**完全な4領域ラベルを持っている**。
v1 で「強教師 2.0%」と書いたのは陽性のみを見た誤った数え方だった。
陰性 bag は「4領域すべて0」という正真正銘の詳細教師である。

ただし**領域どうしを識別する信号**を持つのは 268 bag だけなので、
そこがボトルネックであることは変わらない。

### 268 bag から測れる領域事前分布（07-28 §1 より）

```
R1 body           78/268 = 29.1%
R2 right_foramen  59/268 = 22.0%
R3 left_foramen   72/268 = 26.9%
R4 posterior     157/268 = 58.6%
1 bag あたり陽性領域数の平均 = 366/268 = 1.37
```
（検算: 78+59+72+157 = 366 = 197×1 + 45×2 + 21×3 + 4×4 ✓）

---

## 1. 学習方法（これが本題）

### 1.1 基本の骨格 — 「未知の詳細ラベルは捏造せず、周辺だけを教師する」

bag ごとに、何が既知かで損失を切り替える。**同一のバッチに3種類を混ぜてよい**。

```python
# q: (B, 4) 領域スコア (sigmoid後)
# region_label: (B, 4)  値 0/1、未知は NaN
# y_coarse: (B,)  0/1  全 bag で既知

p_coarse = 1.0 - torch.prod(1.0 - q, dim=1)          # noisy-OR による周辺化

known = ~torch.isnan(region_label).any(dim=1)         # 12,368 bag が True

# (1) 詳細ラベルが既知な bag: 4領域すべてに直接 BCE
L_fine = bce(q[known], region_label[known])

# (2) 詳細ラベルが未知な bag: 周辺にだけ BCE。個別 q_r には直接教師しない
L_marginal = bce(p_coarse[~known], y_coarse[~known])

L = L_fine + lambda_m * L_marginal
```

未知 bag の勾配は **p_coarse を経由してのみ** q_r に届く。これが
「詳細ラベルと一緒に弱ラベル的に学習する」の実体。**個別領域の疑似ラベルを作らない**のが要点。

### 1.2 これが正しいと言える根拠 — Su & Maji (BMVC 2021)【詳細】

**"Semi-Supervised Learning with Taxonomic Labels"** https://arxiv.org/abs/2111.11595
Jong-Chyi Su, Subhransu Maji (UMass Amherst)

#### 設定（Semi-iNat）
生物分類学の7階層 Kingdom→Phylum→Class→Order→Family→Genus→Species。
各階層のクラス数: **Kingdom 3 / Phylum 8 / Class 29 / Order 123 / Family 339 / Genus 729 / Species 810**

| データ | 内容 |
|---|---|
| in-class labeled | 少数。**種（葉）レベル**のラベル付き |
| in-class coarse | 上の **9倍**。粗レベル（門など）のラベルのみ |
| out-of-class coarse | **32倍**。粗ラベルのみ、かつ**未知種**を含む |

テストは種レベルの分類精度。

#### 損失（式1）
```
L_hie^{7,2} = Σ_i H(y_i^7, p_i^7)  +  Σ_j H(y_j^2, q_j^2)
              ~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~
              細ラベルあり            粗ラベルのみ
```
`H` = cross entropy、上付き 7 = Species、2 = Phylum。粗レベルの予測は**学習しない**。
0/1 のエッジ行列 `W_7^2` による線形写像で**周辺化して作る**:
```
q_j^2 = q_j^7 · W_7^2        （各 Phylum 配下の leaf 確率を足し上げる）
```

**重要な設計判断**: 「細ラベルがあるサンプルには**最下層にのみ**教師損失を掛ける」
（原文: *For labeled data, we only add supervised loss on the lowest level*）。
粗レベルにも重ねて掛けることはしない＝**二重計上しない**。

#### leaf parameterization の優位性（実測）

| 方式 | パラメータ | 精度 (ImageNet事前学習) |
|---|---|---|
| **leaf-param（葉のみ予測 → 周辺化）** | 810 | **46.6%** |
| 階層ごとに別ヘッド（multi-head） | — | **42.1%** |
| edge-param（YOLOv2式、各節点で条件付き分布） | 2041 | （不採用） |

**別ヘッド方式は 4.5pt 劣る。** 粗ヘッドを別に学習させてはいけない、という直接の実測根拠。

#### 半教師あり手法との統合（式2-4）
いずれも `L_hie` を**そのまま足すだけ**。

- **Pseudo-Label**: `L = L_hie^{7,2} + Σ_j 1[max(r_i) ≥ τ] H(q̂_i^7, q_i^7)`
- **FixMatch**: `L = L_hie^{7,2} + Σ_j 1[max(q_i^7) ≥ τ] H(q̂_i^7, Q_i^7)`
  - 粗ラベルへの教師損失には**強拡張画像 Q_j** を使う（弱拡張は疑似ラベル生成専用で
    逆伝播しないため）
- **Self-Training**: teacher を labeled のみで学習 → student を
  `L = L_hie^{7,2} + Σ_i H(σ(z_i^t/T), σ(z_i^s/T))` で蒸留

#### 定量結果

**Table 2（in-domain のみ、Phylum レベル教師）** 左=なし → 右=階層損失あり

| 手法 | from scratch | from ImageNet |
|---|---|---|
| Supervised Baseline | 18.5 → **21.7** | 40.4 → **46.6** (+6.2) |
| FixMatch | 15.5 → **25.7** (+10.2) | 44.1 → **47.9** (+3.8) |
| MoCo + Self-Training | 32.0 → **35.4** | 42.6 → **45.8** |

**Table 3（out-of-domain データを混ぜた場合）**

| 手法 | from scratch | from ImageNet |
|---|---|---|
| FixMatch | 11.0 → 21.1 | 38.5 → **41.1** |
| Supervised Baseline + hierarchy | 18.5 → 20.5 | 40.4 → **45.6** |

**階層レベルのアブレーション**: 粗ラベルが**細かいほど利得が大きい**。
Class レベル（29クラス）だと FixMatch **44.1% → 51.8%（+7.7pt）**。
Phylum（8クラス）の +3.8pt より倍近く効く。

#### 負の結果と限界
- **OOD 混入で崩れる**: FixMatch は in-domain のみなら 47.9% だが、
  未知種を含む粗ラベルデータを足すと **41.1%（-6.8pt）**
- さらに強い記述: OOD がある状況で ImageNet 事前学習を使うと、
  **どの半教師あり手法も supervised baseline を上回らなかった**
- 不確実性フィルタ（`max(pred) ≥ 0.8` **かつ** 予測粗ラベルが付与粗ラベルと一致）を
  掛けても **41.1% → 42.0%** しか戻らない。著者自身「単純なヒューリスティックであり
  改善の余地が大きいが、fine-grained 領域では本質的に難しい」と認めている

#### 学習設定
ResNet-50 / 224×224 / SGD (momentum 0.9) / LR グリッド [0.001, 0.03] /
weight decay 1e-3 or 1e-4 / 100k iter (scratch), 50k iter (fine-tune) /
**batch 60 = labeled 30 + coarse 30**。FixMatch は batch = labeled 32 + coarse 160、
τ=0.8、RandAugment。

**注目**: 損失に重み係数 λ は使っていない（等重み）。
バランスは**バッチ組成（30:30）で取っている**。§3.4 の「等重みは細分類を壊す」との
矛盾はここで解消する — **λ ではなくサンプリング比で制御するのが正解**。

### 1.3 未知ラベルの扱いの選択肢 — Cole et al. (CVPR 2021)

**"Multi-Label Learning from Single Positive Labels"** https://arxiv.org/abs/2106.09708

多ラベルで一部の正例しか観測されない設定。**「観測されなかったラベルをどう扱うか」の
選択肢が整理されている**ので、そのまま本件の 1,064 bag に転用できる。

| 手法 | 内容 | 本件への適用 |
|---|---|---|
| **AN** (Assume Negative) | 未観測を全部0とみなす | **やってはいけない**。1,064 bag は必ず1領域以上が1 |
| **WAN** (Weak AN) | 未観測の負例項を γ=1/(L-1) で減衰 | 保守的な代替 |
| **AN-LS** | AN + label smoothing (ε=0.1) | **論文の結論: 単純だが強いベースライン**。fine-tune 設定でしばしば複雑手法に勝つ |
| **EPR** | 1画像あたりの期待陽性数を k に制約する正則化 | 本件では **k = 1.37**（§0で実測済み）を使える |
| **ROLE** | ラベル推定器を同時学習（stop-gradient付き） | 最高性能だが**再構成ラベルへの過学習**が起きると論文自身が指摘 |

**結論として論文が推すのは AN-LS の単純さと ROLE の性能のトレードオフ。**
本件は「少なくとも1つは陽性」という追加情報があるぶん SPML より条件が良いので、
まず §1.1 の周辺化 + EPR から始めるのが妥当。

---

## 2. 【最重要】この学習法が壊れる仕方と、その対策

### 2.1 壊れ方 — Giunchiglia & Lukasiewicz (NeurIPS 2020)

**"Coherent Hierarchical Multi-Label Classification Networks"** https://arxiv.org/abs/2010.10151
（この論文は WSI と無関係。genomics 16 / 医用画像 2 / テキスト 1 の20ベンチマーク）

親を子の **max** で構成すると、標準 BCE の勾配が病理を起こす。論文の例:
`h_A=0.3, h_B=0.1`、真値 `y_A=0, y_B=1`（親は陽性、この子は陰性）のとき

```
∂L/∂h_A ≈ -1.9   ← 陰性であるべき子 A を「上げろ」という勾配
∂L/∂h_B  =  0     ← 親 B には勾配が流れない
```

**間違った子を上げることで親の陽性を満たす** bad local optimum に落ちる。
親が n 個あると `h_A > n/(n+1)` を超えるまで修正が効かない（n=10 で 0.91）。

修正案 MCLoss は `max_{B∈D_A}(y_B · h_B)` として**真に陽性な子だけを max 候補にする**。
合成実験で AU(PRC) 0.938±0.038 → **0.974±0.007**。実データ 17/20 で最良。

### 2.2 本件での現れ方（集約方法によって深刻度が違う）

| 集約 | 未知 bag での勾配の行き先 | 危険度 |
|---|---|---|
| `p = max_r q_r` | **argmax の1領域にのみ流れる** | **最悪**。C-HMCNN の病理に直撃 |
| `p = 1 - Π(1-q_r)` | 全領域に流れる。q_r への係数 ∝ `Π_{s≠r}(1-q_s)` | まし。だが偏りは残る |

noisy-OR でも、モデルが既に検出しやすい領域（面積最大の posterior）を上げるのが
最も安い経路なので、**1,064 bag の陽性がすべて R4 に押し付けられる**。

**07-28 §6.5 で実測した「R4 のみ陽性 41%」「hit@1 = 59%」という prior shortcut の
理論的な発生源がこれ。** MCLoss は y_r が既知である前提なので、
1,064 bag（y_r が全部未知）にはそのまま適用できない。

### 2.3 対策 — 事前分布マッチング

MCLoss が使えない以上、**「どの領域を上げるか」を分布レベルで制約する**のが代替。
半教師あり学習の distribution alignment 系が直接使える。

- **DARP** (NeurIPS 2020) https://arxiv.org/abs/2007.08844
  疑似ラベルを、望ましいクラス分布との整合を制約に置いた上で
  元の疑似ラベルとの KL を最小化する形に「精製」する
- **DebiasPL** (CVPR 2022) — 真の周辺分布を事前知識として必要としない
  （momentum 更新した予測で debias + adaptive margin loss）
- **EPR** (Cole et al., §1.3) — 期待陽性数を制約

**本件での具体形**: 1,064 bag 上での予測平均 `mean(q_r)` を、
268 bag から測った事前分布 **(0.291, 0.220, 0.269, 0.586)** に合わせる正則化を足す。

> **注意**: EPR（陽性数を 1.37 に制約）**だけでは不十分**。
> 「posterior だけを上げる」は陽性数 1.0 で EPR をほぼ満たしてしまう。
> EPR は**個数**を縛るが**どの領域か**は縛らない。分布マッチングが必須。

> **前提の確認が必要**: 事前分布マッチングは「268 が 1,064 の代表標本である」ことに
> 依存する。アノテーション順が骨折の種類と相関していないか要確認。
> ここが崩れると §1.2 の out-of-domain 悪化（-6.8pt）と同じ失敗になる。

---

## 3. その他の関連研究（非WSI）

### 3.1 HierMatch (2021) https://arxiv.org/abs/2111.00164
半教師あり学習にラベル階層を組み込む。粗レベルの疑似ラベルのほうが
信頼度が高いことを利用して細レベルの学習を安定させる。

### 3.2 Li et al., CVPR 2018「Thoracic Disease Identification and Localization with Limited Supervision」
https://arxiv.org/abs/1711.06373

**WSI ではなく胸部X線**。詳細ラベルは空間 bbox なので本件とは詳細ラベルの型が違うが、
**noisy-OR と完全分解の使い分け**は本件にそのまま移植できる。

- 注釈なし画像: `p(y|x) = 1 - Π_j (1 - p_ij)`（noisy-OR、「少なくとも1つ」）
- 注釈あり画像: `p(y|x,bbox) = Π_{j∈N} p_ij · Π_{j∉N} (1 - p_ij)`（積、陰性も明示教師）
- **λ_bbox = 5** で強教師サンプルを重み付け
- アブレーション: 704枚の注釈 + 22,248枚の弱ラベル が 88,892枚の弱ラベルのみに勝つ疾患あり
- 局在は緩い閾値では差が小さく、**T(IoU)=0.6 で 16%→73% と差が爆発**する

### 3.3 Shi et al., MedIA 2021「Marginal loss and exclusion loss」https://arxiv.org/abs/2007.03868
部分ラベル多臓器セグメンテーション。**ラベルされていない臓器は「背景」に merge されており、
その確率は周辺確率だから、それをそのまま CE/Dice に代入すればよい**という定式化。
§1.1 と同じ結論に独立に到達している。exclusion loss（クラス排他）も本件の
4領域が空間的に排他であることに対応するが、椎間孔は面の21-24%で存在しない（07-28 §5）ため
被覆性は成立せず、region_valid の扱いが必要。

### 3.4 「Understanding the Impact of Label Granularity on CNN-based Image Classification」https://arxiv.org/abs/1901.07012
粗/細を等重みで同時最適化すると**細分類が悪化し粗分類だけが良くなる**という報告。
→ **λ_m を 1.0 で始めてはいけない。** 粗側の勾配が支配的にならない重みを探す必要がある。

---

## 4. 実装への推奨（07-28 の実験計画への差分）

07-28 で決めた実験順序（§0-5）は変更不要。以下はその中の**損失設計の確定**。

1. **粗ヘッドは2本持つ**
   - 導出粗ヘッド `p_coarse = 1-Π(1-q_r)`: 1,064 bag の弱教師を受ける経路
   - 独立粗ヘッド（Stage1流 global）: **非劣性 guardrail 専用**。
     §2 の病理を guardrail に持ち込まないため（Codex 相談2 の主張と一致）

2. **陰性 12,100 bag を4領域の詳細陰性教師として明示的に使う**
   これは文献にない本件の強み。ただし陰性が圧倒的多数なので
   領域ヘッドが全0に潰れないようクラス重み（`w_r = clip(n_r0/n_r1, 1, 4)`）が要る

3. **未知 bag には周辺のみ、集約は max ではなく noisy-OR**（§2.2）

4. **事前分布マッチング正則化を入れる**（§2.3）。EPR 単独では不可

5. **λ_m は小さめから探索**（§3.4）。等重みは細分類を壊す

6. **強教師 268 bag は over-sampling**（Li et al. の λ_bbox=5 相当。
   Codex 相談1 の `batch 16 = strong 4 / weak 4 / neg 8` はこれと同じ役割）

### 検証すべき前提（着手前）
- 268 が 1,064 の代表標本か（アノテーション順のバイアス確認）→ §2.3, §1.2 が依存
- z方向（15面）の未知性と領域方向の未知性は**直交する別問題**。両方に
  「未知には教師を与えない」を適用する必要がある

---

## Sources

- [Semi-Supervised Learning with Taxonomic Labels (BMVC 2021)](https://arxiv.org/abs/2111.11595)
- [Multi-Label Learning from Single Positive Labels (CVPR 2021)](https://arxiv.org/abs/2106.09708)
- [Coherent Hierarchical Multi-Label Classification Networks (NeurIPS 2020)](https://arxiv.org/abs/2010.10151)
- [Multi-Label Classification Neural Networks with Hard Logical Constraints (JAIR 2021)](https://arxiv.org/abs/2103.13427)
- [Distribution Aligning Refinery of Pseudo-label (DARP, NeurIPS 2020)](https://arxiv.org/abs/2007.08844)
- [Debiased Learning from Naturally Imbalanced Pseudo-Labels (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.pdf)
- [HierMatch: Leveraging Label Hierarchies for Improving Semi-Supervised Learning (2021)](https://arxiv.org/abs/2111.00164)
- [Marginal loss and exclusion loss for partially supervised multi-organ segmentation (MedIA 2021)](https://arxiv.org/abs/2007.03868)
- [Thoracic Disease Identification and Localization with Limited Supervision (CVPR 2018)](https://arxiv.org/abs/1711.06373)
- [Understanding the Impact of Label Granularity on CNN-based Image Classification (2019)](https://arxiv.org/abs/1901.07012)
- [Acknowledging the Unknown for Multi-label Learning with Single Positive Labels (2022)](https://arxiv.org/abs/2203.16219)
- [Fine-grained Angular Contrastive Learning with Coarse Labels (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Bukchin_Fine-Grained_Angular_Contrastive_Learning_With_Coarse_Labels_CVPR_2021_paper.pdf)
