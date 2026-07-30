# Stage4 実装設計 — mixed supervision（PI仕様準拠）

作成日: 2026-07-29
状態: **次セッションで実装する前提の設計**
上位文書: `.claude/docs/stage4-design-proposal.md`（評価プロトコルの詳細）
関連: `.claude/docs/codex/20260729-stage4-evaluation-protocol.md`

---

## 0. この仕様で外れた制約（先に確認）

私が本セッション前半で立てた制約のうち **2つがこの仕様では不要になった**。設計が単純になる。

| 前半で置いた制約 | 本仕様での扱い | 理由 |
|---|---|---|
| 既存の分割（seed 42）を凍結する | **不要。作り直す** | 凍結の根拠は「Stage1/2/3 の fold-k checkpoint を warm-start に使うから」だった。本仕様は**全arm をゼロから学習**し、Pretrain-to-Mixed でさえ fold ごとに事前学習し直す。既存 checkpoint に依存しない |
| locked test（20% holdout）を置く | **置かない。素の5-fold** | 仕様が明示。結果として OOF 評価対象は 216 → **268 bag 全部**に増える |

分割を作り直せるようになったことで、**仕様が要求する層別（アノテ有無・領域別陽性数・C1–C7分布）が実現可能**になった。
現行分割では R2 が fold によって 5〜15 とばらついていたが、これを 11〜12 に収められる（§1）。

---

## 1. Fold 設計（実測済み・確定）

### 1.1 アルゴリズム

患者（study）単位。1患者の C1–C7 は必ず同一 fold。2段階で組む。

**第1段: アノテーション済み 160 study を層別割り当て**

study ごとに12次元の特徴ベクトルを作る。

```
v = [アノテ済bag数, R1陽性数, R2陽性数, R3陽性数, R4陽性数, C1数, ..., C7数]
```

目的関数は fold 合計と理想値（全体/5）の重み付き絶対偏差。重みは
**アノテ数 3.0 / 領域数 2.0 / レベル数 1.0**（優先順位は §1.3）。

`seed=42` の乱数割り当てを 40,000 回試して最良を取り、その後
**改善が止まるまで2-study交換の局所探索**をかける。決定的で再現可能。

**第2段: 残り 1,849 study**

陽性 bag 数を重み40、総 bag 数を重み1として、貪欲に最も不足している fold へ入れる
（陽性数の多い study から順に処理）。

### 1.2 得られた分割（実測）

`data/rsna_data/stage4_folds.csv` に保存済み（`study_id, fold` の2列、2,009行）。

| fold | study | bag | 陽性 | 陰性 | アノテ済 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 404 | 2,703 | 266 | 2,437 | 53 | 15 | 12 | 14 | 32 |
| 1 | 399 | 2,663 | 267 | 2,396 | 54 | 16 | 11 | 14 | 31 |
| 2 | 399 | 2,663 | 267 | 2,396 | 53 | 16 | 12 | 15 | 32 |
| 3 | 403 | 2,701 | 266 | 2,435 | 54 | 15 | 12 | 14 | 32 |
| 4 | 404 | 2,702 | 266 | 2,436 | 54 | 16 | 12 | 15 | 31 |
| **計** | **2,009** | **13,432** | **1,332** | **12,100** | **268** | 78 | 59 | 72 | 158 |

レベル分布（アノテ済 bag）も fold あたり C1:4-5 / C2:7-8 / C3:6-7 / C4:8-9 / C5:9-10 / C6:10-11 / C7:7-8 に収まった。

**現行分割との比較（領域別の fold 間レンジ）**

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| 現行（seed 42）| 8–24 | **5–15** | 9–13 | 18–32 |
| 提案 | 15–16 | **11–12** | 14–15 | 31–32 |

### 1.3 hard constraint の充足を検証済み

Codex が挙げた必須制約に対する実測。

| 制約 | 目標 | 実測 | 判定 |
|---|---|---|---|
| 各患者ちょうど1 fold | — | — | ✓ |
| fold あたりアノテ患者数 | 32 | 32, 32, 32, 32, 32 | ✓ |
| fold あたりアノテ bag 数 | 54,54,54,53,53 | 53,54,53,54,54 | ✓（同じ多重集合）|
| **各 fold・各領域で陽性 ≥8 かつ陰性 ≥8** | — | 最小は R2 陽性 11 / R4 陰性 21 | **✓ 全充足** |
| 領域 target | R1 15-16 / R2 11-12 / R3 14-15 / R4 31-32 | 完全一致 | ✓ |
| fold あたり患者数 | 402,402,402,402,401 | 404,399,399,403,404 | △ 最大3ずれ |

患者数のずれは、bag 数と陽性 bag 数の均衡を優先した結果。実害は無いが、
気になるなら第2段の重みを調整すれば詰められる。

### 1.4 層別アルゴリズムについての注意（Codex 指摘）

Codex は「ランダム試行から最も綺麗な fold を選ぶ」方式ではなく、**CP-SAT / MILP で一度だけ決める**ことを勧めている。
§1.1 の実装はランダム試行＋局所探索なので、厳密には heuristic である。

ただし Codex の懸念の核心は **「モデル性能を見て split を選ぶな」** であり、
本実装の目的関数は**ラベル分布の均衡だけ**でモデル出力を一切見ていない。
さらに上表のとおり hard constraint と領域 target を全て満たしている。
したがって**このまま使える**が、次の2点は必ず守る。

- `seed=42` 固定・決定的。**split manifest を hash して凍結**する
- **一度決めた split を後から作り直さない**。特に結果を見た後の作り直しは禁止

**lexicographic な優先順位**（衝突したとき、上から順に slack を最小化）

1. 患者分離と各クラスの最低 support
2. アノテ患者数 / アノテ bag 数
3. **R2 陽性数**（最も少ないので最優先）
4. R1・R3 陽性数
5. R4 陽性数
6. 骨折陽性患者数
7. 骨折陽性 bag 数
8. アノテ済みの C1–C7 分布
9. 骨折陽性の C1–C7 分布
10. 全 bag の C1–C7 分布と総 bag 数

**妥当性への寄与度**（Codex）

| 基準 | 重要度 |
|---|---|
| 患者分離 | **必須**。破ると推定量自体が無効 |
| アノテ／領域別の均衡 | 主に分散・fold calibration・指標の定義可能性。多少ずれても pooled OOF が自動的にバイアスするわけではない |
| 骨折有無の均衡 | 椎体指標と学習時 prevalence の安定性に重要 |
| アノテ済みレベル分布 | レベル別解析の精度に有用（§8.1 のとおり本データでは無視できない）|
| 全 bag の C1–C7 分布 | ほぼ全患者に7 level あるので**大部分は cosmetic** |

### 1.5 学習側のサイズ（fold あたり）

| | bag | 陽性 | アノテ済 | 弱陽性 | 陰性 |
|---|---|---|---|---|---|
| fold0 | 10,729 | 1,066 | 215 | 851 | 9,663 |
| fold1 | 10,769 | 1,065 | 214 | 851 | 9,704 |
| fold2 | 10,769 | 1,065 | 215 | 850 | 9,704 |
| fold3 | 10,731 | 1,066 | 214 | 852 | 9,665 |
| fold4 | 10,730 | 1,066 | 214 | 852 | 9,664 |

**領域教師が付く bag は fold あたり 214〜215。** これが本研究の実質的なサンプルサイズ。

---

## 2. データ層（最初に実装するもの）

現状 `fracture_region_labels_dicom.csv` は `train_models/` の**どこからも読まれていない**（確認済み）。

### 2.1 item に足すフィールド

```python
{
    ...,                          # 既存 (study_uid, vertebra, label, *_path)
    "region_label":     [0,0,0,1] # 4次元 int。未アノテ時は [0,0,0,0]（使わない）
    "region_supervision": "strong" | "weak" | "negative",
}
```

| 種別 | 条件 | bag 数 | `L_region` での扱い |
|---|---|---|---|
| `strong` | 人手アノテあり | 268 | 4次元ラベルで BCE |
| `weak` | 陽性だがアノテ無し | 1,064 | **領域教師を与えない**。全0とみなす処理は禁止 |
| `negative` | 椎体陰性 | 12,100 | サンプリングされた分だけ全4領域0で BCE |

CSV は `(study_id, level)` に複数 run があるので **OR 集約**して読む（run はツールが bbox の連続塊を
提示単位に切っただけで、ラベルの意味は椎体単位。詳細は上位文書 §1.3）。

### 2.2 【必須】水平flip時の R2↔R3 入れ替え

`train_models/stage2/src/dataset.py` の `_augment_volume` は現在
`remap_regions_after_horizontal_flip` で**マスクだけ**を入れ替えている。

**flip 1回で同時に行うもの（Codex 指定、すべて必須）**

| # | 対象 |
|---|---|
| 1 | 画像6チャンネル全部を水平反転 |
| 2 | vertebra mask / region mask を水平反転 |
| 3 | **mask ID `2 ↔ 3`**（既存の `remap_regions_after_horizontal_flip`）|
| 4 | **target `y[R2] ↔ y[R3]`**（新規・未実装）|
| 5 | **region-valid フラグ `R2 ↔ R3`**（新規・見落としやすい）|

R1 と R4 は swap しない。vertical flip と transpose は領域IDの意味を変えないので対処不要。

**flip は確率 0.50 のまま維持する。** 正しく swap すれば left/right 識別は壊れない
（label-preserving ではなく **label-equivariant** な augmentation になる）。
R2/R3 の症例数が 59/72 と少ないので 0.25 に弱めるより 0.50 が適切で、
左右 prevalence のショートカットを弱める効果もある。

**主推論では flip / TTA を使わない。** TTA をやるなら secondary とし、
反転側の予測を戻すときに `q_R2 ↔ q_R3` してから平均する。

**必須の unit test 3つ**

1. **double flip** で画像・マスク・ラベルが完全復元される
2. **R2 のみ陽性の合成サンプルが flip 後に R3 のみ陽性**になる
3. 学習サンプル100件で **mask ID と region label の対応不一致が 0 件**。
   1件でもあれば**学習を停止**する

**これを忘れると例外も出ず、静かに全実験が壊れる。**
教師とマスクで左右が逆になり、R2/R3 に関する主張だけが反転する。

### 2.3 検証済み: 教師を付ける先が無い bag は無い

268 bag すべてについて `region_4class.npy` を全数読み、4領域とも**1面以上マスクが存在する**ことを確認した。

| 領域 | mask 有効な bag | 面数 中央値 | 最小 |
|---|---|---|---|
| R1 body | 268/268 | 14 | 9 |
| R2 right | 268/268 | 12 | 7 |
| R3 left | 268/268 | 12 | **1** |
| R4 post | 268/268 | 14 | 8 |

人手が陽性と付けたのにマスクが全15面で空、というケースは**4領域とも0件**。
`region_valid` が False になって教師が捨てられる事故は起きない。
ただし R3 は最小1面の bag があるので、**その bag の R3 予測は1面の特徴だけに依存する**。

---

## 3. モデル（Stage3 から変更しない）

仕様の階層構造は **Stage3 の実装そのもの**なので、アーキテクチャ変更は不要。

```
Stage3Output.instance_evidence_logits  p[z,r]  [B,15,4]   ← 教師を与えない
Stage3Output.region_evidence_logits    q[r]    [B,4]      ← ここに人手ラベルを付ける
Stage3Output.vertebra_logit                    [B]        ← q[r] を smooth-max 集約したもの
```

- 面方向集約: `tied_attention_pool`（既存）
- 領域方向集約: `normalized_smoothmax`（既存）→ これが椎体ロジットを作る
- 4領域は排他でないので **softmax ではなく独立 sigmoid + BCE**（既存の出力形式のまま）
- 人手ラベルは **面集約後の `q[r]` にのみ**適用。`p[z,r]` には与えない（仕様どおり）

**つまり Stage4 の実装差分は「データ層 + 損失1項 + サンプラ」だけ。** モデルコードは触らない。

---

## 4. 損失

### 4.1 全体形

```
L_total = L_vertebra + λ_region · L_region + λ_neg · L_negative
```

`L_vertebra` と `L_negative` は **既に `stage3_loss` に実装済み**（`train_models/stage3/utils/losses.py:118`）。

| 項 | 実装状況 | 対象 | 内容 |
|---|---|---|---|
| `L_vertebra` | **既存** | 全 10,730 bag | `weighted_bce(vertebra_logits, targets, positive_weight=2.0)` |
| `L_negative` | **既存** | 陰性 bag 全部 | `p[z,r]` を全部0へ。有効面で平均 → bag 平均。`lambda_neg=0.1` |
| `L_region` | **新規** | strong 214 + サンプルした negative | `q[r]` に対する4領域 multi-label BCE |

### 4.2 `L_region` の定義

```python
# mask: この bag に領域教師を与えるか (strong=True, サンプル済negative=True, weak=False)
per_region = F.binary_cross_entropy_with_logits(
    q[mask], region_target[mask], pos_weight=w, reduction="none"
)
L_region = per_region.mean(dim=1).mean()   # 4領域で平均 → 教師付きbagで平均
```

- `region_target` は strong なら人手の4次元、negative なら `[0,0,0,0]`
- **weak（陽性・アノテ無し）は mask=False。全0を与えない**（仕様の明示要求）

**クラス重みは「陰性を足した後」に計算する**（重要）。
`L_region` が実際に見る標本分布は strong 214 + 陰性 214 なので、strong だけで重みを作ると補正にならない。

```
w_r = min( (2·N_A − P_r) / P_r , 8.0 )        N_A = strong数, P_r = 領域r の陽性数
```

268 bag 比率での planning 値（**実際は fold の学習側 count から再計算する**）:

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| 陽性 | 78 | 59 | 72 | 158 |
| **pos_weight** | **5.87** | **8.00**（raw 8.08 を cap）| **6.44** | **2.39** |

陰性側の重みは 1.0。
**陰性を足す前に計算した `[2.44, 3.54, 2.72, 0.70]` は使わない**（実際の標本分布を補正できない）。

### 4.3 【要注意】`L_negative` と `L_region` の陰性教師が二重にかかる

仕様は「陰性が多すぎると4領域予測が全部陰性へ倒れる」ことを警戒して `L_region` の陰性をサンプリングする。
しかし **既存の `L_negative` は 9,665 陰性 bag すべてに対して `p[z,r]` を0へ押している**。
`q[r]` は `p[z,r]` の集約なので、これは実質的に**全陰性が領域出力を抑制している**状態である。

つまり **サンプリングで制御できるのは新しい項だけで、古い項は制御されていない**。

**結論（Codex）**: 本実装の `L_negative` は **plane-level（`p[z,r]`）** なので `L_region`（`q[r]`）とは
別の量であり、二重計上ではない。ただし密なので **`lambda_neg` を 0.1 → 0.05 に下げる**。
`L_region` 内で既に q-level の陰性 BCE を使うため、plane-level の罰則は領域損失の **1/20 程度**に留める。

> もし `L_negative` を q-level に変える実装にした場合は `L_region` と完全に二重計上になるので、
> そのときの妥当な値は **`lambda_neg = 0`**。

**必ず記録する診断**: 学習中の `L_vertebra` / `λ_region·L_region` / `λ_neg·L_negative` の
**実効値と勾配ノルムの比**（§8.3 の閾値表）。

### 4.4 λ の具体値（Codex 推奨）

```
λ_region(e) = 0.25 + 0.75 · min(e/4, 1)     # e は 0-indexed epoch
λ_neg       = 0.05                          # step 1 から一定
```

| epoch | 1 | 2 | 3 | 4 | 5以降 |
|---|---|---|---|---|---|
| λ_region | 0.250 | 0.438 | 0.625 | 0.813 | **1.000** |

- **標本数比 10,700 : 216 を λ に掛けてはいけない。** 各損失を平均で正規化している以上、
  標本数は推定分散と露出頻度に効くが、平均損失の係数を50倍する根拠にはならない
- **これは「最初から同時学習」に違反しない**。step 1 の時点で λ_region = 0.25 > 0 だから。
  λ_region = 0 の epoch を置く warmup は仕様と矛盾するので**やらない**

---

## 5. サンプリング

### 5.1 領域損失用の陰性サンプリング（Codex 推奨で確定）

仕様の「約1対1または1対2」について、**1:1 を採用**する。
strong bag の中にも既に多数の領域陰性ターゲットが含まれている（例: R2 陽性59に対し陰性209）ので、
2:1 は過剰。fold あたり strong 214 → **毎エポック 214 bag の陰性**。

実装規則（すべて必須）:

| | 内容 |
|---|---|
| 再抽出 | **毎エポック、学習側 patient からのみ**。復元抽出しない |
| レベル整合 | **C1–C7 構成をアノテ済み集合と完全一致**させる（アノテ済 C3 が37なら陰性 C3 も37）|
| 患者分散 | 可能な限り **1 patient から 1 bag/epoch**。足りないレベルだけ、全 patient 一巡後に2 bag目を許可 |
| 乱数 | `seed = 42 + epoch` に固定し、**選択 manifest を保存**する |
| 禁止 | **run 全体で固定した214 bag だけを使うこと**（subset 固有のショートカットと過学習を起こす）|

レベル完全一致が要る理由は §8.1 のとおり、レベルが領域ラベルの強い予測子だから。
陰性のレベル分布が偏ると、領域出力にレベルバイアスがそのまま乗る。

### 5.2 バッチ構成

層別サンプラで各バッチに strong / weak / negative を一定比で入れる。
Stage3 の `L_vertebra` は層別で歪むので、**bag loss は事前確率で補正する**
（上位文書 §3.3、Codex 07-28 の指摘）。

---

## 6. 実験4本

| arm | 学習データ | 位置づけ |
|---|---|---|
| **Weak-only** | 全 10,730 bag、椎体ラベルのみ（**`λ_region = 0` にするだけ**）| **中心対照** |
| **Mixed-from-scratch** | 全 10,730 bag、最初から両損失 | **主提案手法** |
| **Detail-only** | strong 214 + 陰性 214（fold あたり計 428）| 詳細ラベルだけの上限 |
| Pretrain-to-Mixed | fold ごとに Weak-only 事前学習 → mixed で fine-tune | 事前学習の効果 |
| *level-only*（追加）| CT不使用、レベルの陽性率のみ | **ショートカットの床**（§8.1）|
| *Weak-only-size-matched*（追加）| 214 アノテ bag を弱ラベルとして + 214 陰性 | 標本数を揃えた統制（§8.4）|

**中心主張は Weak-only vs Mixed の1本**。この2 arm は `λ_region` の有無だけが違う構成にする
（§8.2 の条件4-5）。他の対比はすべて secondary。

追加した2 arm は仕様に無いが、無いと結果が解釈できない。
level-only は CPU 数分、Weak-only-size-matched は Detail-only と同コスト。

実装順序は **Weak-only と Mixed-from-scratch を先に**（中心主張がここで決まる）、
level-only は最初に（安いので）、Detail-only と Pretrain-to-Mixed を後から。

---

## 7. 評価

### 7.1 椎体性能（検証 fold の全 bag が対象）

AUROC / AUPRC / F1 / 感度 / 特異度 / C1–C7 別。fold あたり検証 2,663〜2,703 bag（陽性 266〜267）。
**アノテーションが無い bag も椎体評価には含める**（仕様どおり）。

### 7.2 領域性能（検証 fold のアノテ済み bag のみ）

- **主指標: 各領域の AUPRC と macro AUPRC**
- 補助: 各領域 AUROC / macro F1 / 感度 / 特異度 / micro AUPRC
- **pooled OOF**: 5 fold の検証予測を連結して **268 bag に対して一度だけ**計算する。
  fold 別 AP の単純平均は補助として併記（fold あたり 53〜54 bag では単独で意味を持たない）
- **信頼区間は患者単位 bootstrap**。同一患者の複数椎体を独立標本として再標本化しない
  （268 bag / 160 患者 = 1.68）

### 7.3 主仮説の判定（事前登録する形）

```
主仮説:  Δ = macro-AP(Mixed) − macro-AP(Weak-only)   ← pooled OOF 268 bag
優越判定: 患者単位 paired bootstrap 10,000回の 95% CI 下限 > 0
seed:    [42, 43, 44, 45, 46] の5個。bag ごとに5確率を平均してから AP を1回計算
```

**seed を5倍の観測として扱わない。CI を `1/√5` に縮めない。**
bootstrap は **fold 内で患者を復元抽出**し、その患者の全椎体を一緒に複製する。

### 7.4 床の一覧（実測、すべて macro-AP）

| 床 | 値 | 意味 |
|---|---|---|
| prevalence（no-skill）| **0.342** | 何も学習していない |
| **level-only** | **0.458** | レベルの事前分布だけ。**実質的な合格ライン** |
| prevalence（C2除く n=231）| 0.320 | |
| **level-only（C2除く n=231）**| **0.345** | C2 を外すとレベルの上乗せはほぼ消える |

領域別 prevalence: R1 0.291 / R2 0.220 / R3 0.269 / R4 0.590。

---

## 8. リスク（仕様に無いが実測で見つかったもの）

### 8.1 【最重要】レベルだけで macro-AP 0.458 出る

CT画像を一切使わず、**椎体レベル（C1–C7）だけ**から領域陽性率を予測した pooled-OOF AP を実測した。

| | R1 | R2 | R3 | R4 | macro |
|---|---|---|---|---|---|
| prevalence（no-skill）| 0.291 | 0.220 | 0.269 | 0.590 | 0.342 |
| **level-only** | 0.473 | 0.269 | 0.408 | 0.682 | **0.458** |

レベル別の陽性率がまったく違うのが原因。

| lv | n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| C1 | 22 | 40.9% | 13.6% | 27.3% | 50.0% |
| **C2** | 37 | **73.0%** | 45.9% | 54.1% | **18.9%** |
| C3 | 31 | 19.4% | 19.4% | 9.7% | 74.2% |
| C4 | 43 | 14.0% | 16.3% | 14.0% | 79.1% |
| C5 | 46 | 28.3% | 17.4% | 17.4% | 69.6% |
| C6 | 53 | 22.6% | 18.9% | 26.4% | 62.3% |
| C7 | 36 | 13.9% | 22.2% | 41.7% | 50.0% |

**椎体レベルは画像から自明に分かる**（Stage1 が既に識別している）。
つまりモデルはこの 0.458 を**タダで手に入れる**。

→ **「level-only」を必須の対照 arm に加える**（CPU のみ、数分で回る）。
**実質的な合格ラインは 0.342 ではなく 0.458。** これを超えない限り
「4領域を画像から判別できた」とは言えない。仕様の実験計画にこの対照は含まれていないが、
入れないと Mixed の macro-AP が 0.50 でも意味が判定できない。

### 8.1b このショートカットは**ほぼ全部 C2 が作っている**（実測）

C2 を除いて同じ計算をすると、レベル情報の上乗せがほぼ消える。

| 母集団 | level-only macro | prevalence macro | 上乗せ |
|---|---|---|---|
| 全 268 bag | **0.458** | 0.342 | **+0.116** |
| **C2 を除く（n=231）** | **0.345** | 0.320 | **+0.024** |

C2（37 bag、全体の14%）の領域プロファイルが反転している（R1 73% / R4 19% に対し
C3–C7 は R4 が 62–79%）ためで、歯突起骨折の分布がそのまま出ている。

**したがって報告は2本立てにする。**

- **全 268 bag**: 実運用に近い。ただし床が 0.458 と高く、C2 の事前分布を学ぶだけで届く
- **C2 を除く 231 bag**: 床が 0.345 まで下がり prevalence 0.320 とほぼ差が無い。
  **「画像が本当に効いているか」はこちらで判定するのが素直**

C2 単独の解析は 37 bag しかないので指標を出さない（記述統計に留める）。

### 8.1c 領域の共起は弱い（独立 sigmoid の妥当性）

268 bag の領域間 Jaccard は **0.08〜0.22** と全て低い（R1-R2 0.22 / R1-R3 0.21 /
R1-R4 0.13 / R2-R3 0.16 / R2-R4 0.09 / R3-R4 0.08）。
排他でもないが強く共起もしないので、**独立 sigmoid + BCE という仕様の選択は妥当**。

R2/R3 の内訳: 両方0 = 155 / R3のみ = 54 / R2のみ = 41 / 両方1 = 18。
**左右が排他的な 95 bag** が、左右判別能を測れる唯一の部分集合になる。

### 8.2 held-out test が無い → **CV のままで confirmatory にできる**（条件つき）

Codex の結論: **独立 test を切る必要はない。** 患者単位 CV は正式な内部検証であり（TRIPOD）、
160 患者からさらに test を割くより全員を OOF に使うほうがこの規模では効率的。
ただし**外部データへの transportability は評価できない**。

**Weak-only vs Mixed を confirmatory と呼ぶための10条件（すべて実験前に固定）**

1. 主対比は **Mixed − Weak-only の1本だけ**
2. 主評価項目は 268 OOF bag の生確率を連結した4領域 macro average precision
3. **non-interpolated AP** を使う（台形 PR-AUC は使わない）
4. 同一 fold・同一 seed・同一初期値・同一 sampler・**同一の陰性 ID**・同一 augmentation 乱数・同一更新回数
5. **Weak-only も同じバッチを通し、違いは `λ_region = 0` だけ**にする。`λ_neg` も両 arm 同じ
6. **outer validation fold を checkpoint 選択や early stopping に使わない。固定 epoch/step の最終 checkpoint を使う**
7. 領域 OOF を一度見た後に λ・sampler・epoch・アーキを変えて再び confirmatory と呼ばない
8. 5 seed の bag 単位確率を平均してから AP を1回計算する。seed を5倍の観測にしない
9. 患者単位 paired bootstrap 10,000回。`Δmacro-AP` の95% CI 下限が 0 を超えたら優越
10. 他の2対比は secondary。3本すべて形式的に検定するなら Holm 補正

> **条件6は現行 Stage3 の `early_stopping_patience: 15` と衝突する。** Stage4 では早期停止を切り、
> epoch 数を固定する必要がある。両 arm を「λ_region 以外すべて同一」にするための代償。

> **既に 268 bag の領域 OOF を見て λ や構造を選んでいる場合、事後の事前登録では confirmatory 性は戻らない。**
> その場合は正直に exploratory と書く。

**感度解析**: fold 内・領域内の percentile rank を連結した macro-AP も出す。
raw pooled との差が **0.03 以上**、または Mixed/Weak の優劣が反転したら
「fold-scale sensitive」と明記して強い優越主張を避ける。

**どうしても内部 held-out を作るなら**: 最低 **60 annotated patients**（約101 bag、R2陽性約22）。
かつ各領域で「陽性 patient cluster 20以上、陰性 patient cluster 20以上」。
60 未満では R2/R3 の独立検証として弱すぎる。ただし外部検証にはならず、
学習側のアノテ患者が100人まで減るので **Codex は推奨しない**。

### 8.3 椎体ロジットが領域ロジット経由であること（endpoint coupling）

これは統計的リークではない（患者は outer fold から正しく隔離されている）。
**endpoint coupling** であり、独立ヘッドには無い故障が生じる。

| 故障 | 内容 |
|---|---|
| Gradient conflict | 領域陰性で `q` を下げる勾配と、椎体陽性を満たすため `q` を上げる勾配が共有 trunk で衝突 |
| Winner amplification | smooth-max が現在最大の領域に大きな勾配を送り、既に優勢な R4 がさらに優勢化 |
| Compensatory inflation | 複数領域を領域 BCE で下げられると、残る1領域を過大に上げて椎体 BCE を満たす |
| Calibration shift | クラス重みと陰性サンプリングが `q` の切片を変え、そのまま椎体の感度・特異度を変える |
| Annotation-subset shortcut | アノテ集団が易しい骨折に偏ると、その表現だけで椎体性能が上がり未アノテ陽性で悪化 |
| Multi-region under-reward | 領域 BCE は全陽性領域を上げるが、smooth-max の椎体損失は最大の領域しか主に報酬しない |

**100 optimizer step ごとに測る診断と警告閾値**

| 診断 | 警告閾値 |
|---|---|
| 共有 encoder 上の `cos(g_vertebra, g_region)` | epoch 中央値 **< −0.20** が2 epoch 連続 |
| 勾配ノルム比 `λ_region·‖g_region‖ / ‖g_vertebra‖` | 中央値 **> 3.0** が2 epoch 連続 |
| smooth-max 微分重みの最大領域 | 同一領域が陽性 bag の **>75%** で最大 |
| 1領域の pooling weight | `>0.95` の bag が **>80%** |
| `q_r` と領域マスク面積の Spearman | `|ρ| > 0.60` かつ AP 改善が prevalence+0.05 未満 |
| R2–R3 スコア相関 | `ρ > 0.95` かつ両 AP が prevalence+0.03 以下 |
| アノテ陽性の感度改善 | +10pt 以上だが未アノテ陽性で −5pt 以下 |

**椎体 safety gate（非劣性）**

```
ΔAUROC = AUROC_Mixed − AUROC_Weak-only
Pass: 患者 paired 95% CI 下限 > −0.010
Fail: CI 下限 ≤ −0.010（点推定も ≤ −0.010 なら明確な damage）
```

補助ガードレールとして椎体 AUPRC も `Δ ≥ −0.020` を要求する。

**Fail したら**: 独立椎体ヘッドの追加は妥当だが、それは
**同じ confirmatory arm の修正ではなく、新しい事前登録の rescue experiment** として扱う。

### 8.4 Detail-only は 428 bag しか無い

Codex の見解: 小標本で弱いこと自体は交絡ではなく、
「弱ラベル症例を追加する」という介入の一部である。ただし**主張の言い方を限定する**。

> Mixed improved over a model trained on the fixed detail-labelled cohort alone,
> estimating the benefit of **adding weakly labelled cases to that cohort**.

「同じ標本サイズで弱教師が優れていた」とは言えない。

**公平化条件**: 同一アーキ・ImageNet 初期化・augmentation・optimizer。
Mixed と Detail-only で **detail bag の提示回数を完全一致**させる
（"strong epoch" =「学習側 detail bag を1回ずつ提示」と定義し、両 arm で同じ strong epoch 数）。
同じ陰性サンプリング manifest、同じ更新回数（Detail-only は cycling して合わせる）、同じ checkpoint 規則。

**追加する統制 arm: Weak-only-size-matched**（secondary diagnostic）
各 fold で、Detail-only が使う 214 アノテ陽性 bag を**領域ラベルを無視して弱ラベルだけで**使い、
同じ 214 陰性を足した計 428 bag で学習する。更新回数・sampler・seed は Detail-only と同一。

これで三角測量ができる。

| 対比 | 測れるもの |
|---|---|
| Detail-only vs Weak-only-size-matched | **標本数を固定したときの領域ラベルの価値** |
| Mixed vs Detail-only | 弱ラベル症例を足す価値 |
| Mixed vs full Weak-only | 全データ固定での詳細ラベルの価値（**中心主張**）|

---

## 9. 次セッションの実装順序

| # | 作業 | 依存 |
|---|---|---|
| 1 | `--no-bbox` の未commit分をcommit | なし |
| 2 | データ層: 領域ラベル結合 + `region_supervision` 3値 | なし |
| 3 | **水平flip の R2↔R3 swap（mask ID / target / region-valid）+ unit test 3本** | 2 |
| 4 | fold 読み込みを `stage4_folds.csv` ベースに（manifest を hash して凍結）| なし |
| 5 | **level-only 対照**（CPU、数分。合格ライン 0.458 を確定させる）| 2, 4 |
| 6 | `stage4_loss` = `stage3_loss` + λ_region·L_region、λ_neg を 0.1→0.05 | 2 |
| 7 | 陰性サンプラ（1:1、毎epoch再抽出、レベル完全一致、seed=42+epoch、manifest保存）| 2 |
| 8 | **早期停止を切り、epoch 数を固定**（§8.2 条件6）| 6, 7 |
| 9 | 診断ロギング（勾配 cosine・ノルム比・pooling weight 偏り。§8.3 の表）| 6 |
| 10 | **Weak-only と Mixed を 5-fold × 5 seed**（差は `λ_region` のみ）| 3,4,6,7,8,9 |
| 11 | 評価: pooled OOF macro-AP + 患者単位 paired bootstrap 10,000回 | 10 |
| 12 | Detail-only、Weak-only-size-matched、Pretrain-to-Mixed | 10 |

**手順3を飛ばさないこと。静かに壊れる唯一の箇所である。**

手順8と10は「Weak-only と Mixed が `λ_region` 以外まったく同一」を担保するためのもので、
これが崩れると中心主張が confirmatory でなくなる（§8.2）。

---

## 10. Codex で決着した項目

| 項目 | 決着 |
|---|---|
| held-out test | **不要**。CV は正式な内部検証。条件10個を事前登録すれば confirmatory（§8.2）|
| 陰性サンプリング比 | **1:1**（2:1 は過剰）。毎epoch再抽出・レベル完全一致 |
| クラス重み | **陰性を足した後**に計算。`min((2N_A−P_r)/P_r, 8.0)` |
| λ_region | `0.25 + 0.75·min(e/4,1)`（epoch1 で 0.25、epoch5 以降 1.0）|
| λ_neg | **0.05** 一定。plane-level なので二重計上ではない |
| 椎体 非劣性マージン | **−0.010 AUROC**（患者 paired CI 下限）、補助で AUPRC −0.020 |
| 水平flip | **確率 0.50 で維持**。mask ID・target・region-valid の3つを同時 swap。主推論では TTA しない |
| fold 層別 | 現行 split は hard constraint と領域 target を全充足。**hash して凍結**し作り直さない |
| Detail-only の言い方 | 「弱ラベル症例を**追加する**価値」に限定。size-matched 統制を追加 |

## 11. 未確定（ユーザー判断）

1. `train_models/stage4/` 新設か Stage3 に config フラグか
   → 差分がデータ層＋損失1項＋サンプラだけなので、**Stage3 のフラグ拡張で足りる可能性が高い**
2. 固定 epoch 数をいくつにするか（早期停止を切るので事前に決める必要がある）。
   Weak-only の学習曲線を1 fold だけ見て決め、**その後は変えない**のが現実的
3. level-only と Weak-only-size-matched の2 arm を追加することの可否（仕様に無い）
4. C2 を除いた macro-AP を副次指標として報告するか（§8.1b）
   → 差分がデータ層+損失1項+サンプラだけなので、**Stage3 のフラグ拡張で足りる可能性が高い**
