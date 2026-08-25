# region_branch 設計セッション worklog

作成日: 2026-08-25

4領域骨折検出モデル（`fracture_detection/region_branch/`）の設計。**実装は未着手**。
本セッションではコードを書かず、設計の確定と実測による裏取りのみ行った。

---

## 1. 依頼内容

ユーザーから以下の構成の提案があった。

```
5CT + whole mask (6ch)
        ↓
Shared EfficientNetV2-S
        ↓ spatial feature F
   R1〜R4 mask で分離
        ↓
region-normalized pooling
        ↓ z_R1〜z_R4
   Shared BiLSTM
        ↓
   4つの region head
        ↓
   p_R1〜p_R4
```

初期モデルでは attention 系・region専用CNN/BiLSTM・Transformer を使わない。
単一領域モデル4本との比較で「領域ごと独立学習 vs 4領域共有学習」を検証する。

---

## 2. 実測で確定した事実

本セッションで新規に測定した。**この数値は再測定不要。**

### 2.1 領域面積と特徴マップ解像度

サンプル 280〜420 bag、全 C1-C7 水準。

元画像 224×224 での領域面積（非空面の中央値）:

| 領域 | 面積(px) | 画像比 |
|---|---:|---:|
| R1 vertebral_body | 1,328.5 | 2.65% |
| R2 right_transverse_foramen | 401.0 | **0.80%** |
| R3 left_transverse_foramen | 381.0 | **0.76%** |
| R4 posterior_elements | 1,670.5 | 3.33% |

特徴マップ上の実効セル数（面積/stride²、中央値）:

| pooling元 | 格子 | R1 | R2 | R3 | R4 |
|---|---|---:|---:|---:|---:|
| stride 32 | 7×7 | 1.29 | **0.39** | **0.37** | 1.65 |
| stride 16 | 14×14 | 5.18 | 1.55 | 1.48 | 6.61 |
| stride 8 | 28×28 | 20.72 | 6.20 | 5.94 | 26.44 |
| stride 4 | 56×56 | 82.88 | **24.81** | **23.75** | 105.75 |

**結論: stride 32（EfficientNetV2-S 最終層）では R2/R3 が1セル未満**になり、
4つの region feature がほぼ同一ベクトルになる。mask-normalized pooling は
スケール補正であって解像度を復元しないため、正規化だけでは解決しない。

nearest downsample では stride 32 で R2/R3 が **55.7% / 56.5%** の面から消失する。

### 2.2 領域maskの欠損

| 領域 | 面単位で mask が空 | 全15面で空の bag |
|---|---:|---:|
| R1 | 5.68% | 0件 |
| R2 | 16.81% | 0件 |
| R3 | 17.35% | 0件 |
| R4 | 15.62% | 0件 |

bag単位では必ず信号がある。面単位では15-17%が空。

### 2.3 教師信号の母数（3集団は重複なしの完全分割）

| 損失 | 対象 | bag数 |
|---|---|---:|
| 正解ラベル | annotated（**全件が骨折陽性**） | 268 |
| ソフトラベル | 骨折陽性 かつ 非annotated | 1,064 |
| 陰性 | `vertebra_target=0` → 4領域すべて論理0 | 12,100 |
| | 合計 | 13,432 |

annotated 268 に骨折陰性は0件。よって「同じbagに正解とソフトが両方当たる」
優先順位問題は発生しない。

正解ラベルの有効セル: 983/1072（`region_*_target_valid`）。
無効な89セルは「陰性だがアノテーション未完了で0と断言できない」ケース。

領域別の有効bag数と陽性数:

| | R1 | R2 | R3 | R4 |
|---|---:|---:|---:|---:|
| 有効bag | 245 | 243 | 244 | 251 |
| 陽性 | 78 | **59** | 72 | 158 |

### 2.4 疑似ラベルの完全性（セッション冒頭に検証）

`fracture_detection/baseline0/outputs/08_19/pseudo_labels/`

- 40,296行 = 13,432 bag × 3 teacher、過不足・重複・NaN すべて0
- sha256 が metadata の記録値と一致
- fold→teacher 対応は設計どおりの相補パターン
- 温度較正の `n_defined < n_bags` が2件（teacher0/R3: 801/802、teacher1/R3: 794/795）。
  想定内のエッジケース

### 2.5 bbox強制planeリークは解決済み

**過去のmemoryが古く、誤った警告を出した。訂正済み。**

- 全データを作り直して `data/rsna_data/fracture_dataset/` に統合済み（ユーザー確認）
- `fracture_dataset_blind/` は移行時の残骸。参照不要
- `baseline0/data/constants.py` が `fracture_dataset` を指すのは正しい
- 再生成対象 study の ct.npy を blind と照合し全一致を確認
- **注意:** `.claude/docs/experiments/2026-07-13-stage2-region-vs-primary-comparison.md`
  の数値はリーク修正**前**。現行データの基準値として引用しないこと

---

## 3. 確定した設計

| 項目 | 決定 |
|---|---|
| ディレクトリ | `fracture_detection/region_branch/` |
| 特徴抽出 | Shared EfficientNetV2-S、**stage 1-4 を 1×1 conv で 256ch に揃え stride 4 (56×56) へ FPN 融合**してから pooling |
| 領域分離 | mask-normalized pooling。境界誤差は許容（4領域統合で吸収できるかも観点の一つ） |
| 空maskの面 | ゼロmask として前向き、損失からは除外、bag集約は有効面のみ |
| 系列学習 | Shared BiLSTM（4領域を batch 次元に積み、領域間は非干渉） |
| 出力 | 領域別 head 4本（Linear→Act→Dropout→Linear(1)） |
| bag集約 | `logit(mean sigmoid)` |
| whole-vertebra head | baseline0 と同一 |
| 教師信号 | 正解ラベル / ソフトラベル / 陰性の3種。**陰性の扱い（統合するか別項か、母数制御の方法）は未決定**（下記4.2） |
| 比較 | 単一領域モデル4本。feature抽出は共通。**human-only 対照 arm は不要**（決定済み） |
| 検出力 | 許容（268 bag、R2陽性59件で差が解像できない可能性を承知のうえ） |

FPN の採用は `train_models/stage3/src/model.py:137-149` の前例に従う。
解像度とチャンネル数のトレードオフは、片方を選ぶのではなく FPN で両立させる。

---

## 4. 損失（提案済み・**未承認**）

### 4.1 統合モデルと単一モデルは同じ式

```
L = L_whole + λ·( L_exact + α·L_rank )

L_exact = (1/|R|) Σ_{r∈R} [ 領域 r の exact セル平均 ]
L_rank  = (1/|R|) Σ_{r∈R} [ 領域 r の pair 平均 ]
```

`R` が違うだけ。統合 = {1,2,3,4}、単一 = {r}。

**和ではなく平均**にすることが重要。和にすると統合モデルの region 損失が4倍になり、
「共有の効果」と「領域損失を4倍重くした効果」が区別できなくなる。

実装は `active_regions` 一つで切り替わるため、コードは1本で済む。
config 5種（統合1 + 単一4）× 5 fold = **計25 run**。

### 4.2 陰性の扱い（**未決定**）

2026-08-25 時点でこの節全体が**未承認・未決定**に戻った。以下はClaudeの提案であり、
ユーザーの決定ではない。次に詰め直す。

検討していた論点:

- whole-negative の椎体は4領域とも骨折なしが確定なので、人手ラベルの `region_r = 0` と
  同じ種類の教師ではないか、という見立て。項を分けず統合する案
- ただし素朴に統合すると人手268が陰性12,100に希釈される（約1:45）。
  領域ごとの統合後陽性率は R1 0.63% / R2 0.48% / R3 0.58% / R4 1.28%
- 対処案として sampling による母数制御（`pos_weight` ではなく）を提案していたが未承認

再検討時に確認すべき問い:

- 陰性を正解ラベル項へ統合するか、独立した3項目（whole/exact/rank/negativeの4項）にするか
- 統合する場合、希釈（1:45）をどう扱うか
- そもそも陰性椎体の領域ラベルに、whole ラベル以上の情報があるかどうかの是非

### 4.3 層化 batch（**§4.2の未承認案が前提。§4.2が確定するまで無効**）

§4.2で陰性を正解ラベルへ統合する案が撤回されたため、以下も未決定に戻る。
陰性を別項にする場合はこの層化構成そのものを作り直す必要がある。

（撤回前の案、batch size 16 の場合）

| 種別 | bag/batch | fold内プール | 役割 |
|---|---:|---:|---|
| 人手 | 4 | ~214 | `L_exact` の陽性側 |
| 陰性 | 4 | ~9,680 | `L_exact` の陰性側（persistent queue） |
| ソフト | 8 | ~851 | `L_rank` のペア構成 |

- epoch 長はソフトのプールで決める（約106 batch）
- 人手は epoch あたり約2周 → **全 batch に人手の勾配が乗る**
- 陰性は epoch あたり424 bag、約23 epoch で全プール一巡
- `L_exact` 内の人手:陰性が 1:1 に固定され、1:45 の希釈が起きない（という設計意図だった）

### 4.4 λ, α は固定

`λ = 0.25`、`α = 0.25`。既存ドキュメントで事前宣言済みの予算をそのまま使う。

学習前に数バッチで両項の値を実測し、桁が乖離していたときだけ 0.1 か 1 に丸め直す。
**性能は見ない。fold ごとに変えない。以後動かさない。**

過去に沈んだのは「268例の性能を見て λ を選んだ」ことであり、
事前宣言した定数を使うこと自体は問題ない
（`baseline0/pseudo_labeling/scoring.py` の docstring が明示的に警告している）。

### 4.5 λ が効くのは trunk だけ（混乱の元だった点・解決済み）

whole head と region head は **CNN trunk の出口で分岐**する。

| パラメータ | `L_whole` から | region損失から |
|---|---|---|
| CNN trunk | ✓ | ✓ |
| conv_head / bn2 | ✓ | ✗ |
| whole BiLSTM | ✓ | ✗ |
| whole head | ✓ | ✗ |
| FPN | ✗ | ✓ |
| region BiLSTM | ✗ | ✓ |
| region head ×4 | ✗ | ✓ |

**共有されているのは trunk のみ。** よって λ は「領域タスクの学習強度」ではなく
**trunk の表現を whole 寄りにするか領域寄りにするか**を決めている。
λ を小さくしても region head の学習は遅くならない（他に勾配源がないため）。

### 4.6 val loss（§4.2の未承認案が前提の部分を含む）

`L_exact` の中身（陰性を含むか、正解ラベルのみか）は§4.2の決定に従って変わる。
「`L_rank` が val で構造的に計算不能」という構造上の事実そのものは§4.2に依存せず成立する。

```
train loss = L_whole + λ(L_exact + α·L_rank)
val   loss = L_whole + λ·L_exact          ← L_rank は構造的に計算不能
```

疑似ラベルは「教師 k が自分の学習 fold を採点する」設計なので、
student k の held-out bag には教師 k のスコアが存在しない。

結果として **val loss は exact な教師だけで構成される**。
ソフトラベルの質に汚染されない、素直に下がるべき量になる。

- 集約単位を train と val で揃えること（片方だけ直して矛盾させた前例あり）
- val も同じ層化サンプリングで作り、陰性支配を防ぐこと
- `selection_metric` は val の `L_exact` を提案

### 4.7 collapse 監視（統合しても消えない）

機序: exact-negative BCE は4 logit を同方向に下げる / ranking は共通 shift に不変。
過去に `rho ≥ 0.97` の collapse 実績あり。

人手ラベルを使わない固定 diagnostic subset で毎 epoch 記録:

- 領域logit間 Spearman 6組
- 各領域 logit と whole logit の Spearman
- 標準化4-logit行列の第1主成分説明率

事前指定 alarm: off-diagonal 中央値 ≥0.95 かつ region-whole ≥0.95 が3回連続。
**alarm が出ても係数を調整して再実行しない**（268例への再適合になるため）。

---

## 5. 未確定の論点

### 5.1 BiLSTM を2本にするか1本にするか

現在の図では whole 用と region 用で BiLSTM が **2本**。
「whole head は baseline0 と何もかえない」を文字どおり取ると、
baseline0 の BiLSTM（入力1280次元）を残すことになり、region 側（入力256次元）とは別物になる。

一方 `.claude/docs/codex/20260823-pseudo-label-mtl-design.md` には
「BiLSTM は Baseline 0 から初期化し、whole path からも勾配を受ける」＝1本統合案がある。

- **2本** — baseline0 と完全一致、共有点が trunk だけに限定され比較しやすい。パラメータ増
- **1本** — 系列表現も共有。ただし whole path が baseline0 と別物になる

推奨は **2本**。未回答。

### 5.2 §4 の損失仕様の承認

λ/α の値、層化 batch 構成、`selection_metric` を val `L_exact` にする点。
提案済み・未承認。

### 5.3 単一領域4本 × 5fold の実行計画

計25 run の順序・GPU割当は未検討。

### 5.4 陰性の扱い（§4.2）

正解ラベルへの統合案は撤回。次回に詰め直す。詳細は§4.2参照。

---

## 5.5 決定済み：human-only 同一アーキテクチャ arm は追加しない

`20260823-pseudo-label-mtl-design.md` が「mask poolingを入れると、教師CAMとの差が
疑似ラベル効果か領域maskのinductive biasか分離できないため、human-only arm が必要」
と警告していたが、ユーザーが不要と判断（2026-08-25）。単一領域4本 vs 統合1本の
比較のみで進める。

---

## 6. 現在の状態

- **コードは一切書いていない。** 作業ツリーはクリーン
- セッション中に2回、確定前に実装へ入ろうとして中断された
  （`region_pooling/` と `region_branch/` のスケルトンを作成 → 両方撤去済み）
- `fracture_detection/PROGRESS.md` は HEAD の内容へ復元済み
  （セッション開始時、ステージ済み版とディスク版が食い違っていた）

### memory の更新

- `feedback_decide_policy_before_implementing.md` — 再発事例を追記。
  「自分が投げた質問への回答は実装許可ではない」「未決項目の config 化は決定の代替にならない」
- `project_bbox_forced_plane_leak.md` — 【解決済み】を追記。
  古い「blind へ差し替えろ」という指示が誤警告の原因だった

---

## 7. 次にやること

1. §4.2（陰性の扱い）を詰め直す — 統合するか独立項にするか、母数制御の方法
2. §5.1（BiLSTM 2本 or 1本）を決める
3. §4.3〜4.6（層化batch・selection_metric）を§4.2の決定に合わせて作り直す
4. `fracture_detection/region_branch/` の実装に入る
5. README を作成する（`feedback_project_readme_required`）

### 流用できる既存資産

- **ソフトラベル損失は実装済み** — `baseline0/pseudo_labeling/scoring.py`
  （`build_region_pair_batch` / `region_balanced_pairwise_ranking_loss`）。
  `human_target_valid` による正解ラベル優先の除外も入っている
- **FPN + mask pooling の参照実装** — `train_models/stage3/src/model.py:137-213`。
  `region_mode: masked|global|scramble` の対照群付き
- **whole 損失** — `baseline0/modeling/losses.py` の `broadcast_bce_loss` / `bag_probabilities`
- **fold 分割** — `baseline0/data/splits.py`
