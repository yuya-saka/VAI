# Stage4 設計案 — 4領域アノテーションによる局在モデル

作成日: 2026-07-29
状態: **提案（未承認）**。`DESIGN.md` には記載しない。承認後に転記する。
前提: `.claude/docs/work-logs/2026-07/2026-07-28.md`, `2026-07-29.md`、
Codex 3本（`20260728-region-label-model-design.md`, `20260728-clean-slate-localization-architecture.md`,
**`20260729-stage4-evaluation-protocol.md`**）

**教師の単位は椎体（bag）である。スライス単位の情報は持たない。bbox は入力にも教師にも
評価GTにも使わない**（ユーザー確認済み、§1.3-1.4）。

§2 の評価プロトコルは 07-29 の Codex 相談を受けて書き直した。
**当初案から反転した箇所を「【訂正】」として明記してある**（§1.4 / §2.4 / §2.6 / §4.1-4.2）。

---

## 0. Stage4 とは何か

| Stage | タスク | 領域の扱い |
|---|---|---|
| Stage1 | 椎体2値分類 | 使わない |
| Stage2 | 領域マスク付き分類 | 入力の補助 |
| Stage3 | 階層弱教師（直接ヘッドなし） | **潜在変数**（教師なし） |
| **Stage4** | **どの領域が折れているかを出力** | **人手ラベルで強教師** |

Stage4 の主張は「椎体が陽性と分かっているとき、4領域のどこが折れているかを当てられる」。
椎体2値分類は**捨てずにガードレールとして維持する**（局在のために分類性能を落とさない）。

---

## 1. 教師データの実測（2026-07-29 時点）

### 1.1 母集団と既存分割

`fracture_dataset_blind` 上で `collect_items` → `split_test_holdout(0.2, 42)` → `split_items_cv(5, 42)` を実行。

| | bag | study | 陽性 | 陰性 |
|---|---|---|---|---|
| 全体 | 13,432 | 2,009 | 1,332 | 12,100 |
| locked test | 2,703 | 402 | 259 | 2,444 |
| CV pool | 10,729 | 1,607 | 1,073 | 9,656 |

### 1.2 領域アノテーションの fold 配置

`fracture_region_labels_dicom.csv`: 285行 → bag単位OR集約で **268 bag / 160 study**。全て陽性椎体、全て item 集合内。

| fold | n | R1 body | R2 right | R3 left | R4 post | R2⊕R3 | R4のみ | 未アノテ候補 |
|---|---|---|---|---|---|---|---|---|
| 0 | 45 | 24 | 12 | 12 | 24 | 10 | 14 | 23 |
| 1 | **30** | 9 | **5** | 12 | 18 | 11 | 12 | 31 |
| 2 | 47 | 13 | **5** | 13 | 26 | 16 | 21 | 33 |
| 3 | 43 | 9 | 7 | 13 | 26 | 18 | 18 | 27 |
| 4 | 51 | 15 | 15 | 13 | 32 | 20 | 19 | 20 |
| **OOF計** | **216** | 70 | 44 | 63 | 126 | 75 | 84 | 134 |
| locked test | 52 | 8 | 15 | 9 | 32 | 20 | 27 | 29 |

多領域率: 1領域 198 / 2領域 45 / 3領域 21 / 4領域 4。**全ゼロ 0件**（07-28 の1件は解消済み）。
アノテーション可能上限は **431 bag**（bboxを持つ study×level 440 − 除外9）。現在 268/431 = 62%。

### 1.3 ラベルの単位は椎体。z（スライス）情報は持たない

**ユーザー確認済みの前提（2026-07-29）**:

> bbox の情報は一切使わない。椎体部位を4領域に分けてそれぞれにアノテーションしている。
> **スライスごとの情報ではない。**

したがって Stage4 の教師は、bag = (study, level) ごとの **4次元の 0/1 ベクトル1本だけ**である。
「どの面に骨折があるか」の情報は**存在しない**。

`run_id` について: 当初これをアノテータの再試行と読み、17 bag中6件の「不一致」から
信頼性を推定しかけたが誤り。`Unet/dicom_bbox_annotation_tool/server.py:487`
`_split_contiguous_rows` により **run = DICOM系列順で連続する bbox 行の塊**で、
これはツールが提示対象を列挙する単位にすぎない。ラベルの意味は椎体単位なので

- bag 単位の **OR 集約が正しい**。ラベル方針の変更は不要
- **アノテーション再現性のデータは現状ゼロ**。noise ceiling は未測定のまま

### 1.4 bbox は入力にも教師にも評価GTにも使わない

一度は「bbox の z 範囲を plane 単位の教師／z局在の評価GTに使えば、人手アノテーションに
依存せず検出力を稼げる」という案を書いたが、**ユーザー指示により全面撤回した**。

撤回した内容:

- `L_plane`（bbox z範囲外の面を4領域すべて0とする損失項）
- z局在評価（top-1 plane hit、OOF 334 bag、tight 132 bag）とその実行順序上の前倒し

撤回は方針として正しい。07-28〜07-29 の作業は
**bbox を plane サンプリングから除去する（`fracture_dataset_blind` 生成）** ことに費やされた。
そこで入力から追い出した情報を教師や評価GTとして裏口から入れれば、
リーク除去の意味が薄れ、「bbox のある症例でしか成立しない指標」が増えるだけになる。

**bbox の唯一の残存関与**: どの bag にラベルが付き得るか（431 bag = bbox のある study×level）を
決めているのはツールの列挙仕様である。これは**選択バイアスという母集団の性質**であって、
入力・教師・評価GTとしての利用ではない。§5 でこの区別を明記する。

**代償**: 局在の証拠は **OOF 216 bag の領域指標だけ**になる。
人手不要で検出力を稼ぐ道は無くなったので、§2.8 の CI 幅（macro-AP で全幅 0.12–0.22）が
そのまま本設計の制約になる。ショートカット耐性の検証は
**SideAcc（R2⊕R3 の OOF 75 bag）が事実上の主力**になる（§2.11）。

### 1.5 ラベル番号とマスク番号の整合（確認済み）

Stage4 の教師はここが狂うと全部無意味になるので、今回あらためて両側のコードで確認した。

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| アノテーションUI (`index.html:191-194`) | 椎体 | 右椎間孔 | 左椎間孔 | 後方要素 |
| マスク (`rsna_4region_segmentation/constants.py:61`) | body | right_foramen | left_foramen | posterior |

`REGION_NAMES = ("background", "body", "right_foramen", "left_foramen", "posterior")` なので
マスクのクラス番号 1..4 と CSV の `region_1..4` は**そのまま対応する**。remap は不要。

ただし **「右/左」は画像上の左右であって患者基準ではない**（既知）。
論文・報告で "right foramen" と書くと患者の右と誤読されるため、
`image-right foramen` のように基準を明記するか、`R2/R3` のまま扱う。

---

## 2. Fold 設計（評価プロトコル）

### 2.1 分割は既存を凍結して再利用する

**根拠（今回実測）**: `fracture_dataset`（旧）と `fracture_dataset_blind`（新）で

- item 集合の対称差 **0**（13,432 bag 完全一致）
- locked test の study 集合 **完全一致**
- 10,729 bag すべての fold 割り当て **完全一致**

を確認した。つまり bbox 盲データへ切り替えても分割は 1 bag も動かない。

fold を作り直してはいけない理由: Stage1/2/3 の fold-k checkpoint を warm-start や
凍結特徴抽出器として使う設計なので、分割が変わると **その checkpoint の学習 study が
Stage4 の val に混入する**。分割凍結は性能の問題ではなく妥当性の問題である。

```
split_test_holdout(items, test_size=0.2, seed=42)      # 変更禁止
split_items_cv(cv_items, n_splits=5, seed=42)          # 変更禁止
fold-k の warm-start は必ず fold-k の checkpoint から   # 交差させない
```

### 2.2 primary 指標の定義（事前登録する形）

```
macro-AP = (1/4) Σ_r AP(y_r, q̄_r)
```

- 各 bag のスコアは、**その study を一切学習に使っていない** outer-fold モデルから得た OOF 値
- `q̄_r` は 5 seed の確率の算術平均（§2.7）
- AP は **non-interpolated average precision**。台形補間の PR-AUC は使わない（Davis & Goadrich 2006）
- 4 領域を等重みで平均。**閾値は選ばない**

**no-skill 床（実測）**: 領域 prevalence の平均。
OOF は `(70+44+63+126)/(4×216)` = **0.3507**、locked test は `(8+15+9+32)/(4×52)` = **0.3077**。
always-posterior の exact-set 41% / hit@1 59% は**別指標の床**であって macro-AP の床ではない。混同しない。

指標の正式名は
> **Study-grouped OOF conditional anatomical-region macro average precision among bbox-eligible fracture-positive vertebrae**

長いが、条件を名前に埋めないと「無条件の局在性能」と誤読される。短縮するなら `cond-macroAP` とし、
初出で必ず条件を書く。

### 2.3 同じ分割を、2つの評価母集団で使い分ける

| | 母集団 W（弱・全体） | 母集団 S（強・局在） |
|---|---|---|
| 対象 | CV 10,729 bag 全部 | アノテ済み **216 bag のみ** |
| 指標 | 椎体 AUROC | macro-AP、SideAcc |
| 役割 | **ガードレール**（非劣性）＋**候補選択** | **報告する primary** |

### 2.4 【重要な訂正】outer fold は checkpoint 選択からも隔離する

当初案は「fold の val の弱 AUROC で早期停止し、同じ fold の領域ラベルで OOF 評価する」だったが、
**これは厳密な outer-CV ではない**。弱 AUROC と領域性能は相関しうるので、
同じ study を checkpoint 選択と領域評価の両方に使っていることになる。

正しい手順（outer fold *f* ごと）:

1. fold *f* の study を **学習・早期停止・ハイパラ選択のすべてから除外**
2. 残り4 fold ブロックを順に inner validation にして4回まわす
3. checkpoint / ハイパラは **inner の弱 AUROC だけ**で選ぶ
4. epoch 数は4回の best epoch の**中央値**（偶数個なので中央2値の平均を切り上げ）
5. その epoch 数で残り4 fold 全体を refit
6. outer fold を**一度だけ** predict

warm-start に使う checkpoint も **outer fold *f* を学習していないもの**に限る。
outer の分割自体は §2.1 のとおり変更しない。

### 2.5 fold 単位で領域指標を出さない

R2 は fold1・fold2 で **5 bag** しかない。fold 別 AP は分散が支配的で意味を持たない。
→ **5 fold の val 予測をプールしてから、1回だけ**計算する。fold 別の数字は診断ログに留める。

### 2.6 【訂正】プールは生確率のまま。rank 正規化は感度解析に降格

当初案は fold 内 rank 正規化を primary にしていたが、**生確率プールを primary にする**。
同じ loss・同じスコア定義・同じ学習規則から出た確率なら、生スコアが
「cross-fitted な予測手続き」を最も素直に表す。rank 正規化は fold 間の実際のスコア差まで消してしまい、
z-score は fold ごとの location-scale 変換が妥当という根拠のない仮定を置く。

事前登録する順序:

| | 内容 |
|---|---|
| **Primary** | 5-seed 平均の生確率を 216 bag でプールした macro-AP |
| Sensitivity A | fold 内 percentile スコアをプール |
| Sensitivity B | 陽性数重み付きの fold 別 AP 平均 `AP_r = Σ_f (P_fr / Σ_g P_gr)·AP_fr` |
| 診断のみ | z-score 化した logit |

Sensitivity A の percentile 変換は、**アノテ済み部分集合ではなく各 fold の全 bbox-eligible 陽性候補**を
参照集合にする（`u = (平均ランク − 0.5)/N_f`）。**ラベルを使って変換を当てはめてはいけない。**
Sensitivity B で fold を等重みにすると、R2 陽性5件の fold が過大に効くので陽性数重みにする。

**判定**: primary と Sensitivity A の差が **絶対値 0.03 以上**、または4候補の順位が変わったら
「fold-scale sensitive」と宣言し、CV 上の小差で優劣を主張せず locked test を最終判定にする。
val の領域ラベルを使った Platt / temperature / isotonic 較正は**禁止**。

### 2.7 seed 反復（分割は変えない）

- seed を **`[42, 43, 44, 45, 46]` の5個に固定**。全候補・全 fold で同じ5個を使う
- OOF bag ごとに、その outer fold の5確率を算術平均
- **primary は「平均スコアをプールして計算した macro-AP」**

当初案は「seed 毎に AP を出して平均」としていたが、これは**別の estimand**（平均的な単一 seed 性能）である。
デプロイするのが seed アンサンブルなら、決定量は平均スコア側が正しい。
「seed 別 AP の平均・SD・min–max」は secondary の安定性報告として併記する。

**禁止**: seed を独立した5倍の観測とみなしてプールすること。5 seed あっても cluster-bootstrap CI を
`1/√5` に縮めること。

### 2.8 信頼区間は outer fold 内で層別した study-cluster bootstrap

**アノテ済み study の実測数**: OOF **129 study**（fold別 27/24/31/23/24）、locked test 31 study、計 160。
（160 は 268 bag 全体の値であって OOF の値ではない。**OOF の CI で 160 study をリサンプルしてはいけない。**）

- 10,000 回の valid replicate
- **outer fold 内で study を復元抽出**し、fold ごとのアノテ済み study 数を固定
- 選ばれた study の bag はすべて一緒に複製
- replicate ごとに4つの AP と macro-AP を再計算、95% percentile interval
- 候補間の差は同じ bootstrap sample を使う **paired interval**
- ある領域の陽性か陰性が0になる replicate は引き直す。**引き直し率が 1% を超えた領域は CI 不安定と明記**し、
  その領域の confirmatory な主張をしない

**CI 幅の見積もり（Codex の planning range）**

| 対象 | 95% CI の全幅 |
|---|---|
| 216 bag OOF macro-AP | **0.12–0.22**（半幅 ±0.06–0.11）|
| R2 単独 | 0.20–0.35 |
| 52 bag test macro-AP | 0.25–0.40 |
| test の R1(8) / R3(9) | 0.35–0.60 |

半幅が **0.10 を超える macro-AP** は「方向は評価できるが効果量は不精確」として扱う。

**領域別 confirmatory 主張の可否（陽性 study 数 20 未満は禁止、実測値）**

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| OOF 陽性 study | 56 | 36 | 49 | 83 | → **4領域すべて可** |
| test 陽性 study | **8** | **10** | **7** | 22 | → **R4 以外は不可** |

locked test で領域別の主張をするのは R4 だけ。他は macro でしか語れない。

### 2.9 モデル選択の多重性 — region-informed な比較は実質1回まで

216 bag で4候補を比べれば、選んだモデルの OOF 値は必ず楽観的になる。
**安全な比較回数の一般解は存在しない**（Cawley & Talbot 2010, Varma & Simon 2006）。2候補でも optimism は出る。

したがって次のように役割を分ける。

- **候補の絞り込みは母集団 W（弱 OOF AUROC）で行う。** 5-seed アンサンブルの pooled 弱 AUROC が最大のものを選び、
  最大値との差が **0.005 AUROC 未満**なら事前登録した優先順位の上位を採る
- 選択結果・config・seed・manifest hash を**凍結してから**、領域指標を**一度だけ**開く
- 4候補すべての領域 AP を見て最良を選んだ場合、その OOF AP は即座に **exploratory** 扱いになる
- 領域結果を見た後のアーキ / loss / 重みの変更は、1回目から adaptive comparison として数える

**arena（§3.1）は明示的に exploratory と位置づける。** arena の出力は「どのアーキを作るか」という
**設計判断**であって、報告する性能値ではない。報告する OOF 値は、arena 後に事前登録した
**単一モデル**を refit して1回だけ計算したものとする。

### 2.10 locked test は最後に一度だけ

locked test の役割は精密な効果量推定ではなく、**完全凍結後の方向の再現確認**。
事前登録する成功条件:

1. `test macro-AP − test の no-skill 床(0.3077)` が **0.10 以上**
2. その差の one-sided 95% study-cluster bootstrap 下限が **0 より大きい**
3. R2 と R3 の AP がそれぞれの test prevalence を点推定で上回る

満たさない場合の結論は「モデルが無効」ではなく **「小規模 test では確認不十分」**。
test をモデル選択・閾値選択・較正に使ってはならない。

### 2.11 対照（同一 fold 構造で必ず走らせる）

| 対照 | 実測済みの基準値 | 判定 |
|---|---|---|
| always-posterior 定数 | exact-set **41%** / hit@1 **59%** | これを大きく超えないなら無意味 |
| metadata-only（CT不使用） | 未測定 | macro AP **≥0.45 なら重大警告** |
| `region_mode="global"` | Stage3 に実装済み | マスクの寄与を測る |
| `region_mode="scramble"` | Stage3 に実装済み | 空間対応の破壊対照 |
| **SideAcc**（R2⊕R3 の OOF 75 bag）| posterior prior では原理的に解けない | **事実上の主力対照** |

**hit@1 を primary にしてはいけない**（R4 固定で 59% 当たる）。

§1.4 で z軸の対照を落としたので、**ショートカット耐性の検証は SideAcc がほぼ唯一の砦**になる。
SideAcc は「右椎間孔だけ／左椎間孔だけが陽性の bag で左右を当てる」課題で、
posterior prior でも領域 prevalence でも当たらない。ここが chance を超えないなら、
macro-AP が高く出ていても prevalence を学習しただけと判断する。

ただし OOF 75 bag しかないので、SideAcc 自体の CI も広い。
**「SideAcc の点推定が 0.60 を超え、かつ study-cluster bootstrap の下限が 0.50 を上回る」**を条件にする。

---

## 3. モデル設計

### 3.1 まず frozen-feature arena で「作り直すべきか」を決める

fold 対応の Stage1(blind) encoder から stride-2/4 の特徴を**一度だけ**キャッシュし、
head だけを 5-fold 学習して同一特徴上で比較する。encoder 再学習より桁違いに安い。

| 記号 | head | 位置づけ |
|---|---|---|
| E | global pooled + 4 logits | 下限対照（マスクを使わない） |
| **A** | Stage3 の hard mask pooling + 領域教師 | **既存資産の素直な拡張** |
| B0 | region-query cross-attention、prior なし | |
| B1 | region-query + soft mask prior | Codex の本命 |

**撤回条件を先に決めてある**: B が A を macro AP で +0.03 上回らず、mask 摂動耐性も改善しないなら
**素直に A を採る**。その場合アーキテクチャの書き直しは行わない。

A/B とも metadata-only を +0.08 未満しか超えず SideAcc ≤0.60 なら、**高価な end-to-end に進まない**。

**arena は exploratory であると明示的に宣言する**（§2.9）。arena の成果物は
「どのアーキを作るか」という**設計判断**であって、報告する性能値ではない。
4候補の領域 AP を見比べた時点で、その AP は confirmatory ではなくなる。

したがって全体はこう2段になる。

| 段 | 何をするか | 出てくる数字の扱い |
|---|---|---|
| **arena** | E/A/B0/B1 を frozen feature 上で比較 | **exploratory**。論文の主指標にしない |
| **confirmatory** | arena で決めた**単一モデル**を §2.4 の nested 手順で refit し、OOF を1回だけ計算 | **これが報告値** |

confirmatory 段で候補が複数残ってしまった場合は、§2.9 のとおり
**領域指標ではなく弱 OOF AUROC で選ぶ**（差 0.005 未満なら事前登録した優先順位）。

### 3.2 出力仕様

```
vertebra_logit        [B]         椎体2値（ガードレール、独立ヘッド）
region_logits  q[r]   [B,4]       領域multi-label（教師が付く唯一の場所、primaryの対象）
plane_region   p[z,r] [B,15,4]    plane×領域（教師なしの内部表現。集約されて q[r] になる）
```

領域は排他ではない（多領域26%）。softmax ではなく **multi-label sigmoid**。
ガードレールは領域の noisy-OR ではなく**独立した global ヘッド**で取る
（局在の集約方法と分類性能の評価を絡ませない）。

`p[z,r]` は Stage3 の `instance_evidence_logits` に相当し、Stage4 でも **教師は一切付けない**。
ラベルが椎体単位である以上（§1.3）、面ごとの正解は存在しない。
`p[z,r]` は可視化・診断には使えるが、**その品質を主張する根拠は本設計には無い**。

### 3.3 損失（非対称）

```
L = L_bag + α(t)·L_q + 0.25·α(t)·L_strong_neg + 0.1·L_neg_bag
α(t) = min(1, (t+1)/5)
```

| 項 | 対象 | 内容 |
|---|---|---|
| `L_bag` | 全 10,729 bag | 既存の弱教師MIL。層別サンプリングの歪み補正 `(2N⁺·mean⁺ + N⁻·mean⁻)/(2N⁺+N⁻)` |
| `L_q` | アノテ済み 216 | 4領域 multi-label BCE、クラス重み `clip(n₀/n₁,1,4)`。**重みは fold の学習側 bag だけから計算する**（268 bag 全体の値 R1 2.44 / R2 3.54 / R3 2.72 / R4 1.00 は目安） |
| `L_strong_neg` | 陰性椎体 9,656 | 全領域 0（人手不要の強教師） |
| `L_neg_bag` | 陰性椎体 | 既存 |

**plane 単位の教師項は無い**（§1.4 で撤回）。人手が 1 と付けた領域について
「どの面か」は不明のままなので、面方向は Stage3 と同じく MIL 集約に委ねる。
**陽性領域の全 plane を 1 とするのは誤り**であり、これは 07-28 Codex の指摘どおり。

弱陽性 1,073 bag は `max_r q[r] = 1` の制約のみ。これは Stage3 の
`normalized_smoothmax` による椎体ロジット生成が既に体現しているので、追加実装は不要。

**教師の総量（これが Stage4 の実体）**

| 種別 | bag 数 | 教師内容 |
|---|---|---|
| 強陽性（アノテ済み）| **216** | 4領域の完全な 0/1 |
| 弱陽性 | 1,073 | `max_r q[r] = 1` のみ |
| 強陰性（陰性椎体）| 9,656 | 全領域 0（人手不要） |

強陽性と強陰性の比が **1 : 45**。クラス重みとサンプリング比の設計がそのまま効く。

### 3.4 実装前に必ず直すもの

1. **左右反転時の領域ラベル入れ替え**
   `train_models/stage2/src/dataset.py` の `_augment_volume` は水平flip時に
   `remap_regions_after_horizontal_flip` で**マスクだけ**を入れ替えている（現行コードで確認）。
   Stage4 は領域ラベルを持つので、**`label[R2] ↔ label[R3]` を同時に swap** しないと
   教師とマスクが左右逆になる。vertical flip / transpose は領域IDの意味を変えないので対処不要。

2. **augmentation を局在向けに作り直す**
   Stage1 の vertical flip / ±45°回転 / plane permutation は局在には強すぎる
   （解剖学的に不可能な配置を学ばせる）。Stage4 用に別 config を作る。

3. **Stage1 の基準値を blind 上で取り直す（推論のみ。再学習はしない → §6.1）**
   既存 5-fold checkpoint (`v1_parity`) は `fracture_dataset` 上で学習されているが、
   bbox強制は 3.2% の bag の 15面中1面が 0.4mm 動くだけなので再学習は割に合わない。
   非劣性ガードレールに使う基準 AUROC を得るため、**blind データに推論だけかける**。

---

## 4. 残り163件のアノテーションをどう配るか

### 4.1 【訂正】locked test の29件を先に埋める

当初案は「test の29件は後回し、CV に入れるほうが得」としたが、**逆だった**。標準誤差の縮み方は

| 29件をどこに入れるか | 変化 | SE 比 | 縮小率 |
|---|---|---|---|
| **locked test** | 52 → 81 | √(52/81) = 0.801 | **約20%** |
| CV pool | 216 → 245 | √(216/245) = 0.939 | 約6% |

小さい集合に足すほうが限界効用が大きい。**推奨順序: ① test 残29件 → ② CV 残134件 → ③ 全部揃ってから最終学習/OOF。**

ただし81件まで増えても、比率が変わらなければ test の期待陽性数は R1≈12 / R2≈23 / R3≈14 / R4≈50 で、
§2.8 の「陽性 study 20 未満は confirmatory 不可」を R1/R3 は依然として満たさない。
**test を埋めても領域別の主張ができるようにはならない。** macro での方向確認が上限。

### 4.2 【訂正】許される順序と禁止される順序

当初案の「R2（右椎間孔）を優先」は**撤回する**。**アノテーションする前にその bag が R2 かどうかは分からない**ので、
「R2 を優先」は実行不可能か、さもなければ bbox の見た目から領域を推測して選ぶことになり、
それはラベルと相関した選択そのものになる。

| | 内容 |
|---|---|
| **許される** | fold、study ID のハッシュ、現在の fold サンプル数など**ラベルにもモデル出力にも無関係な情報**による順序 |
| | fold 内で study を seed 42 の固定乱数順に並べ、選んだ study 内の全候補 bag をまとめて付ける |
| | 「CV の残り134件を全部終わらせる」目的で fold1 から着手すること |
| **禁止** | 既知・推定の領域（特に「R2 らしさ」）による優先 |
| | モデルスコア・不確実性・モデル間不一致・誤分類による優先 |
| | bbox の位置や形状など、領域と相関しうる特徴による優先 |
| | AP が低かった fold を、結果を見た**後で**追加アノテーションすること |
| | AP が目標に達した時点でアノテーションを止めること |
| | アノテーション後に「曖昧だから」と難例を除外すること |

**途中で止める可能性があるなら**、fold1 集中は危険。残り134件を fold 比例 `[23,31,33,27,20]` に配分し、
固定乱数で抽出する。不均等確率で抽出したなら inverse-probability-weighted AP が必要になる。

### 4.3 アノテーションを追加した後の再実行ルール

| 新ラベルの用途 | 必要な再実行 |
|---|---|
| **評価だけに使う** | モデルの再学習は不要。ただし**新ラベルを見る前に全候補の OOF スコアを凍結**しておくこと。指標と bootstrap だけ再計算 |
| **Stage4 の学習にも使う** | **5つの outer モデルを全部ゼロから再実行**。新 bag は1 fold では評価例、他4 fold では学習例になるため |

**216ラベル時点の OOF 予測と、350ラベル時点の OOF 予測を混ぜてはいけない。**

最善は「アノテーションを完了 → label manifest を hash → 最終 CV を一度だけ走らせる」。

### 4.4 重複アノテーション（提案・要判断）

§1.3 のとおり **アノテーション再現性のデータは現状ゼロ**。ランダムに選んだ 30〜40 bag を
意図的に2回付ければ noise ceiling が測れる。これがないと「モデルの誤り」と「ラベルの揺れ」を区別できない。
ただし §4.1 の優先順位（test → CV）とは予算が競合するので、実施するかは判断が要る。

---

## 5. 主張できること・できないこと

**言える**:
> RSNA の bbox が提供された、既知の fracture-positive vertebrae において、
> 4つの解剖学的領域ラベルを bag 単位スコアで順位付け・識別する性能。

**言えない**（Codex が列挙したもの）:
- 全 vertebra に対する**無条件の**骨折局在性能
- fracture-negative を含む **screening 性能**
- bbox が付かなかった／見逃された／occult な骨折への性能
- **pixel / voxel / bbox レベルの空間精度**（4領域の bag ラベルしか無い）
- study レベルの診断性能
- 外部施設・外部データセットへの汎化
- 領域確率の**較正**（calibration）
- 9,656 の陰性を含む一般母集団での **PPV**

特に注意: positive-only 評価における「領域が陰性」は
**「骨折はあるが、その領域ではない bag」**であって、骨折のない普通の椎体ではない。

AP は class prevalence に依存するので、対象集団を指標名に含めることが必須（§2.2）。

---

## 6. 実行順序と kill 基準

| # | 作業 | 段階 | 進む条件 |
|---|---|---|---|
| 0 | protocol 凍結（本書の承認）| — | — |
| 1 | Stage1 既存checkpointを blind で**推論評価のみ**（再学習しない、§6.1）+ metadata-only shortcut baseline | 準備 | metadata-only の macro AP **<0.45** |
| 2 | frozen-feature arena（E/A/B0/B1 × 5 seed）| **exploratory** | A/B が metadata-only を **+0.08 以上**超え、SideAcc **>0.60** |
| 3 | supervision regime 比較（strong-only / weak-only / weak→strong / joint）| **exploratory** | |
| 4 | partial unfreeze | **exploratory** | 改善 **+0.02 以上**、なければ frozen 確定 |
| 5 | **単一モデルを凍結**（config・seed・manifest hash を固定）| 境界 | — |
| 6 | nested 手順（§2.4）で refit → OOF を一度だけ計算 | **confirmatory** | — |
| 7 | アノテーション完了後、locked test を**一度だけ**開く | **confirmatory** | §2.10 の3条件 |

§1.4 で z局在の手順を落としたので、07-28 案の順序に戻っている。
**人手アノテーションを増やす以外に検出力を増やす手段は無い**、というのが本設計の現状。

### 6.1 【訂正】Stage1 は再学習しない。推論評価だけする

当初案は「Stage1 を blind データで 5-fold 取り直す」だったが、**ユーザー指摘により撤回**した。
影響の大きさを実測したところ、再学習に見合わない。

**bbox強制plane の実際の規模（`processing_metadata` 全数集計）**

| | 値 |
|---|---|
| bbox強制planeを持つ bag | **431 / 13,432 = 3.2%** |
| そのbagでの強制plane数（15面中）| 平均 1.22 / **中央値 1** / 最大 4 |
| 強制plane数の分布 | 1面:352 / 2面:67 / 3面:9 / 4面:3 |
| 陰性椎体で強制されたもの | **全fold 0件** |
| 陽性椎体のうち強制ありの割合 | 31〜38%（fold間でほぼ均一）|

さらに 07-28 の実測で、その1面を bbox盲で選び直したときの位置ずれは
**中央値 0.39mm**（plane間隔 1.61mm、各planeは ±1.25mm を撮像）。
つまり **15面のうち1面が 0.4mm 動くだけ**の変化が、bag の 3.2% に起きている。

**Stage1 の AUROC がこれで動くとは考えにくい。** 5-fold の再学習は割に合わない。

**代わりにやること**: 既存の `v1_parity` 5-fold checkpoint を
`fracture_dataset_blind` の各 fold val に**推論だけかける**（forward のみ、学習なし）。これで

- 非劣性ガードレール（§2.3）に使う **blind データ上の基準 AUROC** が手に入る
- **リークが実際いくらの価値だったか**が直接測れる（leaky 0.921 との差）

が両方得られる。**差が 0.01 AUROC 未満なら既存 checkpoint をそのまま
Stage4 の warm-start / 凍結特徴抽出器として使う。** 0.01 以上開いたときだけ再学習を検討する。

**fold 対応が保たれる根拠**: §2.1 のとおり分割は blind でも 1 bag も動かない。
また fold-k の checkpoint は fold-k の val study を最初から学習していないので、
**ラベルのリークは元々存在しない**。bbox強制は入力サンプリング側の偏りであって、
fold 間の情報漏洩ではない。

**ただし Stage4 では blind データを使い続ける。** 3.2% という全体では小さい数字が、
**アノテーション済み 268 bag では 100%** になる（07-28 実測）。
局在を評価したい bag だけが病変位置由来のサンプリングを受けている状態は、
Stage1 の分類性能には効かなくても **Stage4 の局在指標には直接効く**。
`fracture_dataset_blind` を作った価値はここにあり、そこは変えない。

手順5 の**凍結の線引きが最も重要**。ここより前で見た領域指標はすべて exploratory であり、
論文の主指標として報告できるのは手順6 の1回だけ。手順2〜4 を回した後に
「やっぱりこう変えよう」と戻るのは自由だが、戻るたびに手順6 のやり直しになる。

---

## 7. 未確定（ユーザー判断が要るもの）

1. 重複アノテーション 30〜40 bag に予算を割くか（§4.4）。割くなら test 完成より先か後か
2. Stage4 のコードを `train_models/stage4/` として新設するか、Stage3 に config フラグで載せるか
   （arena 段階は head だけなので新設が素直。end-to-end は Stage3 の派生になる）
3. nested inner-CV（§2.4）は学習コストが5倍になる。frozen-feature arena では確実に払えるが、
   end-to-end の confirmatory 実行でも払うか、それとも単純 holdout で妥協して
   「checkpoint 選択のバイアスが残る」と明記するか

**決着済みとして未確定から外したもの**

| 項目 | 決着 |
|---|---|
| OOF プールを rank 正規化にするか生確率にするか | **生確率が primary**（Codex 相談、§2.6）|
| bbox の z 範囲を教師／評価GTに使うか | **一切使わない**（ユーザー指示、§1.4）|
