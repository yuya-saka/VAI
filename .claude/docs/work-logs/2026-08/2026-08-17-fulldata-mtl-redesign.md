# 2026-08-17: 設計転換 — 全データMTL + missing-label masking

## 決定（ユーザー指示）

旧4アーム計画（Baseline 1/2 + 提案A/B、matched 2,655 bag学習）を廃止し、
全13,928 bagを使う hard parameter sharing型Multi-Task Learningへ全面改訂した。

**転換理由**: 一部症例だけでの学習は過学習する。matched fold 0診断runの実測
（best val AUROC 0.738、epoch 21以降val BCE悪化）と整合。

## 新設計の骨子

- **アーム**:
  - Baseline 0: CT + whole mask → CNN+LSTM → p_whole（4領域情報なし）
  - Baseline 1: 6ch early fusion → shared CNN+LSTM → whole head(1) + region head(4)
  - Proposed: shared CNN → mask-guided 4 branches（各branchに対応maskを注入、各LSTM）
- **学習方法（全アーム固定）**: `L = L_whole + λ·m·L_region`。
  領域ラベル（268 bag、全体の約2%）のないbagはregion lossをマスクする。
  missing labelを0扱いしない。two-stream samplingでannotated bagをバッチへ混入
- **whole出力の2方式**（Baseline 1 / Proposedの比較軸）:
  - 方式A: region aggregation（max / noisy-OR）— whole判定と領域判定の矛盾を構造的に防ぐ
  - 方式B: 独立whole head — 13,928 bagの教師を最大活用できるが矛盾出力があり得る
- **検証項目3点**: ①4領域mask入力の効果（B0 vs B1）、
  ②mask-出力対応branchの効果（B1 vs Proposed）、③方式A vs B

## 廃止

- Baseline 1 matched設定・matched cohort学習（凍結CSVは保持、学習不使用）
- Baseline 2（4独立モデル）、提案A（pseudo-label）、提案B（弱教師）
- 2-stem（image/mask stem）→ 単純early fusionへ

## 存続する既決事項

15面固定 / 完備13,928 bag母集団 / folds.csv凍結 / OR集約 / flip・transposeなし /
回転±40° constant fill / 15面broadcast + mean-sigmoid / val AUROC early stopping /
held-out testなし / full backbone V2-S / mask-average・hard pooling全アーム禁止（PI 2026-08-04）

## ユーザー決定（同日・第2ラウンド）

- **論理的0教師（椎体陰性bagの4領域=0）は使わない**。region lossは268 bagのみ。
  `common/losses.py`の既存の論理的0適用は実装時に削除
- **方式Aの集約関数（max / noisy-OR）はアブレーション**で両方比較
- **Proposedのmask注入はPMGAN方式**（Zhang et al., "Part-Aware Mask-Guided
  Attention for Thorax Disease Classification", Entropy 2021 =
  `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`）。
  精読結果: 共有CNN後に領域ごとのMask-Guided Attention (MA)を置き、MAの
  spatial attention mapを対応臓器maskへRMSE損失L_attで回帰（maskは学習時のみ、
  inference追加計算なし）。特徴再重み付けは残差形式 `(1+m)⊗f`。
  PMGAN自体は global branch + 局所branch群のMax集約 + 独立BCE 2本
  （L = L_ce^global + α·L_ce^local + L_att、α=0.5、L_att = L_att^0 + β·Σ L_att^b）
  で、本研究の方式A/B比較と対応が良い。hard pooling禁止と整合
- **pos_weight=2.0は全アーム固定**
- **fold分割は凍結folds.csvを全アームで再利用**（患者単位層別5-fold）。
  annotated bag・R1〜R4がfold間層別済みで、訓練側約214 / 検証側53〜56の
  annotated bagが各foldに確保される。提案A廃止でfold内teacher制約は消滅、
  nested CVは不要

## 残る未確定（実装前に決定、Codex相談候補）

1. λの値（loss-balance実測で決める）
2. two-stream samplingの混合比率
3. Proposedのattention制約損失重みβ（loss-balance実測）

---

# fold設計とtest分離のレビュー（同日・第3ラウンド）

ユーザー依頼「fold分けが本当にこれでいいか、testデータも分ける必要がないか」への調査。
**以下はすべてCodex推奨＋Claude実測であり、ユーザー承認前の検討段階。設計確定ではない。**
Codex全文: `.claude/docs/codex/20260817-fold-and-testset-design.md`

## Claudeの実測（凍結manifestから算出、検証済み）

### fold自体は健全
- 患者リークなし（複数foldにまたがるstudy 0件）
- fold別prevalence 10.08〜10.13%（ほぼ同一）、bags 2,784〜2,787、studies 402×5
- level別bag数 399〜402で均等、annotated bag 53〜56、R1 15-16 / R2 11-12 / R3 14-15 / R4 31-32
- annotated studyに属するbagの割合も 7.5〜8.2% で均衡
- R2 xor R3 のみ fold別 22/21/21/16/15 と不均衡。ただしpooled OOF評価なので影響しない

### アノテーション160 studyは陽性患者のランダム標本ではない（決定的）
| 指標 | annotated study | 非annotated study | 全体 |
|---|---|---|---|
| 椎体骨折 prevalence | **31.50%** | **8.24%** | 10.09% |
| 陽性患者あたり骨折椎体数 | 2.19（中央値2） | 1.34（中央値1） | — |
| 単一椎体骨折の割合 | 58/160 = 36% | 565/787 = 72% | — |

陽性椎体の annotated 比率も level で大きく偏る: C3 42.5% / C4 40.2% ↔ C7 9.4% / C2 13.2%。
陽性椎体の 268/1,406 = **19.1%** が annotated study に属する。

### test分離の代償（実測）
- ランダム患者20% test → region CV母数 268→**214**（R1 78→62 / R2 59→**47** / R3 72→58 / R4 158→126、SideAcc 95→76）
- 非annotated studyのみからtest → 268は温存できるが、上表の通り case-mix が別集団

## Codexの結論

1. **fold再生成は不要**。凍結5-foldを全アームでそのまま再利用してよい。
   変えるべきなのは分割ではなく fold の**役割**（outer = 評価専用）
2. **固定test setの切り出しは不要**（nested evaluationを入れるなら）。
   ただし「非annotated患者のみからtest」案は confirmatory には**採用不可**。
   annotated患者がtestに入る確率が0なのでpositivity条件を満たさず、IPW/標準化で補正できない。
   陽性の19.1%除外により、仮定なしのAUC boundsは幅0.191以上。
   → domain-shift のサブグループ解析としてのみ有用。外部cohortは将来課題
3. **outer foldでのearly stoppingを中止**。cyclic single-inner-fold を推奨:
   outer=k、inner=(k+1)%5、3 foldで初期fit → innerでepoch選択 → outer以外の4 foldで
   固定epoch再fit → outerを1回だけ推論。学習コスト約2倍。
   選択metricは全アーム共通で inner の椎体AUROC（innerのR2陽性11-12でAP最大化は不安定なため）
4. **primary contrastを1本だけ事前登録**。推奨は「Proposed method A=max、β>0 vs 同一構成でβ=0」、
   endpointは paired OOF 椎体AUROC差。7構成中の最良を選んで同じOOF差をconfirmatoryと呼ばない
5. **現行アームでは検証項目3点を分離できない**。B1 vs B0はmask追加だけでなくregion head・
   region supervision・sampler・パラメータ数が同時に変わる。
   → region maskチャンネルだけを除いた **no-region-mask MTL control** の追加が必要
6. **two-stream samplingの交絡**: annotated streamでも L_whole を平均すると特定の陽性bagが
   whole taskで過重評価される。natural stream = L_whole、annotated stream = L_region + L_att のみ、
   に分離すべき。実効region gradient exposure は λ×混合比 として定義・記録する
7. **SideAcc**: foldは変えない。fold別balanced accuracyの単純平均をやめてpooled 95件で算出。
   判定は事前固定の `argmax(p_R2, p_R3)`、OOF上でthreshold調整しない。
   0.65ゲートの意味（点推定 vs 片側95%下限）を先に固定する。
   Codexの内訳主張（both=18 / R2-only=41 / R3-only=54）は**実データと完全一致を確認済み**

## Claudeが追加で指摘する論点

- **Baseline 0 は region logit を出さない**ため、検証項目①「4領域maskを入力すること自体に
  意味があるか」を**領域エンドポイントでは検定できない**（椎体エンドポイントのみ）。
  Codexの言う no-region-mask MTL control を追加すれば①が領域側でも検定可能になる
- しかもそれは [[project-n268-statistical-power]] が「唯一答えが出る」とした
  **同一backbone・入力チャンネルのみ差**の比較（rho≈0.9 → MDE 0.027）に一致する。
  現行のB1 vs Proposed（構造差、rho 0.5-0.7 → MDE 0.047-0.059）は母数不足側
- 過去にfold 0の結果を見てscheduleを改訂した経緯があるため、Codexは
  これを「pilotによる設計変更」として登録し、以後outer結果を設計変更に使わないよう求めている

## ユーザー決定（同日・第4ラウンド、上記レビューを受けて）

**承認された4点（確定）:**
1. **fold分割は現状維持**。凍結`folds.csv`を全アームでそのまま使用、再生成しない
2. **held-out test setは切り出さない**。「非annotated studyのみからtest」案も不採用
3. **outer foldを評価専用にし、cyclic single-inner-foldで選択**
   （outer=k / inner=(k+1)%5 / 3 foldでStage 1 epoch選択 / 4 foldでStage 2固定epoch再fit /
   outer推論1回）。選択metricは全アーム共通でinnerの椎体AUROC。学習コスト1アーム10 run
4. **Control（no-region-mask MTL）アームを追加**。入力6chはBaseline 0と同一、
   head・region loss・sampler・学習予算はBaseline 1と同一

**実装上の帰結（登録済み）:**
- `ReduceLROnPlateau`はStage 2でouterを監視できないため、Stage 1のLR軌跡（epoch→LR）を
  記録してStage 2で再生する
- Stage 1は3 fold（8,355〜8,359 bag）、Stage 2は4 fold（11,141〜11,144 bag）で学習するため、
  同一epoch数でもStage 2のoptimizer step数が約1.33倍になる。これは仕様として登録
- 既存`baseline1/`はouter early stopping前提なので、nested選択への改修が必要
- nested選択のロジックは全アーム共通なので共通基盤へ実装する

**残る未決4点（学習開始前に確定）:**
1. primary contrastの事前登録（1本）
2. two-stream samplingの損失分離（annotated streamに`L_whole`を含めるか）
3. SideAcc集計とゲート定義（点推定 vs 片側95%下限）
4. λ / β / 混合比の決定規則

## 更新したファイル（第3・第4ラウンド分）

- `memo/計画書/提案手法.md` — Controlアームを第3節へ追加、第5節比較表を4アームへ、
  第6節検証項目を4点へ再構成、第7節を「評価プロトコル」として全面書き換え
  （foldの役割・nested選択・test非分離の根拠・エンドポイント・未確定）
- `fracture_detection/PROGRESS.md` — 全体像表にControl追加とはしご図、
  確定済みの前提を更新（outer評価専用 / nested選択 / test非分離の根拠）、
  未決事項を4件へ縮小、プロジェクト一覧と次タスクを更新
- `.claude/docs/DESIGN.md` — Activeセクションへ評価プロトコル改訂を追記、changelog 1行追加
- メモリ `project_annotation_selection_bias.md` を新規作成

## Codex: 未決4点への回答（2026-08-18）— **ユーザー承認前**

全文 `.claude/docs/codex/20260818-remaining-four-decisions.md`。以下は要約。**確定事項ではない。**

### Q1: primary contrast
- **H1（primary）: `AUROC(Baseline 1–B) > AUROC(Control–B)`**、13,928 bagのpaired pooled-OOF椎体AUROC
- **H2（key-secondary）: `AUROC(Proposed–max, β>0) > AUROC(Proposed–max, β=0)`**
- **固定順序 H1 → H2**。H1が有意なときだけH2を確証的に検定。H1が落ちたらH2は探索的
- 判定は両仮説とも「patient-cluster bootstrapによるpaired差の95%両側CI下限 > 0」
- Control–B vs Baseline 1–B の**領域AP差はkey-secondary family**として別扱い（rho≈0.9を活用）
- ⚠️ 領域APをprimaryにしない理由が重要: **macro化を廃止した以上4仮説familyになり、
  既存のMDEはmacro-APの値なので各regionの検出力を保証しない**

### Q2: two-stream損失構成
- 各stepで `W_t`（全bagからのnatural stream、batch `B_W`）と `A_t`（annotated、**1 bag/step**）
- `L_B0 = mean_{W_t} L_whole`
- `L_Control/B1 = mean_{W_t} L_whole + λ·L_region(A_t)`
- `L_Proposed = mean_{W_t} L_whole + λ·L_region(A_t) + β·mean_{W_t} L_att`
- **`A_t` は `L_region` にのみ寄与**。`L_whole`にも`L_att`にも寄与させない
- **Baseline 0も同一natural sampler・同一 `W_t`・同一optimizer step数**を使う
  （annotated streamのforwardをしないだけ）→ whole taskの分布・勾配が全アーム完全一致
- epoch長はnatural streamの一巡で定義。`L_whole`は常に `B_W` でmean（`B_W+1`で割らない）
- 全アームで同じnatural-stream seed/orderを使う
- ログ: `region_optimizer_steps` / `region_passes` / annotated bag別visit回数のmin/median/max /
  shared CNN block上の `‖∇L_whole‖`・`‖λ∇L_region‖`・`‖β∇L_att‖` とその比

### Q3: 領域別APの床ゲート
- **対象は Proposed–B, β>0 のみ**を事前指定（勝者選択をしない）
- **floorはcross-fitted OOF**: outer fold `k` ごとに、modelが使うのと同じ3 training foldsだけから
  `p̂(r,l) = (x_{r,l} + 0.5) / (n_l + 1)`（Jeffreys平滑化）を作り、outer foldのbagへlevel別に割当て、
  5 foldをpool。**全268からfloorを作り同じ268で検定するのはlabel leakage**と明確に否定
- **評価母集団は268 annotated陽性のみ**。whole-negative bagは1件も足さない
  （region rについてlabel 0のannotated bagが正当なnegative）
- R1〜R4の4検定に**Holm補正**。10,000 patient-cluster bootstrap、model/floorに同一resample、
  bootstrap内で再fitしない、floorを固定真値にしない
- このfloor familyは**独立したkey-secondary family**でQ1のfixed sequenceには入れない
- **SideAcc代替の記述的感度解析2件**（新metricなし）:
  - level removal check: 各level内でscoreをpercentile rankへ変換して4つのAPを再計算
  - **R2/R3 swap negative control**: R2ラベルをR3 scoreで評価し、正しい割当のAPが
    swapped APを上回るかをpaired bootstrap CIで示す

### Q4: λ / β / 混合比
- **grid searchもinner CVも行わない**。outer-training dataだけの固定初期勾配校正（GradNorm簡略版）
- 各outer foldで: 共通seedで初期化 → optimizer更新前に3 training foldsから決定論的に
  **64 calibration batches** → eval mode（BN統計もparameterも更新しない）→
  最後のshared CNN blockで損失別のgradient L2 normを計算（eps `1e-12`）
- `λ_k = clip_[1e-2, 1e2]( 0.5 · exp( median_b log( (g_whole,b + ε) / (g_region,b + ε) ) ) )`
  （referenceは **Baseline 1–B**）
- `β_k` も同型（referenceは **Proposed–B**、`g_att` を使用）
- **同一の λ_k を、そのouter foldの全アーム・全構成へ適用**。
  **arm別チューニングは禁止**（ControlとBaseline 1でλが違えばmask入力だけの比較でなくなる）。
  fold間でλが違うのは事前指定アルゴリズムの一部なので問題なし
- **混合比は調整しない。global optimizer step当たりannotated bag 1件**
- annotated samplerはbag単位のshuffle-without-replacement cycle（全件使い切って再shuffle）
- 非有限gradientが1件でも出たらrunを開始せずimplementation errorで停止
- clipping到達はログするが、結果を見て範囲や重みを変えない
- **追加full runは0**。5 fold × (64 + 64) = 640 calibration batchesが増えるだけ

### Q5: 構成の削減 — **55 run → 6構成 / 30 run**

| # | 構成 |
|---|---|
| 1 | Baseline 0（independent whole head） |
| 2 | Control–B |
| 3 | Baseline 1–B |
| 4 | Proposed–B, β>0 |
| 5 | Proposed–max, β>0 |
| 6 | Proposed–max, β=0 |

- はしご: B0→Control-B（MTL）/ Control-B→B1-B（region mask入力）/
  B1-B→Proposed-B（明示的対応）/ Proposed-B→Proposed-max（whole出力方式）/
  Proposed-max β>0→β=0（attention回帰の新規性）
- 削除: Control/Baseline 1のmax・noisy-OR、Proposedのnoisy-OR、Proposed–B β=0
- **noisy-ORは全アームから削除**。追加runを消費するうえ、whole lossをregion logitsへ直接流すため
  単なる推論時集約ではなく**弱いregion supervision経路まで変えてしまう**
- Controlはmethod Bのみ。method Bははしご全アームで実行

### Q6: 残る制約（結論は無効化しないが表現制限が必須）
1. 60% training下の順位を通常訓練（80%/100%）の順位へ一般化できない。結論は**登録済み60% protocol下の相対比較**
2. 268は非ランダム選択なので、領域AP・floor結果は**annotated-positive populationに条件づけた内部妥当性**のみ
3. outer fold当たり1 seedのため、patient bootstrapは**training stochasticityを含まない**。
   「model-training randomnessを含むCI」と書いてはいけない
4. SideAcc削除後、R2/R3 APが高くても「症例ごとの左右を正しく選んだ」証明にはならない。
   主張可能なのは「R2とR3を個別labelとしてrankingできた」まで。
   **swap negative controlが失敗したらlaterality主張は無効**
5. floorは必ずcross-fitted OOF形式にする（全268でfloorを作るのはleakage）
6. **11構成を見てから6構成へ絞る行為自体がouter OOFをtuning setに変える**。
   6構成・weights・testing order・code hashを凍結してからouter inferenceを開始すること

### Claudeの注記
- Codexが指摘した「macro廃止によりMDEが失効」は重要。既存MDEはmacro-APの値で、
  R1〜R4個別testの検出力はmacroより悪い（macroは4つの相関推定を平均して分散を下げるため）
- 下記の床再計算（fold-out prior）はCodexのcross-fitted OOF仕様に近いが、
  4 fold leave-one-out・Jeffreys平滑化なしなので**Codex仕様とは別物**。要再実装

## 床（level-only floor）の補正ラベルでの再計算（2026-08-18、暫定測定）

未決事項3に直結するため先行して実測した。**まだ凍結していない**（floorの定義自体が
Codex Q3の回答待ちのため）。

### 補正ラベルでの母数（凍結manifestから）

268 bag / 160 study。R1 78（prevalence 0.2910）/ R2 59（0.2201）/ R3 72（0.2687）/
R4 158（0.5896）。level分布 C1 22 / C2 37 / C3 31 / C4 43 / C5 46 / C6 53 / C7 36。

### level-only floor（n_neg=0、268陽性のみ）

| | no-skill | in-sample prior | fold-out prior | 旧記録値 |
|---|---|---|---|---|
| R1 体部 | 0.2910 | **0.5303** | 0.5019 | 0.59 |
| R2 右横突孔 | 0.2201 | **0.3243** | 0.2707 | 0.37 |
| R3 左横突孔 | 0.2687 | **0.4298** | 0.4008 | 0.45 |
| R4 後方要素 | 0.5896 | **0.7259** | 0.7062 | 0.72 |
| macro | 0.3424 | 0.5026 | 0.4699 | 0.451 |

**旧記録値と一致しない。** 差はラベル補正（R1 77→78 / R3 71→72 / R4 155→158）だけでは
説明できない大きさで、旧値の算出条件（prior推定法・AP実装のtie処理）が不明。
**旧値は使わず、定義を決めたうえで算出し直した値を凍結する。**

### 決めるべき点（Codex Q3へ投げ済み）

1. **in-sample prior か fold-out prior か**。差は0.02〜0.05。
   in-sampleは「近道の最良値」なのでゲートとしては保守的（超えにくい）。
   fold-outは実際にlevel-onlyモデルが達成する値
2. **tie処理**。level-only scoreは7水準しかないため大量の同点が発生し、
   APの値がtie処理の規約に強く依存する。AP実装を固定して明記する必要がある

### 陰性混入は不可（実測で確認）

椎体陰性12,522 bagは論理的に4領域とも陰性なので評価に混ぜられるが、混ぜるとAPが機械的に潰れる。

| n_neg | 総数 | R1 | R2 | R3 | R4 | macro |
|---|---|---|---|---|---|---|
| 0 | 268 | 0.5303 | 0.3243 | 0.4298 | 0.7259 | 0.5026 |
| 268 | 536 | 0.2563 | 0.1648 | 0.2145 | 0.3728 | 0.2521 |
| 1,072 | 1,340 | 0.1079 | 0.0699 | 0.0908 | 0.1472 | 0.1039 |
| 12,522 | 12,790 | 0.0107 | 0.0069 | 0.0090 | 0.0155 | 0.0105 |

近道が弱くなったのではなく有病率でAPが潰れているだけなので、異なるn_negの間でAPを比較しても
無意味。**局在の評価は n_neg=0（268陽性のみ）に固定する**のが妥当（memory
[[project-n268-statistical-power]] の旧知見とも一致）。最終判断はQ3の回答を待つ。

⚠️ 再現可能な算出スクリプトは未作成。floorの定義が確定してから実装する。

## nested選択を5 run版に確定（同日・第7ラウンド、ユーザー決定）

「1構成10 run」の内訳を説明した際、run数と計算量を混同していた誤りが判明し、
あわせてプロトコルを見直した。

**採用形（5 run）:**
```text
outer = k、inner = (k+1) mod 5
  学習: 残り3 fold（8,355〜8,359 bag / annotated 159〜162）でinnerを監視しearly stopping
  推論: そのモデルでouterを1回だけ → pooled OOF
```

- **Stage 2の再fit（4 foldで固定epoch再学習）は不採用**。計算資源とのトレードオフ
- **実装が単純化**: `ReduceLROnPlateau`はinnerを監視してそのまま動かせる。
  LR軌跡の記録・再生は不要、「Stage 2のstep数1.33倍」の登録仕様も消滅
- 計算量は旧方式（4 fold学習＋outer early stopping）の**0.75倍**
- ⚠️ 登録すべき限界: 全報告モデルが3 fold（60%）学習で、領域教師が各fold 215→約160 bagへ25%減。
  268が既に母数不足であることと合わせて登録。**絶対性能の主張はしない**。
  handicapは全アームに等しくかかるためアーム間比較の妥当性は保たれる

**訂正:** 「学習コスト約2倍」という以前の記述は run数と計算量の混同だった。
fold-epoch換算では 10 run版でも `0.75 + E_best/E_stop` 倍で、
matched fold 0の実測（early stop epoch 22 / best epoch 7）なら約1.07倍、
`E_best/E_stop = 0.7` でも1.45倍。2倍にはならない。
DESIGN.md・PROGRESS.md・提案手法.md・メモリの該当記述を全て訂正済み。

## SideAcc廃止（同日・第6ラウンド、ユーザー決定）

- **SideAcc（左右balanced accuracy、R2 xor R3の95椎体）を評価指標から削除**
- 局在の評価は **R1〜R4それぞれのAP** で行う（macro平均へ潰さない）。
  左右の判別能も R2 / R3 個別のAPで見る
- 未決事項の「SideAcc集計と0.65ゲート定義」は消滅。代わりに
  「領域別APの床ゲートと多重性補正（4検定Holm、対象は事前選択した1モデルのみ）」を登録
- **登録すべき帰結**: SideAccは近道に唯一耐性のあるエンドポイントだった
  （level-onlyのside accuracyは0.511＝偶然、一方でlevel-onlyの領域macro-APは0.451に達する）。
  廃止により、レベル事前分布による近道への耐性は**領域別APの床ゲートだけ**が担う。
  したがって床の補正ラベル再計算は「あとでやる整理」ではなく**事前登録の前提条件**
- flip/transpose不使用の方針は不変（領域別AP評価にR2/R3のラベル・mask対応が必要なため）
- コード: `common/metrics.py::side_balanced_accuracy`（140行目）と
  `compute_region_metrics`の`side_balanced_accuracy`キー、
  `common/tests/test_metrics.py`の該当assertをendpointから外す。
  実装整理は`common/`改修（次タスク2）へまとめる

## 進捗台帳の分離（同日・第5ラウンド）

ユーザー指示により、旧4アーム計画の記録を `fracture_detection/PROGRESS_ARCHIVE_4arm.md` へ分離。
現行計画とやっていることが根本的に異なるため。

- **archive側**: 旧4アーム全体像、失効した前提（matched cohort・outer early stopping・
  matched backbone区分・2-stem・提案A/B）、進捗ログ 2026-08-07〜2026-08-14、旧「次のタスク」
- **PROGRESS.md側**: 現行計画のみ。確定済みの前提を
  「入力・データ / モデルと損失 / 評価プロトコル / 運用」に再編し、
  旧計画から引き継ぐ成果物を「既存基盤」節に明示。進捗ログは2026-08-17以降
- 検証: 37個のキー文字列（旧SHA256、貪欲法バグ、epoch 51、`--full-auto`事件、
  床の数値、母数など）が両ファイルのいずれかに残存することを確認。**情報欠落なし**

## 更新したファイル

- `memo/計画書/提案手法.md` — 全面書き換え（旧計画との差分表を末尾に記載）
- `fracture_detection/PROGRESS.md` — 全体像・前提・プロジェクト一覧・次タスクを改訂
- `.claude/docs/DESIGN.md` — Activeセクションに2026-08-17更新とchangelog追記

## 次回タスク

1. 未確定事項1-6の設計確定（Codex相談 → ユーザー承認）
2. `common/`へmasking損失とtwo-stream sampler追加
3. Baseline 0（=既存`baseline1/` full設定）の5-fold学習・OOF評価
4. Baseline 1 → Proposedの順で実装
5. 床・検出力の補正ラベル再計算（並行可）
