# fracture_detection 作業進捗

> 頸椎骨折の4領域検出研究（`memo/計画書/提案手法.md`）の実装ディレクトリ。
> 学習モデル・fold定義などは**プロジェクト単位のサブディレクトリ**に分けて作っていく。
> このファイルは各プロジェクトの状態を一覧する進捗台帳。詳細な経緯は
> `.claude/docs/work-logs/2026-08/` と `.claude/docs/DESIGN.md` を参照。
>
> **現行計画は2026-08-17の設計転換以降のもの。**
> それ以前の旧4アーム計画（Baseline 2 / 提案A / 提案B / matched学習）の記録は
> `PROGRESS_ARCHIVE_4arm.md` に分離した。やっていることが根本的に異なるため、
> 旧ファイルの設計判断・数値・タスクは現行計画には適用しない。
>
> 🔧 **2026-08-20時点: 正式校正・profile・凍結は完了。Baseline 0は5 fold完走。**
> **Baseline 1–Bは2GPUで正式学習中**（outer0/1完了、outer2進行中、outer3/4未着手）。
> Control–B / Proposed 3構成は未着手。frozen manifestは科学的に意味のある設定
> （model/augmentation/data.random_seed/学習ハイパラ）のみを凍結し、実験名・GPU割当・
> fold範囲・W&B設定は凍結対象外（2026-08-20修正）。
> 実装経緯は `.claude/docs/work-logs/2026-08/2026-08-20-frozen-manifest-scope-fix-and-baseline1b-launch.md`、
> それ以前の実装だけなら
> `.claude/docs/work-logs/2026-08/2026-08-18-implementation-handoff.md`
> と `memo/計画書/提案手法.md` を読めば足りる（設計の経緯は読まなくてよい）。

---

## 全体像

**設計転換（2026-08-17）**: 一部症例だけでの学習（matched 2,655 bag）は過学習することが
fold 0診断runで実測されたため、**品質除外済み全13,432 bagを使う hard parameter sharing型MTL +
missing-label masking** へ切り替えた（詳細は `memo/計画書/提案手法.md`）。

| アーム | 入力 | 4領域の扱い | 出力 | 領域ラベル268の扱い |
|---|---|---|---|---|
| Baseline 0 | CT + whole mask（6ch） | なし（single task） | whole | 不使用 |
| **Control** | **CT + whole mask（6ch）** | **MTLのみ・領域maskを与えない** | 4 regions + whole | region loss教師（masking） |
| Baseline 1 | CT + whole + 4 masks（10ch） | 6ch Early Fusion（shared backbone + region head 4 logits） | 4 regions + whole | region loss教師（masking） |
| Proposed | CT + whole + 4 masks（10ch） | mask-guided 4 branches（PMGAN式attention制約） | 4 regions + whole | region loss教師（masking） |

**Controlは交絡分離用の対照アーム**（2026-08-17追加）。Baseline 0とBaseline 1は入力・
region head・region supervision・samplerが同時に変わるため、両者の差では
「4領域maskの効果」を主張できない。Controlを挟むと1段につき1要素だけが変わる:

```text
Baseline 0 → Control    差 = マルチタスク化の効果（入力は同一6ch）
Control    → Baseline 1 差 = 4領域mask入力の効果（入力チャンネル以外すべて同一）
```

Baseline 0はregion logitを出さないため領域エンドポイントで比較できないが、Controlは出せる。
Control vs Baseline 1 は rho≈0.9 / macro-AP MDE 0.027 で、268の母数で唯一検出力のある比較型
（Baseline 1 vs Proposed は構造差のため rho 0.5〜0.7 / MDE 0.047〜0.059）

* 学習は全アームとも **品質除外済みfull 13,432 bag**。損失は `L = L_whole + λ·m·L_region`
  （mは領域ラベル有無のマスク。missing labelを0扱いしない）
* 領域ラベル268 bagは全体の約2%のため **two-stream sampling** で各バッチに混入させる
* Control / Baseline 1 / Proposed は whole出力を **方式A（region aggregation）/
  方式B（独立whole head）** の2通りで比較する
* ⚠️ 名称注意: 旧Baseline 1（CT+whole mask）は**新Baseline 0**。
  実装は `baseline0/` へ移行済みで、full設定のみを受け付ける

## 確定済みの前提

### 入力・データ（旧計画から継続、変更不可）

- 入力は統合済み `data/rsna_data/fracture_dataset/`（2.5D、**15面固定**、全アーム共通・変更不可）
- **bag母集団はStage1と同じ品質除外を適用した13,432 bag / 2,009 study / 陽性1,332**。3ファイル完備13,928 bagから`excluded_studies.csv` / `excluded_levels.csv`と交差する496 bagを除外し、除外CSVのSHA256もmanifest metadataへ固定する。領域注釈268 bagは維持
- 領域ラベルCSVは同一椎体の全runを **OR集約**する。`folds/load_labels.py` が唯一の実装
- **領域ラベルは run をまたいだ OR 集約**。run = 同一椎体内で連続するbboxのかたまり＝別々の骨折部位
  （17椎体が複数run、うち6椎体は別部位が別領域に及ぶ）。アノテータ確認済み（2026-08-07）で各runのラベルは正しい。
  確定値 **268 bag / 160 study / R1 78 / R2 59 / R3 72 / R4 158**、複数領域陽性70、R2 xor R3 = 95
- **R2/R3 は横突孔**（椎骨動脈が通る孔）。アノテーションツールのUI文言が「椎間孔」と誤っていたが、
  ラベル自体は横突孔として判定されていることをアノテータが確認（2026-08-07）。文言のみ修正済み
- **R2/R3 の「右」「左」は画像基準**。class2は画像右（平均x=155、class3は66）＝患者の左。
  ラベル・マスク・クラス番号は相互整合しており学習/評価に影響なし。臨床的な左右の記述時のみ反転が必要
- **horizontal flipはlaterality-safe専用実装だけを`p=0.5`で使う**。CT・whole/region maskを同期反転し、
  R2/R3のmask値と教師を同時交換する。`A.HorizontalFlip`の直接使用は禁止
- **vertical flip / transposeは恒久禁止**。brightness、Affine（shift 0.3 / scale 0.7–1.3 /
  rotate ±45° / `BORDER_REFLECT_101`）、blur/noise、distortion、cutoutとnatural stream mixup
  `p=0.2`、`λ∼U(0,1)`はStage1へ揃える
- ステージングはfull学習用のinput-manifest SHA256単位共有`/dev/shm` cacheを使用

### モデルと損失（2026-08-17設計転換で確定）

- **学習は全アームとも品質除外済み full 13,432 bag**（一部症例のみでの学習は廃止）
- 「6ch入力」は実データでは各面 5CT ch + 5mask ch = 10ch。
  **単純early fusion（入力convでconcat）**とする（旧2-stem案は廃止）
- **損失は `L = L_whole + λ·m·L_region`**。領域ラベルのないbagはregion lossをマスクし、
  missing labelを0扱いしない。region lossは通常BCEの4領域平均
- **椎体陰性bagへの論理的0教師は使わない**。region lossは領域ラベルのある268 bagのみで計算する
- **two-stream sampling**: annotated 268 bagが約2%しかないため、
  Whole用/Detail用の2 streamで各バッチにannotated bagを一定割合混入させる
- **whole lossの`pos_weight=2.0`は全アームで固定**。Stage1と同じく陽性損失を2倍し、重み合計で正規化する
- **方式Aの集約関数は max のみ。noisy-ORは全アームから削除**（2026-08-18確定）
- **Proposedのmask注入はPMGAN方式のattention制約**（参考論文:
  `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`）。
  各領域branchのMask-Guided Attentionが出すspatial attention mapを対応領域maskへ
  RMSE損失で回帰させ、特徴は残差形式 `(1+m)⊗f` で再重み付けする。
  maskの直接乗算・poolingをしないためhard pooling禁止と整合。損失重みβはloss-balance実測
- **mask-average pooling / per-region hard pooling は全アーム禁止**（PI決定2026-08-04、継続有効）
- **bag確率は 15面 broadcast + 面ごとBCE + mean-sigmoid**（2026-08-11ユーザー決定。
  Codex推奨のbag-level log-mean-exp は却下）。対応する単一尤度が存在しない点は登録済み限界
- region headは面単位`[B,15,4]`。方式Aも面ごとにregion logitのmaxを取り、
  whole lossと評価の両方でmean-sigmoidを使う
- annotated streamは別forwardとし、BN moduleだけeval、natural backward後にannotated backward、
  optimizer stepは1回。augmentation / mixup / annotated forwardのRNGを分離してcheckpointする
- Proposedは`blocks[4]`で分岐し、各branchの`blocks[5]+conv_head+bn2`を独立複製する。
  `L_att`は14×14 spatial attention `s`へ回帰し、global項と追加SA moduleは使わない
- backboneは `tf_efficientnetv2_s`

### 評価プロトコル（2026-08-17確定）

- **fold分割は凍結済み `folds/outputs/folds.csv` を全アームでそのまま使用**（再生成しない）。
  監査済み: 患者リークなし、fold別prevalence 10.08〜10.13%、level別bag数399〜402、
  annotated bag 53〜56、R1〜R4も層別済み
- **outer foldは評価専用。モデル選択は cyclic single-inner-fold**
  （2026-08-11の「outer foldでval AUROC early stopping」は廃止）:
  outer=k / inner=(k+1)%5 / 残り3 fold（8,048〜8,074 bag / annotated 159〜162）で学習し
  innerでcheckpoint選択（early stopping）→ 事前指定したAUROC-best / PR-AUC-bestを
  outerへ各1回だけ適用しpooled OOFへ
- **1構成あたり5 run**（2026-08-17ユーザー決定）。
  Stage 2再fit（4 foldで固定epoch再学習）を行う10 run版は計算資源とのトレードオフで**不採用**
- checkpoint選択metricは全アーム共通で**innerの椎体AUROC**（innerのR2陽性11-12件では領域AP選択が不安定）
- early stoppingは**innerの椎体BCE**で判定し、15 epoch連続で改善しなければ停止する
- LRはRSNA Type1準拠の固定cosine（`2.3e-4`→`2.3e-5`、75 epoch単一周期）。innerはAUROC checkpoint選択とBCE early stoppingだけに使う
- 計算量は旧方式（4 fold学習＋outerでearly stopping）の**0.75倍**。fold数が4→3に減るだけ
- ⚠️ **登録すべき限界**: 報告する全モデルは3 fold（全データの60%）学習で、
  領域教師は各fold 215→約160 bagへ25%減る。**絶対性能の主張はしない**。
  主張は全アーム同条件での比較（相対差）に限定する。handicapは全アームに等しくかかる
- **held-out test は作らない**。上記nested選択でバイアスを断つ。
  ランダム患者20% testは領域母数を268→214（R2 59→47）に削るため不可。
  **「非annotated studyのみからtest抽出」案も採用不可**（annotated studyはprevalence 31.50% /
  骨折椎体2.19本、非annotatedは8.24% / 1.34本と別集団。positivity違反で補正不能。
  詳細は [[project-annotation-selection-bias]]）。独立外部cohort検証は将来課題として登録。
  研究の表現は「patient-grouped nested internal cross-validationによる評価」とする
- 評価: 椎体AUROC（**13,432 bag / 陽性1,332**、確証的）/
  **領域別AP を R1 / R2 / R3 / R4 個別に**（268のみ、床ゲート付き。macro平均へ潰さない）
- **SideAcc（左右balanced accuracy）は使用しない**（2026-08-17ユーザー決定）。
  左右の判別能は R2 / R3 それぞれのAPで見る。
  `common/metrics.py::side_balanced_accuracy` は実装・返り値から削除済み
  ⚠️ **旧記録の床（R1 0.59 / R2 0.37 / R3 0.45 / R4 0.72）は使わない。**
  確定仕様（**cross-fitted OOF: 3 training foldsからJeffreys平滑化
  `(x+0.5)/(n+1)`**）で再計算し、**R1 0.4946 / R2 0.2863 /
  R3 0.4222 / R4 0.7059**を凍結した。macro値は正式endpointとして出さない。
  APのtie処理はscikit-learn 1.9.0の`average_precision_score`（同一thresholdを一括処理）に固定。
  成果物は`common/outputs/level_floor_metrics.json`と`level_floor_predictions.csv`
- **領域別APの評価母集団は 268陽性のみ**（2026-08-18確定）。
  椎体陰性12,522 bagを混ぜるとlevel-only floorが macro 0.5026→0.0105 まで機械的に潰れ、
  異なる母集団間のAP比較が無意味になる（実測）。局在（どの領域か）と検出（骨折の有無か）は
  別エンドポイントに分ける
  領域別MDEは補正ラベルとcross-fitted床の患者cluster bootstrap SEから再計算済み。
  近似方法・rho別・Holm最悪順序の値は`common/outputs/region_floor_power.json`へ固定
  ⚠️ 旧記載の母数 14,133 / 陽性1,444 は誤り（2026-08-11修正）。確証的評価の分母・陽性数・検出力は
  凍結manifest `common/outputs/input_manifest.csv` から導出すること
- fold / seed / 入力manifest / 集約規則 / 学習予算は全アームで統一

### 運用

- **各プロジェクトに `README.md` を置き、モデル内容を記載する。仕様変更のたびに更新する**

### 検定計画・損失構成・λ校正（2026-08-18確定）

Codexの回答（全文 `.claude/docs/codex/20260818-remaining-four-decisions.md`）を
ユーザー承認のうえ採用。詳細は `memo/計画書/提案手法.md` 第2・4〜7節。

**実行構成は6つ / 30 run**（旧11構成・55 runから削減）:

| # | 構成 | 役割 |
|---|---|---|
| 1 | Baseline 0 | はしごの起点 |
| 2 | Control–B | MTL化の効果 |
| 3 | Baseline 1–B | **primary対比の相手** |
| 4 | Proposed–B, β>0 | 明示的対応 / **床ゲート対象** |
| 5 | Proposed–max, β>0 | whole出力方式 / **secondary対比** |
| 6 | Proposed–max, β=0 | attention回帰の新規性 |

- **noisy-ORは全アームから削除**。whole lossをregion logitsへ直接流すため、
  単なる推論時集約ではなく弱いregion supervision経路まで変えてしまう。方式Aはmaxのみ
- Controlはmethod Bのみ。method Bをはしご全アームの基準にする

**検定計画（固定順序2仮説のみが確証的）**:
- `H1: AUROC(Baseline 1–B) > AUROC(Control–B)` — 4領域mask入力そのものの効果
- `H2: AUROC(Proposed–max, β>0) > AUROC(Proposed–max, β=0)` — attention回帰の新規性
- endpointは両方とも13,432 bagのpaired pooled-OOF椎体AUROC
- **固定順序 H1→H2**。H1が有意なときだけH2を確証的に検定。H1が落ちたらH2は探索的
- 判定は patient-cluster bootstrap の paired差 95%両側CI下限 > 0
- key-secondary: Control–B vs Baseline 1–B の領域別AP差 / 床ゲートfamily
- ⚠️ 領域APをprimaryにしない理由: macro廃止で4仮説familyになり、
  **既存MDEはmacro-APの値なので各領域の検出力を保証しない**（要再計算）

**two-stream損失構成**:
- `A_t`（annotated、**1 bag/step固定**）は **`L_region` にのみ寄与**。`L_whole`にも`L_att`にも寄与させない
- `L_att`はnatural stream上で計算（maskは全bagにあるため）
- **Baseline 0も同一natural sampler・同一 `W_t`・同一optimizer step数**を使う
  （annotated streamのforwardをしないだけ）→ whole taskの分布・勾配が全アーム完全一致
- epoch長はnatural streamの一巡で定義。`L_whole`は常に `B_W` でmean（`B_W+1`で割らない）
- 全アームで同じnatural-stream seed / 順序
- annotated samplerはbag単位のshuffle-without-replacement cycle

**λ / β 校正（grid探索なし・追加full run 0）**:
- 各outer foldで、optimizer更新前に3 training foldsから決定論的に64 calibration batch。
  eval mode（BN統計もparameterも更新しない）、最後のshared CNN blockで損失別gradient L2 norm
- `λ_k = clip_[1e-2,1e2]( 0.5·exp( median_b log((g_whole+ε)/(g_region+ε)) ) )`、reference Baseline 1–B
- `β_k` も同型、reference Proposed–B、`g_att` を使用。ε = 1e-12
- **同一λ_kを全アーム・全構成へ適用。arm別チューニングは禁止**（ControlとB1でλが違えば交絡）
- 混合比は調整しない。非有限gradientが出たらrunを開始せず停止。
  clipping到達はログするが結果を見て変更しない
- 追加コストは5 fold × (64+64) = 640 calibration batchのみ

## 未決事項

なし（2026-08-18に全て解消）。ただし**学習開始前に必ず実施する作業**が次タスク1にある。

## 廃止済み（2026-08-17設計転換）

詳細な記録は `PROGRESS_ARCHIVE_4arm.md`。

- Baseline 1 `matched`設定（固定2,655 bag・1,498患者コホートでの学習）→ 過学習のため廃止。
  `cohorts/`の凍結成果物（SHA256 `91de42ca…`）は削除せず保持するが学習には使わない
- Baseline 2（4領域独立モデル）・提案A（teacher→pseudo-label→student）・
  提案B（弱教師あり、268評価専用）→ 全廃。268はregion lossの教師として直接使用する
- matched用backbone `tf_efficientnetv2_b0` 主解析・V2-S感度分析の区分 → matched自体の廃止に伴い失効
- 2-stem（image stem / mask stem）→ 単純early fusionへ置換
- outer foldでのval AUROC early stopping → outer評価専用化とnested選択へ置換

## プロジェクト一覧

| プロジェクト | ディレクトリ | 状態 | メモ |
|---|---|---|---|
| fold定義 | `folds/` | **完了(検証済)** | folds.csv凍結（seed 20260807）。再生成禁止 |
| 共通基盤 | `common/` | **完了(検証済)** | manifest / dataset / 明示ラベルのみのregion BCE / 領域別AP / deterministic two-stream sampler / nested split / λ・β校正 / cross-fitted床・MDE |
| matched cohort | `cohorts/` | **廃止(成果物保持)** | 2026-08-17設計転換で学習不使用に。凍結CSVは削除しない |
| Baseline 0 | `baseline0/` | **完了(検証済)** | 5 fold全てouter推論完了。outer0〜4のbest_epoch: 44/55/48/53/40 |
| Control (no-region-mask MTL) | `mtl/` | **未着手** | config・実装は完了。学習run未起動（`control_b`） |
| Baseline 1 (Early Fusion MTL) | `mtl/` | **学習中** | 10ch early fusion / whole head + region head `[B,15,4]` / method B。2GPUで正式学習中。outer0(best_epoch=41)・outer1(best_epoch=39)完了、outer2進行中 |
| Proposed (Mask-guided Branch) | `proposed/` | **未着手** | 実装・凍結は完了。3構成とも学習run未起動 |

状態は 未着手 / 実装中 / 学習中 / 完了(検証済) / 保留 のいずれかで更新する。

## 既存基盤（旧計画から引き継ぎ・現在も有効）

旧4アーム計画の下で構築したが、設計転換後も**そのまま使う**成果物。
構築経緯は `PROGRESS_ARCHIVE_4arm.md` を参照。

- `folds/outputs/folds.csv` — 患者単位層別5-fold、seed 20260807、凍結・上書きガード実装済み
- `common/outputs/input_manifest.csv` — 品質除外済み13,432 bag / 2,009 study、SHA256 `9bc0b8b91a5ff719519a63a3b2a7aa7f14476b45fade5582efb58a258ef21ac3`
- `folds/load_labels.py` — 領域ラベルのOR集約（唯一の実装）
- `common/` の dataset（CT 5ch + mask 5chを別テンソルで返す）・評価（椎体AUROC/AP、
  領域別AP、患者cluster bootstrap）。
  SideAccとmacro-APはendpoint・返り値から削除済み
- `baseline0/` の実装基盤 — 6ch adapter、同期augmentation、timm EfficientNetV2-S + BiLSTM、
  15面broadcast BCE / mean-sigmoid、nested outer単位の実験管理、
  full用の共有`/dev/shm` staging、checkpoint別outer 1回制約、pooled OOF整合検証

## 進捗ログ

### 2026-08-20（frozen manifest凍結範囲の修正・各project別CLI・Baseline 1–B起動）

詳細: `.claude/docs/work-logs/2026-08/2026-08-20-frozen-manifest-scope-fix-and-baseline1b-launch.md`

- 自動連鎖していた`.tmp/run_formal_fracture_pipeline.sh`をユーザー指摘で停止。以降、各アームの
  学習はユーザーが個別に手動起動する運用へ変更（自動連鎖は禁止）
- `python -m fracture_detection.{baseline0,mtl,proposed}.cli train --arm <arm>`で各project配下
  から起動できるよう再編。実装は共通`core/`/`cli/`への委譲のみ
- `verify_frozen_manifest`の凍結範囲を「モデル・損失・データ経路・乱数系列に影響する設定」のみに
  限定。実験名・GPU割当（`parallel.*`）・fold範囲・W&B設定は凍結対象外へ変更。
  `fold_to_gpu`の「6アーム全部が同一GPU割当であること」制約も撤廃し、アームごとに異なる
  GPU構成（例: baseline1_bだけ2GPU）を選べるようにした
- `source_tree_sha256`の対象を`.py`のみに変更（従来は`.yaml`も含み、armのconfig編集だけで
  再凍結が必要になっていた）
- 上記修正に伴いλ/β校正・5構成resource profile・frozen manifestを再実行（旧artifactは
  `experiments/archive/stale_20260819/`へ退避、削除はしていない）
- Baseline 0が5 fold完走（詳細は下記プロジェクト一覧）
- Baseline 1–Bを2GPU（`parallel.gpu_ids: [0,1]`）で正式学習開始。outer0/1完了、outer2進行中
- outer0のheld-out実測: AUROC 0.88前後、region_3のAUROC/PR-AUCが他領域より弱い（n=56/foldと
  小さいため1fold単独では未確定）。val_lossのepoch間振れ幅がBaseline 0の3〜4.5倍あり、
  annotated batch=1混入によるgradient分散が主因と推定（未検証の仮説）
- λの`target_ratio=0.5`（2026-08-18確定・grid探索なし・結果を見て変更しない、とPI仕様に明記）
  について「強すぎるのでは」という懸念が本セッションで出たが、確定ルールに従い**変更していない**
- 別プロジェクト`train_models/stage3`（region maskを構造priorとしてのみ使い、region labelは
  教師に使わない設計）がOOF AUROC 0.925とbaseline1_bより高いことを確認。設計差の比較を記録

### 2026-08-19（6構成共有実装とpreflight guard）

- `core/`へarm非依存trainer、optimizer、RNG、artifact、fold-process並列処理を抽出し、Baseline 0 /
  Control–B / Baseline 1–B / Proposed 3構成をadapter経由で統合
- canonical 10ch dataset、laterality-safe hflip、per-sample augmentation seed、独立mixup/annotated RNG、
  BN-only eval、natural→annotatedの逐次2 backward/1 stepを実装
- λ/β raw artifact、loss weights、20-step resource profile、frozen manifestをimmutable hash chainで接続。
  source/dependency/input/fold/config hash driftと重複profileを拒否する
- smoke modeはinner検証とcheckpointまでに限定し、凍結前のouter推論を禁止。正式outer CSVへfrozen manifest
  SHA256を埋め込み、pooled OOF収集時に5 foldで一致検証する
- pooled OOF、H1→H2固定順序、領域floor Holm、Control対Baseline 1領域AP、within-level感度、
  R2/R3 swapのpatient-cluster bootstrap CIを実装
- 実データ1-step GPU smokeはBaseline 0、Control–B、Baseline 1–B、Proposed–B、
  Proposed–max β>0/β=0の全6経路で完走。Control–Bは2 GPU × 2 foldの並列実行とresumeも完走し、
  fold別output/checkpoint/RNGの分離、固定GPU割当、smoke時outer予測なしを確認した
- resume時に`torch.load(map_location=cuda)`がCPU RNG stateまでCUDA tensorへ移す問題を実機で検出し、
  復元境界ですべてのtorch RNG stateをCPU `uint8`へ正規化。回帰を含む111 tests、mypy 31 source files、
  Ruff format/checkが通過
- 正式64-batch校正、5構成20-step profile、2 GPU × 2 foldの20-step preflight、frozen manifest作成、
  full trainingは実装完了後の運用工程として未実行

### 2026-08-19（正式pipeline開始）

- 全6 configをA6000 `[0, 1, 2]`、最大3 fold processへ固定し、fold割当を
  `0→0 / 1→1 / 2→2 / 3→0 / 4→1`へ統一した
- λ校正をGPU 0、β校正をGPU 1で64 batch × 5 outer foldとして開始。初回実機runでwhole-model eval時の
  cuDNN LSTM backward失敗と、FP32 Proposed–B校正の48 GB OOMを検出した
- 校正をrecurrent/Dropout train・BN-only eval、CUDA BF16 autocast、CPU上のstate監査snapshot、
  global RNG完全復元へ修正。単体・型・Ruff検証後に再開し、GPU使用量約21 GB / 38 GBで安定稼働中
- 校正完了後はloss weights結合、同一GPUで5構成20-step resource profile、2 GPU × 2 fold preflight、
  frozen manifest生成、Baseline 0 / Baseline 1–B / Proposed 3構成の正式学習へ自動継続する

### 2026-08-17（設計転換: 全データMTL + missing-label masking）

- ユーザー決定により研究設計を全面改訂。`memo/計画書/提案手法.md` を書き換え
- 転換理由: 一部症例（matched 2,655 bag）での学習は過学習
  （fold 0診断run: best val AUROC 0.738、epoch 21以降val BCE悪化）
- 新構成: 品質除外済み全13,432 bagで学習する hard parameter sharing型MTL
  - Baseline 0: CT+whole mask → CNN+LSTM → whole（現行`baseline0/`）
  - Baseline 1: 6ch early fusion → shared CNN+LSTM → whole head + region head 4 logits
  - Proposed: shared CNN → mask-guided 4 branches → 各LSTM → 4 region出力
  - whole出力は方式A（region aggregation: max / noisy-OR）と方式B（独立head）を比較
- 学習方法は全アーム固定: `L = L_whole + λ·m·L_region`（missing region labelはloss maskで無視、
  0扱いしない）+ two-stream sampling（annotated 268 bagをバッチへ一定割合混入）
- 廃止: Baseline 2 / 提案A（pseudo-label）/ 提案B（弱教師）/ matched学習 / 2-stem

### 2026-08-17（続き・未確定事項のユーザー決定）

- **論理的0教師は使わない**: region lossは領域ラベルのある268 bagのみ。
  `common/losses.py`の椎体陰性への論理的0適用は実装時に削除・整合が必要
- **方式Aの集約関数（max / noisy-OR）はアブレーション**として両方比較
- **Proposedのmask注入はPMGAN方式**（参考論文を精読済み）。
  領域ごとのMask-Guided Attentionのspatial attention mapを対応maskへRMSE回帰
  （L_att、学習時のみ）、特徴再重み付けは残差形式 `(1+m)⊗f`。
  全体損失は `L = L_whole + λ·m·L_region + β·L_att`
- **pos_weight=2.0は全アーム固定**
- **fold分割は凍結folds.csvを全アームで再利用**することを確認。
  提案A廃止でfold内teacher制約は消滅

### 2026-08-17（続き・fold設計とtest分離のレビュー）

- ユーザー依頼でfold分けとtest分離の妥当性を検証。Codex相談＋凍結manifestの実測
- **fold分割自体は健全と確認**（患者リークなし、prevalence 10.08〜10.13%、level均等、
  annotated bag 53〜56層別済み）。再生成しない
- **アノテーション160 studyが陽性患者のランダム標本でないことを実測**
  （prevalence 31.50% vs 8.24%、骨折椎体2.19本 vs 1.34本、level別annotated比率 C3 42.5%〜C7 9.4%）。
  これにより「非annotated studyのみからtest抽出」案は positivity違反で不可と確定
- **ユーザー承認により4点を確定**:
  1. fold分割は現状維持（再生成なし）
  2. held-out testは切り出さない
  3. outer foldを評価専用にし、cyclic single-inner-foldで選択
  4. Control（no-region-mask MTL）アームを追加
- Codexの数値主張（SideAcc内訳 both=18 / R2-only=41 / R3-only=54）は実データと一致を確認
- 残る未決4点（primary contrast / two-stream損失分離 / SideAcc集計と0.65ゲート定義 /
  λ・β・混合比の決定規則）を「未決事項」へ登録

### 2026-08-17（続き・nested選択を5 run版に確定）

- 「1構成10 run」の内訳を確認した結果、**Stage 2の再fitを省く5 run版を採用**（ユーザー決定）
- 採用形: outer=k / inner=(k+1)%5 / 残り3 foldで学習しinnerでcheckpoint選択 → outer推論1回
- **実装が単純化**: 再fitがなく、RSNA Type1準拠の固定cosineを全アームで共有する。
  validation依存のLR軌跡の記録・再生は不要になり、「Stage 2のstep数1.33倍」の登録仕様も消滅
- 計算量は旧方式の**0.75倍**（fold数4→3のみの差）。run数は5で従来と同じ
- ⚠️ 代償として登録: 全報告モデルが3 fold（60%）学習、領域教師は215→約160 bagへ25%減。
  **絶対性能の主張はしない**。主張は全アーム同条件の相対比較に限定
- 併せて、以前の「学習コスト約2倍」という記述は run数と計算量を混同した誤りだったため訂正。
  10 run版でも計算量は `0.75 + E_best/E_stop` 倍（実測値で約1.07〜1.45倍）であり2倍ではなかった

### 2026-08-17（続き・SideAcc廃止）

- ユーザー決定により **SideAcc（左右balanced accuracy）を評価指標から削除**。
  左右の判別能を含め、局在の評価は **R1〜R4それぞれのAP** で見る（macro平均へ潰さない）
- これに伴い未決事項の「SideAcc集計と0.65ゲート定義」は消滅し、
  代わりに「領域別APの床ゲートと多重性補正（4検定Holm）」を未決事項3として登録
- **帰結**: 近道（レベル情報だけで領域を当てる戦略。実測でlevel-only macro-AP 0.451）への
  耐性は、領域別APの床ゲートだけが担うことになった。床の補正ラベル再計算（次タスク8）は
  事前登録の前提条件として必須度が上がった
- `common/metrics.py::side_balanced_accuracy` はendpointから外す。
  実装の整理は次タスク2（`common/`改修）にまとめて行う

### 2026-08-18（未決4点をCodex回答で確定）

- Codexへ未決4点＋構成削減＋妥当性確認を相談（`.claude/docs/codex/20260818-remaining-four-decisions.md`）
- **ユーザー決定（案1）により、Codex推奨をそのまま採用**:
  - primary は `AUROC(Baseline 1–B) > AUROC(Control–B)`、secondary は
    `Proposed–max β>0 > β=0`、固定順序 H1→H2
  - annotated streamは `L_region` のみ。Baseline 0も同一natural sampler・同一step数
  - 床は cross-fitted OOF（3 foldsからJeffreys平滑化）、対象は Proposed–B β>0 のみ、
    母集団は268陽性のみ、R1〜R4にHolm補正
  - λ/βは outer-training のみの固定初期勾配校正、**全アームで同一λ_k**、追加full run 0
  - **6構成 / 30 run へ削減**、noisy-ORは全アーム削除
- **床を補正ラベルで暫定再計算**したところ旧記録値と一致しないことが判明
  （in-sample: R1 0.5303 / R2 0.3243 / R3 0.4298 / R4 0.7259。旧値 0.59/0.37/0.45/0.72）。
  旧値は使わず、Codex仕様どおり実装し直した値を凍結する
- 椎体陰性を混ぜると床がmacro 0.5026→0.0105まで機械的に潰れることを実測し、
  「268陽性のみ」という母集団指定を数値で裏付け
- ⚠️ **macro廃止により既存MDEが失効**していることが判明（既存値はmacro-AP基準）。
  per-region MDEの再計算を次タスク1に追加
- Codex Q6の6つの表現制限（60% training / 268の非ランダム性 / 1 seed /
  laterality主張の条件 / floorのleakage / 凍結タイミング）を計画書へ登録

### 2026-08-17（続き・進捗台帳の分離）

- 旧4アーム計画（Baseline 2 / 提案A / 提案B / matched学習）の記録を
  `PROGRESS_ARCHIVE_4arm.md` へ分離。現行計画とやっていることが根本的に異なるため
- `PROGRESS.md` は現行計画（全データMTL）のみを扱い、確定済みの前提を
  入力・データ / モデルと損失 / 評価プロトコル / 運用 に再編
- 旧計画から引き継ぐ成果物を「既存基盤」節に明示

### 2026-08-18（共通基盤・Baseline 0実装完了）

- 旧`baseline1/`を現行名称`baseline0/`へ移行し、matched分岐・設定を削除
- `common/`へcyclic nested split、deterministic natural sampler、annotated cycle sampler、
  λ/β初期gradient校正、cross-fitted level-only床、患者cluster MDE近似を追加
- region BCEから椎体陰性の論理的0教師を削除。SideAcc・macro-APを正式返り値から削除
- Baseline 0をtrain 3 folds / inner checkpoint選択 / best確定後outer 1回推論へ変更。
  既存outer予測がある場合は再推論を拒否し、旧config checkpointのresumeも拒否
- level-only床を凍結: R1 0.4946 / R2 0.2863 / R3 0.4222 / R4 0.7059
- 検証: 45 unit tests、ruff、mypy、実データ1 bagのV2-S forward/loss/backwardが通過
- Baseline 0のschedulerを旧matched診断由来の`ReduceLROnPlateau`から、RSNA Type1と同じ75 epoch単一cosineへ修正。validation依存の最適化差を全アームから除外
- `baseline0/`直下の10モジュールを`cli/config/data/modeling/training`へ責務別に再配置。直下はREADMEとpackage定義だけに整理し、成果物pathは維持
- 共有`/dev/shm` stagingの無表示区間を撤廃。lock、source走査、cache検証、容量、copy bytes/files/current path/speed、marker、atomic確定、再利用をすべて標準出力へ表示
- NFS→tmpfs copyを設定可能な8 threadへ並列化し、不要なmetadata複製を廃止。同一manifestの中断tmpだけをlock下で自動削除し、完成cacheは再利用
- Baseline 0 protocolを`baseline0-nested-v3`へ更新。best checkpointはinner AUROCのまま、early stoppingだけをinner BCE patience 15へ分離し、再開checkpointへBCE最良値とbad epoch数を保存
- 学習DataLoaderを8 workersへ変更。OpenCVは既定32 threads/workerだったためworker初期化時に1 threadへ制限し、256 threads相当の過剰並列を防止。epochごとのdata wait / compute時間も記録
- `08_18/v1`は旧v2（AUROC停止・4 workers）でouter 0学習が開始されたため、v3の確認用・正式runには再利用しない。成果物は削除せず保持し、新runは`08_18/v2_val_loss_stop`へ分離
- console、progress bar、`history.csv`、W&B、`fold_metrics.json`、validation予測ファイルの表示を`inner`から`val`へ統一。nested split検証用の内部key `runtime.inner_fold`だけ保持
- protocol `baseline0-nested-v4`: `best_model.pt`はval AUROC最大・outer推論用のまま維持し、val PR-AUC（average precision）最大を`best_val_prauc_model.pt`へ独立保存。`last_checkpoint.pt`へ両best epoch/metricsを保存してresume可能にした。PR-AUC-bestは診断用でouter推論には使わない
- protocol `baseline0-nested-v5`: epochログへ固定0.5とval F1最適点のprecision / recall / F1を追加。
  AUROC-bestをprimary、PR-AUC-bestをsecondary感度分析として事前指定し、各checkpoint自身の
  valでF1最大閾値（同率は高い閾値）を決め、対応するouterへ固定適用する。両checkpointで
  AUROC / PR-AUC / precision / recall / F1をfold・pooled OOFに保存し、outer推論は各1回に制限。
  v4の`08_18/v2`成果物と衝突させないため、v5の設定出力先は`08_18/v3`
- v5検証: common + Baseline 0の56 unit tests、Ruff check/format、mypy 27 source filesが通過。
  outerでは最適閾値を再探索せず、valで凍結した閾値の指標だけを正式値として保存する
- protocol `baseline0-nested-v6`: Stage1と同じ品質除外を共通manifestへ適用し、13,432 bag / 2,009 study / 陽性1,332へ更新。flip・transposeを除くStage1 augmentationとmixup `p=0.2`を移植し、whole BCEをStage1と同じ陽性重み合計正規化へ変更。新規runは`08_18/v4`
- protocol `baseline0-nested-v7`: Stage1実装どおり15面×5chを75chへstackし、augmentation呼出しを75回/bagから1回/bagへ削減。Baseline 0で不要なregion mask読み込みを止め、入力をuint8のまま転送してGPU側で正規化し、cuDNN fixed-shape autotuneを有効化。ローカルitem生成は約4.0倍、転送payloadは4分の1。出力先は`08_18/v4`
- 検証: common + Baseline 0の49 tests、Ruff check/format、mypy 27 source filesが通過。sandbox外の軽量DataLoaderで8 workersの起動とbatch取得も通過
- **正式なv3学習・outer推論は未開始**。6構成・λ/β・code/config hash凍結前のouter推論は禁止

### 2026-08-18（続き・v4診断結果とv8正則化改訂）

- `08_18/v4` outer 0の診断結果: epoch 24から train/val BCE が乖離し、
  epoch 38で early stopping（patience 15）が発火。best val AUROC 0.898 は epoch 17、
  停止時の cosine LR は 1.29e-4 で **eta_min へのアニールが49%しか進んでいない**
- 参照Stage1（`train_models/stage1` v1_parity）を実測比較したところ、**全5 foldが75 epoch完走**し
  early stoppingは一度も発火せず、best val AUROCは **epoch 59 / 61 / 61 / 73 / 74**（=低LR収束後）に集中。
  最終 train/val gap は +0.021 / -0.027 / +0.031 / -0.035 / +0.003 で過学習していない
- 差分の主因は **hflip・vflip・transpose（各p=0.5、いずれかが87.5%で発火）の除外**と特定。
  この3つはR1〜R4の向き依存ラベルを壊すため復活させられない
- 副次的に判明した非parity 2点: `gradient_clip_norm=5.0`（Stage1はnull）と、
  weight decayのbias/norm除外（Stage1は全パラメータ一律）
- **dropoutによる代替は却下**。Stage1は`drop_rate=0.0 / drop_path_rate=0.0`で過学習していないため、
  参照実装が使っていない別機構で埋めるのは正当化できない。同じ機構（augmentationの多様性）で戻す
- 領域定義を確認したところ **R2=right_transverse_foramen / R3=left_transverse_foramen は
  左右対称の同種構造**なので、水平反転と同時にラベルとマスク値を入れ替えれば意味論が完全に保存される。
  R1=vertebral_body / R4=posterior_elements は正中構造なので水平反転の影響を受けない。
  一方 vertical flip と transpose は R1 と R4 を入れ替えることになるが、この2つは鏡像関係にない
  別種の構造であり正しい入れ替えが存在しないため、**恒久的に禁止**（schema.pyで設定キーごと拒否）
- protocol `baseline0-nested-v8`: **`horizontal_flip_probability=0.5` を復活**（発火率0%→50%、
  Stage1の87.5%には届かないが3種のうち唯一正当に戻せる）。weight decay をStage1と同じ
  全trainable parameter一律へ統一。`gradient_clip_norm`は`null`（実質無限大）へ。
  `early_stopping_patience`は15→**20**。dropout・mixup・head_dropout・LR・T_max=75は据え置き。
  出力先は`08_18/v5`、GPU 0
- 領域ラベルを持つアーム用に `common.dataset.flip_horizontal` を追加。
  CT・椎体マスク・領域マスク・領域ラベルを反転し、R2/R3のマスク値とラベルを同時に入れ替える。
  Baseline 0はwholeラベルのみで左右依存がないため`A.HorizontalFlip`をそのまま使う
- ⚠️ `min_epoch`は**上げてはいけない**。`trainer.py`の`eligible = epoch >= min_epoch`が
  early stoppingとcheckpoint保存の両方を制御しており、上げるとepoch 17のような早期bestが保存対象から外れる
- ⚠️ この正則化改訂は **inner-valの挙動を見た後のprotocol amendment**。候補グリッド探索は行わず
  事前指定値1点をfreezeしたが、選択バイアスは消えないため報告時にlimitationとして明記する
- 検証: common + Baseline 0の**70 tests**、Ruff check/format、mypy（既存のstub不足のみ）が通過。
  実測で hflip発火率0.5038（4000試行、95%CI [0.488, 0.519]、検出漏れ0）、CTと椎体マスクの同期反転、
  R2↔R3のマスク値・ラベル入れ替えとdtype保持、optimizer 472/472パラメータ（1次元292個含む）が
  同一decay対象、clip閾値infでgrad norm無変更を確認

### 2026-08-19（mixup据え置きの確定・LRとバッチの検討）

- **mixup `p=0.2` を確定。増量案は恒久的に却下**（2026-08-19ユーザー決定）。
  RSNA原典`stage2-type1.ipynb`は`p_mixup=0.5`であり、v5に残るtrain/val gap
  （ep56で+0.087、Stage1は-0.016）への対処として0.5への復帰を検討したが不採用。
  理由は`L_att`との相互作用: `L_att`はspatial attention mapを領域maskへRMSE回帰させる
  **密な空間損失**であり、mixupすると教師が`λ·M_a+(1-λ)·M_b`という別患者2人の
  重ね合わせになる。空間座標は重ね合わせても意味を持たず、「ここが右横突孔」という
  空間的対応を教える`L_att`の役目を壊す。さらにmixup発火率を上げると
  **H2（attention回帰教師の新規性）が検出したい効果そのものを薄める**。
  加えてnatural stream（`L_whole`/`L_att`）だけにmixupがかかりannotated stream
  （`L_region`）にはかからないため、増量は3損失項の相対バランスとλ/β校正をずらす
- **LRは2.3e-4据え置きを確認**。原典`stage2-type1.ipynb`は`batch_size=8`/`init_lr=23e-5`だが、
  実効batch16への倍増は`stage2/parity.yaml`（8×2GPU DDP）の時点で行われ、LRは据え置かれていた。
  Stage1（batch16 + 2.3e-4）が5foldでAUROC 0.909〜0.931・75 epoch完走・過学習なしを実測しており、
  **実効構成としてすでに検証済み**。またmodelはbagを面へ展開するため
  encoder/BNが見るのは16×15=**240サンプル**で、bag単位2倍のLR感度はさらに小さい
- v5のval loss挙動はLR問題を示していない: 全foldがep32〜43でval_bce最良（LRは1.07〜1.54e-4）、
  以降はtrain lossだけ下がりval悪化。val_bceのepoch間変動もLRが高い前半(0.015〜0.030)より
  低い後半(0.011〜0.016)の方が小さく、発散も振動もない。**残る課題はLRではなく正則化不足**
- **batch 32案も却下**。GPU utilizationが既に100%で総演算量は不変のため高速化しない。
  さらにsteps/epochが505→253となり75 epochでの総更新回数が37,875→18,975と半減する。
  高速化が目的ならfoldを複数GPUへ分割する方が無害（`start_outer_fold`/`end_outer_fold`）
- v4(v7旧設定)完走: 5fold平均 outer AUROC 0.8949±0.0153、AP 0.7082、Recall 0.5825、F1 0.6641。
  AUROC-best checkpointはepoch 17/35/37/37/30
- v5(v8)のouter0〜3: 平均 outer AUROC 0.9059±0.0161、AP 0.7367±0.0218、
  Recall 0.6383±0.0223、F1 0.7000。AUROC-best checkpointはepoch 53/49/51/39。
  **精度は全面的に向上しAUROCの安定性は同等**。checkpoint epochが後方へ移動し
  cosineの低LR収束フェーズを使えるようになった
- ⚠️ Precisionのfold間stdはv4 0.0511→v5 0.0981と悪化。原因は**F1最適閾値の選択ブレ**
  （v5で0.342〜0.656）。val陽性が268件でF1曲線が頂点付近で平坦なためargmaxが不安定。
  ただし**primary endpointはAUROCなので主要な結論には影響しない**
- ベースライン設定の最終確定は**後続アーム（Control–B / Baseline 1–B / Proposed 3構成）の
  挙動を見てから判断する**方針（2026-08-19ユーザー決定）。patience・max_epochsは未決

## 次のタスク

1. Baseline 1–Bのouter2〜4完走を待つ（`mtl.cli train --arm baseline1_b`が2GPUで実行中）
2. Control–Bを起動する（`mtl.cli train --arm control_b`）。H1（`AUROC(Baseline1–B)>AUROC(Control–B)`）
   の比較相手であり、まだ未着手
3. Proposed 3構成（proposed_b / proposed_max / proposed_max_beta0）を起動する
4. 全アーム揃ったらpooled OOF解析（`cli/analyze.py`）でH1→H2固定順序検定を実行
5. λの`target_ratio=0.5`見直しは未決着。見直す場合は次の凍結サイクルで正式に意思決定する
6. **凍結後はouter foldの結果を設計変更に使わない**方針は維持（Codex Q6の必須要件）。
   ただし凍結の「範囲」自体（運用設定 vs 科学的設定）は2026-08-20に修正済み
