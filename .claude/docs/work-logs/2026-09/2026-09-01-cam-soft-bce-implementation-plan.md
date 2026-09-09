# CAM疑似ターゲット＋plain soft BCE 実装計画書

作成日: 2026-09-01  
状態: **Phase 0-6完了。Phase 7（outer-0 3アーム本学習）着手中（ユーザー承認済み）**

## 進捗ログ

### 2026-09-01 セッション3: Phase 6完了（4-view artifact本生成）

ユーザーが「やってください」で明示的にGPU実行を承認（[[feedback_manual_execution_trigger]]は
満たされた）。

- 実行前に`cross_validated_calibration_quality`の頑健性ギャップを発見・修正:
  極小fit population（smoke test相当）でGroupKFoldの1分割が1 bagになり
  `fit_shared_logit_share_calibration`がValueErrorを投げていた。分割数を段階的に
  減らし、どれも成立しなければ`oof_available=0.0`で優雅に諦めるよう修正
  （本番規模では発火しない想定だが、実行中クラッシュで数十分のGPU計算を無駄にする
  リスクを避けるため事前修正）。修正後、既存20テストは全green維持。
- GPU smoke test（`--limit-bags 8`）で実チェックポイントでのTTA経路を確認後、
  全データ本生成を実行（`--device cuda:0 --batch-size 16`、他はデフォルト）。
  実行時間は数分、A6000 1台で完了（バックグラウンド実行中に誤って`nohup ... &`と
  `run_in_background`を二重に使い、シェル起動タスクの完了通知と実プロセスの完了を
  混同しかけた。`ps`/`nvidia-smi`で実プロセスの生存を確認し、`until grep`の
  待機loopで実際の完了を検知し直した）。
- **結果（受入条件を全て確認）**:
  - 全5 foldでslope>0かつfinite（1.48〜2.44）、intercept finite。
  - q全件finiteかつ[0,1]（実測範囲9.8e-5〜0.997）。
  - 全whole-positive行でsum(q)>=1（min=1.0416、`projected_fraction=0.0`——
    projectionが実際には一度も発火しなかった）。
  - `(student_outer_fold, study_id, level)`一意（3996行=3996通り）、
    `teacher_outer_fold==student_outer_fold`が全行で成立。
  - fold別`oof_ap` 0.664〜0.731、`oof_brier` 0.155〜0.182、`oof_log_loss` 0.470〜0.549
    ——8/31セッションの研究監査（`.claude/docs/research/2026-09-01-pseudo-target-probability-audit.md`、
    4-view TTA期待値macro AP≈0.70）とよく整合。
  - `n_fit_bags`（fold別138〜145、平均約141.6）は`project_pseudo_label_construction`
    メモリの「約143/fold」という既存推定と一致。
  - `sum(q)`の平均1.384は旧identity-CAM診断値（DESIGN.md記載の中央値1.396）と近い
    オーダーで、4-view化による大きな逸脱はない。
- 係数を凍結: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/`配下に
  `pseudo_region_targets.csv`（3996行・1332 unique bag）、
  `pseudo_target_calibration.csv`、`pseudo_target_generation_metadata.json`を保存。
  旧`pseudo_label_scores.csv`等は無変更のまま残存（ファイル名が異なるため無傷）。
- 次: Phase 7（lambda校正→outer-0 3アームpreflight→本学習→3アーム比較gate判定）。

### 2026-09-01 セッション2: Phase 5完了（Phase 0-5総括）

Phase 4完了後、Claude本体が直接実装を継続（このセッションではフォーク委譲の信頼性問題
（後述）を踏まえ、Phase 5は最初からsubagentを介さず本体が実装）。

- `cli/evaluate.py`: `collect_oof_predictions`のcheckpoint契約を、単一`best_model.pt`
  （role=`best_val_total`）から`best_region.pt`（role=`best_region`）と`best_whole.pt`
  （role=`best_whole`）の2ファイル・2role検証へ変更。両方の存在・role・nested runtime
  一致を個別に検証する。
- 新規`cli/compare_pseudo_arms.py`: outer fold 0限定で`no_pseudo`/`cam_soft`/
  `cam_soft_shuffled`の3アームを比較する受入試験CLI。`compare_arms.py`の
  `paired_cluster_bootstrap_difference`を再利用し、3アーム間でbag集合・
  `vertebra_target`・`{region}_target(_valid)`が完全一致することを事前assert。
  `gate_passed`は「cam_softのmacro AP差分の点推定が両対照を上回るか」の符号判定のみ
  （信頼区間がゼロを跨がないことまでは要求しない、CLI docstringに明記）。
- テスト: `test_evaluate.py`（新規、5件: checkpoint揃い/片方欠落×2/role不正/nested不一致）、
  `test_compare_pseudo_arms.py`（新規、5件: gate合格/不合格/bag集合不一致/hard target不一致/
  outer0限定違反）。region_branchにはこれらのCLIの既存テストが元々存在しなかった
  （前回ログで記録した既知の穴を解消）。
- **通し結果（Phase 0-5最終）**: `fracture_detection/region_branch/tests` 124件全green、
  `fracture_detection/baseline0/tests` 135件green（無変更）。`ruff check`/`ruff format`
  region_branch全体でclean。`poe typecheck`（mypy相当）でregion_branch由来の新規エラー
  0件（既存のimport-untyped系noiseのみ）。`rg`で旧alpha/rank_loss/pairwise/
  pseudo_bags_per_batch/region_lstm_parameters()/val_total/best_model.ptの生きた参照が
  region_branch側にゼロであることを確認
  （`modeling/initialization.py`の`best_model.pt`はBaseline 0自身のcheckpoint名で無関係、
  誤検知ではない）。

**フォーク委譲の信頼性について（重要な運用上の教訓）**: 本セッションでは大規模な
実装作業をsubagent（fork）へ委譲する運用を試みたが、タスクが大きい・詳細すぎる場合に
`tool_uses: 0`〜`1`かつ数十秒で「completed」を自称し、実際には`git diff`が空という
空振りが計3回発生した（最初のPhase 2-5一括委譲、および細分化後のPhase 4b trainer.py
本体）。空振り時の応答テキストはClaude本体が直前にユーザーへ送った状況説明とほぼ同一の
文面であり、実タスクを処理していない可能性が高い。対策として（1）委託は
1ファイル〜数ファイル程度の粒度に細分化する、（2）委託後は必ず`git status`/
`git diff --stat`と実際のpytest実行で独立検証し、報告文だけを信用しない、
（3）細分化してもなお空振りする場合はClaude本体が直接実装する、という運用に切り替えた。
Phase 2・Phase 3・Phase 4a・monitoring.py（Phase 4bの一部）はフォーク経由で成功し
全て独立検証済み。trainer.py本体・cli/train.py・Phase 5一式はClaude本体が直接実装した。

### 2026-09-01 セッション2: Phase 2-4完了

Phase 1完了後、region_branch側（Phase 2〜4）を実装。最初にPhase 2-5を1本の大きなsubagentへ
まとめて委譲したところ`tool_uses: 0`で「完了」を自称する空振りが2回発生（`git diff`で無変更を
確認）。以降はPhase単位（Phase 2 / Phase 3 / Phase 4a / Phase 4bのmonitoring.pyのみ）へ細分化して
再委託し、各回`git status`/`git diff --stat`/pytest出力を自分で独立検証する運用へ切替えた。
Phase 4bのtrainer.py本体（最大・最難関）は3回目の細分化フォークも同型の空振りを起こしたため、
最終的にClaude本体が直接実装した。

- **Phase 2**（dataset契約 + natural-onlyローダー、fork経由・検証済）:
  - `data_pipeline/constants.py`: `SOURCE_*`/`EXPECTED_*_BAGS`/旧pseudo定数を削除、
    `PSEUDO_REGION_TARGETS_CSV`/`REGION_SHARE_COLUMNS`/`REGION_PSEUDO_TARGET_COLUMNS`等を追加。
  - `data_pipeline/pseudo_labels.py`: `load_pseudo_region_targets`/`attach_pseudo_targets`/
    `shuffle_pseudo_target_associations`（`cam_soft_shuffled`負対照用の固定seed置換）に全面置換。
  - `data_pipeline/dataset.py`: `RegionBranchDataset`がcell単位で
    `region_exact_target/valid`・`region_pseudo_target/valid`の4値を返すよう変更
    （whole-negativeは論理0、human-validはhard値優先、それ以外はpseudo。
    `require_pseudo_targets=True`で欠落を`__getitem__`時に即エラー）。
  - `data_pipeline/loaders.py`: `OuterFoldLoaders`を`{natural, steps_per_epoch}`のみに単純化、
    `build_outer_fold_loaders`に`pseudo_arm`引数追加。
  - 新規`data_pipeline/batching.py`（`BatchTensors`/`batch_tensors`を`sampling.py`から分離）。
  - テスト: dataset 10 / pseudo_labels 11 / batching 2 / sampling(暫定) 4 = 27件green。
- **Phase 3**（`modeling/losses.py`、fork経由・検証済）: 旧pairwise ranking依存を全廃し
  `compute_exact_loss`（Baseline0と同じpos_weight・weight-sum正規化、`exact_valid`マスク）/
  `compute_pseudo_loss`（無weightのplain masked mean BCE、native pos_weight=2との
  最適点乖離を明示テスト）を新設。テスト19件green。
- **Phase 4a**（config schema・lambda-only校正・calibrate CLI、fork経由・検証済）:
  - `config/schema.py`: `protocol_version`を`region-branch-v4`へ、`region.pseudo_arm`
    （`no_pseudo`/`cam_soft`/`cam_soft_shuffled`）必須化、`*_bags_per_batch`3keyを廃止、
    `early_stopping_metric`を`val_region_macro_ap`に変更。
    （fork判断の逸脱: `pseudo_label_dir`非null強制は既存の「nullはpackage既定を使う」慣習と
    衝突するため撤回し、`pseudo_arm`の値検証のみに縮小。既存configとの後方整合を優先した
    妥当な判断と判定）。
  - config yaml 8ファイル更新・outer0専用3アーム設定
    （`region_branch_outer0_{no_pseudo,cam_soft,cam_soft_shuffled}.yaml`）新設
    （※region_branch配下の`*.yaml`は`.gitignore`対象でgit差分に出ない。既存の意図的方針）。
  - `training/calibration.py`: alpha関連を全廃し`CalibrationResult`を`lambda_`系のみに縮小。
    `calibration_config_fingerprint`から`pseudo_arm`/`pseudo_label_dir`を除外
    （3アームで同一lambdaを共有するための必須修正、fingerprint不変性テストで確認）。
  - `cli/calibrate.py`: 校正時は`pseudo_arm`を常に`"cam_soft"`へ強制。
  - テスト: test_config.py + test_calibration.py 34件green。
- **Phase 4b**（`training/monitoring.py`・`training/trainer.py`・`cli/train.py`、
  monitoring.pyのみfork経由、trainer.py/cli/train.pyはClaude本体が直接実装・検証）:
  - `monitoring.py`: `select_diagnostic_subset`が`SourcePools`ではなく
    `train_manifest`（whole-positiveかつ人手未確定）を直接受け取る形に変更。
    `compute_diagnostics`は`teacher_scores`→`pseudo_target_array`+`pseudo_valid_array`
    （qは0を正当値として取り得るため`>0`ヒューリスティックを廃止）。
  - `trainer.py`（全面書き換え）: 1 stepは自然分布batch1本のみ処理
    （旧4-loader zipを廃止）。mixup発火時はregion経路を完全skip
    （`region_skip_fraction`で計測）。`pseudo_arm=="no_pseudo"`では`compute_pseudo_loss`を
    一切呼ばない。checkpointを`best_region.pt`（val region macro AP最大化・
    early stopping対象）と`best_whole.pt`（val whole loss最小化・early stoppingに無関係）
    へ分離。`last_checkpoint.pt`は両trackerを含む完全state。outer推論は2 checkpointで
    各1回ずつ実行し、`vertebra_score`はwhole pass、`{region}_score`はregion passから
    マージ（bag集合・vertebra_target一致を事前assert）。
  - `cli/train.py`: 新ローダー・診断subset（`pseudo_arm`に応じてtrain_manifest由来の
    診断subsetへも学習と同じseedでpseudo target/shuffleを反映）に追随。
  - 参照ゼロを確認後`data_pipeline/{sampling,sources}.py`と死んだ`tests/test_sampling.py`を削除。
  - テスト: test_monitoring.py 10 + test_trainer.py 15（新規: 単一batch/optimizer.step 1回・
    mixup時region skip・no_pseudo時pseudo_loss不呼出・dual checkpoint分離・
    outer_predictions列の出所検証・resume往復・early stoppingがregion専用である検証）+
    test_train_cli.py 2 = 27件green。
- **通し結果**: `fracture_detection/region_branch/tests` 114件全green、
  `fracture_detection/baseline0/tests` 135件green（無変更）。ruff check/format green。
  `rg`で旧alpha/rank/pairwise/pseudo_bags_per_batch/region_lstm_parameters()/val_totalの
  生きた参照ゼロを確認（残る一致はdocstring内の「廃止済み」言及とFORBIDDEN_CONFIG_KEYSの
  文字列のみ）。
- **既知の残課題（Phase 5想定内）**: `cli/evaluate.py`が旧`best_model.pt`/
  `checkpoint_role=="best_val_total"`のまま未更新（`best_region.pt`/`best_whole.pt`の
  2ファイル契約へ追随させる必要あり）。region_branchには元々`test_evaluate.py`が
  存在しないため、この破損を検出する既存テストはない。

### 2026-09-01 セッション2: Phase 0-1完了

- GPU復旧確認（`nvidia-smi`でA6000×3台認識、ほぼアイドル）。ただし本番4-view artifact
  生成（Phase 6）はユーザーの手動トリガー待ち（`実行は必ずユーザーが手動トリガー`方針）。
- Phase 0: `git status --short`で既存差分を保護、既存targeted testsが55件green（変更前ベースライン）。
- Phase 1実装完了:
  - 新規 `fracture_detection/baseline0/pseudo_labeling/calibration.py`: 純粋関数のみ
    （`enrichment_to_share`/`average_view_shares`/`shares_to_logit_features`/
    `fit_shared_logit_share_calibration`/`apply_shared_logit_share_calibration`/
    `project_sum_at_least_one`/`cross_validated_calibration_quality`/
    `summarize_probability_distribution`）。20 testで新規カバー。
  - `gradcam.py`に4-view TTA機構を追加: `TTAView`/`DEFAULT_TTA_VIEWS`
    （identity/horizontal_flip/rotation±10°）/`apply_tta_view_to_inputs`/
    `invert_tta_view_on_cam`。回転はcv2.warpAffine、flipは既存
    `cam_audit.flip_planes_horizontally`を再利用。round-trip testで
    合成CAMのregion enrichmentが native frame へ戻ることを確認（rtol=0.1)。
  - `cli/generate_pseudo_labels.py`を全面置換: 旧pairwise/temperature生成を廃止し、
    fold一致teacherの4-view TTA→region density enrichment→share化→
    shared logit-share calibration fit/apply→`project_sum_at_least_one`の
    パイプラインに変更。新artifact3種
    （`pseudo_region_targets.csv`/`pseudo_target_calibration.csv`/
    `pseudo_target_generation_metadata.json`）をatomic writeで生成。
    旧`pseudo_label_scores.csv`/`pseudo_label_temperatures.csv`は無関係のファイル名のため無傷。
  - `test_pseudo_label.py`: 旧`_compute_temperatures`依存testのみ削除し、`scoring.py`
    （廃止済みだが未削除）自体のtestは温存。新規に`_guard_output`/`_atomic_write_*`/
    `_calibrate_fold`（fit成功・fit不足の両分岐）/`run_generation`のCLI end-to-endスモーク
    （5 outer fold、決定論的な最小Baseline0代替モデルで実CAM経路を通す。`--limit-bags=1`で
    意図的にfit-population不足分岐を固定し、flakyな符号依存を回避）を追加。
  - `test_attention.py`にTTA view系のtestを追加（形状/dtype保存、hflip round-trip厳密一致、
    rotation round-tripのregion enrichment近似一致）。
- テスト結果: `fracture_detection/baseline0/tests` 全135件green（新規66件追加: calibration 20 +
  pseudo_label 16→36 (net+20だが一部差し替え) + attention 8件追加）。
  `ruff check`/`ruff format` green。`ty`は環境未導入（バイナリなし）、
  `uv run poe typecheck`（mypy相当）は新規ファイルで`scipy.special`/`sklearn.*`の
  import-untyped 3件のみ（リポジトリ全体で同種の予定された既知gapと同じパターンであり、
  今回変更のロジックエラーではない。対応不要・環境blockerとして記録)。
- 残課題: Phase 1完了条件のうち「CPU mockでgeneration CLIのend-to-end smokeが通る」は
  決定論的トイモデルで達成。実チェックポイントでの`--limit-bags`スモーク（本物のBaseline0
  重みを使う統合確認）は未実施のため、Phase 6直前に一度実施を推奨。

参照する正本:

- `.claude/docs/REGION_MODEL_DESIGN_JA.md` §5「現行の教師・損失・検証契約」
- `.claude/docs/research/2026-09-01-pseudo-target-probability-audit.md`
- `.claude/docs/work-logs/2026-09/2026-09-01-pseudo-label-construction-redesign.md`

本書は、次セッションで設計議論を再開せず、そのままテスト追加と実装へ入るための作業手順である。
疑似ラベル生成、data contract、loss、trainer、校正、outer fold 0の3アーム比較を、依存順に実装する。

---

## 0. 今回は再検討しない確定事項

### 0.1 教師信号

- whole-negative bagは4領域すべてhard 0。
- human valid cellはhard 0/1で、同じcellのCAMを必ず上書きする。
- whole-positiveかつhuman invalidのcellへfold-matched CAM soft target `q`を使う。
- 完全未注釈1,064 bagの4,256 cellと、部分注釈33 bagのunknown 89 cellが対象。
  corpus全体で重複を除いたpseudo-eligible cellは4,345。
- 部分注釈33 bagのknown 43 cellはhard targetのまま。
- confidence filter、hard top-1、cardinality推定、teacher disagreement weightは使わない。

### 0.2 CAMから`q`への変換

fold一致のBaseline 0 teacherだけを使い、各viewのCAMを元座標へ逆変換してから元の領域maskで
density enrichmentを集約する。viewは次の4つに固定する。

1. identity
2. horizontal flip
3. rotation +10 degrees
4. rotation -10 degrees

各view内で先に4領域shareへ正規化し、その後でview平均する。

```text
s_vr = e_vr / sum_j(e_vj)
s_r  = mean_v(s_vr)
x_r  = logit(clip(s_r, 0.01, 0.99))
q*_r = sigmoid(a_k * x_r + b_k)
q_r  = q*_r                         if sum_j(q*_j) >= 1
       q*_r / sum_j(q*_j)           otherwise
```

- `a_k,b_k`はstudent outer fold `k`ごとにfitする。
- studentのtraining folds内にある、4領域すべてhuman validなwhole-positive bagだけをfitに使う。
- 4領域共通のL2正則化logistic model（`C=1`）を使い、study単位grouped OOFで監査する。
- slopeは正でなければ生成失敗とする。
- region identifier、region別intercept、bag probability、CAM total、prevalence、cardinalityは入力しない。
- density enrichmentの面積補正指数は`gamma=1.0`から変更しない。
- 4-view実測後に係数をfitし直す。identity-CAMの暫定係数を本artifactへ流用しない。

### 0.3 損失

```text
L = L_whole + lambda * (L_E + L_P)
```

- `L_E`: human hard cellとwhole-negative logical-zero cellを連結したexact loss。
  Baseline 0と同じ`pos_weight=2.0`とweight-sum normalizationを使う。
- `L_P`: pseudo valid cellだけのmasked mean `BCEWithLogits(z, q)`。
- `L_P`には`pos_weight`、疑似loss係数`mu`、ramp、confidence weight、gradient capを置かない。
- `no_pseudo`は`L_P`を計算して0倍するのではなく、artifactを要求せず`L_P`を構築しない。
- `lambda`はwhole taskに対するregion objective全体の係数であり、pseudo専用係数ではない。
- 旧`alpha=0.5202`、旧`lambda=0.2053`、pairwise ranking、temperatureは使用しない。

### 0.4 Samplingと比較

- Baseline 0と同じnatural-distribution loader一本を使う。
- batch size 16、`EpochShuffleSampler`、約505 step/epochを維持する。
- human / negative / pseudoのsource-balanced loader、cycling、4/4/8構成を廃止する。
- whole mixupが発火したstepはregion mask poolingを定義できないため、region loss全体をskipする。
- 最初はouter fold 0だけで`no_pseudo`、`cam_soft`、`cam_soft_shuffled`を比較する。
- `cam_soft_shuffled`は固定seedでbag単位の`q`ベクトル対応だけを置換する。
  target値の集合、valid mask、hard target、人手優先規則は変えない。
- region checkpointはhuman validation macro AP、whole checkpointは既存`val_whole`定義で別々に選択する。
- `cam_soft`がhuman macro APで両controlを上回るまで、outer 1--4とsingle-region比較を実行しない。

---

## 1. 現状とブロッカー

### 1.1 現行実装との不一致

現行`region_branch`は旧ranking設計のままで、次を置換する必要がある。

| 領域 | 現行 | 置換後 |
|---|---|---|
| pseudo artifact | identity CAM raw score＋region temperature | 4-view share＋共有logit-share校正済み`q` |
| loader | natural＋human＋negative＋pseudoの4 loader | natural loader一本 |
| exact loss | source別0.5/0.5 reduction | hard cellを連結したweighted BCE |
| pseudo loss | bag間pairwise ranking | cell単位plain soft BCE |
| coefficient | `alpha`と`lambda` | `lambda`だけ |
| checkpoint | `val_total`で1本 | region/wholeを別々に保存 |

### 1.2 GPUブロッカー

2026-09-01時点では`nvidia-smi`がdriverと通信できず、SlurmにもGPU GRESがない。
したがって、CPUで実装・unit test・mock CAM smokeまでは進められるが、以下はGPU復旧後に行う。

- 4-view CAMの全対象再生成
- 4-view値によるpatient-grouped確率監査
- fold別`a_k,b_k`とpseudo target artifactの凍結
- CUDA preflightとouter fold 0本学習

次セッション開始時にGPUを再確認する。使えない場合もPhase 0--5は止めない。

### 1.3 現時点の診断値

全13,432 bag、53,728 region cellに対し、hard positiveは367、hard negativeは49,016、
pseudo-eligibleは4,345 cellである。

- `q>0`をbinary positiveとして数えるだけならpositive 4,712、negative 49,016、`10.40:1`、陽性率8.77%。
- identity-CAM暫定`q`のmeanは0.3442。soft positive massは1,862.4、negative massは51,865.6、
  `27.85:1`、positive-mass rate 3.47%。
- これらは観測値であり、pseudo `pos_weight`や`mu`を決める根拠にはしない。
- 旧weighted ratio `18.18:1`は廃止済みで、実装へ入れない。

---

## 2. 実装順序と依存関係

```text
Phase 0: 現行回帰線とtest contract固定
  -> Phase 1: q変換・4-view生成artifact
  -> Phase 2: datasetと単一loader
  -> Phase 3: L_E / L_P
  -> Phase 4: trainer・config・lambda校正
  -> Phase 5: 評価・3アーム制御
  -> Phase 6: GPUでartifact生成・確率監査
  -> Phase 7: outer-0 preflight・3アーム本学習
```

Phase 1--5は小さいunit testを先に追加してから実装する。Phase 6が通る前に学習runを開始しない。

---

## 3. Phase 0 — 回帰線とschemaを先に固定（最優先）

### 作業

1. `git status --short`で既存の未commit変更を記録し、他作業を上書きしない。
2. GPU状態を`nvidia-smi`と利用可能scheduler情報で再確認する。
3. 現行のBaseline 0 pseudo-label testとregion_branch testを実行し、変更前failureを記録する。
4. 新artifactの列、key、metadata versionをtest fixtureとして先に定義する。
5. 最初に次の2系列の失敗testを書く。
   - hard/pseudo targetのcell-wise precedence
   - exact weighted BCEとplain soft BCEの数式一致

### 新artifact契約

旧`pseudo_label_scores.csv`と`pseudo_label_temperatures.csv`は上書きしない。新しいoutput directoryへ、
少なくとも次の3 artifactを生成する。

| artifact | 内容 |
|---|---|
| `pseudo_region_targets.csv` | fold一致の4-view shareと校正済み`q` |
| `pseudo_target_calibration.csv` | fold別`a_k,b_k`、fit件数、study数、slope guard |
| `pseudo_target_generation_metadata.json` | view、式、hash、確率監査、件数、projection率 |

`pseudo_region_targets.csv`の一意keyは`(student_outer_fold, study_id, level)`とし、最低限、
`teacher_outer_fold`、teacher checkpoint hash、`vertebra_target`、各領域の`cam_share`、
各領域の`pseudo_target`を持たせる。`teacher_outer_fold == student_outer_fold`をloaderでも再検証する。

### 完了条件

- 新schema fixtureに旧temperature列が存在しない。
- artifact key重複、fold不一致、欠損、範囲外`q`を拒否するtestが先に存在する。
- 現行whole-path parity testの期待値を変更していない。

---

## 4. Phase 1 — 4-view疑似ターゲット生成

### 4.1 純粋関数を先に分離

`fracture_detection/baseline0/pseudo_labeling/calibration.py`を追加し、次の責務だけを持たせる。

- viewごとのnon-negative enrichmentを4領域shareへ変換
- 4-view shareの平均
- 1% clip後のshared logit-share logistic fit/apply
- `sum(q)>=1` projection
- finite/range/positive-slope guard
- reliability bin、quantile、projection率などmetadata用統計

CAM計算、file I/O、CLI argument処理をこのmoduleへ混ぜない。

### 4.2 TTAとgenerator更新

対象:

- `fracture_detection/baseline0/cli/generate_pseudo_labels.py`
- `fracture_detection/baseline0/pseudo_labeling/gradcam.py`
- `fracture_detection/baseline0/pseudo_labeling/cam_audit.py`
- `fracture_detection/baseline0/pseudo_labeling/__init__.py`
- `fracture_detection/baseline0/tests/test_pseudo_label.py`
- 必要なら新規`fracture_detection/baseline0/tests/test_pseudo_calibration.py`

実装内容:

1. 4 viewを固定順で生成する。
2. hflip/rotation後のCAMをnative frameへ逆変換する。
3. native frameのoriginal whole/region maskで`gamma=1.0`のdensity enrichmentを計算する。
4. **各view内でshare化してから**4 viewを平均する。
5. student outer foldのtraining foldsだけからcomplete whole-positive fit populationを作る。
6. study単位GroupKFoldの監査値を計算し、最後にfit population全体で`a_k,b_k`をfitする。
7. 同じfoldのwhole-positive rowsへ`q`を付け、新versionのartifactへ保存する。
8. `--limit-bags`では係数を凍結せず、artifact metadataへ`smoke_only=true`を記録する。

### test

- 合成CAMをhflip/rotationして逆変換したとき、region enrichmentがnative-frame期待値へ戻る。
- `mean(share(view))`であり、`share(mean(enrichment))`ではない。
- 全領域へ同じ係数が適用され、region-specific interceptを持てない。
- slopeが0以下なら失敗する。
- `sum(q*)<1`だけprojectionし、領域内順位を保存する。
- `q`がfiniteかつ`[0,1]`、projection後のwhole-positiveが`sum(q)>=1`。
- complete-only、whole-positive-only、student train-fold-only、study-groupedのfit populationになる。
- fixed inputとseedからCSV/metadataの内容が再現する。

### 完了条件

- CPU mockでgeneration CLIのend-to-end smokeが通る。
- 旧score/temperature artifactは変更されていない。
- 実GPU値がなくても、変換、fit、validation、serializationがunit test済み。

---

## 5. Phase 2 — Dataset contractとnatural loader一本化

### 5.1 Datasetが返す値

各bagはregion logitに対し、少なくとも次を返す。

| tensor | shape | 意味 |
|---|---|---|
| `region_exact_target` | `[4]` | human hard 0/1またはwhole-negative 0 |
| `region_exact_valid` | `[4]` | exact supervisionを適用するcell |
| `region_pseudo_target` | `[4]` | CAM soft `q` |
| `region_pseudo_valid` | `[4]` | whole-positive、human invalid、artifact有効 |

cell-wise規則:

```text
if vertebra_target == 0:
    exact_target = 0, exact_valid = true, pseudo_valid = false
elif human_valid:
    exact_target = human 0/1, exact_valid = true, pseudo_valid = false
else:
    exact_valid = false, pseudo_target = q, pseudo_valid = true
```

`exact_valid`と`pseudo_valid`は排他的でなければならない。whole-positiveのhuman invalid cellで
artifactが欠けている場合、黙ってskipせずエラーにする。ただし`no_pseudo` armはartifactを読まず、
pseudo supervision自体を要求しない。

### 5.2 Loader

対象:

- `fracture_detection/region_branch/data_pipeline/constants.py`
- `fracture_detection/region_branch/data_pipeline/pseudo_labels.py`
- `fracture_detection/region_branch/data_pipeline/dataset.py`
- `fracture_detection/region_branch/data_pipeline/loaders.py`
- `fracture_detection/region_branch/data_pipeline/sampling.py`
- `fracture_detection/region_branch/data_pipeline/sources.py`
- `fracture_detection/region_branch/tests/test_dataset.py`
- `fracture_detection/region_branch/tests/test_sampling.py`

作業:

1. fold一致`q` loaderとmetadata validationを実装する。
2. armに応じてtrain manifestへ`q`をattachする。
3. `cam_soft_shuffled`はouter foldごとにfixed seedのbag permutationを1回だけ作る。
4. `RegionBranchDataset`へexact/pseudoのtargetとvalid maskを実装する。
5. `OuterFoldLoaders`をnatural train loader＋eval loaderの構成へ単純化する。
6. 汎用batch-to-device helperはsource sampling moduleから`data_pipeline/batching.py`へ移す。
7. 新loaderとtrainerのtestが通ってから`source.py`とsource-specific samplerをruntimeから切り離す。
8. 参照がゼロになった段階で`data_pipeline/sources.py`と`data_pipeline/sampling.py`を削除する。

### test

- fully annotated positive、partially annotated positive、fully unannotated positive、whole-negativeを各1例。
- human valid cellがCAMを上書きする。
- 部分注釈のunknownだけpseudo validになる。
- whole-negativeはexact all-zeroでpseudo invalidになる。
- `no_pseudo`はpseudo artifactなしでloaderを構築できる。
- shuffled armは同seedで同一、別seedで変更、bag-level target vectorのmultisetは保存される。
- train samplerが全training bagの自然分布を一epoch一巡し、source cyclingを行わない。
- corpus manifest上でunique pseudo eligibilityが4,345 cellになる。

### 完了条件

- trainerから`human`、`negative`、`pseudo` loader参照がなくなる。
- temperatureのload、source ID、source poolがruntime data contractから消える。
- `exact_valid & pseudo_valid`が全cellでfalse。

---

## 6. Phase 3 — `L_E`と`L_P`の実装

対象:

- `fracture_detection/region_branch/modeling/losses.py`
- `fracture_detection/region_branch/modeling/__init__.py`
- `fracture_detection/region_branch/tests/test_losses.py`

### `L_E`

- active regionとregion pooling validを含むeffective maskを作る。
- hard targetだけを受け、targetが0/1でなければ失敗する。
- positive elementへ`pos_weight=2.0`を掛ける。
- Baseline 0と同じくweighted loss sumをeffective weight sumで割る。
- humanとwhole-negativeをsource別平均せず、同じmasked tensor上で一度だけreduceする。

### `L_P`

- effective pseudo-valid cellへ`torch.nn.functional.binary_cross_entropy_with_logits`
  の`reduction="none"`を使う。
- `sum(mask * cell_loss) / sum(mask)`のplain masked meanだけを返す。
- `pos_weight`、class weight、confidence weight、`mu`をargumentにも持たせない。
- valid pseudo cellがないbatchでは`logits.sum() * 0.0`型のgraph-connected zeroを返す。

### test

- `L_E`が手計算した`pos_weight=2.0`＋weight-sum normalizationと一致する。
- `L_E`でhumanとlogical zeroが同じreductionへ入る。
- `L_P`がPyTorchのunweighted soft BCE masked meanと完全一致する。
- `L_P`のgradientが`sigmoid(z)-q`に一致し、`sigmoid(z)=q`で0になる。
- native soft-label `pos_weight=2`の結果とは一致しないことを明示する。
- empty pseudo maskのlossがfinite zeroでbackward可能。
- active region外、pooling invalid、human override cellへgradientが入らない。
- `region_balanced_pairwise_ranking_loss`とtemperatureをimport/callしない。

### 完了条件

- public loss APIに`alpha`、pair、temperature、pseudo coefficientが存在しない。
- exact/pseudoの数式testがCPUで通る。

---

## 7. Phase 4 — Trainer、config、`lambda`校正

### 7.1 Trainer

対象:

- `fracture_detection/region_branch/training/trainer.py`
- `fracture_detection/region_branch/training/experiment.py`
- `fracture_detection/region_branch/training/monitoring.py`
- `fracture_detection/region_branch/tests/test_trainer.py`
- `fracture_detection/region_branch/tests/test_monitoring.py`

1 optimizer stepを次の順序へ固定する。

```text
natural batch取得
-> whole forward / L_whole / backward
-> mixup非発火なら同じnatural bagでregion forward
-> L_Eを構築
-> armがcam_soft系ならL_Pを構築
-> lambda * (L_E + L_P)をbackward
-> clip（設定されている場合）/ optimizer.step / scheduler.step
```

- `no_pseudo`ではsoft-loss functionを呼ばない。
- `cam_soft_shuffled`もloss関数は`cam_soft`と同一で、datasetへattachするcase対応だけを変える。
- whole/region backward後にoptimizer updateを1回だけ行う。
- mixup発火stepはregion forwardと`L_E/L_P`をskipし、その件数を記録する。
- validation lossは利用可能なexact/pseudo targetを同じ式で観測するが、model選択には使わない。
- `best_region.pt`はhuman validation macro AP最大、`best_whole.pt`は既存`val_whole`最大で保存する。
- early stoppingはprimary endpointであるhuman validation macro APに従い、既存patienceを維持する。

epoch logへ最低限、次を追加する。

- `train_whole_loss`, `train_exact_loss`, `train_pseudo_loss`, `train_region_loss`
- exact/pseudo valid cell数、hard positive/negative数、`sum(q)`、pseudo `q` quantile
- human valid cellが0のstep数、pseudo valid cellが0のstep数、region-skip step数
- `val_region_macro_ap`, region別AP/AUROC、`val_whole`
- 低頻度に観測した`||grad L_E||`, `||grad L_P||`と比

勾配比は診断ログだけに使い、pseudo weightの調整、cap、学習停止条件には使わない。

### 7.2 Config

対象:

- `fracture_detection/region_branch/config/schema.py`
- `fracture_detection/region_branch/config/region_branch_all.yaml`
- outer-0用の3 arm config（新規）
- `fracture_detection/region_branch/tests/test_train_cli.py`
- `fracture_detection/region_branch/tests/test_cli_runtime.py`

作業:

- protocol versionを上げ、`pseudo_arm`を`no_pseudo|cam_soft|cam_soft_shuffled`に限定する。
- `human_bags_per_batch`、`negative_bags_per_batch`、`pseudo_bags_per_batch`を削除する。
- `alpha`、rank、temperature、pseudo coefficient、ramp、confidence weightに相当するkeyを禁止する。
- `pseudo_label_dir`はCAM系armだけ必須、`no_pseudo`では不要にする。
- 3 arm間でarchitecture、augmentation、optimizer、schedule、seed、batch size、`lambda`が一致することをtestする。
- single-region configはouter-0 gate通過まで実行対象外とし、今回のrun範囲を0だけに固定する。

### 7.3 `lambda`校正

対象:

- `fracture_detection/region_branch/training/calibration.py`
- `fracture_detection/region_branch/cli/calibrate.py`
- `fracture_detection/region_branch/tests/test_calibration.py`

- `alpha`校正を削除する。
- model更新前のdeterministic calibration batchesでshared trunk上の`L_whole`と`L_E+L_P`の
  gradient normを測り、既存のregion-vs-whole target比からfold別`lambda`だけを再計算する。
- outer foldごとに統合4領域`cam_soft` objectiveで1回だけ校正し、同じfoldの3 armへ同じ`lambda`を使う。
- armごとの再校正やhuman validation性能を使ったgrid searchはしない。
- 旧calibration artifactは読み込まず、新versionとconfig fingerprintで保存する。

### test

- 1 batchにつきnatural loaderの`next()`が1回だけ。
- sequential backward後のparameter gradientが合成lossのgradientと一致する。
- optimizer stepは1回、mixup stepはregion loss 0回。
- `no_pseudo`でpseudo loss functionが呼ばれない。
- checkpoint trackerがregion/wholeで独立する。
- calibration resultに`alpha`がなく、`lambda`だけが存在する。
- 3 armが同じ`lambda` artifactを参照する。

### 完了条件

- runtime logとcheckpointに`alpha`、rank loss、temperature、source pool countが出ない。
- `best_region.pt`と`best_whole.pt`のepochが独立に記録される。
- 旧`val_total`がcheckpoint selectorではない。

---

## 8. Phase 5 — 評価と3アーム制御

対象:

- `fracture_detection/region_branch/evaluation/metrics.py`
- `fracture_detection/region_branch/cli/evaluate.py`
- 新規`fracture_detection/region_branch/cli/compare_pseudo_arms.py`
- 対応するCLI/evaluation test

作業:

1. region評価はhuman valid cellだけでregion別AP/AUROCとmacro APを計算する。
2. `best_region.pt`からregion prediction、`best_whole.pt`からwhole predictionを出す。
3. 3 armのouter-0 predictionを`(study_id, level)`で一対一joinする。
4. `cam_soft - no_pseudo`と`cam_soft - cam_soft_shuffled`のmacro AP差を保存する。
5. study単位paired bootstrap CIを副次的に保存するが、未登録の有意差閾値を後付けしない。
6. artifact hash、code/config fingerprint、seed、checkpoint epochを比較reportへ保存する。
7. 既存の統合model対single-region比較CLIは変更せず、outer-0 gate後まで使用しない。

### 完了条件

- 3 armの評価母集団とhuman valid maskが完全一致する。
- shuffled armはtarget値分布を保ち、case対応だけが異なることをreportで監査できる。
- `cam_soft`が両controlを上回ったかを1つのJSON/CSVで判定できる。

---

## 9. Phase 6 — GPU復旧後のartifact生成と確率監査

### 実行前

- 全5 teacher checkpointの存在、role、SHA-256を確認する。
- input manifestとdatasetのhashを記録する。
- 新しい空のoutput directoryを使い、旧`pseudo_labels` directoryへ`--overwrite`しない。
- 小数bagのGPU smokeで4-view逆変換とVRAM/時間を確認する。

### 本生成

1. 全student outer foldについて、fold一致teacherで対象whole-positive bagの4-view CAMを生成する。
2. complete human positiveだけでpatient-grouped OOF auditを再実行する。
3. 各foldのfinal`a_k,b_k`をfitする。
4. 全pseudo対象へ`q`を生成し、新artifactへ保存する。
5. `.claude/docs/research/scripts/2026-09-01-audit_pseudo_target_probabilities.py`を
   新artifact入力へ対応させ、確率値を実測する。

### artifact受入条件

- 全foldで`a_k > 0`かつfinite。
- すべての`q`がfiniteかつ`0 <= q <= 1`。
- whole-positive vectorはprojection後に`sum(q)>=1`。
- `(student_outer_fold, study_id, level)`が一意でfold matchingが成立する。
- human override前の対象が欠落せず、source manifestでunique 4,345 pseudo-eligible cellを説明できる。
- complete calibration bag数・study数が監査表と一致するか、差分理由がmetadataに記録される。
- reliability bins、Brier、log loss、ECE、AP/AUROC、`q` quantile、`sum(q)` quantile、projection率を保存する。
- 4-view後の確率分布をidentity-CAM診断値と比較し、係数とartifactをここで初めて凍結する。

いずれかを満たさない場合は学習へ進まず、生成artifactを`rejected`として残して原因を調べる。
失敗を隠すためのclip、係数差し替え、confidence filterは追加しない。

---

## 10. Phase 7 — outer fold 0 preflightと3アーム本学習

### preflight

1. `no_pseudo`で数step実行し、artifactなし、`L_P`未構築、finite lossを確認する。
2. `cam_soft`でpseudo cell count、`q` mass、`L_E/L_P`、gradient ratioを確認する。
3. `cam_soft_shuffled`でfixed permutationとtarget multiset保存を確認する。
4. 3 armのeffective config diffが`pseudo_arm`とartifact association以外にないことを確認する。
5. region/wholeの2 checkpointが別metric・別epochで保存可能なことを確認する。

### 本学習

- outer fold 0だけを、同じcode、seed、schedule、`lambda`で3 arm実行する。
- primary判定はhuman validation macro AP。
- validation BCEは観測するが、soft target entropy floorがあるためarm採否に使わない。
- whole metric、collapse diagnostic、region別AP/AUROC、学習曲線を併記する。

### 次段階へ進むgate

```text
AP(cam_soft) > AP(no_pseudo)
and
AP(cam_soft) > AP(cam_soft_shuffled)
```

gateを満たすまではouter 1--4、single-region 4本、TTA uncertainty weightingを開始しない。
満たさない場合は、まずtarget-case対応に有効信号がないのか、学習実装が信号を使えていないのかを、
shuffled control、gradient、human-only曲線から切り分ける。pseudo係数を後付けして救済しない。

---

## 11. 検証コマンド

すべてrepo rootから`uv`経由で実行する。NFS cache問題を避けるためcacheは`/tmp`へ置く。

### 変更箇所のunit test

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run pytest -q \
  fracture_detection/baseline0/tests/test_pseudo_label.py \
  fracture_detection/baseline0/tests/test_pseudo_calibration.py \
  fracture_detection/region_branch/tests/test_dataset.py \
  fracture_detection/region_branch/tests/test_losses.py \
  fracture_detection/region_branch/tests/test_calibration.py \
  fracture_detection/region_branch/tests/test_trainer.py
```

### package test

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run pytest -q \
  fracture_detection/baseline0/tests \
  fracture_detection/region_branch/tests
```

### lint・format・type check

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run ruff check \
  fracture_detection/baseline0 \
  fracture_detection/region_branch
UV_CACHE_DIR=/tmp/vai-uv-cache uv run ruff format --check \
  fracture_detection/baseline0 \
  fracture_detection/region_branch
UV_CACHE_DIR=/tmp/vai-uv-cache uv run ty check \
  fracture_detection/baseline0 \
  fracture_detection/region_branch
```

`ty`が現環境に未導入なら、依存関係を勝手に変更せず環境blockerとして記録し、
既存`uv run poe typecheck`の結果も併記する。既知の無関係failureは修正対象に含めない。

### 旧runtime経路が残っていないことの静的確認

```bash
rg -n "alpha|rank_loss|pairwise|pseudo_bags_per_batch|load_pseudo_temperatures|SOURCE_" \
  fracture_detection/region_branch
```

研究archive、旧artifact readerの完全削除前履歴、明示的な禁止key test以外にactive参照がないことを確認する。

---

## 12. Rollback計画

### artifact

- 旧`fracture_detection/baseline0/outputs/08_19/pseudo_labels`を上書きしない。
- 新しいversion付きoutput directoryだけへ書く。
- CSV/metadataは一時fileへ書いてからatomic renameし、途中生成を完成品として読ませない。
- 受入失敗artifactは削除せず`rejected` metadataと理由を残す。

### code

- `sources.py`、`sampling.py`、旧temperature readerは、新しいtestとruntime切替がgreenになるまで削除しない。
- Phaseごとにtestを通し、failure時はそのPhaseの変更だけを戻せる粒度に保つ。
- whole-path architecture、Baseline 0 checkpoint、manifest、augmentationの凍結値は変更しない。
- 旧`region-branch-v3` artifactを新protocolで暗黙に読むfallbackは作らない。

### experiment

- 各armは別output directoryへ保存し、resume時はeffective configとartifact hash一致を必須にする。
- preflight outputとfull run outputを分離する。
- 失敗runのcheckpointを別armへ流用しない。
- outer-0 gate失敗時は残りfoldを走らせず、原因分析へ戻る。

---

## 13. 変更予定ファイル一覧

### Baseline 0 pseudo generation

- `fracture_detection/baseline0/cli/generate_pseudo_labels.py`
- `fracture_detection/baseline0/pseudo_labeling/calibration.py`（新規）
- `fracture_detection/baseline0/pseudo_labeling/gradcam.py`
- `fracture_detection/baseline0/pseudo_labeling/cam_audit.py`
- `fracture_detection/baseline0/pseudo_labeling/__init__.py`
- `fracture_detection/baseline0/tests/test_pseudo_label.py`
- `fracture_detection/baseline0/tests/test_pseudo_calibration.py`（新規候補）

### Region data/loss/training

- `fracture_detection/region_branch/data_pipeline/constants.py`
- `fracture_detection/region_branch/data_pipeline/pseudo_labels.py`
- `fracture_detection/region_branch/data_pipeline/dataset.py`
- `fracture_detection/region_branch/data_pipeline/loaders.py`
- `fracture_detection/region_branch/data_pipeline/batching.py`（新規候補）
- `fracture_detection/region_branch/data_pipeline/sources.py`（切替後に削除）
- `fracture_detection/region_branch/data_pipeline/sampling.py`（切替後に削除）
- `fracture_detection/region_branch/modeling/losses.py`
- `fracture_detection/region_branch/training/calibration.py`
- `fracture_detection/region_branch/training/trainer.py`
- `fracture_detection/region_branch/training/experiment.py`
- `fracture_detection/region_branch/training/monitoring.py`
- `fracture_detection/region_branch/config/schema.py`
- `fracture_detection/region_branch/config/*.yaml`

### Evaluation/tests/docs

- `fracture_detection/region_branch/evaluation/metrics.py`
- `fracture_detection/region_branch/cli/calibrate.py`
- `fracture_detection/region_branch/cli/train.py`
- `fracture_detection/region_branch/cli/evaluate.py`
- `fracture_detection/region_branch/cli/compare_pseudo_arms.py`（新規）
- `fracture_detection/region_branch/tests/test_*.py`の関連test
- `.claude/docs/research/scripts/2026-09-01-audit_pseudo_target_probabilities.py`
- `.claude/docs/REGION_MODEL_DESIGN_JA.md`（実装後に実体と照合）
- `.claude/docs/DESIGN.md`（重要な差分が出た場合だけ更新）

---

## 14. 次セッションの開始チェックリスト

- [ ] 本計画書と`REGION_MODEL_DESIGN_JA.md` §5を読む。
- [ ] `git status --short`で既存変更を保護する。
- [ ] GPU可否を再確認する。
- [ ] 現行targeted testsのbaseline結果を保存する。
- [ ] **最初の編集としてtarget precedence testとloss testを書く。**
- [ ] Phase 1の純粋関数から実装する。
- [ ] 各Phase完了時に本書へ実施内容、test結果、残課題を追記する。
- [ ] GPUがなければPhase 5までで止め、4-view係数を仮定して学習しない。
- [ ] GPU復旧後はPhase 6のartifact gateを通してからouter-0を実行する。
- [ ] outer-0の3 arm gate通過前に残りfoldへ拡張しない。

### 次セッションで最初に実行するコマンド

```bash
git status --short
nvidia-smi
UV_CACHE_DIR=/tmp/vai-uv-cache uv run pytest -q \
  fracture_detection/baseline0/tests/test_pseudo_label.py \
  fracture_detection/region_branch/tests/test_dataset.py \
  fracture_detection/region_branch/tests/test_losses.py \
  fracture_detection/region_branch/tests/test_calibration.py \
  fracture_detection/region_branch/tests/test_trainer.py
```

