# 2026-08-18: 共通基盤・Baseline 0実装

## 実施内容

- 旧`fracture_detection/baseline1/`を現行名称`baseline0/`へ移行し、matched設定を廃止
- `memo/参考コード_RSNA`を基準に、EfficientNetV2-S、2層BiLSTM、15面、broadcast BCE、mean-sigmoid、75 epoch単一cosineを固定
- outer=k / inner=(k+1)%5 / 残り3 folds学習を共通化し、outerはbest checkpoint確定後に1回だけ推論
- natural streamとannotated streamのshuffle-without-replacement samplerを決定論的に実装
- region BCEを明示的なregion label validityだけに限定し、椎体陰性の論理的0教師を削除
- lambda/betaの64 batch初期gradient-norm校正を追加し、校正前後のmodel・optimizer state一致を検証
- 268陽性だけを対象とするcross-fitted level-only床と患者cluster MDE成果物を凍結
- `baseline0/`直下の実装を`cli/config/data/modeling/training`へ責務別に分離し、直下をREADMEとpackage定義だけに整理
- 共有stagingのlock待機からatomic確定までを可視化し、copy中は総bytes・file count・current path・速度を継続表示
- NFS→tmpfs内容copyを既定8 threadへ並列化し、同一manifestの中断tmpだけをlock下で自動削除

## 凍結成果物

- `fracture_detection/folds/outputs/`: 13,928 bagのmanifest・fold割当・監査レポート
- `fracture_detection/common/outputs/level_floor_predictions.csv`: 5 outer pooled予測
- `fracture_detection/common/outputs/level_floor_metrics.json`: R1 0.4946387 / R2 0.2863489 / R3 0.4222059 / R4 0.7058684
- `fracture_detection/common/outputs/region_floor_power.json`: 10,000回patient-cluster bootstrap SEとper-region MDE
- AP tie規約: scikit-learn 1.9.0 `average_precision_score`の同一threshold grouping

## 検証

- `uv run --no-sync pytest fracture_detection/common/tests fracture_detection/baseline0/tests`: 45 passed
- `uv run --no-sync ruff check fracture_detection/common fracture_detection/baseline0`: passed
- `uv run --no-sync mypy --ignore-missing-imports --exclude '/tests/' fracture_detection/common fracture_detection/baseline0`: 27 source files passed
- 実データ1 bagでV2-S forward / broadcast BCE / backwardを実行し、finite gradientを確認

## 実行状態

旧v2の`08_18/v1`でouter 0学習が開始されたが、AUROC停止・4 workersのprocessであり、正式runには使わない。正式なv3学習とouter推論は未開始。全6構成、outer別lambda/beta、検定順序、code/config hashを凍結してからouter推論を実行する。Baseline 0だけ先にouter結果を見ることは禁止する。

## Scheduler訂正

初回実装の`ReduceLROnPlateau`は、廃止済みmatched fold-0診断で未知の有効epoch数へ対応するために導入した設定であり、full 13,928 bagのBaseline 0へ引き継ぐ根拠がなかった。参考Type1は初期LR `2.3e-4`、最小LR `2.3e-5`、75 epochの`CosineAnnealingWarmRestarts(T_0=75)`で、学習期間内にrestartは起きない。したがって同じ単一周期を`CosineAnnealingLR(T_max=75)`で実装し、protocolを`baseline0-nested-v2`へ更新した。これによりLR軌跡はinner BCEやarmごとの学習挙動に依存せず、全比較構成でepochごとに一致する。

## Package整理

実装責務を`cli/`（entry points）、`config/`（schemaとYAML）、`data/`（datasetとstaging）、`modeling/`（networkとloss）、`training/`（optimizer、trainer、experiment management）へ分離した。互換性用のroot-level wrapperは置かず、CLIは`baseline0/cli/`を正規経路とする。成果物rootは移動前と同じ`baseline0/outputs/`を維持する。作業中に設定された`experiment.phase/name = 08_18/v1`は実験識別子として保持し、空文字・`.`・`..`・path区切りだけを拒否する可変値とした。

## Staging進捗

共有`/dev/shm` cacheの準備中に停止と誤認しないよう、`data/staging.py`はmanifest/確定先、lock待機・取得・解放、source inventory、既存cache検証、容量内訳、copyのbytes/file/current path、経過時間・速度、READY marker、atomic rename、再利用を標準出力へ表示する。copyは一時directory内で完結してからrenameする従来の完全性契約を維持する。

68.3 GiB / 41,784 filesの逐次`copy2`はNFS上で約45〜60 MiB/sに留まったため、内容だけを`copyfile`で転送する8-thread `ThreadPoolExecutor`へ変更した。worker数はYAMLの`data.stage_copy_workers`（1〜32）で制御する。`/dev/shm`全消去は完成cacheの再利用を失わせるため行わず、同じmanifest hashのPID別一時directoryだけをmanifest lock取得後に削除する。

## Early stopping・学習throughput訂正

ユーザー指定により停止判定をinner BCEへ変更し、AUROCはbest checkpoint選択専用とした。protocolは`baseline0-nested-v3`、patienceは15。resume stateへ`early_stopping_best_loss`とbad epoch数を保存し、history/W&Bにも両状態を分けて記録する。旧`08_18/v1`はv2 processとして既に起動しており、コード変更は反映されないため、成果物を保持したまま`08_18/v2_val_loss_stop`へrunを分離した。

旧v1のW&B system metricsではepoch 2/3のtrain区間が約346〜349秒、inner validationが約25秒だった。GPU 0利用率はtrain中でも平均60.4% / 55.6%、50%未満のsampleが37.5% / 43.5%あり、GPU memory allocationは約40.8%だったため、主ボトルネックを入力pipelineと判定した。DataLoaderを4から8 workersへ増やす一方、OpenCV既定値が32 threads/processだったため、worker初期化時に`cv2.setNumThreads(1)`を適用して過剰並列を防ぐ。batch 16・prefetch factor 2・8 workersの入力Tensor上限は1 loaderあたり約4.3 GiBで、hostのavailable RAM約148 GiBおよび68.3 GiB cache配置後のtmpfs余裕に収まる。sandbox外の軽量datasetで8 workersの起動とbatch取得を確認した。

ユーザー向け表示は`inner`ではなく`val`へ統一した。対象はconsole、progress bar、`history.csv`、W&B、`fold_metrics.json`、`val_predictions.csv`。nested splitの整合検証に使う内部key `runtime.inner_fold`だけは変更しない。

ユーザー指定によりval PR-AUC最大checkpointを追加し、protocolを`baseline0-nested-v4`へ更新した。`best_model.pt`は従来どおりval AUROC最大でouter推論に使用し、`best_val_prauc_model.pt`はaverage precision最大の診断用重みとして独立保存する。`last_checkpoint.pt`はAUROC-bestとPR-AUC-best双方のepoch/metricsを保持する。history/W&Bには`val_prauc`、`is_best_val_auroc`、`is_best_val_prauc`を分離記録する。PR-AUC-bestをouter推論や正式なarm比較へ使用しない。

実行識別子とdeviceはユーザーが設定した`experiment.phase/name = 08_18/v2`、`gpu_id = 1`を保持した。v4成果物はまだ作成されておらず、旧v1成果物とは衝突しない。

更新後の検証はcommon + Baseline 0の49 unit tests、Ruff check/format、mypy 27 source filesがすべて通過した。

## Threshold-based formal evaluation

ユーザー指定によりprotocolを`baseline0-nested-v5`へ更新した。epochログは固定閾値0.5の
precision / recall / F1と、val内F1最大閾値およびそのprecision / recall / F1をconsole、
`history.csv`、W&Bへ記録する。正式評価ではAUROC-bestとPR-AUC-bestの各checkpoint自身の
val予測でF1を最大化し、同率なら高い閾値を選ぶ。その閾値を対応するouterへ固定適用し、
両checkpointについてAUROC / PR-AUC / precision / recall / F1をfold別・pooled OOFで保存する。
AUROC-bestはprimary、PR-AUC-bestはsecondary感度分析であり、outer結果を見た選び直しは禁止。
各checkpointのouter推論は1回に制限し、v4の`08_18/v2`と分離してv5は`08_18/v3`へ出力する。
outer上のF1最適閾値は計算も保存もせず、valで凍結した閾値の適用結果だけを正式値とする。
検証はcommon + Baseline 0の56 unit tests、Ruff check/format、mypy 27 source filesが通過した。

## Stage1 parity: augmentation・品質除外・loss

ユーザー指定によりprotocolを`baseline0-nested-v6`へ更新した。`train_models/stage1`と同じ`excluded_studies.csv` / `excluded_levels.csv`を共通manifestへ適用し、3ファイル完備13,928 bagから496 bagを除外した。新しい母集団は13,432 bag / 2,009 study / 陽性1,332、領域注釈268 bagは不変。manifest SHA256は`9bc0b8b91a5ff719519a63a3b2a7aa7f14476b45fade5582efb58a258ef21ac3`で、除外CSVのSHA256と適用件数もmetadataへ保存する。既存`folds.csv`は再生成せず、除外後も同じpatient-fold割当を使う。

augmentationはR2/R3の向きを維持するためhorizontal/vertical flipとtransposeだけを引き続き禁止し、それ以外をStage1へ揃えた。brightness `p=0.7`、Affine（shift 0.3、scale 0.7–1.3、rotate ±45°、`BORDER_REFLECT_101`、`p=0.7`）、4種blur/noise `p=0.5`、optical/grid distortion `p=0.5`、112 px cutout `p=0.05`を全15面・全channelへ同じReplayで適用する。batch-level mixupはStage1と同じ`p=0.2`、一様`λ∼U(0,1)`、共有permutationで、適用率をconsole/history/W&Bへ記録する。

whole BCEはPyTorch `pos_weight`の要素数平均を廃止し、Stage1と同じく陽性要素のlossとnormを2倍して`weighted_loss.sum() / weight.sum()`で正規化する。train/validation epoch lossもStage1と同じbatch loss平均へ揃えた。v5以前のloss値・checkpoint・early-stopping状態とは数値定義が異なるためresumeせず、新規出力先`08_18/v4`を使う。

## Stage1 input-path高速化

protocol `baseline0-nested-v7`では、Stage1の`_augment_volume`と同じく15面×5 CT channelを75 channel、15面maskを15 channelへstackし、Albumentationsを1 bagにつき1回だけ呼ぶ。旧Baseline0は同一Replayを面・CT channelごとに75回呼んでいた。224×224 synthetic bagの中央値はaugmentation単体`0.440→0.145秒`（3.0倍）、実bagのloadからinput生成までは`0.434→0.109秒/item`（4.0倍）へ短縮した。

Baseline 0で使わない`region_4class.npy`のload・5 mask channel構築も省き、CTとwhole maskはuint8のままDataLoaderからnonblocking転送し、GPU上でfloat32・0〜1へ正規化する。whole maskは転送前に0/255へ変換するため、モデル入力時の0/1値は従来と同一。1 itemのDataLoader payloadは`17.23→4.31 MiB`となる。固定224×224 shape向けにStage1と同じ`cudnn.benchmark=true`も有効化した。v6の`08_18/v3/config.yaml`はstaging容量不足で学習開始前に終了した記録として残し、v7は`08_18/v4`へ分離する。
