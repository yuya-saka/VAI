# Baseline 0：椎体単位骨折分類

4領域情報を教師として使わず、CTと椎体全体maskだけから椎体単位の骨折確率を予測する研究全体の起点です。

## ディレクトリ構成

| ディレクトリ | 責務 |
|---|---|
| `cli/` | 学習・OOF評価のエントリポイント |
| `config/` | 凍結YAMLと設定schema |
| `data/` | manifest読み込み、augmentation、共有staging |
| `modeling/` | networkとloss |
| `training/` | optimizer、学習loop、実験成果物管理 |
| `tests/` | 上記責務のunit tests |

直下には案内用`README.md`とpackage定義`__init__.py`だけを置き、実装は責務別packageへ分離します。

## 入力とモデル

- 入力: `float32[B, 15, 6, 224, 224]`
- 各面: CT 5 channel + 椎体全体mask 1 channel
- backbone: ImageNet事前学習済み `tf_efficientnetv2_s`
- sequence model: hidden 256、2層の双方向LSTM
- head: `Linear -> BatchNorm -> Dropout(0.3) -> LeakyReLU -> Linear`
- 出力: 15面のlogit。椎体確率は各面のsigmoid平均

RSNA 2022 1位解法のType1を基準に、15面・6ch・EfficientNetV2-S・BiLSTM・75 epoch・初期LR `2.3e-4`・最小LR `2.3e-5`・75 epochの単一cosine周期を採用します。参考コードは`CosineAnnealingWarmRestarts(T_0=75)`ですが、75 epoch内にrestartは発生しないため、同じ軌跡を`CosineAnnealingLR(T_max=75)`で実装します。研究全体で固定した上書き条件として、batch size 16、BF16、weight decay `1e-4`、gradient clip `5.0`、`pos_weight=2.0`を使います。BCEはStage1と同じく陽性要素を2倍した後、要素数ではなく重み合計で正規化します。

学習母集団はStage1と同じ`excluded_studies.csv` / `excluded_levels.csv`を適用した`13,432 bag / 2,009 study / 陽性1,332`です。品質除外前の共通manifestから496 bagを除外し、除外CSV自体のSHA256もmanifest metadataへ保存します。領域注釈済み268 bagは除外対象に含まれません。

## データ拡張

全15面×5 CT channelを75 channelへ積み、15面maskもchannelへ積んで、Stage1と同じく1 bagにつき1回のtransformで同期拡張します。Stage1のうちorientationを変えるhorizontal flip、vertical flip、transposeだけを除外し、brightness `p=0.7`、Affine（shift `0.3`、scale `0.7–1.3`、rotate `±45°`、`BORDER_REFLECT_101`、`p=0.7`）、blur/noise `p=0.5`、distortion `p=0.5`、112 px cutout `p=0.05`を使います。batch-level mixupもStage1と同じ適用確率`0.2`、一様分布`λ∼U(0,1)`です。

CPU側はCTとmaskを`uint8`のままDataLoaderへ返し、nonblocking GPU転送後にfloat32・0〜1へ正規化します。Baseline 0では不要な`region_4class.npy`を学習時に読みません。旧実装比のローカル計測ではdataset item生成が中央値`0.434→0.109秒`（約4.0倍）、batch転送payloadが`17.23→4.31 MiB/item`です。固定shapeのcuDNN autotuneも有効化します。

## Nested選択

凍結済み5-foldを再生成せず、outer fold `k` ごとに次の役割を割り当てます。

```text
outer = k
val = (k + 1) mod 5
train = 残り3 fold
```

trainのnatural streamは`EpochShuffleSampler`により`seed + outer + epoch`だけで順序を決めます。LR軌跡はvalidationに依存しない固定cosineとし、primary checkpointはval椎体AUROC、secondary checkpointはval PR-AUC（実装はaverage precision）、early stoppingはval BCEで決定します。内部のnested split設定ではこのval foldを`inner_fold`として保持します。

epochごとにAUROC、PR-AUCに加えて、固定閾値0.5のprecision・recall・F1と、val内でF1を最大化した閾値およびそのprecision・recall・F1をconsole、`history.csv`、W&Bへ記録します。正式評価では各checkpoint自身のval予測からF1最大の閾値を決め、同率なら高い閾値を採用します。その閾値を対応するouter予測へ固定適用し、AUROC・PR-AUC・precision・recall・F1を保存します。AUROC-bestをprimary、PR-AUC-bestをsecondary感度分析として事前に区別し、結果を見てcheckpointを選び直しません。各checkpointのouter推論は1回だけ許可します。

## 実行

```bash
uv run python fracture_detection/baseline0/cli/train.py \
  --config fracture_detection/baseline0/config/baseline0.yaml

uv run python -m fracture_detection.baseline0.cli.evaluate \
  --config fracture_detection/baseline0/config/baseline0.yaml
```

一部outerだけ実行する場合は`--start-outer-fold`と`--end-outer-fold`を指定します。`--resume`は同一の実効configに限り許可され、旧Baseline 1やmatched checkpointは再開できません。

### 共有キャッシュの進捗表示

`/dev/shm/vai-fracture-dataset`へのstagingでは、無表示の工程を作りません。標準出力へ次を順次表示します。

NFSからの内容コピーは`data.stage_copy_workers`で並列化し、既定値は8です。`1`〜`32`を指定できますが、共有NFSへの過負荷を避けるため通常は8を上限目安とします。cache全消去は初回約65.9 GiBのコピーを毎回繰り返すため高速化には使いません。同じmanifestの中断済み一時directoryだけをlock取得後に自動削除します。

学習DataLoaderも既定8 workersです。各workerでOpenCVを1 threadへ制限し、8 workers × OpenCV 32 threadsの過剰並列を防ぎます。batch size 16・prefetch factor 2では、実入力Tensorだけで最大約4.3 GiBのprefetch量です。約65.9 GiB cache配置後のtmpfs余裕とhost RAM余裕の範囲内です。`history.csv`とW&Bへ`train_data_wait_seconds` / `train_compute_seconds` / `train_mixup_fraction`を記録し、worker追加後もGPU待ちが残るか確認します。

- manifest SHA256と確定先
- process間lockの待機・取得・解放
- 同一manifestの未完了一時cache確認・削除
- source全ファイルの走査件数、現在path、総容量
- 既存cacheのmarker・ファイル数検証
- data・予約領域・必要量・空き容量
- copy済みbytes、ファイル数、現在path、速度、経過時間
- READY marker作成、一時directoryから確定先へのatomic移行、再利用判定

既に起動中のPython processへコード変更は反映されないため、旧版でstaging中の場合はそのrunを完走させるか、停止後に新しいprocessで再実行する必要があります。

## 成果物

`outputs/{phase}/{name}/outer{k}/`へ以下を保存します。

- `effective_config.yaml`
- `best_model.pt`（val AUROC最大、outer推論用）
- `best_val_prauc_model.pt`（val PR-AUC最大、追加保存）
- `last_checkpoint.pt`（resume用）
- `history.csv` / `training.log`
- `val_predictions.csv`
- `outer_predictions.csv`
- `val_predictions_prauc_checkpoint.csv`
- `outer_predictions_prauc_checkpoint.csv`
- `fold_metrics.json`

各予測CSVは`decision_threshold`と`vertebra_prediction`を含みます。5 outer完了後、実験rootへprimaryの`oof_predictions.csv` / `oof_metrics.json`とsecondaryの`oof_predictions_prauc_checkpoint.csv` / `oof_metrics_prauc_checkpoint.json`を保存します。ローカル成果物を正とし、W&Bへcheckpointや個票予測は送信しません。

## 検証

```bash
uv run pytest fracture_detection/common/tests fracture_detection/baseline0/tests -q
uv run ruff format --check fracture_detection/common fracture_detection/baseline0
uv run ruff check fracture_detection/common fracture_detection/baseline0
uv run mypy fracture_detection/common fracture_detection/baseline0 \
  --exclude tests --ignore-missing-imports
```
