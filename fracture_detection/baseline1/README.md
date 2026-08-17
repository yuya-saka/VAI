# Baseline 1：椎体単位骨折分類器

Baseline 1は、C1--C7の各椎体について二値の骨折スコアを1つ予測します。共通の2.5D入力を使用し、学習時に領域ターゲットは使用しません。

## モデル

入力は`float32[B, 15, 6, 224, 224]`です。

- CTチャネル0--4は共通の`FractureDataset`から取得します。
- チャネル5は二値の椎体全体マスクです。
- 15枚の各面をtimmのEfficientNetV2バックボーンに入力します。
- 隠れ次元256の2層双方向LSTMで、順序付けられた面列をモデル化します。
- `Linear -> BatchNorm -> Dropout -> LeakyReLU -> Linear`により、椎体ごとに15個の面logitを出力します。

バックボーンの特徴次元はtimmから動的に取得します。事前学習済みの6チャネルstemには、timm 1.0.22標準の入力畳み込み適応を使用します。

## 目的関数とスコア

椎体ターゲットを15個の面logitへ複製し、`pos_weight=2.0`の`BCEWithLogitsLoss`で学習します。推論スコアは15個のsigmoid確率の平均です。`pos_weight`は`matched`・`full`の全設定で2.0に固定します。focal loss、クラス均衡サンプリング、mixup、反射、転置、歪み、cutout、EMAは使用しません。

## 最適化設定

`matched`は2,122--2,125 training bag/fold（batch size 16で約133 step/epoch）です。以前の200 epoch・10 epoch backbone freezeは、約430 training bag/foldを前提に決めた値をそのまま流用しており、現行コホートではhead-only更新が約1,330 stepに膨張していました。固定cosine期間を事前に決めず、validationの停滞から学習率を調整します。

| 設定 | `matched` | `full` |
|---|---:|---:|
| 最大epoch（安全上限） | 100 | 75 |
| backbone freeze | 0 epoch | 0 epoch |
| warmup | 2 epoch | なし |
| scheduler | `ReduceLROnPlateau` | `ReduceLROnPlateau` |
| scheduler monitor | `val_bce`（minimize） | `val_bce`（minimize） |
| backbone initial / minimum LR | `1e-4` / `1e-6` | `2.3e-4` / `2.3e-5` |
| LSTM・head initial / minimum LR | `3e-4` / `3e-6` | `2.3e-4` / `2.3e-5` |
| warmup開始係数 | 0.1 | 1.0 |
| LR reduction | factor 0.5、patience 4、cooldown 1 | 同左 |
| relative threshold | 0.1% | 0.1% |
| early stopping patience | 15 | 15 |
| checkpoint選択開始 | epoch 1 | epoch 1 |
| gradient clip global norm | 5.0 | 5.0 |

`matched`は2 epochのstep単位warmup後、`val_bce`が相対0.1%以上改善しない状態を4 epoch許容してLRを0.5倍にします。cooldownは1 epochです。checkpoint選択とearly stoppingは従来どおり`val_auroc`を使うため、schedulerの滑らかな最適化指標と主評価指標を分離しています。100 epochは到達目標ではなく安全上限で、通常はAUROC patience 15により先に終了します。AdamW、weight decay `1e-4`、batch size 16、bf16、陽性重み2.0は維持しています。

## データ設定

| 設定 | コホート | バックボーン | ステージング |
|---|---:|---|---|
| `matched` | 2,655 bag / 1,498 study | B0主解析、V2-S感度解析 | NFSを直接読み込み |
| `full` | 13,928 bag / 2,010 study | V2-S | 共有`/dev/shm`キャッシュ |

`matched`は固定済みの`cohorts/outputs/matched_cohort.csv`を読み込みます。アノテーション済み陽性268 bagをすべて保持し、`full`の陽性率10.095%を再現するよう陰性2,387 bagを抽出します。陰性bagは`full`のfold × 頸椎level分布に比例させます。`full`は固定済みの共通マニフェストを読み込みます。どちらの設定も固定済みの5分割割当を使用します。

## 設定ファイル

- `config/matched_b0.yaml`：matchedの主解析です。現在の改訂runは`test-2/matched_b0_v2`へ出力します。
- `config/matched_s.yaml`：matchedのV2-S感度解析です。
- `config/full_s.yaml`：全データによるV2-S実験です。

設定検証では、固定seed、15面、6チャネル、`pos_weight=2.0`、反射なしのデータ拡張方針、および上表の学習スケジュールを強制します。`data.n_folds`は固定済み分割の総数5、`data.start_fold`と`data.end_fold`は実際に学習する包含範囲です。例えばfold 2だけを学習する場合は両方を`2`にします。

## 実験管理

`Unet/line_only`に倣い、全設定で`experiment.phase`と`experiment.name`を必須とします。出力は`outputs/{phase}/{name}/fold{N}/`に分離し、実験ルートには`config.yaml`を、各foldには`effective_config.yaml`、checkpoint、履歴、log、指標、検証予測を保存します。

付属設定ではW&Bを既定で有効にします。1 foldを1つのW&B runとし、実効設定、epochごとのBCE/AUROC/AP/LR、clip率、陽性・陰性平均scoreとscore gap、および最良・最終summaryを記録します。ローカルファイルを正とし、checkpoint、CTデータ、bag単位予測はW&B artifactとして送信しません。

## 実行方法

```bash
uv run python -m fracture_detection.cohorts.make_matched_cohort

uv run python fracture_detection/baseline1/train.py \
  --config fracture_detection/baseline1/config/matched_b0.yaml

uv run python -m fracture_detection.baseline1.evaluate \
  --config fracture_detection/baseline1/config/matched_b0.yaml
```

学習コマンドはリポジトリルートから実行してください。通常はYAMLの`data.start_fold`と`data.end_fold`で学習対象を指定します。`--start-fold`、`--end-fold`は一時的な上書き、`--gpu-id`はGPUの上書きに使用します。`--resume`は、既存foldを`last_checkpoint.pt`から再開する場合だけ指定します。CLI上書き後のfold範囲も実効configへ保存されます。

旧`matched_b0_test/08_12` checkpointは改訂前scheduleの診断runなので、改訂設定から`--resume`してはいけません。新しい出力先でepoch 1から開始してください。

コンソールにはマニフェスト読込、fold分割、DataLoader作成、モデル初期化、W&B初期化、epoch開始を順に表示します。学習・検証中はbatch進捗と実行中の平均BCEを表示します。各epochの最初のbatchはDataLoader workerの起動、NFSからの読込、prefetch、データ拡張を含むため、後続batchより時間がかかることがあります。

起動時にmultiprocessingの一時領域を`/tmp/vai-baseline1-{uid}`へ切り替えます。プロジェクト側の`TMPDIR=.tmp`はNFS上にあり、DataLoader worker終了時に`.nfs*`ファイルを削除できず`multiprocessing.util._remove_temp_dir` tracebackが多数出るため使用しません。

`full`では、ステージングが入力マニフェストのSHA256をキーとする読み取り専用キャッシュを`/dev/shm/vai-baseline1`に1つ作成します。並行するfoldプロセスはこのキャッシュを共有し、未完成のキャッシュは再利用せず失敗させます。

## 検証

```bash
uv run pytest fracture_detection/baseline1/tests fracture_detection/common/tests fracture_detection/cohorts/tests -q
uv run ruff format --check fracture_detection/baseline1 fracture_detection/cohorts fracture_detection/common
uv run ruff check fracture_detection/baseline1 fracture_detection/cohorts fracture_detection/common
uv run mypy fracture_detection/baseline1 fracture_detection/cohorts fracture_detection/common --exclude tests --ignore-missing-imports
```
