# 2026-08-11 Baseline 1 実装計画

> 状態: **実装完了・学習未着手**
> 仕様: `2026-08-11-baseline1-design.md`
> 前提: `2026-08-11-fracture-common-and-baseline-plan.md`

## 1. 実装境界

- 対象は固定 matched cohort、Baseline 1、椎体 OOF 評価までとする
- Baseline 2、提案 A/B、学習結果に基づくハイパーパラメータ再調整は含めない
- `fracture_detection/common/` にはモデルや Baseline 1 固有 loss を置かない
- 旧 `train_models/stage1/` は挙動の参照だけに使い、直接変更しない
- 既存の `folds.csv` と `input_manifest.csv` は再生成しない

## 2. 実装前に解消する契約差分

### 2.1 比較用コホートの陽性率

陽性268 bagと陰性268 bagの50:50コホートは、`full`の陽性率10.095%から大きく外れ、
比較用データだけ異なる分類問題になる。アノテーション済み陽性268 bagは全件保持し、
`full`の陽性1,406 / 陰性12,522に対応する陰性2,387 bagを抽出する。陰性のfold×level分布も
`full`の陰性分布に比例させ、患者ごと1椎体という人工的な制約は置かない。

### 2.2 Baseline 1 の評価 API

`common.metrics.evaluate_prediction_frame` は4領域 score を必須とするため、領域出力を持たない
Baseline 1 には直接使えない。`common/metrics.py` に椎体専用の
`evaluate_vertebra_prediction_frame` を追加し、既存の総合 evaluator も内部でこれを再利用する。
ダミー領域 score は生成しない。

### 2.3 明示依存

`uv.lock` には `timm==1.0.22` があるが `pyproject.toml` に直接依存として記載されていない。
Baseline 1 実装時に同バージョンを明示し、6ch pretrained stem の初期化をこのバージョンへ固定する。

### 2.4 full staging の共有

full データを fold ごとに複製すると `/dev/shm` 容量を超える。manifest SHA256 をキーにした
read-only 共有 cache を1個だけ作り、複数 fold プロセスが再利用する。lock と完了 marker により
コピー途中の cache は学習に使わない。

## 3. Phase 1: fixed matched cohort

作成先: `fracture_detection/cohorts/`

| ファイル | 責務 |
|---|---|
| `constants.py` | 入力・出力 path、seed、固定列 |
| `make_matched_cohort.py` | 2,655 bag の決定的選択、検証、凍結出力 |
| `README.md` | 選択規則、schema、再生成禁止条件 |
| `tests/test_matched_cohort.py` | 選択規則と上書き guard の検証 |
| `outputs/matched_cohort.csv` | B1 matched / B2 共通の exact ID |
| `outputs/matched_cohort_meta.json` | 入力 hash、seed、件数、出力 SHA256 |

手順:

1. `has_region_target == True` の268 bagを正例側として全件採用する
2. `full`の陽性率から必要な陰性件数2,387を算出する
3. `vertebra_target == 0` の完備bagをfold×level別に集計する
4. `full`の陰性fold×level分布に比例する必要件数を最大剰余法で割り当てる
5. 各セル内をseed付き固定ハッシュで順位付けし、annotated 268 + negative 2,387を保存する
6. 既存出力と byte 単位で一致しない再生成は失敗させる

受入条件:

- 2,655 bag / 1,498患者 / annotated 268 + negative 2,387
- 陽性率10.094%が`full`の10.095%と丸め誤差内で一致
- negative はすべて `vertebra_target == 0`
- negative のfold×level分布が`full`の陰性分布に比例
- 同一入力・seedで同じCSV byte列とSHA256を再現
- Baseline 1 matched loader はこの固定 path 以外を受け付けない

Baseline 2 は未実装なので、B1/B2両 consumer の同一性テストは Baseline 2 実装時に追加する。
それまでは両者が import する唯一の cohort path を `cohorts/constants.py` に固定する。

## 4. Phase 2: Baseline 1 data layer

作成先: `fracture_detection/baseline1/`

| ファイル | 責務 |
|---|---|
| `config.py` | YAML読込、型変換、未知・禁止設定の検証 |
| `dataset.py` | common dataset の6ch化、train augmentation |
| `staging.py` | full 用共有 local cache |
| `tests/test_config.py` | mode別契約と禁止設定 |
| `tests/test_dataset.py` | shape、同期変換、mask二値性、向き保持 |
| `tests/test_staging.py` | cache再利用、marker、容量不足、競合 |

データ契約:

- `data.mode` は `matched | full` の必須項目とし、暗黙 default を置かない
- `data.n_folds=5`を固定分割の総数とし、実行する包含範囲を必須の`data.start_fold` / `data.end_fold`で指定する
- CLIの`--start-fold` / `--end-fold`はconfig値の任意上書きとし、CLI側に暗黙のfold既定値を置かない
- matched は固定2,655行、full は pinned 13,928行を読む
- frozen `fold` 列だけで train/validation を分け、再 stratify しない
- `common.FractureDataset` の `ct` と `masks[:, 0:1]` を結合し
  `float32[15,6,224,224]` を返す
- train loader は通常 shuffle のみ。weighted sampler / balanced batch は実装しない
- validation は完全 deterministic、augmentation なし
- seed は config の `20260807` と fold 番号から決定的に導出し、worker も固定する

augmentation 契約:

- 15面×5CTchを1画像、15面の全体maskを1 maskとして一度だけ変換し、幾何を共有する
- `Affine` は CT bilinear、mask nearest、constant fill 0、変換後 mask を再二値化する
- brightness / contrast / blur / noise は CT のみに適用する
- flip / transpose / distortion / cutout / mixup のコード path 自体を作らない
- matched/full の確率・shift・scale・rotate・強度範囲は確定 YAML と一致させる

## 5. Phase 3: model and objective

| ファイル | 責務 |
|---|---|
| `model.py` | timm backbone + 2-layer BiLSTM + per-plane head |
| `losses.py` | label broadcast BCE と mean-sigmoid bag probability |
| `tests/test_model.py` | B0/S、可変batch、出力shape、freeze |
| `tests/test_losses.py` | 数式一致、勾配、shape/error |

モデル契約:

- `timm.create_model(..., in_chans=6, num_classes=0)` を使う
- 特徴次元は `model.num_features` から取得し、backbone名の表は持たない
- 入力 `[B,15,6,224,224]` を `[B*15,6,224,224]` に展開する
- BiLSTM は hidden 256、2層、bidirectional、batch-first
- head は `Linear -> BN -> Dropout -> LeakyReLU -> Linear`
- 出力は plane logits `[B,15]` のみ。患者補助headは移植しない
- 6ch pretrained stem は timm 1.0.22 の標準変換を使用し、独自初期化を加えない

objective 契約:

- target `[B]` を `[B,15]` へ broadcast し、固定`pos_weight=2.0`の`BCEWithLogitsLoss`を適用する
- 推論 bag score は `sigmoid(plane_logits).mean(dim=1)` とする
- `pos_weight`は2.0以外を拒否し、focal、class weight、mixup、EMAの引数は公開しない

## 6. Phase 4: optimization and trainer

| ファイル | 責務 |
|---|---|
| `optimization.py` | AdamW parameter group、freeze、LR policy |
| `trainer.py` | train/validation loop、AMP、early stopping、checkpoint |
| `experiment.py` | 出力path、epochログ、W&B run・summary管理 |
| `tests/test_optimization.py` | LR境界、weight decay、BN固定 |
| `tests/test_trainer.py` | early stopping、best保存、resume、非有限値 |
| `tests/test_experiment.py` | 出力分離、W&B無効時・mock runの検証 |

実装規則:

- backbone/head と decay/no-decay を分離した AdamW group を作る
- bias と normalization parameter には weight decay を適用しない
- LR は optimizer step 単位で更新し、確定した epoch 境界値を厳密に再現する
- matched epoch 1-10 は backbone を freeze し、BatchNorm を eval に固定する
- epoch 11 で unfreeze し、epoch 11-15 warmup、16以降 cosine とする
- full は 2.3e-4 から 2.3e-5 への75 epoch cosine とする
- CUDAではbf16 autocast、CPU testではfloat32へ安全にfallbackする
- backward前後の非有限 loss/gradient を検出し、global norm 1.0でclipする
- val AUROCをstrict improvementで選び、同値なら先に保存したepochを維持する
- matched の checkpoint 候補と patience count は epoch 20 以降に限定する
- full は config の `min_epoch`、patience 15を使用する
- 1 GPU / 1 process / 1 foldを基本単位とし、fold並列は外側から実行する

実験管理は `Unet/line_only/src/experiment.py` の責務分離を踏襲する:

- YAMLに必須の `experiment.phase` / `experiment.name` を置き、出力rootを
  `baseline1/outputs/{phase}/{name}/` から一意に導出する
- CLI上書き後の実効configを実験rootの `config.yaml` に保存する
- `wandb.enabled` で明示的にON/OFFし、無効時はimportも通信も行わない
- `wandb.project: null` の既定projectは `fracture-{phase}-{name}` とする
- `wandb.run_name: null` の既定run名は `fold{N}` とし、1 foldを1 runに対応させる
- `wandb.init(config=effective_config, reinit=True)` で実効config全体を記録する
- epochごとに train/val BCE、val AUROC/AP、backbone/head LR、grad norm、経過時間を記録する
- 起動準備の各段階を即時表示し、train/validationはbatch進捗と実行中平均BCEを表示する
- best更新時に `best_epoch`、`best_val_auroc`、`best_val_ap`、`best_val_loss` をsummaryへ保存する
- 終了時に停止epoch、最終指標、train/val件数をsummaryへ保存して `wandb.finish()` する
- W&B未導入・初期化失敗時は警告してローカル記録を継続し、学習自体は失敗させない
- W&Bへのcheckpoint・CT・個票prediction uploadは行わず、ローカル成果物を正本とする
- W&Bの単体テストはmockのみを使い、外部通信しない

fold成果物:

- `effective_config.yaml`
- `best_model.pt`
- `last_checkpoint.pt`
- `history.csv`
- `val_predictions.csv`
- `fold_metrics.json`
- `training.log`

各標準configの `experiment.name` は `matched_b0` / `matched_s` / `full_s` とし、
出力先は `outputs/baseline1/{experiment.name}/fold{N}/` とする。異なるmode/backboneの
混在を防ぎ、既存成果物への上書きは `--resume` または明示的な新実験名なしでは拒否する。

## 7. Phase 5: CLI and pooled OOF evaluation

| ファイル | 責務 |
|---|---|
| `train.py` | config、fold、device、stage、resume のCLI統合 |
| `evaluate.py` | 5 fold予測の完全性検証とpooled OOF評価 |
| `tests/test_evaluate.py` | 重複、欠損、fold漏洩、pooled metric |
| `config/matched_b0.yaml` | matched主解析 |
| `config/matched_s.yaml` | matched感度分析 |
| `config/full_s.yaml` | full別実験 |
| `README.md` | モデル、入出力、損失、設定、実行、成果物 |

3つの標準configはすべて `experiment` と `wandb` セクションを持つ。ローカル出力名と
W&B project/run名を同じ実効configから導出し、CLI上書き後も両者の対応を維持する。

OOF evaluator は次を assert する:

- cohort の各 `(study_id, level)` がちょうど1回だけ存在する
- prediction の fold と manifest の fold が一致する
- checkpoint が当該 validation fold を学習に含めていない
- score が有限かつ `[0,1]`
- matched は2,655行、full は13,928行

出力は実験rootの `oof_predictions.csv`、`oof_metrics.json`、`all_folds_summary.json`。
5 foldを反復標本として平均せず、全OOF行をpoolして椎体AUROC/APと患者cluster bootstrap
95% CIを計算する。`all_folds_summary.json` はfold別情報とpooled指標をまとめるが、fold指標の
単純平均を主結果として出さない。

## 8. Phase 6: validation and rollout

### 静的・単体検証

```bash
uv run pytest fracture_detection/cohorts/tests fracture_detection/baseline1/tests fracture_detection/common/tests -q
uv run ruff format --check fracture_detection/cohorts fracture_detection/baseline1 fracture_detection/common
uv run ruff check fracture_detection/cohorts fracture_detection/baseline1 fracture_detection/common
uv run mypy fracture_detection/cohorts fracture_detection/baseline1 fracture_detection/common --exclude tests --ignore-missing-imports
git diff --check
```

### 実データ smoke test

1. cohortを生成し、件数・陽性率・fold×level表・SHA256を確認する
2. matchedのtrain/valから各1 batchを読み、forward/loss/backwardを1回実行する
3. augmentation後のCT/mask重ね合わせを固定サンプルPNGで目視確認する
4. matched B0 fold 0を短い隔離runで実行し、resumeと成果物schemaを確認する
5. full stagingを1回作り、複数プロセスから同じready cacheを読めることを確認する

### 本学習順

1. matched B0 5-fold（主解析）
2. matched V2-S 5-fold（感度分析）
3. full V2-S fold 0で運用確認後、残りfoldを並列実行
4. 各実験のpooled OOFを生成し、`PROGRESS.md` と `DESIGN.md` を更新

## 9. Rollback plan

- cohort生成失敗時: frozen fold/common manifestは触らず、`cohorts/outputs` の新規成果物だけを破棄する
- Baseline実装不具合時: `baseline1/` は独立しているため旧Stage1や他armへ影響させず戻せる
- common evaluator変更時: 既存 `evaluate_prediction_frame` の入出力を保持し、追加APIだけを戻せる構造にする
- staging不具合時: `/dev/shm` cacheは派生成果物として削除可能。NFS原本はread-onlyで扱う
- 学習失敗時: fold単位の隔離出力を保持し、`last_checkpoint.pt` から明示resumeする
- 設定変更が必要になった場合: 既存runを上書きせず新run名で分離し、READMEと実効configを同時更新する

## 10. 完了条件

- fixed matched cohortが凍結され、SHA256と全不変条件が検証済み
- 3 configすべてが同じデータ・fold・loss契約を満たす
- unit/static/smoke testが通る
- mode/backbone/foldごとの成果物が再現可能
- pooled OOF evaluatorが欠損・重複・fold漏洩を拒否する
- `README.md`、`PROGRESS.md`、`DESIGN.md` が実装と同期している
