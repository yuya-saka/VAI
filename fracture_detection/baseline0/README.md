# Baseline 0：椎体単位骨折分類

4領域教師を使わず、CT 5 channelと椎体全体mask 1 channelから椎体骨折を分類する比較の起点です。
正式な再学習は旧`baseline0`専用loopではなく、6構成共通の`fracture_detection/core/`を使います。

## モデル

- 入力: canonical 10 channelから先頭6 channelを選択した`[B,15,6,224,224]`
- backbone: ImageNet事前学習済み`tf_efficientnetv2_s`
- sequence: hidden 256、2層BiLSTM
- head: `Linear -> BatchNorm -> Dropout(0.3) -> LeakyReLU -> Linear`
- 出力: 15面logit。bag確率は面sigmoidの平均

canonical datasetは全アーム共通でCT、whole mask、4領域maskを同期変換します。Baseline 0のmodel adapterは
4領域channelを入力へ渡さず、annotated streamも実行しません。これによりnatural sampler、augmentation、
mixup、optimizer step数を他アームと共有します。

## 学習契約

- 品質除外済み`13,432 bag / 2,009 study / 陽性1,332`
- 凍結済み5-fold、`outer=k / inner=(k+1)%5 / train=残り3 fold`
- natural batch 16、BF16、AdamW、weight decay `1e-4`
- LR `2.3e-4 -> 2.3e-5`、75 epoch単一cosine周期
- `pos_weight=2.0`、mixup `p=0.2`、gradient clipなし
- laterality-safe horizontal flip `p=0.5`。R2/R3のmask値・教師を同時交換
- vertical flipとtransposeは禁止

正式runはλ/β校正、5構成resource profile、6構成config、source/dependency/input/fold hashを含む
`frozen_experiment_manifest.json`が揃わない限り開始できません。smokeはcheckpointとinner検証まで実行し、
outer推論を行いません。

## 実行

このproject専用のentry pointから起動します。configはproject内の`shared_core.yaml`へ
固定済みで、armは1つなので`--arm`は省略できます。

```bash
uv run python -m fracture_detection.baseline0.cli train \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

`--outer-fold`を省くと`parallel.gpu_ids`へ5 foldを並列割当します。resource profileは
`profile`、W&B同期は`sync-wandb` subcommandです。実装は共通の`fracture_detection/core`と
`fracture_detection/cli`にあり、entry pointは委譲のみを行います。

```bash
uv run python -m fracture_detection.baseline0.cli train --resume
uv run python -m fracture_detection.baseline0.cli profile \
  --output fracture_detection/profiling/outputs/<phase>/baseline0.json
```

共通CLIを直接叩く旧来の形式も同じ実装へ到達します。

```bash
uv run python -m fracture_detection.cli.train \
  --config fracture_detection/baseline0/config/shared_core.yaml \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

正式runでは`--smoke-steps`を外します。run開始時とresume時にfrozen manifestを照合し、hash driftを拒否します。

W&Bは全共有アームで既定ONです。foldごとに`{experiment.name}-outer{k}`を作成し、
`history.csv`と同じepoch指標を記録します。過去にW&B無効で実行したrunは、再学習せず同期できます。

```bash
uv run python -m fracture_detection.baseline0.cli sync-wandb
```

`sync-wandb`はconfigから成果物rootを解決します。任意のディレクトリを指定する場合は
共通CLIを使います。

```bash
uv run python -m fracture_detection.cli.sync_wandb \
  --experiment-dir fracture_detection/baseline0/outputs/08_19/baseline0_shared_core
```

同期前に`uv run wandb login`でW&Bへログインしておく必要があります。

## 成果物

`fracture_detection/baseline0/outputs/{phase}/{name}/outer{k}/`へ次を保存します。

- `effective_config.yaml`
- `best_model.pt` / `best_val_prauc_model.pt` / `last_checkpoint.pt`
- `history.csv`
- `outer_predictions.csv` / `outer_predictions_prauc_checkpoint.csv`（正式runのみ）
- `summary.json`
- `wandb_run_id.txt`（W&B有効時）

正式outer予測には参照したfrozen manifestのSHA256を埋め込み、pooled OOF解析で全foldの一致を再検証します。

## 検証

```bash
uv run pytest fracture_detection -q
uv run ruff format --check fracture_detection
uv run ruff check fracture_detection
```
