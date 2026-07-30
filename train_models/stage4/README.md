# Stage4 mixed supervision

Stage4 は `fracture_dataset_blind`、固定 `stage4_folds.csv`、DICOM 由来の
4領域ラベルを使う独立パイプラインです。confirmatory run は早期停止せず、
Weak-only / Mixed の両 arm を75 epochで学習します。outer validation は
過学習監視のため毎epoch記録しますが、checkpoint選択には使わず、両armとも
固定75 epochの最終checkpointを評価します。

固定比batchは `strong / weak / negative` の条件付き損失をfold母集団比へ
戻してから椎体損失を構成します。領域損失は、epochごとに抽出した
level-matched negativeを反復し、各batchで strong と negative の実効教師数を
必ず1:1にします。

## 確認

```bash
uv run pytest -q train_models/stage4/tests
uv run python train_models/stage4/scripts/stage4_level_only_baseline.py
```

## Smoke

```bash
uv run python train_models/stage4/train.py \
  --config train_models/stage4/config/smoke.yaml
```

## Confirmatory

```bash
uv run python train_models/stage4/scripts/run_stage4_confirmatory.py
```

中断後は同じコマンドで再開できます。各 run は
`train_models/stage4/outputs/{arm}/seed{seed}/fold{fold}/` に final/latest
checkpoint、epoch別陰性 manifest、学習ログ、OOF予測を保存します。

## 評価

```bash
uv run python train_models/stage4/scripts/stage4_evaluate.py
```

5 seed の確率を bag ごとに平均し、pooled OOF macro-AP、C2除外値、
fold内患者 paired bootstrap、椎体 safety gate、fold percentile-rank
感度解析を1つのJSONレポートに出力します。
