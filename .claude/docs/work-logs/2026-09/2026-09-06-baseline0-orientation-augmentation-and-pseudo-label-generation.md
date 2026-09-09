# Baseline 0向き拡張再学習と疑似ラベル生成

作成日: 2026-09-06  
状態: **Baseline 0の全5 fold学習・outer推論・疑似ラベル生成・検証まで完了。region branch学習は未開始**

参照する正本:

- `.claude/docs/DESIGN.md`
- `.claude/docs/REGION_MODEL_DESIGN_JA.md`
- `fracture_detection/baseline0/config/baseline0.yaml`
- `fracture_detection/region_branch/config/region_branch_all.yaml`

---

## 1. 今回の方針決定

Baseline 0を標準5-foldの80%学習で作り直す案も検討したが、region実験の目的はwhole分類の
絶対精度最大化ではなく、疑似ラベルを用いるregion fine-tuningの比較である。そのため、
最終的に次の契約を維持した。

- outer fold 20%、inner validation fold 20%、training fold 60%のnested splitを維持する。
- 同じfold-matched Baseline 0 checkpointを、疑似ラベル教師、whole分類baseline、
  region branch初期値の3用途で使用する。
- checkpointはinner validationだけで選択し、outer foldをcheckpoint選択に使用しない。
- 疑似ラベル教師には主結果の`best_model.pt`、すなわちinner validation AUROC最大の
  checkpointを使用する。
- early stoppingは従来どおりvalidation BCE、patience 20とする。checkpoint選択指標と
  early stopping指標は別である。

80%学習を採用しなかった理由は、region studentのinner foldが初期化時点ですでにwhole分類
教師へ使われると、inner指標を完全な未見性能とは呼べなくなるためである。現行60%設計は、
精度を多少犠牲にしてもnested比較を単純かつ解釈可能に保つ。

## 2. データ拡張と2 GPU実行の修正

Stage1 parityとの差を踏まえ、Baseline 0とregion branchのtraining augmentationへ次を復元した。

- `VerticalFlip(p=0.5)`
- `Transpose(p=0.5)`

CT、whole mask、4領域maskは1回の`ReplayCompose`で同期変換する。領域IDは交換せず、
各領域maskの画素だけを画像と同じ幾何変換で移動する。これにより、transposeや上下反転後も
CTと教師maskの対応が崩れない。

また、以前使用していたfold単位の2 GPU並列実行を復元した。

```yaml
parallel:
  mode: fold
  gpu_ids: [0, 1]
  max_concurrent_folds: 2
```

1 foldを1 process・1 GPUへ割り当て、round-robinで
`outer0 -> GPU0`、`outer1 -> GPU1`、`outer2 -> GPU0`、`outer3 -> GPU1`、
`outer4 -> GPU0`と実行する。DDPではないため、global batch size、BatchNorm、sampling、
統計的protocolは変化しない。

主な実装箇所:

- `fracture_detection/baseline0/data/dataset.py`
- `fracture_detection/baseline0/training/parallel.py`
- `fracture_detection/baseline0/cli/train.py`
- `fracture_detection/region_branch/data_pipeline/dataset.py`
- `fracture_detection/region_branch/cli/train.py`
- `fracture_detection/region_branch/config/*.yaml`

## 3. Baseline 0再学習

実験名と出力先:

```text
fracture_detection/baseline0/outputs/09_04/baseline0_aug追加/
```

全5 foldが完了し、各foldに`best_model.pt`、`best_val_prauc_model.pt`、
`outer_predictions.csv`、`outer_predictions_prauc_checkpoint.csv`、`fold_metrics.json`が
保存された。

### 3.1 主checkpoint（inner validation AUROC-best）

| outer fold | checkpoint epoch | outer AUROC | outer AP | outer F1 |
|---:|---:|---:|---:|---:|
| 0 | 55 | 0.918605 | 0.772599 | 0.710084 |
| 1 | 50 | 0.910591 | 0.723800 | 0.680412 |
| 2 | 53 | 0.916679 | 0.759082 | 0.701439 |
| 3 | 45 | 0.883413 | 0.699275 | 0.647948 |
| 4 | 50 | 0.916865 | 0.759551 | 0.702355 |
| **全OOF** | | **0.908486** | **0.740479** | **0.689007** |

F1は各checkpointのinner validationでF1最大となる閾値を選び、その閾値を対応するouterへ
固定適用した値である。

### 3.2 PR-AUC-best checkpoint

| outer fold | checkpoint epoch | outer AUROC | outer AP | outer F1 |
|---:|---:|---:|---:|---:|
| 0 | 65 | 0.904751 | 0.747535 | 0.694143 |
| 1 | 70 | 0.918976 | 0.731275 | 0.658333 |
| 2 | 68 | 0.918533 | 0.757269 | 0.714012 |
| 3 | 60 | 0.888397 | 0.711797 | 0.653226 |
| 4 | 50 | 0.916865 | 0.759551 | 0.702355 |
| **全OOF** | | **0.906405** | **0.738122** | **0.684536** |

今回はPR-AUC-bestよりAUROC-bestの方が、全OOFのAUROC、AP、F1のすべてでわずかに高い。
inner validationのAP最大epochがouterでも最大APになるとは限らないため、これは矛盾ではない。

## 4. データ拡張修正前との比較

比較対象:

```text
fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/
```

| checkpoint | 旧AUROC | 新AUROC | 差 | 旧AP | 新AP | 差 | 旧F1 | 新F1 | 差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC-best | 0.898949 | 0.908486 | +0.009537 | 0.716385 | 0.740479 | +0.024094 | 0.685666 | 0.689007 | +0.003341 |
| PR-AUC-best | 0.897605 | 0.906405 | +0.008800 | 0.717970 | 0.738122 | +0.020152 | 0.681680 | 0.684536 | +0.002856 |

AUROC-bestのfold別差分:

| outer fold | AUROC差 | AP差 | F1差 | 解釈 |
|---:|---:|---:|---:|---|
| 0 | +0.019814 | +0.045043 | +0.051045 | 明確に改善 |
| 1 | -0.002357 | -0.013376 | -0.002846 | わずかに低下 |
| 2 | +0.001026 | -0.001338 | -0.034334 | 順位性能は同等、閾値依存F1は低下 |
| 3 | +0.015207 | +0.016663 | +0.016818 | 改善 |
| 4 | +0.011759 | +0.030519 | -0.009973 | 順位性能は改善、F1は低下 |

foldごとの改善は一様ではないが、全OOFでは特にAPが約0.024改善した。したがって、
maskを同期したvertical flip・transpose追加は全体として有効と判断した。

## 5. 疑似ラベル生成

教師run名が実際の出力名と一致するよう、残っていた
`baseline0_orientation_aug`参照を`baseline0_aug追加`へ修正した。Baseline 0の疑似ラベルCLI、
CAM audit、README、region branchの初期化config/schemaが同じrunを参照する。

本番前に全5教師・各fold 1 training bag + 1 held-out bagのGPU smoke testを行い、10行の
artifact生成まで完走した。その後、次のコマンドで本番生成した。

```bash
uv run python -m fracture_detection.baseline0.cli.generate_pseudo_labels \
  --experiment-dir 'fracture_detection/baseline0/outputs/09_04/baseline0_aug追加' \
  --output-dir fracture_detection/baseline0/outputs/09_04/pseudo_labels \
  --checkpoint-name best_model.pt \
  --device cuda:0 \
  --batch-size 16
```

生成物:

- `fracture_detection/baseline0/outputs/09_04/pseudo_labels/pseudo_region_targets.csv`
- `fracture_detection/baseline0/outputs/09_04/pseudo_labels/pseudo_target_calibration.csv`
- `fracture_detection/baseline0/outputs/09_04/pseudo_labels/pseudo_target_generation_metadata.json`

生成結果:

- 5,328 rows
- 1,332 unique bags
- duplicate key 0
- pseudo targetのNaN 0
- teacher outer fold 0〜4をすべて包含
- metadata上の`checkpoint_name`: `best_model.pt`
- metadata上の`smoke_only`: `false`
- targets SHA-256先頭12文字: `7e65d5fed491`
- calibration SHA-256先頭12文字: `73e6223eb40e`

student outer foldごとのloader読込行数は、outer0から順に
`1070 / 1064 / 1063 / 1069 / 1062`で、全foldが正常にロードできた。

### 5.1 Shared logit-share calibration

| student outer fold | slope | intercept | fit bags | fit studies | OOF AP | OOF Brier | OOF log loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5868 | 1.1559 | 144 | 93 | 0.6716 | 0.1762 | 0.5417 |
| 1 | 1.5933 | 1.1435 | 145 | 91 | 0.7453 | 0.1559 | 0.4840 |
| 2 | 1.8191 | 1.3780 | 138 | 86 | 0.7468 | 0.1528 | 0.4626 |
| 3 | 0.9290 | 0.4234 | 140 | 86 | 0.4632 | 0.2104 | 0.6059 |
| 4 | 1.7776 | 1.3622 | 141 | 88 | 0.7075 | 0.1685 | 0.5211 |

全foldでslopeは正であり、CAM shareのfold内順位方向を維持している。fold 3の校正OOF APは
他foldより低いため、後続のregion実験ではfold別のばらつきとして監視する。

## 6. 検証

データ拡張・parallel launcher実装後の全体検証は253 tests passed、lintとdiff checkも通過した。

疑似ラベル生成後は、pseudo-label pipelineとregion branchのfocused testを再実行した。

```bash
python -m pytest \
  fracture_detection/baseline0/tests/test_pseudo_label.py \
  fracture_detection/baseline0/tests/test_pseudo_calibration.py \
  fracture_detection/region_branch/tests \
  -q
```

- **168 tests passed**
- warningはsynthetic testの定数入力に対する既知の`ConstantInputWarning`と、ネットワーク制限下の
  Albumentations version checkのみ

## 7. 次の作業

疑似ラベルはregion branchからそのまま利用可能である。次工程は以下。

1. `region_branch`の各configが
   `fracture_detection/baseline0/outputs/09_04/baseline0_aug追加`を初期化元としていることを確認する。
2. `fracture_detection/baseline0/outputs/09_04/pseudo_labels`を疑似ラベル入力にする。
3. 必要なfoldについてregion lossのlambda calibrationを再生成する。
4. 現行60% nested foldを維持したまま、region branchをfold-process方式で学習する。
5. whole checkpointとregion checkpointを独立選択し、outer endpointを評価する。

本ログ作成時点では、今回生成した疑似ラベルを使うregion branch本学習はまだ開始していない。
