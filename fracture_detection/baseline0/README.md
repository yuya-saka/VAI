# Baseline 0

`fracture_detection/` の骨折分類教師モデルと疑似ラベル生成基盤。

## モデル

15面固定の各planeについて、CT 5chと椎体全体mask 1chをEfficientNetV2-Sへ入力する。
面特徴をBiLSTMで文脈化し、15個の面logitへbroadcast BCEを適用する。
bag確率は面sigmoidの平均で求める。

## 実行

```bash
uv run python -m fracture_detection.baseline0.cli.train \
  --config fracture_detection/baseline0/config/baseline0.yaml

uv run python -m fracture_detection.baseline0.cli.evaluate \
  --config fracture_detection/baseline0/config/baseline0.yaml
```

対象foldやGPUは学習CLIの引数で上書きできる。

```bash
uv run python -m fracture_detection.baseline0.cli.train \
  --start-outer-fold 0 \
  --end-outer-fold 0 \
  --gpu-id 0
```

## 疑似ラベル

疑似ラベルは現行の主要機能であり、`pseudo_labeling/` にGrad-CAM、CAM品質監査、領域スコア、監査レポートをまとめている。

```bash
uv run python -m fracture_detection.baseline0.cli.cam_audit \
  --experiment-dir fracture_detection/baseline0/outputs/08_19/baseline0_shared_core

uv run python -m fracture_detection.baseline0.cli.generate_pseudo_labels \
  --experiment-dir fracture_detection/baseline0/outputs/08_19/baseline0_shared_core \
  --output-dir fracture_detection/baseline0/outputs/08_19/pseudo_labels
```

疑似ラベル生成は各outer foldの教師を、その教師が学習したfoldのbagへ対応付ける。生成済みファイルがある場合は、意図した再生成時だけ `--overwrite` を付ける。

## 検証

```bash
uv run pytest fracture_detection/baseline0/tests -q
uv run ruff check fracture_detection/baseline0
uv run ruff format --check fracture_detection/baseline0
```

## 成果物

現行5-fold教師成果物は `outputs/08_19/baseline0_shared_core/` に保持する。
疑似ラベル生成物の既定出力先は `outputs/08_19/pseudo_labels/`。失敗したMTL、Proposed、Type2、smoke run、旧診断runは現行treeから削除済み。
