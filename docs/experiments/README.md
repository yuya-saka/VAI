# 実験記録

実験の設定・条件・記録を管理します。

## ファイル命名規則

- 実験記録: `exp_<id>_<description>.md`
- 例: `exp_001_unet_baseline.md`

## テンプレート

```markdown
# Experiment <ID>: <Title>

**Date:** YYYY-MM-DD
**Status:** Planning / Running / Completed
**Goal:**

## Dataset
- Name:
- Size:
- Preprocessing:

## Model
- Architecture:
- Parameters:

## Training
- Optimizer:
- Learning rate:
- Batch size:
- Epochs:
- Loss function:

## Results
See: `../results/exp_<id>/`

## Notes
```
