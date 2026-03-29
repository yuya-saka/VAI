# Codex Analysis: fold1 training regression
Date: 2026-03-29

## Question

リファクタリング後に fold1 の角度誤差が 8.60° → 17.77° に悪化した原因を調査。
fold4 は 8.11° で完全一致しており、fold1 だけが大きく変わっている。

具体的に問い合わせた内容:
- evaluate() の valid_mask 変更 (NaN チェック → confidence > 0) が早期停止の best checkpoint 選択に影響するか
- augmentation ライブラリ変更 (ShiftScaleRotate → Affine) が fold1 に影響するか
- float64 → float32 の精度変更の影響
- なぜ fold1 のみ影響を受け fold4 は影響を受けないか

## Codex Response

Most likely causes, ranked:

1. **Changed validation logic altered checkpoint selection (highest likelihood).**
   `evaluate()` moved from `pred_valid = not-NaN` to `pred_valid = confidence > 0`, and confidence computation also changed (float64+old formula -> float32+new formula + moment regularization).
   This can change which samples are counted in val angle error, so early stopping can pick a different "best" epoch in fold1. Fold4 can remain identical if its val predictions stay confidently anisotropic and epoch ranking is unchanged.

2. **Numerical behavior shift in line extraction (strong contributor to #1).**
   `extract_pred_line_params_batch` changed invalid handling (`NaN` -> `0/conf=0`), precision (`float64` -> `float32`), and confidence formula.
   For borderline/isotropic heatmaps (common in harder folds), tiny numeric differences change validity gating and metric stability. That is exactly the kind of fold-specific trigger that can double error in one fold but not another.

3. **Augmentation op change (`ShiftScaleRotate` -> `Affine`) (possible but less likely as primary).**
   This is the main train-data-path difference, so it can change learned weights. But if this were the dominant cause, you'd usually expect broader multi-fold drift, not one fold collapsing while another is effectively unchanged. It's still a plausible secondary effect if fold1 is harder/more scale-sensitive.

4. **Shared `torch.Generator` bug across train/val/test (very unlikely for this regression).**
   It is a real bug, but present in both old and new code with same loader creation order, so it does not explain the old-vs-new fold1 delta by itself.

Direct answers to questions:

1. **Yes**: evaluation differences can absolutely cause a different best checkpoint for fold1 and not fold4.
2. **Yes, possible**: augmentation change can alter training trajectories, but likely secondary here.
3. Most subtle extra factor is the **precision/confidence/valid-mask coupling** (float32 + new conf + `confidence>0` gate), which amplifies fold-specific instability.
4. Fold1 is likely a harder/more ambiguous split (more near-isotropic outputs), so it is sensitive to these threshold/numerical changes; fold4 is easier, so both pipelines converge/select similarly.

## Key Technical Differences Identified

| 項目 | 旧コード (shim/) | 新コード (src/) |
|------|-----------------|----------------|
| 精度 | float64 (共分散行列) | float32 |
| 無効判定 | NaN を返す | 0.0 を返す (conf=0) |
| confidence 計算式 | `1 - lam2/lam1` | `(lam1-lam2)/(lam1+lam2+1e-8)` |
| valid_mask (evaluate) | `~isnan(pred_params)` | `confidence > 0` |
| 正則化 | sxx, syy のみ | sxx, syy に +1e-6 |
| augmentation | A.ShiftScaleRotate | A.Affine |

## Recommendations

1. **最優先**: evaluate() の early stopping が MSE のみに基づいていることを確認 (val_loss_mse のみで判断されているはずだが、新しい valid_mask が MSE ではなく angle error 計算に影響していないか検証)
2. 旧コードの NaN based valid_mask を新コードで再現して fold1 の挙動が回復するか確認
3. confidence 計算式の差が fold1 の early stopping epoch 選択に影響しているかログで確認
