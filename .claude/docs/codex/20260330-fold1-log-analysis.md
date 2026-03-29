# fold1 回帰調査: ログ比較分析
Date: 2026-03-30

## 問題

sig2.0_base（新 src コード）で fold1 角度誤差が **8.60° → 17.77°** に悪化。

## 排除済みの原因

| 仮説 | 排除根拠 |
|------|---------|
| augmentation 変更（SSR→Affine） | 実験で差 0.22°（test_aug_fold1.py） |
| wandb が torch RNG を消費 | Codex 確認済み（20260329-wandb-randomness.md） |
| fold1 のデータが本質的に難しい | 旧コードは同じデータで 8.60° を達成 |

## ログ比較結果

### baseline（旧 shim）fold1 収束パターン

wandb run: run-20260325_175319-ytzzh4kv

- epoch 1-108: peak=33-82px, angle=35-55°（ランダム、未収束）
- epoch 109-112: peak 24→20px に急降下、angle 23°→19°
- epoch 113: peak=21.65px, **angle=5.87°**（突然収束！）
- epoch 114-143: angle 5-7° で**安定推移**
- best_epoch=128, val_mse=0.001649, angle=5.70°
- test 結果: 8.60°

### sig2.0_base（新 src）fold1 収束パターン

wandb run: 2026-03-29 の run

- epoch 1-95: peak=50-80px, angle=40-50°（未収束、baseline と同様）
- epoch 96: angle=6.43°（一時的に収束）
- epoch 97-100: angle=29°→27°→31°→37°（**すぐ不安定化**）
- epoch 103-170: angle 13°→31°→37°→28°→37° と振動継続
- best_epoch=170, val_mse=0.001880, angle=13.04°（val）
- test 結果: 17.77°

## 重要な観察

1. **旧コードでも fold1 の収束は遅い**（epoch 108 まで未収束）
2. **旧コードは一度収束すると安定**する（5-7° で維持）
3. **新コードは fold1 で収束後も不安定**（13°→37° で振動）
4. fold4 は新旧どちらも安定収束（epoch 130-140 頃、5-8°）

## 旧→新コードの変更点

| 変更 | 旧コード | 新コード |
|------|---------|---------|
| confidence 計算式 | `1 - lam2/lam1` | `(lam1-lam2)/(lam1+lam2+1e-8)` |
| 無効判定 | NaN を返す | 0.0 + conf=0 |
| valid_mask（evaluate） | `~isnan(pred_params)` | `confidence > 0` |
| 精度 | float64 | float32 |
| 正則化 | sxx, syy のみ | sxx, syy に +1e-6 |

## 現時点の仮説

**evaluate() の valid_mask 変更が fold1 の early stopping 選択を不安定にしている**

- 新コードの `confidence > 0` は旧コードの `~isnan()` より厳しい条件
- fold1 は fold4 より isotropic な heatmap が多い（harder fold）
- confidence が閾値周辺を振動すると、有効サンプル数が epoch ごとに変動
- 有効サンプル数が変動すると val_angle_error が不安定になる
- これが epoch 96 以降の角度誤差振動を引き起こす可能性

## 次のステップ

1. `confidence > 0` → `~isnan()` に戻して fold1 を再学習し、収束が安定するか確認
2. 精度を float64 に戻して evaluation のみ実行
3. `data_utils.py` の共有 generator バグを修正（別問題だが再現性のために）

## 参考ファイル

- `Unet/line_only/src/trainer.py` - evaluate(), valid_mask の定義
- `Unet/line_only/utils/losses.py` - _compute_moments_batch(), confidence 計算
- `Unet/line_only/shim/line_losses.py` - 旧コードの confidence 計算
- wandb: run-20260325_175319-ytzzh4kv（baseline fold1 ログ）
