# Unet/ 作業ログ

日々の作業内容を記録します。セッション終了時に追記してください。

---

## 2026-03-13

### やったこと

**1. 作業管理の仕組み構築**
- `Unet/CLAUDE.md` を作成 - 作業管理方法をドキュメント化
- `unet-work-log.md` を作成 - 作業ログの仕組みを構築
- セッション継続性を保つ仕組みを整備

**2. line-only 実装完了（`docs/notes/ideas/Unet_2_plan.md` に基づく）**
- `line_only/` ディレクトリに直線検出関連コードを実装
  - `line_detection.py` - モーメント法による (φ, ρ) 計算
  - `line_losses.py` - 角度損失 + ρ損失の実装
  - `line_metrics.py` - 評価指標（角度誤差、法線方向距離、垂線距離）
  - `train_heat.py` - 段階的損失の訓練ループ
  - `test_line_losses.py` - 単体テスト

**3. 実装の主要ポイント**
- GT 直線の定義: アノテーション2点から法線形式 (φ, ρ) を計算
- 予測: ヒートマップ → モーメント → 主軸 → (φ, ρ)
- 符号合わせ: 法線ベクトルの内積で向きを統一
- 段階的学習:
  - Phase 1: ヒートマップ MSE のみ
  - Phase 2: + 角度損失（warmup後に重み増加）
  - Phase 3: + ρ損失（弱めに追加）
- 評価指標: 角度誤差、法線方向距離、GT線分からの平均垂線距離

**4. チェックポイント保存**
- `/checkpointing --full` でセッション全体を保存
- `.claude/checkpoints/2026-03-13-070134.md` 作成
  - 5コミット記録
  - Codex相談16件（角度検出バグ分析など）
  - Gemini調査4件（matplotlib日本語対応など）

### 現状

**実装済み:**
- `line_only/` の全機能実装完了（Unet_2_plan.md 準拠）
- 評価指標の幾何量ベース化
- 損失関数の段階的追加メカニズム
- 符号合わせロジック

**未完了:**
- 訓練実行と精度検証
- 比較実験（Baseline / +Angle / +Angle+Rho）
- 訓練/評価パイプラインのリファクタリング（DESIGN.md 参照）

### 次にやること

**優先度：高**
- [ ] `line_only/train_heat.py` の動作確認
- [ ] Phase 1 訓練実行（MSE baseline）
- [ ] Phase 2-3 訓練実行（角度損失、ρ損失追加）
- [ ] 3条件での比較評価

**優先度：中**
- [ ] 評価メトリクスの検証
- [ ] ハイパーパラメータ調整（warmup、損失重み）
- [ ] 可視化スクリプト作成

### メモ

- matplotlib の日本語対応仕様が決定済み

---

## 2026-03-25

### やったこと

**1. 線方向の誤差根本原因調査（完了）**

症状: 予測ヒートマップは位置的に正しいが、抽出された線の方向が大きくずれる

**原因1: GT角度の誤計算 (`line_losses.py` → `extract_gt_line_params`)**
- アノテーションのV字型ポリライン（折れ線）に対して、先頭・末尾点のみで角度を計算していた
- V字ポリラインは全データの **57.3%**（3639アノテーション中）に存在
- これにより **46.4%** のアノテーションで角度誤差 >10°（ランダムに近い）

**原因2: `detect_line_moments` のしきい値なし問題**
- ヒートマップのsigmoid出力は背景全体に低いノイズを持つ
- しきい値なしでモーメント計算すると mu20/mu02 が ~1000-2000（背景ノイズ支配）
- しきい値=0.2 を適用すると ~10-200（線ブロブのみ）に絞られ正確な方向抽出が可能

**2. 修正方針決定**

| 修正対象 | 修正内容 | 効果 |
|---------|---------|------|
| GT計算: `extract_gt_line_params` | 端点法 → PCA法（全ポリライン点を使う） | 平均角度誤差 22.52° → 4.64° |
| 予測抽出: `detect_line_moments` | `threshold=0.2` をデフォルト引数に追加 | 平均角度誤差 28.26° → 4.64° |

**3. 実装済み**
- `line_detection.py`: `detect_line_moments` に `threshold: float | None = 0.2` 引数追加
- 診断スクリプト群 (`Unet/.tmp/`): V字分析、精度比較、可視化
- 可視化画像: `Unet/.tmp/vis_threshold_compare/` (20枚、6パネル)

### 現状

**検証済み:**
- しきい値修正の効果をテストセットで確認（mean 28.26° → 4.64°）
- PCA法GTの方が端点法GTより正確（57.3%のV字ポリライン問題）
- 学習済みモデル自体は良好（PCA GTで比較すると91.4%が10°以内）

**未適用:**
- `line_losses.py` の `extract_gt_line_params` をPCA法に置換（次セッションで実施）
- PCA法GTで再学習（必要に応じて）

### 次にやること

**優先度：高**
- [ ] `line_losses.py` の `extract_gt_line_params` をPCA法に置換
- [ ] 修正後のGT + しきい値修正済み抽出で再評価
- [ ] 必要なら角度損失・ρ損失を有効化して再学習

### 作成したテストコード (`Unet/.tmp/`)

| ファイル | 目的 | 主な出力 |
|---------|------|---------|
| `diagnose_direction.py` | GT heatmap抽出精度の確認（端点法 vs モーメント法の比較） | テキスト出力 |
| `diagnose_polylines.py` | V字型ポリラインの分析（endpoint_dist/total_length の統計） | テキスト出力 |
| `diagnose_summary.py` | データセット全体の統計（V字型57.3%、角度誤差46.4%が>10°） | テキスト出力 |
| `test_pca_fix.py` | PCA法GTの正確性を直線・V字の両ケースで検証 | テキスト出力 |
| `test_full_statistics.py` | 全データの誤差分布と図（fig1〜3） | `vis_*.png` |
| `eval_angle_compare.py` | 端点法GT vs PCA法GT の角度誤差比較（22.52° vs 4.64°） | テキスト出力 |
| `eval_detailed.py` | PCA法GTでのライン別詳細精度評価 | テキスト出力 |
| `eval_visualize.py` | 3パネル可視化（Heatmap / GT+Pred / polyline+Pred） | `vis_output/*.png` |
| `eval_worst_pca.py` | PCA法GTでも誤差が大きいワースト20ケースの可視化 | `vis_worst_pca/*.png` |
| `eval_extraction_debug.py` | チャンネル別ヒートマップ + モーメント値 + GT/Pred重ね表示 | `vis_extraction_debug/*.png` |
| `test_threshold_fix.py` | しきい値なし/0.15/0.2 の精度比較（28.26°/4.74°/4.64°） | テキスト出力 |
| `vis_threshold_compare.py` | しきい値なし vs 0.2 の予測線を6パネルで比較可視化 | `vis_threshold_compare/*.png` |

### メモ

- 現行モデルはMSE損失のみで学習（`use_angle_loss: false`, `use_rho_loss: false`）
- モデル自体は十分学習できている（問題は評価側・抽出側のバグ）
- `detect_line_moments`（numpy/可視化用）と `extract_pred_line_params_batch`（torch/eval用）は同じ結果を返す
- `train_heat.py` は削除され、`line_only/train_heat.py` に統合
- 実装は `Unet_2_plan.md` の方針に完全準拠
- Codex 相談で角度計算のバグ分析を実施済み

---

## テンプレート（以下をコピーして使用）

```markdown
## YYYY-MM-DD

### やったこと

-
-

### 実験結果（あれば）

**Setup:**
-

**Results:**
-

### 次にやること

- [ ]
- [ ]

### メモ

-
```

## 2026-03-18 Session: Critical Bugs Fixed & Y-Axis Coordinate Bug Discovery

### 🔧 Critical Issues Fixed (Codex Analysis)

**3 critical bugs fixed in line_losses.py and train_heat.py:**

1. **NaN Handling Order (line_losses.py:191-278)**
   - Problem: Computed `cos/sin/exp` on NaN-containing tensors before masking
   - Impact: `NaN * 0 = NaN` poisoned batch loss, caused training collapse
   - Fix: Extract valid entries FIRST, then compute loss only on valid data
   - Added: `.detach()` on confidence to prevent gradient collapse

2. **Double Smoothing in Rho Loss (line_losses.py:265)**
   - Problem: Applied smooth minimum, then `F.smooth_l1_loss()` on result
   - Impact: Mathematically incorrect, unpredictable gradient scale
   - Fix: Removed `F.smooth_l1_loss()` call

3. **Warmup Weight Formula (train_heat.py:843)**
   - Problem: `loss = loss_mse + warmup_weight * line_loss` (MSE always constant)
   - Impact: Gradient shock at epoch 50 when geometry loss activates
   - Fix: `loss = (1 - 0.5*warmup_weight) * loss_mse + warmup_weight * line_loss`

**Results After Critical Fixes:**
- test_mse: 0.011 → **0.0017** (6.5x improvement)
- peak_dist: 57.59 → **21.29 px** (2.7x improvement)
- perpendicular_dist: 25.25 → **5.37 px** (4.7x improvement)
- angle_error: 41.66 → **34.30 deg** (7.4 deg improvement)
- rho_error: 18.74 → **15.75 px** (3.0 px improvement)

### 🐛 Y-Axis Coordinate System Bug Discovery

**Root Cause Identified (line_losses.py:96):**
```python
# WRONG (image convention):
y_grid = torch.arange(H, ...) - H / 2.0
# row 0 (top) → -H/2, row H-1 (bottom) → +H/2 (Y increases downward)

# CORRECT (math convention):
y_grid = -(torch.arange(H, ...) - H / 2.0)
# row 0 (top) → +H/2, row H-1 (bottom) → -H/2 (Y increases upward)
```

**Why 40-50° Errors (Not 90° or 180°)?**
- Flipped Y-axis causes non-linear angle distortion in moment calculations
- For 20° line: mu11 sign flips asymmetrically → ~43° error
- For 125° line: similar mechanism → ~52° error

**Fix Impact (Verified with Unit Tests):**
- sample5_C1_slice029 line_1: 42.72° error → **0.03° error**
- sample5_C1_slice029 line_3: 52.64° error → **0.02° error**
- All test angles (0°-180°): max error **0.38°**

### 📊 Expected Full Re-training Impact

**Current (with Critical Fixes + Y-axis Bug):**
- angle_error: 34.30 deg
- rho_error: 15.75 px

**After Re-training (Y-axis Fix):**
- angle_error: **<10 deg** (expected 5-10 deg)
- rho_error: **<10 px** (with lambda_rho tuning)

### 📝 Codex Strategy Recommendations

**Next Steps (Codex Priority):**
1. **Re-train with Y-axis fix** - Verify angle error drops to <10 deg
2. **Increase lambda_rho gradually** - 0.05 → 0.1 → 0.2 (NOT 0.5 jump)
3. **Consider adaptive warmup** - Only if geometry loss plateaus

**NOT Recommended Now:**
- ❌ Jump lambda_rho to 0.5 (10x) - Too risky
- ❌ Implement adaptive warmup immediately - Complexity without proven need
- ❌ Multiple changes at once - Can't isolate effects

### 🧪 Unit Tests Created

**Location:** `Unet/line_only/test/`
- `test_moment_extraction.py` - Basic angle tests (0°, 45°, 90°, 135°)
- `test_sample_fix.py` - Problem case verification
- `visualize_fix.py` - Visual comparison generator

**All tests passing with <1° error.**

### 📁 Files Modified

1. `Unet/line_only/line_losses.py:96` - Y-axis coordinate fix
2. `Unet/line_only/line_losses.py:191-233` - NaN-safe angle_loss
3. `Unet/line_only/line_losses.py:235-278` - NaN-safe rho_loss, removed double smoothing
4. `Unet/line_only/train_heat.py:843` - Warmup formula fix

### 🎯 Next Session Tasks

- [ ] Run full re-training: `cd Unet && uv run python train.py`
- [ ] Verify angle_error <10 deg, rho_error <10 px
- [ ] If successful, test lambda_rho=0.1 (2x increase)
- [ ] Document final results

---

## 2026-03-23 Session: Threshold Processing Investigation

### 📊 Problem Analysis

**Current Test Results (Y-axis fix applied):**
- angle_error: 32.70° (still high, expected <10°)
- Confidence: 0.214 (very low, indicates poor line quality)

**Root Cause Hypothesis:**
- Low confidence → unstable angle calculation
- When mu20 ≈ mu02, denominator in `atan2(2*mu11, mu20-mu02)` becomes small
- Heatmap includes low-intensity "tails" that spread variance isotropically

### 🧪 Threshold Processing Experiment

**Test Setup:**
```python
# Test script: Unet/line_only/test/test_threshold_effect.py
# Compare: threshold=0.0 (no filter) vs threshold=0.2 (filter low values)
```

**Results (411 test samples):**

| Metric | No Threshold | Threshold ≥ 0.2 | Improvement |
|--------|--------------|-----------------|-------------|
| Mean Error | 32.70° | **23.73°** | -8.97° (27.4%) |
| Median Error | 28.10° | **10.09°** | -18.01° (64.1%) |
| Max Error | 89.61° | 89.49° | ~same |
| Confidence | 0.214 | **0.941** | +0.726 (340%) |

**Key Findings:**
- ✅ 98.8% of samples improved or unchanged
- ✅ Dramatic confidence boost (0.214 → 0.941)
- ⚠️ 1.2% of samples degraded (5 out of 411)
  - Degraded samples: originally low heatmap response
  - Threshold cut important low-intensity regions

### 🎨 Visualization Scripts Created

**Location:** `Unet/line_only/test/threshold_effect/`

1. **test_threshold_effect.py**
   - Compare threshold=0.0 vs 0.2 across test dataset
   - Generate statistics and improvement analysis
   - Output: `analysis_summary.png`

2. **visualize_threshold_comparison.py**
   - Visual comparison of improved/degraded samples
   - 3-panel format: Heatmap / Pred Lines / GT Lines
   - Key features:
     - Shows thresholded heatmap (>= 0.2 only)
     - Pred lines clipped to thresholded region + 10px margin
     - Background: original CT image

**Generated Images (11 files, 1.9MB):**
- `analysis_summary.png` - Overall statistics
- `improved_1-5_delta*.png` - Top 5 improved samples (70-88° improvement)
- `degraded_1-5_delta*.png` - Top 5 degraded samples (52-72° degradation)

### 📈 Sample Analysis

**Best Improvement (sample22_C3_slice044_ch1):**
- Error: 89.6° → 0.9° (**88.6° improvement**)
- Threshold removed noisy low-intensity regions
- Pred line almost perfectly matches GT

**Worst Degradation (sample22_C4_slice046_ch1):**
- Error: 6.3° → 78.2° (**71.9° degradation**)
- Original heatmap had weak but correct response
- Threshold cut too much, leaving fragmented information

### 🎯 Next Steps

**Immediate (This Direction):**
- [ ] Implement threshold processing in training pipeline
  - Modify `extract_pred_line_params_batch()` in `line_losses.py`
  - Add `threshold=0.2` parameter
- [ ] Re-train model with threshold processing
- [ ] Verify angle_error drops to <15° (ideally <10°)

**Alternative Approaches (If Needed):**
- [ ] Adaptive threshold per channel (instead of fixed 0.2)
- [ ] Soft weighting instead of hard threshold
- [ ] Improve heatmap quality through better training

### 📝 Technical Details

**Threshold Implementation:**
```python
# Before moment calculation:
if threshold > 0:
    heatmaps = torch.where(heatmaps >= threshold, heatmaps, 0.0)
# Then compute moments as usual
```

**Why It Works:**
1. Removes low-intensity noise → cleaner heatmap
2. Reduces mu20 ≈ mu02 cases → more stable atan2
3. Increases λ1/λ2 ratio → higher confidence

**Trade-off:**
- 98.8% improve, but 1.2% degrade (acceptable)
- Degraded cases: weak heatmap response (training issue, not threshold issue)

### 📁 Files Created

- `Unet/line_only/test/test_threshold_effect.py` - Effect analysis
- `Unet/line_only/test/visualize_threshold_comparison.py` - Visualization
- `Unet/line_only/test/threshold_effect/*.png` - 11 visualization images

### 💡 Key Insight

**The angle error is not caused by coordinate bugs or formula errors.**
**It's caused by mathematical instability when computing angles from noisy, isotropic heatmaps.**
**Threshold processing is an effective post-processing solution.**

---
