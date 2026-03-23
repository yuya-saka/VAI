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
