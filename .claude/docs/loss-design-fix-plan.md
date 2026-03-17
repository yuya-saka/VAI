# Loss Design Fix: Implementation Plan

**Date:** 2026-03-18
**Based on:** Codex Analysis (20260318-0254-loss-design-complete.md, 20260318-0251-loss-design-review.md)

---

## 📋 Executive Summary

Codex分析により、現在の学習設計に5つの重大なバグを発見：
1. Warmup formula の誤り → Gradient shock
2. Rho loss の double smoothing → 勾配歪み
3. Rho loss weight が弱すぎる → Rho学習不可
4. LR scheduler の誤作動 → 早期LR削減
5. NaN handling 順序バグ → NaN汚染

---

## 🎯 Implementation Phases

### Phase 1: Critical Bugs（訓練失敗の原因）

**Priority:** 🔴 Immediate
**Goal:** 訓練が正常に完了するようにする

#### 1.1 Warmup Formula Fix
**File:** `Unet/line_only/train_heat.py:840`

**Current:**
```python
loss = loss_mse + warmup_weight * line_loss_dict["total"]
```

**Fix:**
```python
# MSEを逆warmup、Line lossをwarmup
mse_weight = 1.0 - 0.5 * warmup_weight
loss = mse_weight * loss_mse + warmup_weight * line_loss_dict["total"]
```

**Rationale:**
- 現在はEpoch 50でgradient shock（line lossが突然フル強度）
- MSEを1.0→0.5に減衰させることで、スムーズな移行を実現
- Epoch 1: MSE=1.0, Line=0.0 → MSE主導
- Epoch 50: MSE=0.5, Line=1.0 → バランス
- Epoch 100+: MSE=0.5, Line=1.0 → Geometry重視

**Impact:** Moderate (20 lines)
**Risk:** Low (数式変更のみ)
**Test:** 訓練開始時のloss値とwarmup推移を確認

---

#### 1.2 Remove Double Smoothing
**File:** `Unet/line_only/line_losses.py:265`

**Current:**
```python
# smooth最小値（微分可能）
alpha = 10.0
exp1 = torch.exp(-alpha * err1)
exp2 = torch.exp(-alpha * err2)
loss = (err1 * exp1 + err2 * exp2) / (exp1 + exp2 + 1e-8)

# ❌ SmoothL1（不要な2回目の平滑化）
loss = F.smooth_l1_loss(loss, torch.zeros_like(loss), reduction="none")
```

**Fix:**
```python
# smooth最小値（微分可能）
alpha = 10.0
exp1 = torch.exp(-alpha * err1)
exp2 = torch.exp(-alpha * err2)
loss = (err1 * exp1 + err2 * exp2) / (exp1 + exp2 + 1e-8)

# SmoothL1削除（smooth minで十分）
```

**Rationale:**
- SmoothL1は生の誤差を期待するが、事前平滑化された値に適用している
- 意図しない非線形変換（loss < 1 → 0.5*loss², loss > 1 → loss - 0.5）
- 勾配が歪み、MSEとのスケール不一致

**Impact:** Small (1 line削除)
**Risk:** Very Low
**Test:** Rho loss値のスケールを確認（~0.01程度になるはず）

---

#### 1.3 Fix NaN Handling Order
**File:** `Unet/line_only/line_losses.py:191-232, 235-278`

**Current (angle_loss):**
```python
pred_phi = pred_params[..., 0]  # NaN含む
gt_phi = gt_params[..., 0]

pred_nx = torch.cos(pred_phi)   # NaN伝播
dot = pred_nx * gt_nx + ...
loss = 1.0 - torch.abs(dot)

weights = confidence * valid_mask.float()
weighted_loss = loss * weights  # NaN * 0 = NaN
```

**Fix:**
```python
# 有効エントリのみを抽出（NaN汚染防止）
if not valid_mask.any():
    return torch.tensor(0.0, device=pred_params.device, requires_grad=True)

valid_pred_phi = pred_params[..., 0][valid_mask]
valid_gt_phi = gt_params[..., 0][valid_mask]
valid_conf = confidence[valid_mask].detach()  # 勾配消失防止

# 有効エントリのみで計算
pred_nx = torch.cos(valid_pred_phi)
pred_ny = torch.sin(valid_pred_phi)
gt_nx = torch.cos(valid_gt_phi)
gt_ny = torch.sin(valid_gt_phi)

dot = pred_nx * gt_nx + pred_ny * gt_ny
loss = 1.0 - torch.abs(dot)

weighted_loss = (loss * valid_conf).sum() / (valid_conf.sum() + 1e-8)
return weighted_loss * 0.01
```

**同様の修正をrho_lossにも適用**

**Rationale:**
- NaNを含んだまま計算すると、`NaN * 0 = NaN` でバッチ全体が汚染
- 有効エントリのみを抽出してから計算することで、NaN伝播を防止
- Confidenceをdetachして勾配消失（zero-heatmap trivial solution）を防止

**Impact:** Medium (40 lines)
**Risk:** Low（ロジックは同じ、順序のみ変更）
**Test:** NaN発生時の挙動を確認

---

### Phase 2: Loss Balance Adjustment

**Priority:** 🟡 Important
**Goal:** Loss weightingを適切にバランス

#### 2.1 Increase Rho Loss Weight
**File:** `Unet/config/config.yaml`

**Current:**
```yaml
lambda_rho: 0.005  # 実効重み 0.00005
```

**Fix:**
```yaml
lambda_rho: 0.05  # 10倍増加
```

**Rationale:**
- 現在の実効重み: 0.005 × 0.01 = 0.00005（MSEの1/200）
- ネットワークはrho値を学習できていない
- 10倍に増やしてもMSEの1/20なので、まだ控えめ

**Impact:** Minimal (config変更のみ)
**Risk:** Very Low
**Test:** Rho error metricsの改善を確認

---

#### 2.2 Disable LR Scheduler During Warmup
**File:** `Unet/line_only/train_heat.py:863`

**Current:**
```python
# validation lossに基づいてスケジューラを更新
scheduler.step(val_metrics["val_loss_mse"])
```

**Fix:**
```python
# Warmup後のみスケジューラを適用
if ep > warmup_epochs:
    scheduler.step(val_metrics["val_loss_mse"])
```

**Rationale:**
- Warmup中はlossが増加傾向（geometry loss ramping up）
- ReduceLROnPlateauが「訓練が悪化している」と誤認
- Warmup完了後にschedulerを開始することで、正常なLR削減

**Impact:** Small (1 line追加)
**Risk:** Very Low
**Test:** LR削減のタイミングを確認

---

### Phase 3: Configuration Tuning

**Priority:** 🟢 Optional
**Goal:** 訓練効率とパフォーマンス向上

#### 3.1 Adjust Warmup Parameters
**File:** `Unet/config/config.yaml`

**Current:**
```yaml
warmup_epochs: 50
warmup_mode: "linear"
```

**Option A (Conservative):**
```yaml
warmup_epochs: 50
warmup_mode: "cosine"  # よりスムーズ
```

**Option B (Aggressive):**
```yaml
warmup_epochs: 20  # 短縮
warmup_mode: "linear"
```

**Rationale:**
- Phase 1でwarmup formulaを修正すれば、50 epochs linearでも安定
- Cosineはより滑らかな移行
- 20 epochsに短縮すれば早期にgeometry制約が効く

**Impact:** Minimal (config変更のみ)
**Risk:** Low
**Test:** 複数設定で比較実験

---

#### 3.2 Increase Angle Loss Weight (Optional)
**File:** `Unet/config/config.yaml`

**Current:**
```yaml
lambda_theta: 0.1  # 実効重み 0.001
```

**Suggested:**
```yaml
lambda_theta: 1.0  # 10倍増加、実効重み 0.01 = MSEと同等
```

**Rationale:**
- 現在はMSEの1/10なので、angle制約が弱い
- MSEと同等にすることで、ヒートマップとgeometryをバランス良く学習
- ただし、Phase 1の修正で既に改善する可能性あり

**Impact:** Minimal (config変更のみ)
**Risk:** Low
**Test:** Angle error metricsの改善を確認

---

## 📊 Testing Strategy

### Unit Tests
```bash
# Loss関数のNaN handling
uv run pytest Unet/line_only/test_line_losses.py -v -k "nan"

# Warmup weight計算
uv run pytest Unet/line_only/test_line_losses.py -v -k "warmup"
```

### Integration Tests
```bash
# 短期訓練（10 epochs）でNaN発生しないか確認
uv run python Unet/line_only/train_heat.py --epochs 10
```

### Full Validation
```bash
# 1 foldで完全訓練
uv run python Unet/line_only/train_heat.py
```

---

## 🔄 Rollback Plan

各Phaseは独立しているため、問題が起きた場合：

1. **Phase 1で問題発生:**
   - Git revert
   - 個別の修正を無効化して原因特定

2. **Phase 2で問題発生:**
   - Config値を元に戻す
   - LR scheduler行をコメントアウト

3. **Phase 3で問題発生:**
   - Config値を元に戻す

---

## 📈 Expected Improvements

### Phase 1完了後:
- ✅ NaN collapseが解消
- ✅ Gradient shockが解消
- ✅ 訓練が安定してepoch 200まで完走

### Phase 2完了後:
- ✅ Rho値の精度向上（現在: ~15px → 目標: ~5px）
- ✅ LR削減のタイミングが適切になる

### Phase 3完了後:
- ✅ Angle errorの改善
- ✅ 訓練効率の向上（早期収束）

---

## 📝 Implementation Checklist

- [ ] Phase 1.1: Warmup formula fix (train_heat.py:840)
- [ ] Phase 1.2: Remove double smoothing (line_losses.py:265)
- [ ] Phase 1.3: Fix NaN handling order (line_losses.py:191-278)
- [ ] Phase 2.1: Increase lambda_rho (config.yaml)
- [ ] Phase 2.2: Disable scheduler during warmup (train_heat.py:863)
- [ ] Unit tests実行
- [ ] Integration test実行
- [ ] Full validation (1 fold)
- [ ] Phase 3調整（必要に応じて）

---

**Estimated Time:**
- Phase 1: 1-2 hours (実装 + テスト)
- Phase 2: 30 minutes
- Phase 3: Optional, iterative tuning

**Ready to proceed with Phase 1?**
