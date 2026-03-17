# Codex Analysis: Loss Design Complete Review
Date: 2026-03-18 02:54

## Question
Review the training loss design in Unet/line_only/ focusing on loss combination formula, weighting, warmup, and rho loss implementation.

## Codex Response

### 🔴 CRITICAL ISSUES (Must Fix Immediately)

1. **Warmup Weight Formula is Incorrect (Line 840, train_heat.py)**
   - Current: `loss = loss_mse + warmup_weight * line_loss_dict["total"]`
   - Problem: Only line losses are weighted, MSE remains constant
   - Impact: Creates gradient shock at epoch 50 when geometry losses suddenly activate
   - Fix: Apply inverse weighting to MSE: `loss = (1 - 0.5*warmup) * loss_mse + warmup * line_loss`

2. **Double Smoothing in Rho Loss (Lines 254-265, line_losses.py)**
   - Applies smooth minimum, then applies SmoothL1 to the result
   - This is mathematically incorrect - SmoothL1 expects raw errors, not pre-smoothed values
   - Fix: Remove the second `F.smooth_l1_loss()` call - soft minimum is already smooth

3. **Rho Loss Too Weak**
   - Effective weight: 0.00005 (200x smaller than MSE at 0.01)
   - Network will never learn accurate rho values with such weak signal
   - Fix: Increase `lambda_rho` from 0.005 to 0.05 (10x increase)

### 🟡 IMPORTANT ISSUES (Should Fix Soon)

4. **LR Scheduler Confusion**
   - ReduceLROnPlateau will see increasing loss during warmup (geometry loss ramping up)
   - May trigger premature LR reduction thinking training is degrading
   - Fix: Disable scheduler until after warmup: `if epoch > warmup_epochs: scheduler.step()`

5. **Warmup Too Short for Current Formula**
   - 50 epochs is too short for zero-to-full transition with current formula
   - If keeping current formula, need 100+ epochs
   - If fixing MSE weighting, 50 epochs with cosine schedule is acceptable

### 🟢 MONITORING RECOMMENDATIONS

6. **Add Logging for Debugging**
   - Log valid sample ratio: `valid_ratio = valid_mask.float().mean()`
   - Log weighted loss contributions (not just raw values)
   - Monitor gradient norms per loss component

## Files That Need Changes

1. **train_heat.py** (line 840)
   - Fix warmup weight application to MSE

2. **line_losses.py** (lines 254-265)
   - Remove double smoothing in rho loss

3. **config.yaml**
   - Increase `lambda_rho: 0.05`
   - Consider `warmup_epochs: 100` and `warmup_mode: "cosine"`

## Recommendations

**Immediate action items:**
1. Fix warmup formula to weight MSE inversely
2. Remove double smoothing in rho loss
3. Increase rho loss weight by 10x
4. Disable LR scheduler during warmup

**Expected improvements:**
- Smoother training without gradient shock
- Better rho value learning
- More stable loss progression
- Fewer NaN/divergence issues
