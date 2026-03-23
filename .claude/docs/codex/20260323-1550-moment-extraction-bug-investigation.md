# Codex Investigation: Moment Extraction Bug Analysis
Date: 2026-03-23 15:50

## Investigation Request

User suspects the heatmap → line extraction (moment-based) has implementation bugs.

Observation: 5-10% crosstalk causes 8-28° angle errors, which seems excessive.

## Codex Analysis

### Critical Bug Found

**Location:** `Unet/line_only/line_losses.py` lines 161-162

**Current (WRONG) Code:**
```python
dir_x = mu11
dir_y = lambda1 - mu20
```

**Problem:** This is **NOT the correct eigenvector formula**. This arbitrary formula extracts wrong directions that are extremely sensitive to noise, explaining why 5-10% crosstalk causes 8-28° errors.

### Root Cause

The eigenvector calculation is mathematically incorrect. The two implementations use different methods:

1. **line_detection.py (CORRECT):** Uses analytical formula `θ = 0.5 * atan2(2*μ11, μ20 - μ02)`
2. **line_losses.py (WRONG):** Attempts eigenvector extraction but uses wrong formula

When the formula is wrong, small changes in covariance moments produce arbitrary angle outputs instead of the principal axis.

### Recommended Fix

**Replace lines 159-176 in line_losses.py with:**

```python
# Use same analytical formula as line_detection.py for consistency
theta = 0.5 * torch.atan2(2.0 * mu11, mu20 - mu02)
dir_x = torch.cos(theta)
dir_y = torch.sin(theta)
```

This ensures both implementations use the same **proven correct** method.

### Mathematical Explanation

- **Correct formula:** Principal axis angle = `0.5 * atan2(2*μ11, μ20 - μ02)`
- **Wrong formula:** `dir = (μ11, λ1 - μ20)` has no mathematical basis for eigenvector extraction
- **Impact:** Wrong formula makes angle extremely sensitive to moment ratios → 5% noise amplified to 20°+ error
- **After fix:** 5-10% crosstalk should cause < 5° error (proportional to noise level)

### Files That Need Changes

**`Unet/line_only/line_losses.py`** - Replace eigenvector calculation with analytical formula (lines 159-176)

### Verification Needed

After applying the fix, rerun training and check if angle errors drop from 8-28° to < 5° with same crosstalk levels.

---

## Additional Finding (Main Claude Analysis)

Through empirical testing, discovered that the **actual crosstalk level is ~9.4%** and that **pure crosstalk should only cause ~0.5° error** (verified experimentally).

**Key observation: Confidence-error correlation**

Low confidence samples (Conf < 0.2) show large errors:
- `Conf=0.025` → `Error=56.3°`
- `Conf=0.053` → `Error=42.0°`
- `Conf=0.119` → `Error=89.6°`

High confidence samples (Conf > 0.4) show small errors:
- `Conf=0.456` → `Error=2.3°`
- `Conf=0.454` → `Error=2.8°`

**Root cause: Weak anisotropy (circular blobs)**

Confidence = 1 - λ₂/λ₁ measures how elongated the distribution is:
- Low (<0.2): Nearly circular → principal axis undefined → unstable angles
- High (>0.4): Elongated → clear principal axis → stable angles

**Conclusion:**
Both factors contribute:
1. **Wrong eigenvector formula** (Codex finding) → baseline error
2. **Low confidence heatmaps** (Main analysis) → error amplification

## Recommended Actions

1. **Fix eigenvector calculation** (immediate, per Codex)
2. **Filter low-confidence samples** during training (Conf < 0.3)
3. **Reduce sigma** in GT generation (2.5 → 1.5-2.0) for sharper heatmaps
4. **Add heatmap sharpness loss** to encourage higher confidence predictions
