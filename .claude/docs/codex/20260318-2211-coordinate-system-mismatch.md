# Codex Analysis: Coordinate System Mismatch in Line Extraction
Date: 2026-03-18 22:11

## Problem Summary

Heatmaps are correctly predicted, but extracted lines point in completely wrong directions (90+ degree errors). This indicates a bug in the line extraction logic, not in training.

## Investigation

### Coordinate System Comparison

| Function | File | Line | Y-axis Setup | Purpose |
|----------|------|------|--------------|---------|
| `extract_pred_line_params_batch()` | line_losses.py | 97 | `y_grid = -(torch.arange(H) - H/2.0)` | **Training loss, evaluation** |
| `detect_line_moments()` | line_detection.py | 81 | `ys = np.arange(H)` | **Visualization** |

### The Bug

**line_losses.py (Line 97)**: Uses **math coordinate system** (origin at center, Y-up)
```python
y_grid = -(torch.arange(H, device=device, dtype=torch.float32) - H / 2.0)  # Y-axis points UP
x_grid = torch.arange(W, device=device, dtype=torch.float32) - W / 2.0
Y, X = torch.meshgrid(y_grid, x_grid, indexing="ij")
```

**line_detection.py (Line 81-83)**: Uses **image coordinate system** (origin at top-left, Y-down)
```python
ys = np.arange(H, dtype=np.float64)  # 0, 1, 2, ..., H-1 (Y-down)
xs = np.arange(W, dtype=np.float64)  # 0, 1, 2, ..., W-1
X, Y = np.meshgrid(xs, ys)
```

## Codex Response

```
(φ, ρ) を損失で扱うなら、`line_losses.py` のような「画像中心原点・`x`右向き・`y`上向き」の数学座標系を正とするのが正解です。
`line_detection.py` は今 `y`下向きの画像座標なので角度が一致せず、`ys = -(np.arange(H) - H/2.0)`（必要なら `xs` も `np.arange(W) - W/2.0`）にそろえるか、出力角を `theta_math = (-theta_img) % np.pi` に変換してください。
要するに、抽出・GT・評価は同一座標系で統一し、可視化時だけ画像座標へ戻すのが正しい修正です。
```

**Translation:**
If you're using (φ, ρ) in the loss function, the math coordinate system (center origin, x-right, y-up) like in line_losses.py is correct. line_detection.py currently uses y-down image coordinates, causing angle mismatch. Fix it by using `ys = -(np.arange(H) - H/2.0)` (and `xs = np.arange(W) - W/2.0` if needed), or convert the output angle with `theta_math = (-theta_img) % np.pi`. In summary, extraction, GT, and evaluation should use the same coordinate system, and only convert to image coordinates for visualization.

## Root Cause

**Coordinate system inconsistency:**
- Training uses math coords (Y-up)
- Visualization uses image coords (Y-down)
- When evaluation calls detect_line_moments(), it gets image coords angles
- But GT and metrics expect math coords angles
- Result: 90+ degree angle errors

## Recommended Fix

**Option 1 (Recommended): Unify to math coordinate system**

Modify `line_detection.py` lines 81-83:

```python
# OLD (image coords):
ys = np.arange(H, dtype=np.float64)
xs = np.arange(W, dtype=np.float64)
X, Y = np.meshgrid(xs, ys)

# NEW (math coords to match training):
y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)  # Y-up, center origin
x_grid = np.arange(W, dtype=np.float64) - W / 2.0     # X-right, center origin
X, Y = np.meshgrid(x_grid, y_grid)
```

## Expected Outcome

After fix:
- Pred lines will match GT directions (angle error < 5 degrees)
- Visualization will show correct line orientations
- Evaluation metrics will be consistent with training

## Next Steps

1. Apply fix to line_detection.py
2. Re-run evaluation on test set
3. Verify pred lines match GT visually
4. Check angle_error_deg_mean drops from ~41° to < 5°
