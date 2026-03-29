# Codex Analysis: Line-Matching Loss Design for U-Net Heatmap Training
Date: 2026-03-28

## Question
Design a line-matching loss for a TinyUNet trained to output 4-channel sigmoid heatmaps for
detecting 4 boundary lines per cervical vertebra in CT images (224×224). The current MSE-only
baseline achieves angle_error=5.7–6.5° but plateaus there. We want to break this plateau by
adding a geometry-aware line loss on top of MSE.

**Line representation:** Hesse normal form xcosφ + ysinφ = ρ, φ ∈ [0,π), ρ normalized by D=√2·224

**Training context:**
- 4-channel sigmoid heatmap, 224×224 images
- MSE val_mse 0.219 → 0.0018 over 136 epochs
- Angle plateaus at 5–7° with MSE only
- Angle extraction unreliable until val_mse < 0.002 (~epoch 96)

**Proposed vectorized pred extraction (no for-loop, no NaN):**
```
h = sigmoid_output.view(B,C,-1)
M = h.sum(-1).clamp(1e-6)
cx, cy = weighted centroid
sxx, syy, sxy = weighted covariance / M
disc = sqrt(clamp((sxx-syy)^2 + 4*sxy^2, min=0))
lam1 = (sxx+syy+disc)/2, lam2 = (sxx+syy-disc)/2
confidence = (lam1-lam2)/(lam1+lam2+1e-8)
dir_x=sxy, dir_y=lam1-sxx
nx=-dir_y, ny=dir_x  (normal direction)
sign = where(ny>=0, 1, -1)
phi = atan2(ny*sign, nx*sign)
rho = (nx*sign*cx + ny*sign*cy) / D
```

**Proposed loss:**
```
L = L_mse + w(t) * [λ1*(1 - |cos(φ_pred - φ_gt)|) + λ2*SmoothL1(ρ_pred - ρ_gt)]
gated by: confidence > 0.5 AND gt_valid
```

## Codex Response

### 1. Gradient Issues in Vectorized Extraction

#### a. `sqrt(disc)` near 0
- `clamp(min=0)` avoids NaN from tiny negative roundoff, but does NOT fix the sharp derivative at 0.
- Use `disc = torch.hypot(sxx - syy, 2*sxy)` (more stable than `sqrt(a^2+b^2)`), then `disc_safe = disc + eps` where needed.
- **Key advice**: avoid relying on eigenvalue gap for direction when isotropic; gate those cases by confidence.

#### b. `torch.where` sign flip at `ny=0`
- Hard branch: gradient is undefined at boundary and discontinuous around it.
- **Do NOT canonicalize with `ny` branch during training.** Canonicalize only for logging/export.
- Use **orientation-invariant loss** (`|dot|` or `dot^2`) and align sign to GT with a **detached sign** for `rho`.

#### c. `atan2` at `(0,0)`
- Gradient is undefined.
- Either **avoid `atan2` in loss** (recommended), or mask low-anisotropy predictions before angle loss.
- If kept, guard with confidence/anisotropy mask and never backprop angle where vector norm is tiny.

---

### 2. `SmoothL1(rho_pred - rho_gt)` vs smooth-min

- If `(phi, rho)` canonicalization is truly consistent, plain `SmoothL1` is correct.
- In practice, branch jitter can still flip sign near degenerate cases.
- **Robust approach**: align predicted normal to GT normal with **detached sign**, then apply `SmoothL1` on `rho`; this removes need for smooth-min.

```python
# Robust sign alignment
sgn = torch.sign((n_pred * n_gt).sum(-1).detach())   # (B, C)
n_aligned = sgn.unsqueeze(-1) * n_pred                # align normal direction
rho_pred = (n_aligned * centroid).sum(-1) / D         # consistent rho
L_rho = F.smooth_l1_loss(rho_pred, rho_gt)
```

---

### 3. Loss Alternative Comparison

| Option | Description | Verdict |
|--------|-------------|---------|
| **A** | angle+rho (proposed) | Best match to moment-based extraction |
| **B** | Avg perpendicular distance at GT line points | Clean gradients, good geometric regularizer |
| **C** | Centroid-to-GT-line only | Weak: barely constrains orientation |
| **D** | IoU/Dice on thresholded heatmap | Worst: thresholding / thin-line imbalance |

**Recommendation:**
- **Primary**: A (angle+rho) — but implement angle as **vector-dot loss**, not raw angle subtraction.
- **Optional**: Add small B term as geometric regularizer.
- Avoid C (under-constrained) and D (numerically fragile).

**Best angle loss formulation:**
```python
L_ang = 1 - (n_pred_unit * n_gt_unit).sum(-1).pow(2)  # pi-periodic, smooth, no abs cusp
```
This is `1 - cos²(φ_pred - φ_gt)` = `sin²(φ_pred - φ_gt)`, which handles 180° ambiguity cleanly.

---

### 4. Loss Weight Formula

- `L = L_mse + w * L_line` **is better** than `(1-0.5w)*L_mse + w*L_line`.
- Keep MSE weight fixed so heatmap quality does not collapse as `w` increases.
- Normalize `L_line` by valid gated count (not batch size) to avoid scale sensitivity.
- Ramp `w` gradually after extraction is reliable.

---

### 5. Concrete Step-by-Step Implementation Plan

**Stage 1: Implement stable extraction (no for-loop)**
```python
# Use torch.hypot for disc (more stable)
disc = torch.hypot(sxx - syy, 2 * sxy)          # (B, C)
disc_safe = disc + 1e-8                           # avoid zero-disc gradient

lam1 = (sxx + syy + disc_safe) / 2
lam2 = (sxx + syy - disc_safe) / 2

# Line direction eigenvector (for lam1)
dir_x = sxy                                       # (B, C)
dir_y = lam1 - sxx                               # (B, C)
dir_norm = torch.hypot(dir_x, dir_y).clamp(1e-8)
dir_x = dir_x / dir_norm
dir_y = dir_y / dir_norm

# Normal = perpendicular to direction
nx = -dir_y
ny =  dir_x

# Confidence (anisotropy ratio, detach for gating)
confidence = ((lam1 - lam2) / (lam1 + lam2 + 1e-8)).detach()
```

**Stage 2: Orientation-invariant angle loss (no atan2, no sign flip)**
```python
# GT normal: (nx_gt, ny_gt) unit vector from GT (phi, rho)
nx_gt = torch.cos(phi_gt)   # (B, C)
ny_gt = torch.sin(phi_gt)   # (B, C)

# dot^2 loss: 0 when parallel, 1 when perpendicular
dot = nx * nx_gt + ny * ny_gt  # (B, C)
L_ang = 1 - dot.pow(2)         # [0, 1], pi-periodic, smooth
```

**Stage 3: Aligned rho loss (detached sign)**
```python
# Align normal sign to GT before computing rho
sgn = torch.sign(dot.detach()).clamp(min=1)    # (B, C), avoid zero
nx_a = sgn * nx
ny_a = sgn * ny
rho_pred = (nx_a * cx + ny_a * cy) / D        # (B, C)
L_rho = F.smooth_l1_loss(rho_pred, rho_gt, reduction='none')  # (B, C)
```

**Stage 4: Soft confidence gate (no hard threshold)**
```python
# Soft gate: 0 when conf < 0.3, 1 when conf > 0.6
gate = ((confidence - 0.3) / 0.3).clamp(0, 1).detach()  # (B, C)
gate = gate * gt_valid.float()                            # mask invalid GT

# Gated losses
L_line = (gate * (lambda1 * L_ang + lambda2 * L_rho)).sum() / gate.sum().clamp(1)
```

**Stage 5: Curriculum warmup**
```python
# Trigger warmup when val_mse drops below threshold
if val_mse < 0.0025 and warmup_started is False:
    warmup_start_epoch = current_epoch
    warmup_started = True

w = 0.0
if warmup_started:
    w = min(1.0, (current_epoch - warmup_start_epoch) / ramp_epochs)

loss = L_mse + w * L_line
```

**Stage 6: Monitoring**
- Log per-epoch: `L_mse`, `L_ang`, `L_rho`, `gate_ratio` (fraction of valid gates), `conf_mean`
- Log `grad_norm` of extraction block and decoder head
- Track NaN count: `torch.isnan(L_line).sum()`
- Validation: angle MAE + rho MAE + failure rate on low-confidence samples

**Stage 7: Ablation plan**
1. Baseline: MSE only (current)
2. +A (vector angle + rho): `L_mse + w * (L_ang + λ*L_rho)`
3. +A+B: add small point-to-line geometric regularizer
4. Compare per-fold angle_error, rho_error, perp_dist

---

## Summary of Key Recommendations

1. **Do NOT use `torch.where` for sign flip during training** — use orientation-invariant `dot^2` loss instead
2. **Do NOT use `atan2` in the loss** — compute loss directly on normal vectors
3. **Use `torch.hypot` instead of `sqrt(a^2+b^2)`** for better numerical stability
4. **Align rho sign with detached GT dot product** — eliminates smooth-min need
5. **Soft confidence gating** (sigmoid ramp 0.3→0.6) instead of hard threshold
6. **MSE weight stays fixed at 1.0** — only add line loss additively
7. **Angle loss**: `1 - (n_pred · n_gt)^2` (handles 180° ambiguity, smooth, no cusp)
8. **Lambda scale**: start with λ1=λ2=1.0 (since L_ang ∈ [0,1] and L_rho ∈ similar range after normalization by D)
9. **Warmup trigger**: val_mse < 0.0025, ramp over 20–30 epochs
10. **Normalize L_line by valid gate count**, not batch size
