# Codex Analysis: line_loss tuning
Date: 2026-03-29

## Question

TinyUNet heatmap regression で line_loss を加えると性能が悪化する原因と修正方法。

RESULTS: Baseline MSE: angle_err=5.23deg, rho=2.90px. With line_loss ON: angle_err=6.50deg, rho=3.34px.
LOSS: L_mse~0.002, L_ang~0.5 (range 0-1), L_rho~0.05 (SmoothL1 normalized), lambda_angle=1.0, lambda_rho=1.0.
Warmup starts epoch 90, 30 epochs, 200 total.

## Codex Response

### 1. Root cause of `line_loss` hurting

- The line terms are dominating optimization by ~2 orders of magnitude.
- Current weighted loss magnitude breakdown:
  - `L_mse = 0.002`
  - `λ_ang * L_ang = 1.0 * 0.5 = 0.5`
  - `λ_rho * L_rho = 1.0 * 0.05 = 0.05`
  - Total `≈ 0.552`, so MSE is only `0.36%` of total.
- The model is no longer optimizing heatmap quality; it is mostly chasing noisy/unstable geometric terms (especially angle), which explains worse `angle_err` and `rho`.

### 2. Are lambdas mismatched? What values fix it?

- Yes, strongly mismatched.
- Equalize each line term to MSE scale (`~0.002`):
  - `λ_angle ≈ 0.002 / 0.5 = 0.004`
  - `λ_rho ≈ 0.002 / 0.05 = 0.04`
- Practical safer start (to avoid over-regularization):
  - `λ_angle = 0.002`, `λ_rho = 0.02` (line total ~ MSE total)
- Keep roughly `λ_rho ≈ 10 * λ_angle` because raw `L_ang` is ~10x larger than `L_rho`.

### 3. Fixed vs dynamic warmup trigger (`val_mse < 0.0025`)?

- Prefer **dynamic trigger** over fixed epoch 90.
- Recommended trigger:
  - Start line warmup when `val_mse < 0.0025` for **3–5 consecutive epochs**, with `min_epoch=20`.
  - Ramp over **20 epochs** (not 30) to final lambdas.
- Why: fixed epoch 90 is late and arbitrary; dynamic trigger ties geometry loss activation to heatmap readiness.

### 4. Concrete 3–4 run experiment plan

- **Run A (control):** baseline MSE only (reconfirm `5.23° / 2.90px`).
- **Run B:** fixed schedule, but rescaled lambdas: `λ_ang=0.004`, `λ_rho=0.04`, warmup start 90, ramp 30.
- **Run C:** dynamic trigger (`val_mse<0.0025` x 5 epochs), `λ_ang=0.004`, `λ_rho=0.04`, ramp 20.
- **Run D:** dynamic trigger + conservative final weights: `λ_ang=0.002`, `λ_rho=0.02`, ramp 20.
- **Success criterion**: beat baseline on both metrics (`angle_err < 5.23°` and `rho < 2.90px`).

### 5. Alternative formulations

- Use line-invariant angle loss: `L_ang = 1 - cos(2Δθ)` (handles line direction ambiguity better than plain angle diff).
- Delay geometry gradients to head only (or smaller LR multiplier for backbone, e.g. `0.1x`) during warmup.
- Use uncertainty weighting instead of fixed lambdas:
  `L = L_mse + exp(-s1)*L_ang + s1 + exp(-s2)*L_rho + s2` (learns task weights automatically).
- Replace `(θ,ρ)` regression with endpoint regression + line fit (often more stable for heatmap-based models).

## Summary

**Root cause**: lambda=1.0 makes L_line (~0.55) dominate L_mse (~0.002) by 250x. MSE becomes only 0.36% of total loss.

**Fix**: Scale lambdas so each term matches MSE magnitude:
- `λ_angle = 0.004` (= 0.002/0.5)
- `λ_rho = 0.04` (= 0.002/0.05)
- Or conservative: `λ_angle = 0.002`, `λ_rho = 0.02`

**Priority experiment**: Run B first (rescaled lambdas, same fixed schedule) to confirm scale was the issue.
