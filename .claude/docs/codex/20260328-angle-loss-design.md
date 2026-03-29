# Codex Analysis: angle_loss design for line detection U-Net
Date: 2026-03-28

## Question

You are analyzing a PyTorch U-Net training setup for line detection in medical CT images. Here is the full context:

TRAINING DYNAMICS (MSE-only baseline):
- Phase 1 (epoch 1-50): val_mse 0.219->0.006, angle_error=35-50 degrees with many NaN values (heatmap below threshold=0.2)
- Phase 2 (epoch 50-95): val_mse 0.006->0.002, angle_error 25-40 degrees
- Phase 3 (epoch 95-136): sudden improvement angle 40 degrees -> 7-10 degrees, val_mse ~0.002->0.0018
- Best test result: angle=5.32 degrees, rho=3.11px

CURRENT CODE:

losses.py extract_pred_line_params_batch:
- Python for-loop over (B,C) batch/channel dimensions
- Uses .double() conversion for precision
- Assigns to output[b,c] = phi.float() and output[b,c] = float('nan') for invalid
- confidence = 1 - lambda2/lambda1

angle_loss:
- Loss = 1 - |dot(n_pred, n_gt)| where n = [cos(phi), sin(phi)]
- valid_mask gates which samples contribute
- confidence.detach() is used as weight
- Returns weighted_loss * 0.01

trainer.py:
- loss = (1 - 0.5 * warmup_weight) * loss_mse + warmup_weight * line_loss_dict['total']
- line_loss_dict['total'] = lambda_theta(=0.1) * angle_loss_raw
- warmup_epochs=50 (linear from 0 to 1)

QUESTIONS:
1. Does the for-loop with .double() and tensor assignment output[b,c] = phi.float() properly propagate gradients through PyTorch autograd? What about the output[b,c] = float('nan') assignments for invalid samples?

2. Given angle only becomes reliable at epoch ~96 (val_mse < 0.002), is warmup_epochs=50 too early? What warmup strategy fits this dynamic?

3. Is the loss formula (1-0.5w)*MSE + w*line_loss balanced given MSE~0.002 and line_loss~0.001 at epoch 100+?

4. Compare these angle_loss design alternatives:
   a. Current: differentiable moment extraction -> angle loss (potential gradient flow issues)
   b. Anisotropic Gaussian target: make GT heatmap elliptical along line direction, MSE naturally trains angle
   c. Direction field: add 2-channel angle output (cos_phi, sin_phi) as separate head
   d. Detached heatmap moment + angle supervision only when confidence high (current but with fixes)

5. Recommend concrete values: warmup_epochs, lambda_theta, loss formula, min_confidence for Phase 2.

## Codex Response

1. **Autograd correctness (most important)**
- The Python `for` loop itself does **not** break gradients. It is slow, but autograd can still track ops inside it.
- `.double()` / `.float()` casts are differentiable. They do not cut gradient flow by themselves.
- `output[b, c] = phi.float()` is usually tracked (`CopySlices` path), so gradients can flow back to `phi`.
- `output[b, c] = float('nan')` for invalid entries is the dangerous part:
  - Those entries have no useful gradient (constant assignment).
  - If NaNs touch later ops (`cos`, `sin`, `dot`, reductions), they can poison the whole loss/grad even with masking (`0 * NaN = NaN`).
- `confidence.detach()` means weighting does not backprop through confidence (fine if intentional).
- Recommendation: **do not store NaNs in training tensors**. Use `torch.where(valid, phi, 0.0)` and mask before trig/reduction.

2. **Warmup timing**
- Yes, `warmup_epochs=50` is early for your observed dynamics (angle only becomes reliable around epoch ~96).
- Better: delayed, metric-gated warmup.
  - Start angle term when `EMA(val_mse) < 0.0025` **and** valid-ratio/confidence is stable for 3 epochs.
  - Fallback fixed start if you want deterministic: `start_epoch=85–90`.
  - Ramp length: `20–30` epochs (cosine or linear).

3. **Loss balance**
- Current formula at full warmup is `0.5*MSE + line`. That explicitly downweights pixel fit.
- Given your scales (`MSE~0.002`, `line~0.001`), it is not catastrophically off, but I'd avoid reducing MSE.
- Prefer:
  L = L_mse + w * lambda_theta * L_theta
  This keeps localization pressure constant while adding orientation gradually.

4. **Design alternatives**
- **a) Current differentiable moments**: workable, minimal changes, but unstable near isotropic blobs (`lambda1≈lambda2`) and NaN-prone if invalid handling is weak.
- **b) Anisotropic Gaussian target**: easiest robust upgrade for single-head setup; orientation signal appears earlier via plain MSE; good practical choice.
- **c) Direction-field head (`cos, sin`)**: best long-term for angle quality/stability; clearer supervision; more implementation cost.
- **d) Detached moment + gated supervision**: good as a stabilizer only if you still have a gradient path for angle learning elsewhere. Fully detaching moment-based `phi` removes that path.

5. **Concrete settings**
- `warmup_start`: metric-gated, typically around epoch `90` on your run.
- `warmup_ramp`: `25` epochs.
- `lambda_theta`: start `0.02`, ramp to `0.15` (try `0.2` if angle plateaus).
- Loss formula: `L = L_mse + w * lambda_theta * L_theta` (drop `(1 - 0.5w)` factor).
- `min_confidence` (Phase 2): start `0.6`, relax to `0.45` after ramp completion.
- Also change confidence to a numerically safer form:
  conf = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)
  then clamp [0,1].

If you want, I can draft a drop-in `angle_loss`/moment-extraction patch that removes NaNs and applies this gated schedule.
