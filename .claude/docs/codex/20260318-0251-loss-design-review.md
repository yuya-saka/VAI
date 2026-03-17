# Codex Analysis: Loss Design Review
Date: 2026-03-18 02:51

## Question
Review the loss computation in Unet/line_only/line_losses.py focusing on NaN handling.

Key concerns:
1. Lines 218-232 (angle_loss): Are NaN params properly excluded BEFORE trigonometric ops?
2. Lines 250-278 (rho_loss): Same NaN exclusion check?
3. Lines 310-316 (compute_line_loss): Is valid_mask computed correctly?
4. Line 222 (angle_loss): Is confidence detached to prevent gradient collapse?

## Codex Response

**Critical Issues (bugs causing training failure)**
- NaNs are not excluded before trigonometric ops in `angle_loss`. `pred_phi`/`gt_phi` are used directly in `cos/sin` and dot-product math, then masked afterward. If any invalid entry is `NaN`, `loss` becomes `NaN`, and `NaN * 0` stays `NaN`, so batch loss can poison training even when `valid_mask=False` for that entry.
  References: [line_losses.py:206](line_losses.py:206), [line_losses.py:210](line_losses.py:210), [line_losses.py:222](line_losses.py:222), [line_losses.py:228](line_losses.py:228)

- Same NaN-order bug exists in `rho_loss`: `err1/err2`, `exp`, and smooth-min are computed before valid-only filtering. Mixed valid/invalid channels can produce `NaN` final loss.
  References: [line_losses.py:251](line_losses.py:251), [line_losses.py:260](line_losses.py:260), [line_losses.py:268](line_losses.py:268), [line_losses.py:274](line_losses.py:274)

- Repro check confirms failure mode: one valid + one NaN/invalid channel currently returns `angle_loss=NaN`, `rho_loss=NaN`.

**Potential Problems (instability risks)**
- `valid_mask` in `compute_line_loss` is logically reasonable (`gt_valid & pred_valid & conf_valid`), but it is not sufficient to prevent NaN propagation because filtering happens too late in downstream losses.
  Reference: [line_losses.py:313](line_losses.py:313)

- `confidence` tensor is not detached in `angle_loss`/`rho_loss`. This creates gradients w.r.t. moment-derived confidence that may encourage zero-heatmap trivial solutions (confidence drops to zero → loss weight becomes zero → gradient vanishes → network learns to output blank heatmaps).
  References: [line_losses.py:222](line_losses.py:222), [line_losses.py:268](line_losses.py:268)

- Double smoothing in `rho_loss` (smooth-min + smooth_l1_loss) is redundant and distorts gradient scale, making rho loss contribute unpredictably compared to MSE and angle loss.
  Reference: [line_losses.py:265](line_losses.py:265)

**Recommendations (concrete fixes)**
1. **Fix NaN masking order**: Apply valid_mask BEFORE computing loss components:
   ```python
   # angle_loss
   def angle_loss(pred_params, gt_params, confidence, valid_mask):
       # Extract only valid entries
       valid_pred_phi = pred_params[..., 0][valid_mask]
       valid_gt_phi = gt_params[..., 0][valid_mask]
       valid_conf = confidence[valid_mask]

       if valid_pred_phi.numel() == 0:
           return torch.tensor(0.0, device=pred_params.device, requires_grad=True)

       # Compute loss only on valid entries
       pred_nx = torch.cos(valid_pred_phi)
       pred_ny = torch.sin(valid_pred_phi)
       gt_nx = torch.cos(valid_gt_phi)
       gt_ny = torch.sin(valid_gt_phi)

       dot = pred_nx * gt_nx + pred_ny * gt_ny
       loss = 1.0 - torch.abs(dot)

       # Weight by confidence
       weighted_loss = (loss * valid_conf).sum() / (valid_conf.sum() + 1e-8)
       return weighted_loss * 0.01
   ```
   Same pattern for `rho_loss`.

2. **Detach confidence**: Change line 222 and 268 to:
   ```python
   weights = confidence.detach() * valid_mask.float()
   ```
   This prevents gradient flow through confidence, avoiding trivial zero-heatmap solutions.

3. **Remove double smoothing in rho_loss**: Delete line 265:
   ```python
   # DELETE this line
   # loss = F.smooth_l1_loss(loss, torch.zeros_like(loss), reduction="none")
   ```
   The smooth-min is already differentiable and appropriately scaled.

4. **Alternative: Replace smooth-min with direct torch.minimum**:
   ```python
   loss = torch.minimum(err1, err2)  # Simple, clear, differentiable
   ```

## Priority
🔴 Critical - Training will fail with NaN collapse without fixes #1 and #2
🟡 Important - Loss balance issues without fix #3
