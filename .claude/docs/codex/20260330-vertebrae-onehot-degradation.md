---
# Codex Analysis: vertebrae-onehot conditioning degradation
Date: 2026-03-30

## Question

Analyze why bottleneck one-hot vertebra conditioning hurts U-Net performance when vertebra mask is already in input.

Context:
- TinyUNet for spinal X-ray line detection (phi, rho via heatmaps)
- Input: 2ch (CT image + vertebra MASK image) -- mask already encodes vertebra identity
- One-hot conditioning: concat 7-class one-hot to bottleneck (128ch), project back to 128ch via 1x1 conv
- Identity init: weight[:, :128, 0, 0] = eye(128), weight[:, 128:, 0, 0] = 0, bias = 0
- Result: val_mse 21% worse (0.001996 vs 0.001637), test_angle 9% worse (5.94° vs 5.43°)
- Data: <100 patients, 7 vertebra classes, early_stopping_patience=15

Questions:
1. Is the identity init correct for nn.Conv2d(135, 128, kernel_size=1)?
2. Why does redundant conditioning hurt rather than being neutral?
3. Is AdamW weight_decay=1e-4 potentially causing issues?
4. Root cause ranking
5. Recommended fix

## Codex Response

1. Yes, your identity init is mathematically correct.

Given concat input
z = cat(x4, c) ∈ R^{B×135×Hb×Wb},
with x4 ∈ R^{B×128×Hb×Wb}, c ∈ R^{B×7×Hb×Wb},
and W ∈ R^{128×135×1×1}, b ∈ R^{128},

y_o = sum_{i=1}^{128} W_{o,i} x4_{i} + sum_{k=1}^{7} W_{o,128+k} c_k + b_o.

If you set:
- W_{o,i} = delta_{oi} for i <= 128,
- W_{o,128+k} = 0,
- b_o = 0,

then y_o = x4_{o} exactly.
That matches your code in model.py with concat order [x4, cond].

2. Why it hurts even though neutral at t=0:

Neutral is only initialization, not a constraint. During training, W drifts.

For conditional channels:
dL/dW_{o,128+k} = sum_{b,h,w} dL/dy_{b,o,h,w} * 1[v_b = k].

Since one-hot is constant over all (h,w), this learns a strong class-specific global offset quickly.

So bottleneck becomes:
y ≈ A x4 + u_v + b,

where u_v is a vertebra-specific bias vector. On small data, this is a shortcut (class prior)
and overfits, especially because mask is already in input (dataset.py).
With batch size 8 and BN in decoder blocks, class-dependent shifts can also worsen BN running stats noise.

3. AdamW weight_decay=1e-4 is probably not the main culprit.

- No "spurious momentum" update happens without gradient.
- Zero-initialized cond weights stay zero if grads are zero.
- Decoupled decay on identity weights is tiny: per-step factor ≈ 1 - lr*wd ≈ 1 - 2e-8 (lr=2e-4, wd=1e-4).

So optimizer is not creating corruption by itself; it is just optimizing an unnecessary branch.

4. Root-cause ranking (most likely first):

1. (a) Redundant conditioning creates shortcut/overfit pressure when mask already provides vertebra information.
2. (c) Small data (<100 patients, 7 classes) amplifies that overfit and instability.
3. (d) AdamW effects are secondary (can modulate speed, not primary cause).
4. (b) Identity-init bug is least likely (your init is correct).

5. Recommended fix:

Primary recommendation: remove one-hot conditioning (keep use_vertebra_conditioning: false)
because mask already carries the same information and your ablation already shows degradation.

If you still want to keep label conditioning for experimentation:
- Use gated residual form x4' = x4 + alpha * f(c) with alpha=0 init.
- Freeze alpha, f for first N epochs, then unfreeze with 10x lower LR.
- Optionally add regularization toward alpha=0 or W_cond=0.

This keeps true neutrality during early training instead of "neutral only at step 0".
