# Codex Analysis: Hyperparameter Analysis for Angle Accuracy Improvement
Date: 2026-03-30

## Question

You are analyzing a U-Net training configuration for medical image line detection (vertebral X-ray boundary detection). The task is to detect 4 boundary lines as heatmaps, then extract (phi, rho) line parameters.

Current Setup:
- Model: TinyUNet, features=[16,32,64,128], dropout=0.1, in_channels=2 (CT + mask), out_channels=4
- Vertebra conditioning: use_vertebra_conditioning=false (config.yaml) - BUT work-log says it should be true (C1-C7 one-hot at bottleneck). This discrepancy needs investigation.
- Loss: MSE-only (use_line_loss: false) - L_line loss is implemented but disabled
- Optimizer: AdamW, lr=2e-4, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau, mode=min, patience=8, factor=0.5
- Epochs: 230, early_stopping_patience=15
- lr_patience=8, lr_factor=0.5
- grad_clip=1.0
- Augmentation: rotation±10°, scale±5%, brightness/contrast±10% (very weak), horizontal_flip=false, gaussian_noise=false
- sigma: 2.0 (heatmap gaussian sigma in pixels), image_size=224
- batch_size=8

L_line loss details (currently disabled):
- lambda_angle=0.004, lambda_rho=0.04
- warmup_start_epoch=90 (waits for MSE convergence first)
- warmup_epochs=30, warmup_mode=linear
- confidence_gate_low=0.1, confidence_gate_high=0.8

Current results (5-fold CV):
- fold0: 5.57°, fold1: 7.60°, fold2: 5.08°, fold3: 4.26°, fold4: 7.99°
- avg: 6.10°
- Target: < 5°

Questions:
1. What are the most impactful hyperparameter changes to improve angle accuracy? Consider: model capacity (features=[16,32,64,128] might be too small?), augmentation strategy (very weak currently), loss function (should L_line be enabled?), learning rate schedule, dropout, sigma value.
2. The vertebra conditioning is disabled in config but work-log says it helps. Should it be enabled?
3. For the L_line loss - are lambda_angle=0.004 and lambda_rho=0.04 appropriate weights? Current MSE is around 0.0001-0.001 scale. What warmup strategy is best?
4. The high variance between folds (4.26° to 7.99°) suggests overfitting or data variance issues. What can help?
5. Rank the top 5 parameter changes by expected impact.

Please provide specific recommended values for each parameter change.

## Codex Response

Mismatch investigation result:
- Runtime config is currently `use_vertebra_conditioning: false` in config.yaml.
- Work-log says `true` in unet-work-log.md.
- Recent weak-aug 5-fold W&B configs also show `false`.
- `*.yaml` is gitignored in .gitignore, so doc/config drift is expected.

### 1) Most impactful changes for angle accuracy (recommended values)

- `data.sigma: 3.5` (strongest historical angle gain in your own ablation)
- `loss.use_line_loss: true`
- `loss.lambda_angle: 0.008`, `loss.lambda_rho: 0.08` (start here; if stable, try `0.012/0.12`)
- `loss.warmup_start_epoch: 70`, `loss.warmup_epochs: 20`, `loss.warmup_mode: "cosine"`
- `augmentation.rotation_limit: 15`, `scale_limit: 0.10`, `brightness_limit: 0.2`, `contrast_limit: 0.2`, `gaussian_noise: true`, `noise_var_limit: [5, 20]`
- `model.features: [24, 48, 96, 192]`, `model.dropout: 0.05`

### 2) Should vertebra conditioning be enabled?

- Default recommendation: keep `use_vertebra_conditioning: false` for now.
- Reason: CT+mask input already carries vertebra identity; bottleneck one-hot is redundant and can overfit on small-N.
- Only enable after controlled A/B (same fold, same seed, same aug) and keep it if it improves mean angle by at least ~0.3° consistently.

### 3) Are lambda_angle=0.004, lambda_rho=0.04 appropriate? Warmup best?

- They are conservative/safe, but likely weak for breaking the <5° barrier.
- Recommended: `0.008/0.08` first, then `0.012/0.12` if stable.
- Best warmup: plateau-aware start is ideal; with current fixed config use `start=70`, `epochs=20`, `cosine`.

### 4) High fold variance (4.26° to 7.99°): what helps?

- Run each fold in a separate process: `--start_fold f --end_fold f`.
- For reproducibility check, temporarily set `num_workers: 0` (then restore).
- Increase regularization slightly: `weight_decay: 2e-4`.
- Increase `early_stopping_patience: 35` when line loss is on (current 15 can stop before geometry gains).

### 5) Top 5 changes by expected impact (ranked)

1. `sigma: 3.5` — strongest single-parameter angle gain historically
2. Enable line loss + `lambda_angle/lambda_rho = 0.008/0.08` + warmup `70/20/cosine`
3. Stronger augmentation (`rotation 15°`, `scale 0.10`, `brightness/contrast 0.2/0.2`, gaussian noise on)
4. Increase model capacity to `[24,48,96,192]` and reduce dropout to `0.05`
5. Fold isolation + reproducibility controls (per-fold process, workers check, `early_stopping_patience: 35`)
