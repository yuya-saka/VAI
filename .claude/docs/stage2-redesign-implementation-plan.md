# Stage2 Redesign Implementation Plan

Date: 2026-07-01

## Goal

Rebuild `train_models/stage2/` as an unchanged Stage1 primary classifier plus a
weakly supervised four-region auxiliary branch.

The primary contract is:

- Stage1 final `1280`-dimensional EfficientNetV2-S feature
- Stage1 two-layer BiLSTM and per-plane classification head
- weighted BCE on all 15 plane logits
- mean sigmoid across planes as the primary vertebra probability

The region contract is:

- region masks are side inputs for feature pooling, not CNN input channels
- FPN features are mask-pooled per plane and per region
- one shared temporal region head returns `[B, 15, 4]` logits
- stable Noisy-OR combines the four regions within each plane
- weighted BCE supervises every valid plane using only the vertebra label
- mean probability across valid planes gives the auxiliary vertebra score

## Phase 1: Lock Stage1 Parity with Tests

Files:

- `train_models/stage2/test_model.py`
- `train_models/stage2/test_losses.py`

Tasks:

1. Add a parity test that copies Stage1-compatible `encoder`, `lstm`, and `head`
   weights into Stage2 and verifies identical per-plane logits in evaluation
   mode.
2. Verify that `region_loss_weight=0` reproduces Stage1 weighted BCE exactly.
3. Verify that Stage2 primary inference uses mean sigmoid, not sigmoid of a
   mean logit.

Acceptance:

- Stage1 and Stage2 primary logits and loss agree within floating-point
  tolerance on the same input.

## Phase 2: Replace the Model

File:

- `train_models/stage2/src/model.py`

Tasks:

1. Create the backbone with `features_only=False`, matching Stage1.
2. Use `forward_intermediates(indices=(1, 2, 3, 4))` once per flattened plane
   batch. With `timm==1.0.22`, this returns the final `1280`-channel map and the
   four FPN inputs in one encoder pass.
3. Obtain the exact Stage1 vector with
   `encoder.forward_head(final_map, pre_logits=True)`.
4. Preserve Stage1 module structure as `encoder`, `lstm`, and `head`.
5. Build the region branch from the four intermediate maps:
   lateral `1x1` projections, resize to `56x56`, concatenate, and fuse.
6. Apply mask-normalized pooling to produce `[B, 15, 4, C]`.
7. Run one FP32 shared BiLSTM/head over each region sequence and retain all
   per-plane logits as `[B, 15, 4]`.
8. Return a named output containing:
   `slice_logits`, `region_logits`, and `valid_region_planes`.
9. Remove the old global head, temperature parameter, region-before-plane
   aggregation, and vertebra-level LSE output.

Acceptance:

- One CNN pass produces both exact Stage1 logits and finite region logits.
- Invalid region masks never contribute to pooling or later losses.

## Phase 3: Replace the Loss

File:

- `train_models/stage2/utils/losses.py`

Tasks:

1. Keep the Stage1 sample-weighted BCE semantics.
2. Expand the scalar vertebra label to `[B, 15]` for the primary plane loss.
3. Implement stable per-plane Noisy-OR over valid regions:
   `p_any_slice = 1 - product(1 - sigmoid(region_logit))`.
4. Compute Noisy-OR in log space using `logsigmoid` and `expm1`.
5. Apply weighted BCE to every valid `p_any_slice`.
6. Define:
   `total = stage1_loss + region_loss_weight * region_slice_loss`.
7. Preserve mixup by evaluating both target permutations against the same mixed
   image and mixed soft region masks.

Acceptance:

- Direct BCE is never applied independently to the four region logits.
- Noisy-OR is never applied across the 15 planes.
- Extreme logits, missing regions, and mixed soft masks produce finite losses
  and gradients.

## Phase 4: Update Training and Configuration

Files:

- `train_models/stage2/src/trainer.py`
- `train_models/stage2/src/data_utils.py`
- `train_models/stage2/config/config.yaml`

Tasks:

1. Replace `global_loss_weight` with `region_loss_weight` and start at `0.5`.
2. Remove temperature and global-head configuration.
3. Keep BF16/FP32 safety, gradient clipping, DDP, resume, and diagnostics.
4. Log `stage1_loss`, `region_slice_loss`, and total loss separately.
5. Keep the Stage1 probability as `pred_prob`.
6. Add `region_pred_prob` as the valid-plane mean of `p_any_slice`.
7. Save the four valid-plane mean region evidence scores using anatomical names.
8. Add a checkpoint architecture version and reject incompatible old Stage2
   checkpoints with a clear error.

Acceptance:

- Checkpointing, mixup, DDP, AMP, early stopping, and ensemble prediction use
  the new output contract.
- Early stopping and primary metrics continue to use `pred_prob`.

## Phase 5: Correct Dataset Semantics

Files:

- `train_models/stage2/src/dataset.py`
- `train_models/stage2/test_dataset.py`

Tasks:

1. Define region IDs as:
   `1=body`, `2=right_foramen`, `3=left_foramen`, `4=posterior`.
2. Correct horizontal-flip remapping to `[0, 1, 3, 2, 4]`.
3. Keep region masks synchronized across all spatial augmentation replay.
4. Continue validating integer and soft-mask input forms.

Acceptance:

- Horizontal flip swaps only right and left foramina.
- Body and posterior labels remain unchanged.

## Phase 6: Update Evaluation

Files:

- `train_models/stage2/src/evaluation.py`
- `train_models/stage2/src/trainer.py`
- `train_models/stage2/train.py`
- `train_models/stage2/test_evaluation.py`

Tasks:

1. Report primary Stage1-compatible metrics from `pred_prob`.
2. Report separate auxiliary metrics from `region_pred_prob`.
3. Report region evidence distribution, valid-plane rate, entropy, saturation,
   and dominant-region frequency.
4. Remove manual region-label evaluation and `region_labels_path`, because no
   region labels exist.
5. Update ensemble output keys to anatomical names.

Acceptance:

- Primary metrics remain directly comparable with Stage1.
- Region outputs are explicitly named evidence scores, not calibrated
  probabilities.

## Phase 7: Validation and Rollout

Commands:

```bash
UV_CACHE_DIR=.tmp/uv-cache uv run --offline pytest train_models/stage2 -q
UV_CACHE_DIR=.tmp/uv-cache uv run --offline ruff check train_models/stage2
UV_CACHE_DIR=.tmp/uv-cache uv run --offline ruff format --check train_models/stage2
```

Additional checks:

1. CPU forward/backward smoke test with missing regions and extreme logits.
2. One real DataLoader batch smoke test.
3. One short single-GPU training run before DDP.
4. Confirm `region_loss_weight=0` parity, then run the first auxiliary
   experiment with `region_loss_weight=0.5`.

Rollout:

- Use a new experiment name such as `redesign_v1`.
- Do not reuse old Stage2 checkpoints or output directories.
- Do not modify Stage1 implementation or existing Stage1 outputs.

Rollback:

- Revert only `train_models/stage2/` and this design-plan entry.
- Existing Stage1 code, checkpoints, and outputs remain untouched.
