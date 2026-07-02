# Stage2 Consistency Audit

Date: 2026-07-01

## Scope

This audit compares the current Stage2 implementation, runtime configuration,
tests, experiment outputs, and design documents against the stated contract:
keep the Stage1 primary classifier unchanged and add only a weakly supervised
anatomical-region auxiliary branch.

## Executive Summary

The Stage2 primary architecture is structurally compatible with Stage1, and the
core region-loss direction is implemented correctly. However, the current
`test_v2` run is not a controlled Stage1-versus-Stage2 experiment.

The largest problems are:

1. Stage1 and Stage2 use almost entirely different holdout and fold assignments.
2. Stage2 changed multiple training controls beyond adding the region branch.
3. The differential backbone LR is based on a removed BatchNorm-freezing premise.
4. The required `region_loss_weight=0` operational baseline was never run.
5. Validation component losses are not recorded.
6. The Stage2 config and tests are ignored by Git.

## Verified Correct

| Contract | Result |
|---|---|
| Stage1-compatible `encoder`/`lstm`/`head` structure | Eval-mode copied-weight parity test passes |
| Primary output uses mean sigmoid across 15 planes | Pass |
| Stage1 weighted BCE semantics at `region_loss_weight=0` | Unit test passes |
| Noisy-OR combines regions only within each plane | Pass |
| Horizontal flip swaps only right/left foramina | Pass |
| BF16-sensitive recurrent/head paths run in FP32 | Pass |
| Checkpoint architecture version is validated | Pass |
| Current Stage2 test suite | 29 tests passed |

## Critical Findings

### C1. Existing Stage1 and Stage2 metrics are not split-compatible

Measured from the current dataset and exclusions:

| Quantity | Stage1 | Stage2 |
|---|---:|---:|
| Items | 14,077 | 13,433 |
| Studies | 2,012 | 2,009 |
| Study fracture-label changes after level exclusion | - | 55 |
| Holdout studies | 403 | 402 |
| Common holdout studies | - | 100 |
| Fold-0 validation studies | 322 | 322 |
| Common fold-0 validation studies | - | 43 |
| Fold-0 train items | 9,007 | 8,586 |
| Fold-0 validation items | 2,250 | 2,144 |

Split stratification is recomputed after Stage2 exclusions. The current Stage1
and Stage2 AUROC/AUPRC values therefore do not measure the same patients.

Required correction:

- Persist one canonical study split manifest.
- Apply it to both Stage1 and Stage2.
- Compare only the common Stage2-eligible item population.

### C2. The run changes many controls beyond the region branch

| Control | Stage1 baseline | Stage2 `test_v2` |
|---|---|---|
| Backbone LR | `2.3e-4` | `2.3e-5` |
| Head LR | `2.3e-4` | `2.3e-4` |
| LR floor | `2.3e-5` | `2.3e-6` for both groups |
| Mixup probability | `0.2` | `0.02` |
| Gradient clipping | None | Global norm `1.0` |
| CNN autocast | FP16 | BF16 |
| Primary BiLSTM/head | autocast path | forced FP32 |
| Spatial augmentation | independently sampled per plane | replayed across all planes |
| Vertical flip | `0.5` | `0.0` |
| Transpose | `0.5` | `0.0` |

Some changes are defensible, but together they prevent attribution of a
performance difference to the region branch.

The common scalar `eta_min=2.3e-6` also gives different decay ratios:

- backbone: `2.3e-5 -> 2.3e-6` (10x)
- heads: `2.3e-4 -> 2.3e-6` (100x)

### C3. Backbone LR rationale is stale

The June 29 work log selected a 10x lower backbone LR because backbone BatchNorm
would be frozen. The July 1 redesign removed BatchNorm freezing to restore
Stage1 behavior, but the lower LR remained.

Required correction:

- Use Stage1 optimizer/scheduler settings for the controlled parity baseline.
- Introduce differential LR only as a separately named ablation.

### C4. Required zero-region baseline was skipped

The rollout plan requires an operational `region_loss_weight=0` run before the
`region_loss_weight=0.5` experiment. Unit-level parity does not validate
optimizer, augmentation, AMP, DDP, or data-split parity.

### C5. Config and tests are not versioned

`.gitignore` ignores every `tests/` directory and every `*.yaml` file.
Consequently, `train_models/stage2/tests/` and the canonical Stage2 config are
not tracked.

## High-Severity Findings

### H1. Validation loss components are discarded

Training records total, Stage1, and region losses. Validation discards the
component dictionary and stores only total loss. The failing branch cannot be
identified from the current run.

### H2. Required auxiliary evaluation is incomplete

The evaluator emits `region_pred_prob`, but epoch, OOF, and test metrics are
computed only from primary `pred_prob`. The planned saturation metric is also
missing.

### H3. Resume is not an exact continuation

- Only the best-AUROC checkpoint is saved, not the latest epoch.
- Scheduler state is saved before `scheduler.step()`.
- Saved and current configs are not compared.
- Startup overwrites output `config.yaml` before resume validation.
- Training JSONL is always appended.

### H4. DDP reporting and validation are inconsistent

- Every rank evaluates the complete validation set.
- Training statistics are rank-0 local values, not all-reduced global values.
- Primary-head BatchNorm uses per-GPU batch statistics, so global-batch equality
  does not establish Stage1 parity.

### H5. Invalid-region masking is not finite for infinite logits

`region_log_survival()` multiplies `logsigmoid(-logit)` by a validity mask.
For an invalid region with `logit=+inf`, this becomes `-inf * 0 = NaN`. This was
reproduced directly. Use `torch.where(valid_mask, log_not_region, 0)`.

### H6. Experiment output reuse is not guarded

Starting without `--resume` in an existing experiment directory appends to the
existing JSONL and can mix run histories.

## Medium-Severity Findings

### M1. ReplayCompose also synchronizes intensity augmentation

The complete augmentation pipeline is replayed across all 15 planes. This
synchronizes brightness, blur/noise, and distortion choices, although the
earlier work log specifies shared spatial transforms and per-plane brightness.

### M2. Region-collapse diagnostics are availability-biased

Entropy is normalized by `log(4)` when fewer than four regions may be valid, and
argmax distribution does not adjust for each region's opportunity to be valid.

### M3. Validation loss is a mean of batch means

The final short batch receives the same weight as a full batch. Stage1 has the
same behavior, so this is parity-compatible but not an exact dataset-level BCE.

### M4. Documentation overstates rollout completion

The work log says phases 1-7 are complete while also listing the real-data smoke
test, single-GPU short run, and operational zero-region parity run as unperformed.

## Interpretation of `test_v2`

The validation-loss behavior cannot be attributed specifically to the region
branch because validation components are absent, the patient split differs, the
training controls differ, and no zero-region operational control exists.

`test_v2` is a useful Stage2 training smoke experiment, but it is not a valid
controlled comparison against the existing Stage1 baseline.

## Required Remediation Order

1. Version the canonical Stage2 config and tests.
2. Persist a common-population split manifest.
3. Add validation component logging.
4. Fix invalid-region finite masking.
5. Add config-safe latest-checkpoint resume and output reuse guards.
6. Remove DDP duplicate validation and all-reduce training statistics.
7. Run `region_loss_weight=0` with Stage1-matched controls.
8. Run `region_loss_weight=0.5` changing only the auxiliary branch/loss.
9. Tune differential LR, mixup, clipping, or augmentation only as named ablations.

