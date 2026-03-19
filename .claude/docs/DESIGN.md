# Project Design Document

> This document tracks design decisions made during conversations.
> Updated automatically by the `design-tracker` skill.

## Overview

<!-- Project purpose and goals -->

## Architecture

<!-- System structure, components, data flow -->

```
[Component diagram or description here]
```

## Implementation Plan

### Patterns & Approaches

<!-- Design patterns, architectural approaches -->

| Pattern | Purpose | Notes |
|---------|---------|-------|
| Provider fallback wrapper | Rate-limit resilience across Claude/Gemini | Centralized error parsing, retry, and provider switch |
| Canonical conversation store | Preserve context across provider switches | Single source of truth (history + system prompt + tool logs) |
| Heatmap-moment line geometry loss | Add geometric supervision for line detection | Normal-form (phi, rho) from GT endpoints and heatmap moments; sign alignment; staged warmup |

### Libraries & Roles

<!-- Libraries and their responsibilities -->

| Library | Role | Version | Notes |
|---------|------|---------|-------|
| | | | |

### Key Decisions

<!-- Important decisions and their rationale -->

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| Adopt flat package layout with explicit package list in hatchling | Aligns with requirement to remove src/ while keeping deterministic builds | Dynamic discovery; keep src layout | 2026-01-31 |
| Keep distribution name configurable via `[project].name` while package import name remains stable | Enables identity switch without code moves | Rename package directory per release | 2026-01-31 |
| Implement rate-limit fallback at a single provider wrapper layer | Reduces duplication and keeps switching logic consistent | Hook-based or per-agent fallback | 2026-01-31 |
| Prefer wrapper-based detection over Claude hook reliance for rate-limit fallback | Hooks may not fire on API errors and only run inside Claude lifecycle | Hook-based detection; log scraping | 2026-01-31 |
| Use semi-automatic fallback (prompt user to switch) by default | Prevents unexpected provider switches and preserves user intent | Fully automatic switching | 2026-01-31 |
| Persist canonical conversation state for cross-provider reuse | Enables Gemini to continue with full context | Ad-hoc copy/paste | 2026-01-31 |
| Align angle evaluation by using the same estimator on GT and predictions; default to heatmap-moment angle for both and keep polyline PCA as an oracle metric | Avoids systematic bias from representation mismatch and separates estimator error from model error | Compare raw polyline PCA vs predicted heatmap moments | 2026-02-18 |
| Refactor training/evaluation into smaller, single-purpose functions (data split/load, model/opt setup, train loop, eval metrics, visualization) | Improve readability, testability, and reduce regression risk during future changes | Keep monolithic functions; split only by file | 2026-03-04 |
| Minimal line-eval reporting: angle_error_deg + rho_error_px as core, keep perpendicular_dist_px for interpretability, and one heatmap peak metric (e.g., peak_dist) for debugging | Keeps task metrics primary while preserving a light diagnostic signal for heatmap quality | Drop all heatmap metrics; report only angle/rho | 2026-03-16 |
| Set Gaussian heatmap sigma based on output resolution and annotation uncertainty; start with sigma ≈ 2–3 px on the output map (scale by stride), and ensure L^2/12 ≫ sigma^2 for stable moment-based angle; consider anisotropic sigma (smaller perpendicular) if angle is unstable | Balances training stability vs geometric precision for moment-based line estimation | Fixed sigma without scaling; isotropic-only kernels | 2026-03-16 |
| Default to no hard thresholding for moment-based line estimation; if background noise dominates, prefer soft weighting or percentile-based gating to suppress low-confidence regions while preserving Gaussian tails | Hard thresholding can bias centroid/covariance and increase angle jitter when signal is blurred or low-SNR | Always-threshold; binary mask moments | 2026-03-16 |
| Use staged training for heatmap+geometry: short heatmap-only pretrain, then enable angle loss, then add rho loss | Early geometry signals are noisy when heatmaps are weak; angle is more stable than rho and can shape representation before full constraints | Linear warmup from epoch 1; two-stage heatmap-only then all geometry | 2026-03-16 |
| Detach prediction-derived confidence in line losses and avoid confidence-gradient gating during training to prevent trivial zero-heatmap minima | Self-weighting by confidence lets the model reduce loss by shrinking/flattening heatmaps; detaching preserves the weighting signal without incentivizing collapse | Remove confidence weighting; use GT-only masks; add explicit mass regularizer | 2026-03-17 |
| Never propagate NaN sentinel values through loss arithmetic; keep a separate validity mask, index only valid entries for angle/rho losses, and add finite-guard checks before backward | `NaN * 0` is still `NaN`; mixed valid/invalid channels can poison batch loss and collapse training when geometry warmup becomes dominant | Keep NaN sentinels and rely on weighting/masking after elementwise ops | 2026-03-17 |
| Treat epoch-48 NaN collapse as a NaN-masking arithmetic failure amplified by late warmup; prioritize NaN-safe valid-only reduction and finite guards before hyperparameter retuning | Current code computes trigonometric losses on NaN-marked entries and then masks (`NaN * 0`), which can immediately poison loss/gradients once invalid channels appear | Tune LR/clip/warmup first; rely on confidence thresholding alone | 2026-03-17 |
| Implement loss-time valid-index slicing (`idx = valid_mask & isfinite`) before trig/exp operations in `angle_loss` and `rho_loss` | Prevents NaN propagation from invalid channels and avoids empty reductions (`mean of empty slice`) | Multiply-by-mask after full-tensor arithmetic; post-hoc `nan_to_num` | 2026-03-17 |
| Sequence loss-design fixes as: (1) NaN handling in angle/rho loss, then (2) warmup formula, then (3) rho double-smoothing removal; rollout as two commits (`#3` first, then `#1+#2`) | NaN-safe arithmetic is a hard prerequisite for stable training; warmup and smoothing tune gradient shape after numerical safety is guaranteed | Apply warmup first; apply all three in one opaque change | 2026-03-18 |
| Prefer phased rollout order `#3 -> #2 -> #1` with validation gates after each commit; if speed is required, still isolate `#3` then batch `#2+#1` | `#3` is a hard stability prerequisite, while `#2` changes rho-loss curvature/scale and should be established before global reweighting in `#1` to keep effects attributable and tuning interpretable | `#3 -> #1 -> #2`; one-shot three-fix commit | 2026-03-18 |
| For 200-epoch heatmap+geometry runs, prefer adaptive geometry warmup triggered by robust `val_mse` plateau detection (EMA + relative-improvement patience + hysteresis) over a fixed 50-epoch ramp | Typical MSE plateau appears around epochs 30-50; adaptive trigger reduces wasted MSE-only epochs while avoiding premature geometry activation under noisy validation curves | Fixed 50-epoch warmup; pure gradient-threshold trigger; immediate full geometry weight at plateau | 2026-03-18 |
| Standardize plateau trigger as dual-gate detector: EMA-smoothed `val_mse`, relative-improvement threshold, patience, min-epoch guard, and one-way latch (no deactivation) | Reduces false triggers from noisy validation while still reacting near true MSE saturation; one-way latch avoids oscillation once geometry warmup starts | Raw metric slope only; absolute-delta only; trigger/revert oscillating gate | 2026-03-18 |
| After plateau trigger, ramp geometry weight with bounded cosine over ~20 epochs and keep fixed full geometry weight thereafter; enforce fallback latest-start epoch (e.g., 60) | Cosine ramp avoids gradient shock at activation; fallback start prevents never-trigger cases and guarantees geometry training budget in 200-epoch runs | Immediate step to full weight; linear ramp without fallback; fixed 50-epoch start | 2026-03-18 |
| Fix adaptive warmup defaults for current 200-epoch setup: EMA(0.9) `val_mse`, trigger when relative improvement stays below 0.2% for 5 epochs after epoch >= 20, force start at epoch 60 if not triggered; use `w(e)=0.5*(1-cos(pi*clip((e-e_start)/20,0,1)))` and total loss `L=(1-0.5w)L_mse+wL_geo` | Aligns trigger with observed MSE plateau window (30-50), limits false positives, and preserves at least ~140 epochs of geometry training even in worst case | Keep fixed 50-epoch linear warmup; use absolute-delta trigger only; immediate full geometry after trigger | 2026-03-18 |
| Given latest stable 200-epoch run (large MSE gain, but angle/rho still high), prioritize geometry-loss reweighting experiments before implementing adaptive warmup; test `lambda_rho` increase in small staged steps and tune `lambda_theta` next | Fixed 50-epoch warmup is no longer the main bottleneck after critical fixes; direct geometry signal strength is the most likely limiter for `rho_error_px`/`angle_error_deg`, while adaptive warmup adds complexity with uncertain incremental gain | Implement adaptive warmup immediately; broaden scheduler changes first | 2026-03-18 |
| From post-fix metrics (`test_mse=0.0017`, `angle=34.3°`, `rho=15.8px`), choose staged geometry reweighting as highest-ROI next step: increase `lambda_rho` gradually (e.g., 0.05->0.1->0.2), then decide on adaptive warmup only if MSE plateaus while angle/rho stall | Current gap is geometry accuracy, not heatmap fit; direct rho gradient strengthening is lower-cost and more attributable than introducing a new warmup controller, while avoiding instability risk from a 10x single jump | Immediate 10x jump to 0.5; implement warmup adaptation first; do both at once | 2026-03-18 |
| Do not disable `ReduceLROnPlateau` during warmup in current code path because scheduler is stepped with `val_loss_mse` only | Warmup changes train-time total loss mix, but scheduler monitor remains `val_loss_mse`, so disabling risks missing useful LR reductions without fixing an observed issue | Disable scheduler until warmup ends based on older assumption of total-loss monitoring | 2026-03-18 |
| Mandate explicit CLI invocation in subagent prompts via a `CRITICAL: You MUST call [Tool] CLI` prefix | Ensures subagents leverage specialized Codex/Gemini CLI tools instead of returning self-analysis | Implicit delegation instructions; allow subagents to decide whether to call CLI | 2026-03-17 |
| For moment-based line extraction, treat eigenvector-formula mistakes as low-likelihood; prioritize coordinate-convention checks (`x=col`, `y=row`), anisotropy/confidence gating, and background-mass suppression before changing core eigensolver math | For symmetric covariance `[[mu20, mu11], [mu11, mu02]]`, `(mu11, lambda-mu20)` and `(lambda-mu02, mu11)` are equivalent eigenvectors; observed 40-50° errors are more consistent with axis mixups or low-SNR/near-isotropic moments than with a pure eigensystem bug (which often yields 90°/180° patterns) | Rewrite eigenvector formula only; force minor-axis/major-axis swap without diagnostics | 2026-03-18 |
| For `2x2` symmetric eigensolver checks, accept only eigenvector forms derived from `(a-λ)x + by = 0` or `bx + (d-λ)y = 0`: `(b, λ-a)` and `(λ-d, b)`; reject component-swapped variants | Confirms `line_losses.py` current form `(mu11, lambda1-mu20)` is mathematically valid and equivalent to `(lambda1-mu02, mu11)` up to scale/sign, so large angle errors should be debugged in convention/anisotropy/noise handling instead | Replace with `(lambda-a, b)` or `(b, lambda-d)` due to apparent mismatch | 2026-03-18 |
| Standardize robust phi extraction as `theta = 0.5*atan2(2*mu11, mu20-mu02)`, `phi = wrap_pi(theta + pi/2)` with explicit degeneracy guard on anisotropy (`lambda1-lambda2`) | Closed-form `atan2` avoids component-order ambiguity in manual eigenvector construction and improves debuggability; anisotropy guard prevents unstable angles when covariance is near-circular | Keep manual eigenvector path as sole estimator without degeneracy flag; hard-threshold only | 2026-03-18 |
| **Y-axis coordinate system in moment-based line extraction must use standard math convention (Y increases upward)** - Fix: negate y_grid in line_losses.py:96 | Image convention (Y down) causes 40-50° angle errors in moment calculations; flipping Y-axis to math standard eliminates systematic angle bias and achieves <1° precision | Accept image convention and compensate in normal-vector computation; apply sign correction post-extraction | 2026-03-18 |
| Root cause of 40-50° angle errors was Y-axis inversion at coordinate grid creation (line_losses.py:96), NOT eigenvector formula | Y-grid defined as `range(H)-H/2` maps row 0→-H/2 (top, Y increases downward); math requires Y increase upward; fix: `y_grid = -(range(H)-H/2)` eliminates non-linear angle distortion in moments | Modify eigenvector calculation; add post-hoc angle correction; redesign normal-vector logic | 2026-03-18 |
| When resolving GT-vs-pred convention mismatch in `line_losses.py`, keep prediction extraction in math coordinates (Y-up) and apply Y-flip to GT center coordinates (`p*_c[1] = -(p*[1]-center)`) | Preserves validated moment-extraction fix on prediction side and restores `(phi, rho)` consistency between GT and prediction without reverting known angle-accuracy improvements | Remove Y negation in prediction grid (revert to image coordinates) | 2026-03-18 |
| Validate moment-based fixes with synthetic line tests (0°, 45°, 90°, 135°) achieving <0.5° error before full re-training | Unit tests with known-angle synthetic heatmaps confirm coordinate-system fix eliminates systematic errors and validates eigenvector math correctness | Skip unit tests and validate only on real data after full training | 2026-03-18 |
| Keep `(phi, rho)` in math coordinates (`y` upward) end-to-end and convert GT polylines from image coordinates at extraction time (flip centered `y` in `extract_gt_line_params`), rather than removing prediction-side `y_grid` negation | Predicted moment extraction, synthetic-angle tests, and phi/rho visualization already assume Cartesian convention (`row = -y + H/2` for rendering). Converting GT once preserves annotation format while removing pred/GT frame mismatch that yields ~90° diagonal errors | Remove `y_grid` negation and keep all phi/rho math in image coordinates | 2026-03-18 |
| Adopt a boundary-adapter coordinate pattern: keep raster operations (annotation pixels, heatmaps, overlays) in image coordinates (Y-down), while defining geometric line parameters `(phi, rho)` in math coordinates (Y-up) with explicit one-way converters at GT extraction and visualization boundaries | Aligns with CV data conventions for pixel-domain tasks and with Hesse-normal stability for geometry; minimizes refactors by localizing conversions to interfaces and preventing hidden frame drift across train/eval/vis | Force pure math coordinates everywhere; force pure image coordinates everywhere; ad-hoc mixed conversions without explicit adapters | 2026-03-18 |
| Make `Unet/line_only/line_detection.py` the single source of truth for `detect_line_moments` and evaluation logic; keep `Unet/line_detection_moments.py` only as a short-lived compatibility shim before removal | The legacy file still uses image-space moments and older centroid/angle metrics, which can reintroduce coordinate drift and duplicate maintenance. A shim-first migration preserves external callers while preventing future divergence | Directly copy patches between two files indefinitely; immediate hard delete without compatibility window | 2026-03-19 |
| Do not "copy only `detect_line_moments`" into legacy `Unet/line_detection_moments.py`; if migration is needed, route callers to `Unet/line_only/line_detection.py` via a thin shim and deprecate | Legacy evaluation compares predicted angle against polyline PCA in image coordinates (`centroid_dist/angle_diff`), while canonical implementation evaluates `(phi, rho)` in math coordinates (`angle/rho/perpendicular`). Partial copy creates mixed-frame metrics and silent regressions | Keep legacy evaluator and patch only moment grid; maintain two divergent evaluators long-term | 2026-03-19 |
| In `line_only` metrics, convert GT points to math coordinates before perpendicular-distance evaluation (`x'=x-center`, `y'=-(y-center)`), not image-centered coordinates (`y'=y-center`) | Predicted `(phi, rho)` already lives in math coordinates; using image-space `y` in distance checks inflates errors (especially diagonals) and masks true geometry progress | Keep current mixed-frame distance calculation; drop perpendicular metric entirely | 2026-03-19 |
| Prefer a two-phase unification rollout: compatibility shim now (`line_detection_moments.py` re-export/deprecation warning), hard removal only after zero-usage scan and parity tests | Reduces immediate break risk for ad-hoc scripts while still enforcing a single canonical implementation (`line_only`) and preventing further coordinate drift | Immediate delete with potential hidden import breaks; continued dual-maintenance | 2026-03-19 |

## TODO

<!-- Features to implement -->

- [ ] 
 - [ ] Flatten package layout (move package to repo root, update hatchling/pytest/mypy paths)
 - [ ] Add provider wrapper that detects rate-limit errors and switches to Gemini CLI
 - [ ] Define wrapper CLI contract (args, env, exit codes) for Claude/Gemini switching
 - [ ] Define canonical conversation state persistence format and storage location
 - [ ] Implement context export command for Gemini (system + history + last tool results)
 - [ ] Decide on user confirmation flow for fallback
 - [ ] Unify coordinate conventions (x=col, y=row) across GT heatmap generation, angle extraction, and evaluation
 - [ ] Deprecate and remove legacy `Unet/line_detection_moments.py` after migrating any external callers to `Unet.line_only.line_detection`
 - [ ] Fix `line_only/line_metrics.py::compute_perpendicular_distance` to flip centered GT `y` into math coordinates before distance computation
 - [ ] Ensure major-axis selection in moment-based angle extraction (eigenvalues or variance check)
 - [ ] Add analytic-covariance unit tests for moment extraction (known angles, axis-swap sentinel, near-isotropic degeneracy handling)
 - [ ] Add evaluation metrics beyond angle error (e.g., perpendicular distance, Hausdorff, heatmap overlap)
- [ ] Implement minimal eval reporting set (angle_error_deg, rho_error_px, perpendicular_dist_px + one peak metric)
- [ ] Refactor train/eval pipeline: split train_one_fold/evaluate, extract line-eval/visualization utilities, and standardize metric aggregation/logging
- [ ] Refactor train_heat2.py: separate dataset validation/split, dataloader factory, model+optim+scheduler setup, epoch train step, epoch eval step, checkpoint/early-stop, and reporting
- [ ] Extract evaluation utilities: metric aggregation (overall + per-vertebra), heatmap visualization helpers, and line-detection evaluation wrapper
- [ ] Normalize non-English code comments to English; move Japanese explanations to docs

## Open Questions

<!-- Unresolved issues, things to investigate -->

- [ ] 
 - [ ] Exact Claude rate-limit error signatures to detect (exit codes/stderr text/HTTP status)
 - [ ] Gemini CLI invocation contract (flags, input format, output parsing)
 - [ ] Best source for Claude Code conversation logs (hooks vs local logs directory)
 - [ ] How much bias is introduced by GT polyline point spacing vs heatmap-based moment estimates?
- [ ] Do image-space augmentations (rot/flip/resize) apply consistently to GT angle evaluation?
- [ ] What bias does UNet smoothing/blur introduce in moment-based angle estimates, especially for short or curved lines?

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-31 | Added packaging and fallback strategy decisions; noted TODOs and open questions |
| 2026-01-31 | Added wrapper-based fallback decisions and context persistence follow-ups |
| 2026-02-18 | Recorded decision to align angle evaluation methods and added TODOs/open questions for line-angle analysis |
| 2026-03-04 | Planned refactor of training/evaluation pipeline and comment language normalization |
| 2026-03-04 | Added concrete refactor TODOs for train_heat2.py decomposition and eval utilities |
| 2026-03-10 | Added plan pattern for heatmap-moment line geometry loss (phi, rho) with sign alignment and warmup |
| 2026-03-16 | Recorded minimal line-eval metric recommendation (angle/rho core + perpendicular distance + one heatmap diagnostic); added sigma selection guidance for heatmap regression |
| 2026-03-16 | Added staged training decision for heatmap regression with angle-then-rho geometry constraints |
| 2026-03-17 | Added decision to detach prediction-derived confidence in line losses to avoid trivial collapse |
| 2026-03-17 | Added NaN-safe line-loss decision: avoid NaN sentinels inside arithmetic, use valid-only indexing, and add finite-guard training checks |
| 2026-03-17 | Recorded epoch-48 collapse diagnosis: NaN-mask arithmetic failure likely dominates, with warmup near 1.0 amplifying impact |
| 2026-03-17 | Added concrete implementation decision: use `isfinite` + valid-index slicing before angle/rho loss arithmetic to avoid NaN propagation and empty reductions |
| 2026-03-18 | Recorded implementation sequence for loss fixes: apply NaN-handling first, then warmup + smoothing adjustments (two-commit rollout) |
| 2026-03-18 | Refined loss-fix sequencing to prefer `#3 -> #2 -> #1` with per-step validation, while keeping `#3` isolation mandatory even in fast rollout |
| 2026-03-17 | Added delegation rule decision to require explicit `CRITICAL: You MUST call [Tool] CLI` prefix in subagent prompts (.claude/rules/codex-delegation.md, .claude/rules/gemini-delegation.md) |
| 2026-03-18 | Established agent workflow protocol (.claude/rules/agent-workflow.md): mandatory 6-phase process for Codex/Gemini consultation (info gathering → subagent invocation → wait for completion → read results → create implementation plan → report) |
| 2026-03-18 | Created detailed Phase 1 implementation plan (.claude/docs/loss-design-fix-plan.md): 3 critical fixes (warmup formula, double smoothing removal, NaN handling) with impact/risk assessment, test strategy, and rollback plan |
| 2026-03-18 | Verified implementation order via Codex (.claude/docs/codex/20260318-0327-implementation-order.md): confirmed #3 (NaN) → #2 (smoothing) → #1 (warmup) sequence with strong dependency rationale |
| 2026-03-18 | Added warmup strategy decision: recommend robust adaptive trigger for geometry-loss ramp (EMA + relative-improvement patience + hysteresis) instead of fixed 50-epoch warmup for 200-epoch runs |
| 2026-03-18 | Refined adaptive warmup spec with dual-gate plateau detector (EMA + rel-improvement + patience + min-epoch + latch) and post-trigger cosine ramp plus latest-start fallback for stability |
| 2026-03-18 | Added concrete adaptive warmup defaults for 200-epoch training: EMA=0.9, rel-improve threshold 0.2%, patience=5, min-epoch=20, forced latest-start=60, and 20-epoch cosine geometry ramp with `L=(1-0.5w)L_mse+wL_geo` |
| 2026-03-18 | Updated next-step priority from latest experiment outcome: defer adaptive warmup, first run geometry-loss reweighting (`lambda_rho` staged increase, then `lambda_theta`) and keep scheduler-on-warmup because monitor is `val_loss_mse` |
| 2026-03-18 | Recorded post-fix ROI decision: prefer staged `lambda_rho` increase first (not 10x jump), evaluate adaptive warmup only if geometry metrics stall after reweighting |
| 2026-03-18 | Added moment-extraction diagnostics decision: eigenvector formula is likely not the primary bug; prioritize coordinate checks, anisotropy guards, background suppression, and closed-form `atan2` phi extraction with dedicated covariance unit tests |
| 2026-03-18 | Added explicit `2x2` eigenvector-form validation: `(b, λ-a)` and `(λ-d, b)` are valid; component-swapped variants are invalid for general symmetric matrices |
| 2026-03-18 | Added convention-resolution decision for `line_losses.py`: keep prediction Y-up grid and flip GT Y around center to align `(phi, rho)` coordinate systems |
| 2026-03-18 | Clarified coordinate-convention consistency: keep pred `y_grid` negation (math coordinates) and add GT-side centered-y flip in `extract_gt_line_params` to unify `(phi, rho)` across loss/eval/visualization |
| 2026-03-18 | Added canonical boundary-adapter pattern for coordinates: keep pixel-domain processing in image frame (Y-down), keep `(phi, rho)` in math frame (Y-up), and confine conversions to GT extraction/visualization interfaces |
| 2026-03-19 | Added migration decision for line detection unification: treat `Unet/line_only/line_detection.py` as canonical, keep `Unet/line_detection_moments.py` as temporary compatibility shim, then remove |
| 2026-03-19 | Added guardrail for migration: avoid partial copy of `detect_line_moments` into legacy evaluator; use shim-based call redirection to canonical `line_only` path to prevent mixed-coordinate metrics |
| 2026-03-19 | Added evaluation-frame consistency decision: `compute_perpendicular_distance` must convert GT points to math coordinates (flip centered `y`) before distance computation |
| 2026-03-19 | Added rollout strategy detail: perform unification in two phases (shim/deprecate -> remove after usage scan + parity tests) to minimize breakage while eliminating duplicate line-detection logic |
