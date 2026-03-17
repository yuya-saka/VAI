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
| Mandate explicit CLI invocation in subagent prompts via a `CRITICAL: You MUST call [Tool] CLI` prefix | Ensures subagents leverage specialized Codex/Gemini CLI tools instead of returning self-analysis | Implicit delegation instructions; allow subagents to decide whether to call CLI | 2026-03-17 |

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
 - [ ] Ensure major-axis selection in moment-based angle extraction (eigenvalues or variance check)
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
