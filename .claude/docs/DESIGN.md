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
| 2026-03-16 | Recorded minimal line-eval metric recommendation (angle/rho core + perpendicular distance + one heatmap diagnostic) |
