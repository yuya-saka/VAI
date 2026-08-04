# line_surface_3d Training Design Audit

Date: 2026-08-03
Scope: `baseline-v1` training, evaluation and geometry code, as committed in 2805538.
Status: **All findings addressed on 2026-08-04.** See §10 for what was done.
The findings below describe the code *before* that rewrite and are kept as the record
of why it changed.

Every claim below was verified by running against the real data or the real code.
Reproduction commands are in §8.

---

## 1. Verdict

Four defects are unambiguous bugs and must be fixed regardless of what happens next.
Two are design choices that need a decision. One suspected problem turned out to be minor
and should not be prioritized.

| # | Finding | Severity | Kind |
|---|---|---|---|
| 1 | `fit_ribbon` ignores its own `valid` flag, attenuating tilt ~16× | Critical | Bug |
| 2 | `peak_dist` is degenerate for ridge-shaped targets | High | Wrong metric |
| 3 | Evaluation counts each annotated slice ~13× and never aggregates windows | High | Wrong protocol |
| 4 | `collect_rho_errors` sign-invariance can hide ~31 px failures | High (latent) | Wrong metric |
| 5 | Heatmap loss is MSE on sigmoid, not BCE-with-logits | Medium | Design choice |
| 6 | `blob_iou` ignores the configured adaptive threshold | Low | Bug |
| 7 | Checkpoint selection on angle alone | **Minor — do not prioritize** | Checked |

---

## 2. Finding 1 — `fit_ribbon` ignores `valid`, and it destroys the tilt

`utils/ribbon.py::fit_linear_values` regresses over **all** slab positions. It takes no
mask, and `fit_ribbon` computes `moments.valid` but only passes it through to the output
dataclass — it is never used in the fit.

Unannotated slices have an all-zero GT heatmap. `compute_heatmap_moments` then returns
`centroid = (0, 0)` (mass clamped to `min_mass`) and a degenerate doubled angle of
`(-1, 0)`. Those values enter the regression with full weight.

Verified on a synthetic surface with a known slope, 6 annotated slices out of 15:

| Quantity | Value |
|---|---:|
| True slope from the 6 annotated slices | −1.0000 px/slice |
| Slope returned by `fit_ribbon` | −0.0625 px/slice |
| **Attenuation** | **16×** |

The `valid` flags were correctly computed (`[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0]`) and correctly
ignored.

**Consequences.** `evaluation.py:186` builds `target_fit` from GT heatmaps this way, so
every `surface_fitted_*` and `surface_raw_*` number ever logged or saved — including in
`metrics/test_fold*.json` — was computed against a corrupted target and is unusable. The
`ribbon-n15` config (`loss.ribbon.enabled: true`) would train against that corrupted target
as well; `baseline-v1` has it disabled, so baseline-v1's *training* is unaffected, but its
reported surface metrics are not.

This is also the single most important defect for the plane work, because the attenuated
quantity is exactly the tilt.

## 3. Finding 2 — `peak_dist` does not measure line accuracy

The GT heatmap is a Gaussian ridge along the line, so its value is near-constant along the
line and `argmax` is decided by numerical noise. On a real annotated slice:

- 2 pixels attain the exact maximum, 6 px apart
- 23 pixels are within 0.001 of the maximum
- 111 pixels are within 0.05 of the maximum

So `peak_dist` measures where along the line the two argmaxes happened to land. The reported
`peak_dist_mean` of 18–22 px is expected behaviour for a correct model, not evidence of
failure. The metric was inherited from `line_only` and is not informative here.

It should be replaced by perpendicular distance from GT polyline points to the predicted
line, which is what `plane_feasibility.py` already uses.

## 4. Finding 3 — evaluation measures the wrong unit

With `train_stride: 1` and windows requiring ≥3 labeled slices, each annotated slice appears
in many overlapping windows, and `evaluate()` treats every occurrence as an independent
observation. On fold 0's test split (9 samples, 875 windows):

| Quantity | Value |
|---|---:|
| Unique annotated slices | 288 |
| Observations actually counted | 3,814 |
| **Multiplicity** | **13.2× (range 10–15)** |
| Per-vertebra contribution spread | 3× between the largest and smallest |

Two separate problems follow.

First, the weighting is an artifact of vertebral geometry: a slice near the middle of a tall
vertebra is counted more often than one near the edge. No clinical reason supports that.

Second, and more importantly, the model produces **15 different predictions for the same
slice** — one per slab position — and `evaluate()` averages the errors of all 15 instead of
aggregating the predictions and then measuring the error once. `inference.py` does aggregate
(`OnlineRibbonAggregate`), so **the reported metrics do not measure what the deployed
pipeline outputs.** The headline 4.963° / 3.116 px are per-window, pre-aggregation numbers.

This matters beyond bookkeeping: aggregation averages down independent noise, so the
deployed per-slice rho error is probably better than 3.116 px. The identifiability analysis
in `line-surface-plane-tilt-design.md` §3 used 3.116 px as σ_ρ and is therefore
**pessimistic by an unknown factor**. Measuring the aggregated error is a prerequisite for
trusting that SNR argument in either direction.

## 5. Finding 4 — the rho metric can report a gross failure as zero error

`utils/metrics.py::collect_rho_errors` uses `min(|ρ_p − ρ_g|, |ρ_p + ρ_g|)`.

Both `extract_gt_line_params` and `moments_to_phi_rho` already canonicalize the normal to
the upper half-plane (`n_y ≥ 0`), so ρ is already signed consistently. The sign-invariant
form is only defensible near the φ ≈ 0 / φ ≈ π wraparound, but it is applied
unconditionally.

Measured GT |ρ| over 120 vertebrae: median 15.7 px, p90 23.8 px, max 32.8 px. A predicted
line landing on the opposite side of the image centre is therefore reported as **≈0 error
instead of ≈31 px** at the median.

Whether this currently triggers has not been measured, so treat it as a latent hazard rather
than a demonstrated one. It becomes fatal for the plane work regardless, because there the
sign of ρ *is* the signal. The correct form aligns by angle first:
`s = sign(n_p · n_g)`, then compare `ρ_p` against `s·ρ_g`.

## 6. Finding 5 — MSE on sigmoid instead of BCE on logits

`trainer.py:112` applies `sigmoid` and `losses.py` takes MSE against the target heatmap.
For a positive ridge pixel with target 1:

| logit | p | MSE gradient | BCE gradient | ratio |
|---:|---:|---:|---:|---:|
| −8 | 0.00034 | 6.7e−04 | 1.00 | 1491× |
| −6 | 0.00247 | 4.9e−03 | 0.998 | 203× |
| −4 | 0.018 | 3.5e−02 | 0.982 | 28× |
| −2 | 0.119 | 1.9e−01 | 0.881 | 5× |

Early in training, when the model outputs near-zero everywhere, MSE gives ridge pixels a
gradient up to three orders of magnitude weaker than BCE-with-logits. The `p(1−p)` factor
suppresses exactly the pixels that need to move.

This is a design choice rather than a bug — the model did converge — but it is a plausible
reason the model plateaus at a rho error larger than the tilt signal, which is the binding
constraint for everything downstream. Note that switching to BCE-with-logits invalidates
`baseline-v1` as a comparison point and requires re-running all five folds.

## 7. Findings 6 and 7 — one small bug, one non-issue

**Finding 6.** `evaluation.py` passes the configured `heatmap_threshold` (adaptive,
min 0.10, peak_ratio 0.4) to `extract_pred_params_cc_batch` but calls `collect_blob_ious`
without it, so blob IoU silently uses the hardcoded default of 0.1. The reported
`blob_iou: 0.685` is therefore not computed at the configured operating point.

**Finding 7 — checked and dismissed.** `selection_metric: angle_error_deg` selects the
checkpoint on angle while `early_stopping_metric: val_loss_mse` controls stopping, which
looked like it might select a checkpoint with poor rho. Measured across all five folds, the
angle-selected checkpoint costs on average **+0.059 px of rho (2%)** relative to the
rho-optimal epoch. That is not worth changing on its own. The real cost is wasted compute:
folds 1 and 4 continued for 38 and 60 epochs after the selected checkpoint.

---

## 8. Proposed fix order

**Stage A — correctness, no retraining needed.** Fixes 1, 2, 3, 4 and 6 are all in the
evaluation and geometry code. Applying them and re-scoring the existing five checkpoints
gives, for the first time, trustworthy numbers for the model that already exists — and
produces the aggregated rho error that §4 shows is needed before the tilt SNR argument can
be trusted. No GPU training required.

**Stage B — decide on the loss.** Whether to switch to BCE-with-logits (Finding 5) is a
scope decision, because it invalidates `baseline-v1` and costs a full 5-fold re-run. It
should be decided after Stage A reveals the true aggregated accuracy, since that is what
determines whether the current accuracy is actually the blocker.

**Stage C — plane and tilt work**, per `line-surface-plane-tilt-design.md`. Note that its
§3 SNR figures must be recomputed with the Stage A numbers.

## 9. Reproduction

```bash
# Findings 2-6 (as measured before the rewrite)
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline python -m \
  Unet.line_surface_3d.analysis.training_audit

# Finding 1 is now a regression check on the replacement geometry
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline python -m \
  Unet.line_surface_3d.analysis.training_audit --check plane-fit
```

---

## 10. What was done (2026-08-04)

The pipeline was rewritten around a strict plane with exactly three parameters per
surface: in-slice angle, signed offset, and the z tilt. The model itself is unchanged —
still `TinyUNet` at 505,740 parameters, no extra heads.

| Finding | Resolution |
|---|---|
| 1 `fit_ribbon` ignores `valid` | `utils/ribbon.py` deleted. `utils/plane.py::fit_plane` weights each slice by a detached confidence, so empty slices carry zero weight. Verified: known slope recovered to 1.7e-5 relative error. |
| 2 `peak_dist` degenerate | Removed. Replaced by `plane_rho_error_px`, an actual offset error. |
| 3 evaluation counts windows | `evaluate()` now aggregates overlapping windows into one plane per vertebra×surface before measuring. Test `test_each_surface_is_counted_once` locks this in. |
| 4 sign-invariant rho | Removed. `aligned_rho_error_px` aligns by `sign(n_p·n_g)` first. |
| 5 sigmoid+MSE | **Not changed.** Deliberately deferred — it invalidates the baseline and costs a full 5-fold re-run. Decide after the plane arm reports. |
| 6 blob IoU threshold | Now taken from `evaluation.blob_iou_threshold`. |
| 7 checkpoint selection | Selection and early stopping both use `plane_rho_error_px`, ending the wasted epochs. |

A separate defect surfaced during the rewrite: `qc_scores.json` is a dict keyed by slice
index, but `load_manual_labels` iterated it as a list, so **7 slices marked `exclude` were
never excluded**. Fixed in `_load_qc_excluded`.

### Loss weighting is measured, not guessed

All three geometry terms are converted to a line-position error in pixels and share one
Huber scale, but the gradient that reaches the heatmap still differs per term by three
orders of magnitude. Measured after a heatmap-only warmup, averaged over 10 batches:

| Term | Gradient norm at weight 1.0 | Weight for 50% of the heatmap gradient |
|---|---:|---:|
| heatmap | 0.0255 | — |
| angle | 35.2 | 0.00012 |
| rho | 1.99 | 0.00214 |
| tilt | 0.157 | 0.0272 |

`config/plane.yaml` uses 0.0001 / 0.002 / 0.03. Re-measure with
`--check loss-balance` whenever a loss term changes; weights set by intuition here are
wrong by 1000×, and the first attempt (all weights 1.0) stalled the heatmap loss
completely.

### Status

Implemented and verified structurally: 31 tests pass, ruff and mypy clean on the changed
files, and an end-to-end run on real data reduces both the heatmap loss (0.251 → 0.027)
and the plane angle error (60.7° → 22.7°). **No 5-fold experiment has been run yet**, so
there are no accuracy claims. The `baseline.yaml` config is now the plane-off control arm,
differing from `plane.yaml` only in `loss.plane.enabled`.
