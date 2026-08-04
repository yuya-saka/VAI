# Plane Tilt: Learning Constraint and Evaluation Design

Date: 2026-08-03
Status: **Proposal. Not approved, not implemented.**

Companion documents:

- `line-surface-plane-feasibility.md` — whether a plane GT can be built at all (answered: yes)
- `.claude/docs/codex/20260803-2211-plane-tilt-loss-eval.md` — full Codex analysis with formulas
- `Unet/line_surface_3d/analysis/tilt_identifiability.py` — reproduces every number below

This document answers a different question from the feasibility study: given that a plane
GT can be constructed, **can the tilt be learned and can it be measured honestly?**

---

## 1. Summary

Four measurements change the plan that §6–§8 of the 2026-08-03 work-log sketched.

1. The current ribbon parameterization mixes a nuisance quantity into the tilt signal at
   roughly twice the amplitude of the signal itself. This must be fixed before any tilt
   loss is meaningful.
2. Post-hoc plane fitting on frozen `baseline-v1` predictions is below the noise floor.
   The tilt must be learned, not fitted afterwards.
3. The anatomical prior is weak (about 60% sign accuracy). This is good news: the tilt is
   genuinely patient-specific, and the control baseline for evaluation is clean.
4. `transverse_foramen_crossing_rate` as written in the work-log is **not computable** from
   the existing masks. A GT-backed substitute exists.

---

## 2. Measurement 1: the parameterization contaminates the signal

`utils/ribbon.py::fit_ribbon` regresses `centroid_x` and `centroid_y` independently
against z, and additionally regresses the doubled angle against z.

Two problems follow. A strict plane requires the in-slice angle to be **constant** along z,
so the angle slope is a free parameter the plane formulation does not have. More seriously,
only the component of centroid motion **along the surface normal** is boundary movement.
Motion along the line direction is an artifact of how far the annotator drew the polyline
(and, for predictions, of where the heatmap happens to have mass).

Decomposing the per-slice polyline centroid into along-line and perpendicular components,
over the full annotated band, across 708 surfaces:

| Component | Median displacement over the band |
|---|---:|
| Perpendicular (the real signal) | 1.285 px |
| Along-line (pure nuisance) | 2.978 px |

The nuisance is **2.32× the signal**, and exceeds it on **71.8%** of surfaces.

Fitting `(x, y)` jointly therefore lets the nuisance dominate the fitted slope. The fix is
to fit only `ρ = n·c` against z, using one shared normal `n` per surface, and to discard the
along-line component entirely. Codex reached the same conclusion independently and adds the
correct mechanism: never canonicalize the normal per slice (a sign flip destroys `k`);
take one confidence-weighted doubled-angle mean for the whole surface, canonicalize once,
then derive every `ρ_i` from that single shared normal.

---

## 3. Measurement 2: post-hoc fitting is below the noise floor

For a linear fit of `ρ` against z, the slope standard error is `σ_ρ / sqrt(S_zz)`.

| Quantity | Value |
|---|---:|
| Tilt signal, median \|k\| | 0.187 px/slice |
| GT annotation SE(k), median | 0.084 px/slice |
| GT SNR | 2.23 |
| Model SE(k) over the annotated band, at σ_ρ = 3.116 px | 0.680 px/slice |
| Model SE(k) over one 15-slice window | 0.199 px/slice |
| **Implied model SNR over the annotated band** | **0.30** |
| **Implied model SNR over one 15-slice window** | **0.94** |

Two consequences.

First, fitting a plane to model predictions **within the annotated band** is hopeless: the
band is only 7 slices wide at the median, so `S_zz` is small. Any procedure that estimates
tilt only where annotations exist will return noise.

Second, even over a full 15-slice window the SNR is about 1. To reach SNR 2 by post-hoc
fitting alone, `baseline-v1` would need its per-slice rho error to improve from
**3.12 px to 1.46 px** — a 53% reduction that no part of the current plan delivers.

Both numbers assume per-slice errors are **independent** along z, which is optimistic:
a 15-slice slab shares one set of convolutional features, so residuals are likely
correlated and will not average down as `1/sqrt(N)`. The figures above are therefore an
upper bound on what post-hoc fitting can achieve.

**Conclusion: the tilt has to be an explicit learning target, not a post-processing step.**
This matches Codex's independent verdict that post-hoc fitting on frozen predictions is a
likely dead end.

### Unmeasured, and worth measuring cheaply

The along-z correlation of `baseline-v1`'s rho residuals has not been measured. The five
fold checkpoints exist (`Unet/outputs/line_surface_3d/baseline-v1/checkpoints/`) but the
`predictions/` directories are empty. Running inference with the existing checkpoints —
no training — would settle how much the correlated-noise caveat bites, and would give the
concrete "post-hoc fit" control arm that the evaluation needs anyway.

---

## 4. Measurement 3: the anatomical prior is weak, which is good

Signed tilt direction is only meaningful against a control. If tilt direction were a
per-level anatomical constant, a model could score well by memorizing it.

Leave-one-**sample**-out, restricted to the 422 surfaces moving at least 1 px:

| Predictor | Sign accuracy |
|---|---:|
| Per (level, line) mean tilt | 59.2% |
| Per line mean tilt | 60.2% |
| Global majority class | 60.2% |

The per-(level, line) prior is **no better than the global majority class**. Codex assumed
level+line would be the strongest baseline; the data says the strongest prior is simply
**60.2%**.

This is the most encouraging result in the analysis. Tilt direction is patient-specific,
not stereotyped, so there is real information to learn — and any model that beats ~60%
out-of-sample is demonstrably using image evidence rather than a memorized prior.

---

## 5. Measurement 4: the reliability rule works, and the study is powered

Applying Codex's proposed per-surface reliability rule (N≥5, span≥4 slices,
point RMS ≤2.0 px, angle RMS ≤5.0°, movement ≥1 px, LOO sign agreement ≥0.8, plus either
movement ≥2 px or a significant t-statistic with odd/even sign agreement):

| | Count |
|---|---:|
| Surfaces with ≥5 annotated slices | 700 |
| Passing basic QC | 385 (55.0%) |
| **Reliable signed tilt** | **269 (38.4%)** |

Spread is adequate: 60–75 surfaces per line, 21–53 per level, and every one of the 40
samples contributes. Codex's Go-condition of ≥100 reliable surfaces with ≥20 per line is
satisfied with margin.

Power, by simulation — 269 surfaces in 40 sample-level clusters, paired sample-clustered
bootstrap, testing whether the 95% CI lower bound on the improvement over a 60.2% prior
excludes zero:

| True model sign accuracy | Power (ICC 0.05 / 0.15 / 0.30) |
|---:|---|
| 70% | 0.66 / 0.66 / 0.63 |
| **75%** | **0.95 / 0.95 / 0.94** |
| 80% | 1.00 / 1.00 / 1.00 |

Codex's proposed threshold of ≥75% sign accuracy is well powered and robust to
within-patient correlation. A 70% effect would be missed a third of the time, so 75% is the
right bar to state in advance — it is where the study can actually deliver a verdict.

---

## 6. Measurement 5: the foramen metric as written is not computable

The work-log lists `transverse_foramen_crossing_rate` as a metric and §8-12 leaves its
definition open. Codex proposed extracting enclosed background components inside the
vertebra mask and selecting two lateral foramen candidates.

That approach fails on the existing data. The masks in `data/dataset_zprop/*/masks/` are
**binary** (0/255) whole-vertebra masks with no foramen label. Surveying 1,200 randomly
sampled annotated slices for enclosed background holes, after excluding the largest hole
(the spinal canal):

| Detected lateral holes ≥30 px | Slices |
|---:|---:|
| 0 | 679 (57%) |
| 1 | 264 (22%) |
| **2 (usable)** | **118 (10%)** |
| no hole at all | 137 (11%) |

Only about 10% of slices would support the metric, and the rate varies from 26% (C1) to 1%
(C6). The transverse foramina are mostly not preserved as closed holes. **This metric
cannot be built from the current masks**, and Codex's own caveat — that it cannot be
promoted to a clinical endpoint without dedicated foramen labels — should be treated as
decisive rather than conditional.

### The substitute that does work

`data/dataset/*/C*/gt_masks/` contains **4-region label maps (values 1–4)** on annotated
slices, and `utils/region_eval.py` already names the regions
`("body", "right_foramen", "left_foramen", "posterior")`.

So the clinical failure mode — a boundary cutting through a transverse foramen — can be
measured against real GT as **per-region agreement between the plane-induced partition and
`gt_masks` on annotated slices**, reporting `right_foramen` and `left_foramen` IoU and a
foramen truncation rate. This uses existing labels, needs no new annotation, and stays
inside the central band where the expert says the boundary is actually visible.

Outside the annotated band there is no region GT, so any out-of-band claim remains a
comparison against the constructed plane GT and must be labelled as such.

---

## 7. Recommended plan

Adopting Codex's design with the four corrections above.

**Architecture.** Fine-tune `baseline-v1` per fold. Replace `fit_ribbon` with a strict-plane
projection: shared normal from a confidence-weighted doubled-angle mean, canonicalized once;
per-slice `ρ_i = n·μ_i` from that shared normal; ridge-regularized weighted least squares of
`ρ` against normalized z, shrinking to `k = 0` when effective slice count is low. Add a
small zero-initialized head on the pooled bottleneck predicting a correction `Δv` to the
fitted tilt vector `v = k·n`. Confidence weights are stop-gradient, so the model cannot
flatten heatmaps to escape the loss.

**Loss.** `L_sparse + λ_p·L_projected + λ_t·L_tilt + λ_e·L_extrap`, with λ = 0.005 / 0.005 /
0.002 and staged cosine ramps. Delete the existing ribbon residual term — strict projection
already imposes the constraint. Tilt targets are weighted by the annotation-derived
uncertainty (detached, not model-predicted). Vertical-fallback surfaces are pulled toward
zero only outside a dead zone of `1 px / band span`, since `v = 0` is an operational
default, not an observation. Virtual extrapolation at ±4 mm, with 2/4/6 mm as sensitivity.

**Guard.** Only checkpoints passing central non-inferiority (angle ≤5.463°, rho ≤3.416 px,
IoU ≥0.665) are eligible; among those, select on ±4 mm virtual line-position error over
reliable surfaces.

**Evaluation.** Pool OOF predictions, bootstrap over sample clusters (never a t-test on 5
fold means), and report every tilt metric against the 60.2% prior and against the post-hoc
fit control. Report reliable-GT and vertical-fallback subsets separately.

**Go / No-Go.** Codex's seven criteria, with the sign-accuracy bar at ≥75% and ≥10 pp over
the measured 60.2% prior — now confirmed as adequately powered. If the hybrid cannot beat
the prior, the honest conclusion is that subject-specific signed tilt is not predictable
from this data and this input, **not** that a 3D model would solve it.

---

## 8. Open decisions for the user

1. Run the zero-training diagnostic first (inference with the five existing checkpoints, to
   measure residual correlation along z and establish the post-hoc control arm)? It is cheap
   and would either confirm or undercut the SNR argument in §3 before any training starts.
2. Accept replacing `transverse_foramen_crossing_rate` with region IoU against `gt_masks`,
   or commission dedicated foramen labels?
3. Include the 2–4 annotation vertebrae (§8-6 of the work-log) in `L_sparse` only? They
   cannot support a reliable plane fit but do carry central line evidence.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline python -m \
  Unet.line_surface_3d.analysis.tilt_identifiability
```

Outputs `Unet/outputs/line_surface_3d/tilt_identifiability/summary.json`.
The foramen-hole survey and the power simulation in §5–§6 are ad-hoc and not yet scripted.
