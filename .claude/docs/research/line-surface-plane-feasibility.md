# Line Surface Plane Feasibility

Date: 2026-08-03

## Scope

This analysis tests whether the existing manual 2D boundary lines can define one
3D plane per anatomical boundary while preserving the signed direction of its
z-axis tilt. It does **not** test whether a newly trained model can predict those
planes.

The evaluated plane has a constant in-slice normal and a signed line-position
trend along z. Comparators are a plane without z tilt and the existing ribbon
formulation that permits line-angle change along z.

## Data and Protocol

- Manual annotations: `data/dataset/sample*/C*/lines.json`
- QC exclusions: `bad_slices_all.json` and per-vertebra `qc_scores.json`
- Minimum observations: 5 slices per surface
- Spacing: isotropic 0.4 mm
- Evaluated data: 175 vertebrae, 700 surfaces, 4,868 line observations
- Validation:
  - Full-observation plane residual
  - Leave-one-slice-out prediction
  - Fit central slices and predict both edge slices
  - Fit one half of the annotations and extrapolate to the other half
  - Tilt-sign consistency under leave-one-out, odd/even, and independent-half splits

## Results

### Single-plane representability

| Metric | Median | P90 | P95 |
|---|---:|---:|---:|
| In-slice angle RMS residual | 1.908 deg | 4.121 deg | 4.637 deg |
| Polyline-to-plane RMS distance | 0.994 px | 1.790 px | 2.015 px |
| Absolute z tilt | 10.597 deg | 27.334 deg | 31.733 deg |
| Absolute movement over the manual z span | 1.296 px | 3.546 px | 4.429 px |

- 96.9% of surfaces have angle RMS residual at most 5 degrees.
- 94.7% have point RMS residual at most 2 px.
- C1 is the most difficult level: median point residual 1.40 px and P90 2.12 px.

The existing manual lines can therefore be converted to one plane in the large
majority of cases. The remaining disagreement is concentrated in a small tail
and should be handled as annotation/QC inconsistency, not as curved-surface
capacity.

### Signed z-tilt direction

Across all 700 surfaces:

- All leave-one-out fits retain the full-fit slope sign for 77.7%.
- Odd/even slice fits agree on the sign for 86.0%.
- A Student-t 95% slope interval excludes zero for 42.3%.
- Independent lower/upper-half fits agree on the sign for 50.0%.

The all-surface rates are reduced by nearly untilted surfaces. Stratifying by
the total signed movement over the annotated z span gives:

| Absolute movement | Surfaces | All LOO signs agree | Odd/even sign agrees |
|---|---:|---:|---:|
| At least 1 px | 422 (60.3%) | 93.6% | 94.8% |
| At least 2 px | 215 (30.7%) | 96.7% | 96.7% |
| At least 4 px | 49 (7.0%) | 100.0% | 100.0% |

Thus, the signed direction is stable when the plane has a measurable z shift.
For shifts below 1 px over the complete annotation span, the sign is dominated
by annotation variation and should be treated as near-zero/uncertain rather
than forced into a positive or negative class.

### Extrapolation

When fitting the central observations and predicting the two annotated edges:

| Representation | Median angle error | Median point error |
|---|---:|---:|
| No z tilt | 1.932 deg | 1.303 px |
| Plane with signed z tilt | 1.932 deg | 1.181 px |
| Twisted ribbon | 1.730 deg | 1.118 px |

The plane improves point error in 56.5% of edge predictions. The ribbon lowers
the median slightly but has worse tails because it estimates unnecessary angle
change.

The stricter half-to-half extrapolation is not reliable if every surface is
forced to have a nonzero slope: overall median point error is 1.429 px for the
plane versus 1.382 px without tilt. For the subset whose full-fit slope interval
excludes zero, the plane improves to 1.185 px versus 1.419 px and is better in
66.4% of comparisons. The benefit also increases with actual movement.

### Required annotation count

The annotations were subsampled by count and the resulting slope sign was
compared with the plane fitted from every available manual slice. The following
results are restricted to surfaces moving at least 1 px over the full manual
annotation span.

When annotations are distributed across the complete available z span:

| Annotation slices | Slope-sign agreement | Held-out point error median |
|---:|---:|---:|
| 2 | 97.6% | 1.375 px |
| 3 | 98.6% | 1.167 px |
| 4 | 99.1% | 1.125 px |
| 5 | 100.0% | 1.041 px |

When the same number of annotations are consecutive and cover only a narrow
part of the available z span:

| Consecutive annotation slices | Slope-sign agreement | Held-out point error median |
|---:|---:|---:|
| 2 | 70.0% | 2.012 px |
| 3 | 77.5% | 1.717 px |
| 4 | 85.3% | 1.564 px |
| 5 | 90.0% | 1.419 px |
| 6 | 90.5% | 1.327 px |

Two separated slices are mathematically sufficient to define a slope, but they
provide no redundancy for detecting one bad annotation. Three slices are the
practical mathematical minimum, and five are recommended for robust fitting.
Here, "distributed" means distributed across the **central band in which the
boundary is visually identifiable**, not across the full vertebral height.
Uncertain superior/inferior boundaries should not be annotated merely to widen
the z span. If only a short central band is visible, annotating five consecutive
central slices is acceptable, with the measured limitation that tilt-sign
agreement was about 90% for surfaces moving at least 1 px over the available
manual span.

## Conclusion

1. **Geometric target construction is feasible.** One plane represents the
   existing annotations within approximately 1 px median residual.
2. **Signed z tilt is recoverable when it is materially nonzero.** The target
   should remain a continuous signed quantity, with near-zero slopes treated as
   uncertain rather than assigned an arbitrary direction.
3. **A short partial annotation range is insufficient for every case.** Robust
   plane construction should use the complete available manual band.
4. **Direct learned prediction remains unverified.** A held-out learned-model
   experiment is still required before concluding that image features can
   predict the plane and its signed z tilt.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline python -m \
  Unet.line_surface_3d.analysis.plane_feasibility \
  --annotation-root data/dataset \
  --output-dir Unet/outputs/line_surface_3d/plane_feasibility \
  --min-slices 5
```

Generated artifacts:

- `Unet/outputs/line_surface_3d/plane_feasibility/summary.json`
- `Unet/outputs/line_surface_3d/plane_feasibility/surfaces.csv`
- `Unet/outputs/line_surface_3d/plane_feasibility/held_out_predictions.csv`
