# Codex Analysis: GT Validation Investigation Plan
Date: 2026-03-31

## Question
We have a medical image dataset for cervical vertebra CT line annotation quality investigation. Dataset: 40+ patient samples, 7 cervical vertebrae C1-C7 each, multiple axial CT slices 224x224 pixels, GT has 4 polylines per slice in lines.json format each with 2-6 xy points. GT is processed into Gaussian heatmaps via distance transform with sigma=4 and phi rho Hesse normal form via PCA on polyline points. Provide a comprehensive investigation plan: 1) quality metrics per annotation 2) statistical analyses to reveal systematic problems 3) visualization strategies 4) thresholds to flag suspicious annotations 5) domain-specific medical imaging checks 6) how to present results. Be specific about formulas and thresholds.

## Codex Response

Use this as a concrete QC protocol for your cervical CT line GT.

**0) Setup and notation**
- Annotation unit: one polyline (`1 of 4 lines`) on one `224x224` slice for one vertebra (`C1-C7`) in one patient.
- Polyline P={p_i=(x_i,y_i)}_{i=1}^n, n in [2,6], length L=sum_{i=1}^{n-1}||p_{i+1}-p_i||_2.
- Fit line by PCA on all polyline points:
  - centroid mu, covariance C, eigenvalues lambda_1 >= lambda_2, principal direction u_1.
  - Hesse normal n=(-u_{1y},u_{1x}), rho=n^T * mu, phi=atan2(n_y,n_x), canonicalize so rho >= 0.

## 1) Quality metrics per annotation

### Integrity checks
- Point count valid: 2 <= n <= 6 (hard fail otherwise).
- All points in bounds: 0 <= x,y <= 223 (hard fail otherwise).
- Duplicate/near-duplicate points: any segment < 1 px (hard fail), 1-2 px (warn).

### Geometric line-likeness
- Orthogonal residual: d_i = |x_i*cos(phi) + y_i*sin(phi) - rho|, RMSE_perp = sqrt(1/n * sum(d_i^2)).
- Linearity index: LI = lambda_1 / (lambda_1 + lambda_2).
- Curvature/jaggedness: mean absolute turn angle kappa_bar = 1/(n-2) * sum|theta_{i+1} - theta_i|.
- Self-intersection check (hard fail if true).

### Heatmap consistency (sigma=4)
- Rebuild expected GT heatmap: H_exp(x) = exp(-d(x,P)^2 / (2*sigma^2)).
- Stored-vs-expected mismatch: NRMSE_H = ||H_stored - H_exp||_2 / ||H_exp||_2.
- Peak-to-line distance: d_peak = min_{x in argmax H} d(x,P).
- Area consistency at tau=0.5: A_0.5 = |{x: H(x) >= 0.5}|, expected A_0.5_exp ≈ 2*sigma*sqrt(2*ln2)*L = 9.42*L, ratio R_0.5 = A_0.5 / (9.42*L).
- Mass ratio: R_M = sum_x H(x) / (sqrt(2*pi)*sigma*L) (ideal around 1, boundary effects allowed).

### Inter-slice continuity (same line, same vertebra)
- Delta_phi_t = min(|phi_t - phi_{t-1}|, pi - |phi_t - phi_{t-1}|) in degrees.
- Delta_rho_t = |rho_t - rho_{t-1}|.
- Angular jerk J_{phi,t} = |phi_t - 2*phi_{t-1} + phi_{t-2}|.

## 2) Statistical analyses for systematic problems

### Robust outlier scoring per metric
- z* = 0.6745*(x - median) / MAD, computed within each (vertebra, line_id) stratum.

### Hierarchical modeling
- Mixed model for each metric y:
  y = beta_0 + beta_vertebra + beta_line + beta_slice_position + u_patient + epsilon, u_patient ~ N(0, sigma_u^2).
- Identifies vertebra-level or line-ID-specific bias while accounting for repeated slices/patients.

### Systematic drift tests
- Trend across cranial->caudal slice index using GAM/LOESS; flag monotonic drifts.

### Group comparisons
- Kruskal-Wallis (or ANOVA if normal) across C1-C7 and line IDs.
- Multiple testing control: Benjamini-Hochberg (FDR q < 0.05).

### Uncertainty
- 1,000x bootstrap CIs for flag rates and median errors (important with ~40 patients).

## 3) Visualization strategies
- Slice overlay panel: CT + 4 polylines + heatmap contours + per-line QC badges.
- "Trajectory plots" per vertebra: phi and rho vs slice index with outliers highlighted.
- Distribution plots by C-level and line ID: violin/box for RMSE_perp, LI, Delta_phi, R_0.5.
- Heatmap residual maps: H_stored - H_exp to detect GT conversion bugs.
- Outlier gallery: top 50 worst annotations, sorted by composite score, with reason codes.
- Patient-level heatmap matrix: rows=patients, cols=(C1..C7 x line1..4), color=flag rate.

## 4) Thresholds to flag suspicious annotations

Use dual criteria: absolute thresholds + robust z*.

### Hard fail
- Invalid point count/bounds, self-intersection, line crossing that violates topology.

### Warn / Fail cutoffs (initial)
| Metric       | Warn           | Fail           |
|--------------|----------------|----------------|
| LI           | < 0.95         | < 0.90         |
| RMSE_perp    | > 2.5 px       | > 4.0 px       |
| kappa_bar    | > 10 deg       | > 20 deg       |
| NRMSE_H      | > 0.02         | > 0.05         |
| d_peak       | > 1.5 px       | > 3.0 px       |
| R_0.5        | outside [0.7, 1.3] | outside [0.5, 1.6] |
| Delta_phi    | > 8 deg        | > 12 deg       |
| Delta_rho    | > 6 px         | > 10 px        |
| z* (any metric) | > 2.5       | > 3.5          |

### Composite QC score
- S = 0.35*S_geom + 0.25*S_cont + 0.20*S_heat + 0.20*S_domain, each subscore in [0,1].
- Suggested triage: Pass < 0.30, Review 0.30-0.50, Suspicious >= 0.50 or any hard fail.

## 5) Domain-specific medical imaging checks
- Enforce vertebral order consistency (C1->C7 along z; hard fail if swapped labels).
- Define expected topology for the 4 lines per slice (non-crossing, consistent pairing). Violations: fail.
- Adjacent-line angular relation (e.g., near-parallel paired cortical/endplate lines): |Delta_phi| > 10 deg warn, > 15 deg fail.
- CT edge support score:
  ESS = mean|grad I| on line band / mean|grad I| in local annulus.
  ESS < 1.1 warn, < 1.0 fail.
- HU plausibility (if raw HU available): fraction of nearby pixels > 250 HU too low suggests off-bone placement (< 0.30 warn, < 0.20 fail).
- Relax continuity thresholds at vertebra start/end slices (first/last 10%) for partial-volume effects.

## 6) How to present results

### Executive page
- Total annotations, flagged %, fail %, by C-level and line ID.
- Top 3 systematic issues with effect size and CI.

### Technical appendix
- Metric definitions/formulas, thresholds, robust-z method, model specs.

### Actionable tables
- Per-annotation CSV: identifiers + all metrics + reasons + final status.
- Priority queue: "relabel now" cases ranked by S.

### Validation
- Manually review ~200 annotations stratified by score to tune thresholds.
- Report precision/recall of suspicious flagging after adjudication.
