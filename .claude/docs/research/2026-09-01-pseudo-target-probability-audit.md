# CAM Pseudo-Target Probability Audit

Date: 2026-09-01  
Status: **Identity-CAM audit complete; four-view TTA probability audit pending GPU availability**

## Question

Define four soft binary regional targets for every fracture-positive cell without
a human regional label, while preserving the fold-matched CAM location signal,
avoiding regional-prevalence shortcuts, and retaining Baseline 0
hard-label behavior.

## Provisional Construction

For each fold-matched teacher and TTA view `v`, let `e_vr` be the non-negative
CAM density enrichment for region `r`. Normalize **inside each view**, then
average the four evaluated views:

```text
s_vr = e_vr / sum_j(e_vj)
s_r  = mean_v(s_vr),  v in {identity, hflip, rot(+10 deg), rot(-10 deg)}
x_r  = logit(clip(s_r, 0.01, 0.99))
q*_r = sigmoid(a_k * x_r + b_k)
q_r  = q*_r                         if sum_j(q*_j) >= 1
       q*_r / sum_j(q*_j)           otherwise
```

`a_k,b_k` are fitted separately for student outer fold `k`, using its
fold-matched teacher scores and only whole-positive bags whose four human region
targets are all valid. The fit is one L2-regularized logistic model (`C=1`)
shared by all four regions. It receives no region identifier, region-specific
intercept, class prevalence, estimated cardinality, teacher bag probability, or
raw CAM total. The fitted slope must be positive; otherwise generation fails.

The `sum(q)>=1` projection is a logical guard for a known whole-positive bag. It
preserves regional ordering. It did not activate for any current identity-CAM
OOF or pseudo-pool prediction, but prevents an incoherent artifact after TTA or
future checkpoint changes.

Human labels always take precedence cell by cell. The 1,064 fully unannotated
whole-positive bags contribute all four CAM targets. The 33 partially annotated
bags contribute CAM targets only for their 89 unknown cells; their 43 known
cells stay hard human targets. Thus 4,345 unique positive cells are
pseudo-eligible. No confidence filter removes a case.

This is provisional because the stored four-view CAM values were removed after
the earlier discrimination audit, and the current host has no working CUDA
driver or GPU Slurm resource. The same calibration audit must be rerun on the
four-view `s_r` values before coefficients or pseudo-label artifacts are frozen.

## Calibration Population

Only complete labels are used to learn a four-output probability scale. Using
valid cells from partially annotated bags caused selection bias and inflated
the inferred mean cardinality, so that alternative is rejected.

| Student outer fold | Complete bags | Studies | Mean positive cardinality |
|---:|---:|---:|---:|
| 0 | 144 | 93 | 1.410 |
| 1 | 145 | 91 | 1.372 |
| 2 | 138 | 86 | 1.377 |
| 3 | 140 | 86 | 1.379 |
| 4 | 141 | 88 | 1.411 |

Each candidate was evaluated by five-fold `GroupKFold` on `study_id` inside
each student outer fold. Metrics below are means over the five student outer
folds.

## Candidate Results on Stored Identity CAM

| Candidate | AP | Macro AP | AUROC | Brier | Log loss | ECE | SD of sum(q) | rho(sum(q), K) | Own-region direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean(K) * share` | **0.6489** | 0.6441 | 0.7615 | 0.1819 | 0.5485 | 0.0487 | 0.038 | -0.174 | 19/20 |
| Predicted-cardinality share | 0.6392 | 0.6550 | 0.7695 | 0.1808 | 0.5643 | 0.0548 | 0.179 | 0.233 | 18/20 |
| Isotonic shared share | 0.6130 | 0.6043 | 0.7459 | 0.1842 | 0.5842 | 0.0462 | 0.115 | -0.074 | 19/20 |
| Raw-share logistic | 0.6441 | 0.6410 | 0.7582 | 0.1862 | 0.5545 | 0.0643 | 0.044 | -0.246 | 19/20 |
| **Shared logit-share logistic** | 0.6442 | 0.6421 | 0.7600 | **0.1809** | **0.5390** | **0.0354** | 0.066 | -0.094 | 18/20 |
| Shared Beta calibration | 0.6430 | 0.6405 | 0.7596 | 0.1811 | 0.5398 | 0.0364 | 0.064 | -0.089 | 18/20 |
| Logit-share + bag probability + CAM total | 0.6454 | **0.6555** | **0.7705** | **0.1778** | **0.5315** | 0.0374 | 0.185 | 0.217 | 18/20 |

The shared logit-share map is selected as the initial probability map:

- it preserves the within-bag CAM ordering because every region uses the same
  positive-slope monotone map;
- it improves log loss by 0.0095 versus `mean(K) * share`; a patient bootstrap
  gave a mean 95% interval `[-0.0197, -0.0019]` and `P(delta<0)=0.996`;
- its Brier improvement is small and uncertain (`-0.0010`, 95% interval
  `[-0.0023, +0.0004]`), but ECE falls from 0.0487 to 0.0354;
- its lower stitched-OOF AP is not a within-model ranking loss: each fitted map
  is monotone, while concatenating predictions from five separately fitted
  inner calibrators changes their cross-split scale;
- raw-share logistic and isotonic regression are worse, while the more flexible
  Beta map provides no measurable benefit.

The bag-probability/CAM-total model remains a diagnostic challenger, not the
initial contract. Its improvement comes from varying total positive mass with a
weak cardinality proxy (`rho=0.217`), below the already rejected cardinality
criterion; its extra evidence also lacks a frozen aggregation rule across TTA
views. It must not be introduced post hoc into the first outer-0 training arm.

The 1% share floor is a numerical guard, not a confidence threshold. One human
R3-positive cell had exactly zero R3 CAM share. With no floor, that single
`logit(0)` observation materially changed the fold-0 and fold-1 slopes. Floors
from 0.5% through 5% formed a broad metric plateau; 1% is the smallest round
value that removes the singular leverage while retaining low-share ordering
above the floor.

## Observed Probability Values

Fitting the provisional map on all complete identity-CAM labels in each student
outer fold produced positive slopes `a_k = 1.294, 1.284, 1.682, 1.890, 0.861`
and intercepts `b_k = 0.887, 0.825, 1.227, 1.411, 0.393`.

Across the fold-matched positive pool, including partially annotated bags before
the cell-wise human override:

| Generated target | Mean | p1 | p5 | Median | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 0.371 | 0.052 | 0.103 | 0.350 | 0.709 | 0.849 |
| R2 | 0.318 | 0.006 | 0.023 | 0.263 | 0.798 | 0.887 |
| R3 | 0.361 | 0.006 | 0.028 | 0.320 | 0.829 | 0.913 |
| R4 | 0.332 | 0.029 | 0.073 | 0.289 | 0.745 | 0.889 |
| `sum(q)` | 1.383 | 1.167 | 1.262 | 1.396 | 1.458 | 1.472 |

The OOF human cells had median `q=0.469` when hard target was one and
`q=0.254` when hard target was zero. This overlap is expected: CAM is an
imperfect auxiliary teacher, not a replacement annotation.

These values are **identity-CAM diagnostics**, not the final four-view target
artifact. Four-view TTA previously improved mean regional AUROC from 0.763 to
0.803, so its calibrated probability distribution must be observed rather than
assumed equal to this table.

## Pseudo-Loss Decision

Native `BCEWithLogitsLoss(pos_weight=2)` must not be passed directly a soft
target `q`. Its cell loss

```text
-[2q log(p) + (1-q) log(1-p)]
```

is minimized at `p=2q/(1+q)`, not `p=q`. On the observed pseudo pool, that would
raise the target by a median 0.146 and as much as 0.172. It is therefore not used
for pseudo targets.

The decided pseudo loss is only the ordinary soft BCE:

```text
L_P_cell = BCEWithLogits(z, q)
L_P = sum(valid * L_P_cell) / sum(valid)
```

There is no pseudo-loss coefficient, ramp, confidence weight, or
target-preserving outer weight. `pos_weight=2.0` remains on `L_E`, the hard human
and logical whole-negative term, and on the Baseline 0 whole path. The soft BCE
has gradient `sigmoid(z)-q`, so its optimum remains the calibrated target `q`.

## Whole-Corpus Imbalance

Across 13,432 bags and four regions there are 53,728 cells. Human hard targets
contain 367 positives and 616 negatives; the 12,100 whole-negative bags add
48,400 hard negatives; 4,345 cells receive a pseudo target.

If every `q>0` is counted as one positive, all pseudo cells count as positive
because a sigmoid-calibrated target is strictly positive. This gives 4,712
positives and 49,016 negatives: negative-to-positive ratio `10.40:1` and an
8.77% positive rate. This binary count is misleading for soft BCE because it
treats `q=0.006` and `q=0.9` identically.

Using the actual identity-CAM probability mass instead, mean pseudo `q` is
0.3442. The whole-corpus positive mass is 1,862.4 and negative mass is 51,865.6,
giving `27.85:1` and a 3.47% positive-mass rate. This is the relevant descriptive
imbalance for plain soft BCE, but it still does not define a pseudo coefficient:
`L_E` and `L_P` are separate normalized terms and no `mu` is used.

## Acceptance Before Training

1. Regenerate four-view CAM shares for all complete human bags and all
   pseudo-eligible positive bags with the fold-matched teacher.
2. Refit the shared calibration map using complete labels only and repeat the
   patient-grouped OOF table.
3. Fail generation if a slope is non-positive, any `q` is non-finite/outside
   `[0,1]`, or a whole-positive vector remains below `sum(q)=1` after projection.
4. Record per-fold coefficients, reliability bins, `q` quantiles, `sum(q)`, and
   projection rate in the generation metadata.
5. Use all pseudo-eligible cells in outer fold 0, then compare `no_pseudo`,
   `cam_soft`, and case-shuffled `cam_soft_shuffled`. No remaining outer fold
   runs until `cam_soft` improves human validation macro AP over both controls.

## Primary References

- Niculescu-Mizil and Caruana, [Predicting Good Probabilities with Supervised Learning](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf), ICML 2005. Platt and isotonic calibration require probability-specific evaluation; discrimination alone is insufficient.
- Kull, Silva Filho, and Flach, [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html), AISTATS 2017. Beta calibration was included because it is a principled richer parametric alternative; it did not improve this dataset.
- PyTorch, [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html). The documented unreduced formula gives the native soft-target optimum shift derived above.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  .claude/docs/research/scripts/2026-09-01-audit_pseudo_target_probabilities.py
```
