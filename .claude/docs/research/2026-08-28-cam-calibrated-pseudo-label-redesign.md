# CAM-Calibrated Pseudo-Label Redesign

Date: 2026-08-28

## Decision

Keep CAM as the pseudo-label source, but do not restore the failed inter-case
CAM-density ranking loss. Convert each bag's regional CAM evidence into four
independent soft pseudo labels calibrated against scarce human regional labels.

This does not revive the rejected exact-only region teacher. The teacher remains
the independently trained Baseline 0 whole-fracture classifier; human regional
GT is used only to calibrate its CAM score-to-probability relationship.

## Why the Earlier Formulation Failed

The outer-0 pilot optimized two targets with different semantics:

- human BCE asked for an absolute binary probability per bag and region;
- pairwise rank loss asked only that two bags preserve the teacher CAM ordering.

By epoch 24, training human loss had fallen to 0.101 while validation human loss
had risen to 1.030, and student-teacher Spearman agreement continued to improve.
Thus the model increasingly satisfied the CAM ranking without improving the
held-out human endpoint. In addition, the effective loss-value share of the
fixed rank term increased as the exact loss shrank. The scarce human pool was
also cycled many times per epoch to match larger streams.

## Pseudo-Label Construction

For student outer fold `k`, use only `Teacher_k` CAM scores and human regional
labels from the student's three training folds.

For each region `r`, fit a regularized monotone calibration model:

```text
x_ir = log(max(CAM_density_ir, epsilon))
q_ir = sigmoid(a_kr * x_ir + b_kr),  with a_kr >= 0
```

Use patient-grouped cross-validation inside the three training folds to report
AUROC, AP, Brier score, calibration slope/intercept, and bootstrap uncertainty.
After the calibration family and regularization are fixed, refit it on all
eligible train-fold human cells and apply it to region-unlabeled,
whole-positive bags in the same folds.

The output artifact must contain the raw CAM density, calibrated `q_ir`, teacher
ID/checkpoint hash, calibration fold IDs, calibrator coefficients, and a validity
flag. Inner- and outer-fold regional labels must never fit the calibrator.

Rules:

- create four independent sigmoid targets; regions are not mutually exclusive;
- do not use softmax, argmax, or normalize targets to sum to one;
- do not compare CAM values between different cases during student training;
- human regional labels always override CAM targets;
- use exact all-zero targets, not CAM, for whole-negative bags;
- do not force the highest-CAM region to one when all four scores are uncertain.

## Outer-0 Feasibility Probe

A patient-grouped three-fold probe used the 159 human bags available to
`Student_0` and the corresponding `Teacher_0` CAM scores.

| Region | Valid / positive | CAM AUROC | AP | OOF Brier |
|---|---:|---:|---:|---:|
| R1 | 149 / 48 | 0.799 | 0.679 | 0.168 |
| R2 | 149 / 36 | 0.831 | 0.574 | 0.135 |
| R3 | 147 / 44 | 0.789 | 0.622 | 0.187 |
| R4 | 151 / 95 | 0.795 | 0.873 | 0.191 |

This supports fitting a simple one-feature calibrator. It does not support hard
high-confidence labels as the primary target: at `q >= 0.8`, the cross-validated
sample contained only two R1 positives, no R2/R3 examples, and fourteen R4
examples. Therefore the primary pseudo label is the calibrated soft probability,
not a thresholded 0/1 target.

After refitting on all outer-0 train human cells, the 643 whole-positive unknown
bags had mean soft targets R1 0.372, R2 0.284, R3 0.310, and R4 0.499; mean
expected cardinality was 1.465. These are diagnostics, not priors to be forced.

## Training Loss and Sampling

```text
L_region = L_GT + beta_N * L_N + mu(t) * L_CAM
```

- `L_GT`: masked BCE on observed human cells, reduced within each region and
  then averaged across active regions.
- `L_N`: exact all-zero BCE on sampled whole-negative bags, with a supporting
  coefficient rather than corpus-count weighting.
- `L_CAM`: soft BCE between the student probability and `q_ir`, reduced within
  each region before regions are averaged. It is present at a small weight from
  the first region epoch, ramps gradually, and remains auxiliary to `L_GT` by
  monitored gradient ratio.

Do not add a logical-OR loss in the initial experiment. Record
`P_any = 1 - product_r(1 - p_ir)` on whole-positive bags as a non-gradient
diagnostic, including its distribution and the fraction of near-zero cases. If
the CAM-soft arm collapses on this diagnostic, reject the arm first; evaluate
OR only later as an explicit independent ablation.

The CAM target is attached to the unaugmented bag identity while the student
may receive the existing spatial augmentation. The four-class anatomical mask
values retain region identity during horizontal flipping, so R2/R3 scalar
targets are not exchanged.

Define a region epoch by one shuffled pass through the human pool. Pair each
human batch with CAM-labeled whole-positive and sampled whole-negative batches;
rotate the larger pools across epochs instead of cycling human bags to exhaust
them in one epoch. Do not choose `mu` from corpus size. Log the epoch-wise
`||grad L_CAM|| / ||grad L_GT||` ratio and decay or stop the CAM ramp if the
pseudo term becomes dominant.

## Required Ablation and Gate

Run only outer fold 0 until the design is accepted:

1. `no_cam`: the same sampling and objective with `mu = 0`;
2. `cam_soft`: calibrated soft CAM targets;
3. `cam_shuffled`: shuffle `q_ir` within region and train fold as a negative
   control.

Use the same initialization, optimizer, human exposure, and checkpoint rule.
The CAM arm passes only if inner-human macro AP improves over `no_cam`, human
loss does not show the previous sustained divergence, and `cam_shuffled` does
not reproduce the gain. `P_any` on whole-positive bags must remain non-collapsed,
but it is not optimized directly. Teacher agreement alone is not a success
metric.

## Evidence Basis

- PseudoSeg showed that calibrated soft CAM-derived targets can outperform raw
  CAM or hard filtering while a pseudo-majority corpus is batch-balanced with
  scarce exact labels.
- The IPMI fracture study used a whole-image fracture teacher, many more pseudo
  maps than expert boxes, and adaptive soft-label sharpening; excessive
  sharpening degraded performance.
- CAP and distribution-aware multi-label FixMatch support class-specific
  confidence handling rather than one threshold or an assume-negative rule for
  every missing cell.

These studies support calibrated, controlled per-case pseudo supervision. None
supports the retired inter-case regional CAM-density ranking objective.
