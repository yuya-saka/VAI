# Review of negative-inclusive CAM evaluation

Date: 2026-09-07

Status: Read-only artifact analysis completed. No training or model changes.
The observations below do not adopt a replacement architecture or selection rule.

Reviewed source:
`2026-09-07-negative-inclusive-cam-evaluation-and-pseudo-label-reliability.md`.

## Reproduced measurements

Recomputed AP using `baseline0/resources/input_manifest.csv`, all five
`09_04/baseline0_aug追加/outer*/outer_predictions.csv` files, the held-out
teacher rows of `09_04/pseudo_labels/pseudo_region_targets.csv`, and
`09_04/pseudo_labels/cam_audit_negative/fold*.csv`. Joins were one-to-one on
study and level, covering all 13,432 bags before region-specific GT filtering.

| Region | Whole score | Whole x CAM share | Whole x calibrated CAM q |
|---|---:|---:|---:|
| R1 | 0.393675 | 0.182736 | 0.206375 |
| R2 | 0.289347 | 0.283756 | 0.274511 |
| R3 | 0.170344 | 0.273847 | 0.281542 |
| R4 | 0.460404 | 0.422268 | 0.426674 |
| Macro AP | 0.328442 | 0.290652 | 0.297276 |

The third column of scores uses the calibrated predictions in the negative CAM
audit, not the known-zero targets in the negative-inclusive training artifact.
Its values are descriptive; no new significance claim is made for this comparison.
The original whole-versus-share AP measurements reproduce. Calibration does not
reverse their macro ordering, but raw share and the actual training target q are
different quantities. Neither teacher score directly evaluates a trained student.

This evaluation combines annotated whole-positive bags with all whole-negative
bags, excluding unannotated positives. Its AP is not automatically representative
of the original corpus or deployment prevalence.

## Corrections to causal interpretations

For R1, the whole-versus-share AP gap is 0.210939. Setting whole-negative scores
to zero changes it to 0.050038, reproducing source section 2.3. About 76% of the
gap disappears under this intervention. AP is nonlinear, so this is not an
additive causal attribution, but it contradicts dismissing negative ranking as
unimportant. Errors within positive bags also remain. A low mean negative score
does not establish that its upper tail is harmless.

The 7x7 feature map is a plausible contributor to localization errors, not a
demonstrated unique cause. Likewise, increasing baseline validation AP does not
establish that the student's increase is solely a training-duration artifact.
Attribution requires a matched continued-training control from the same teacher
checkpoint, with the same duration and selection protocol.

## Conditional and marginal objectives

The current region objective fits P(region | whole-positive, image). Excluding
whole-negative bags from that conditional loss is consistent with its definition;
those bags still supervise the whole path and shared encoder. Their exclusion is
not, by itself, discarded supervision or a bug. Adding negative region targets
changes the estimand to a marginal region probability. Increasing the number of
negative gradients does not replace missing positive localization supervision.

Since a positive region implies a positive vertebra, the probability factorization
P(region | image) = P(whole | image) P(region | whole, image) is appropriate.
Raw normalized CAM share is not guaranteed to equal its conditional factor,
particularly when multiple regions can be positive.

## Leakage and teacher-fold accounting

`baseline0/data/splits.py:25` and both the 08_19 and 09_04 pseudo-generation
metadata confirm a 60% training / 20% inner / 20% outer split. Teacher 0 trains on
folds 2, 3, 4 and uses fold 1 for checkpoint selection. Thus the source statement
that each teacher trains on all but its own outer fold is incorrect.

The actual problematic path nevertheless exists: student 0's validation targets
come from teacher 1, whose training/calibration folds include student test fold 0.
Training targets use teacher 0 and exclude that outer test. Selection using the
contaminated validation targets compromises the selected region model's outer
evaluation even though the outer evaluator itself attaches no pseudo targets.
The previously reported v2-to-v3 conditional AP increase, 0.690928 to 0.850140,
is therefore an observed saved-model difference, not clean confirmation of
generalization improvement.

Using teacher 0 plus its train-only calibrator to score student 0's inner fold 1
is a candidate that excludes outer fold 0 without training 20 new teachers.
It requires newly generated inner scores and revised routing. Fold 1 already
selected the teacher checkpoint, so it is not wholly untouched upstream;
the outer test would still be excluded. This candidate is not implemented.
Requiring every training pseudo target to come from a teacher that never saw
that bag is a separate cross-fitting requirement, not a prerequisite for excluding
the student's outer test from the pipeline.

Predeclared fixed training duration and region checkpoint selection could also
remove this selection path in a future run. Disabling log fields alone cannot.
The current region loss also stops the shared training loop, affecting the set
of epochs available to whole-checkpoint selection. Its whole validation labels
are clean, but that alone does not prove the whole procedure is independent of
region validation. This does not establish numerical bias in the existing
epoch-5 best-whole checkpoint.

## Confidence filtering

Reproduced the source's argmax correctness convention and applied paired
patient-cluster bootstrap with 10,000 draws, seed 20260907, and fixed observed
confidence thresholds. Both methods were resampled with the same study weights.

| Subset | Bags | CAM correctness | Always R4 | Paired difference 95% CI |
|---|---:|---:|---:|---|
| All | 268 | 0.694030 | 0.589552 | [0.010909, 0.198473] |
| Upper half | 134 | 0.776119 | 0.529851 | [0.119652, 0.374047] |
| Upper quarter | 67 | 0.835821 | 0.567164 | [0.088235, 0.452055] |

Upper-half minus all-bag correctness has CI [0.030337, 0.136855] under this
convention. This supports an exploratory association, not a training benefit.

Missing-label caveat: 12/268 argmax regions have unknown human labels, represented
as zero in the raw label columns. There are 3 such cases in the upper half and
none in the upper quarter. Always-R4 has 17 unknown labels overall. Therefore
these are not fully observed correctness rates. Restricting to complete GT bags
gives 0.703390 (236 bags), 0.776860 (121), and 0.819672 (61), respectively, with
the same fixed thresholds. The descriptive trend remains.

An argmax hit validates one selected positive region, not all four soft BCE
targets or the completeness of multi-region localization. A Mann-Whitney p=0.82
does not prove equivalence or transportability of accuracy. Correlation 0.14
with whole score does not exclude confounding by anatomy, label completeness,
fracture multiplicity, or other difficulty factors. Thresholds chosen on these
GT patients cannot subsequently be presented as independent holdout decisions.

## Literature and next interpretation

[Eyuboglu et al. (2021)](https://www.nature.com/articles/s41467-021-22018-1)
does use an all-exam regional cross-entropy objective (equation 6), report-derived
weak labels, and a >=10% prevalence criterion for its 26 joint tasks before
single-task fine-tuning. These are choices in that experiment, not proof that
conditional modeling is invalid or that negative-inclusive training must work
here. Its binary whole-abnormality comparison concerns weak versus full labels,
not adding a region loss to an otherwise matched whole classifier.

Current evidence supports a conditional localization signal and a confidence
association, while whole improvement remains unestablished. Prioritize correcting
outer-test access in model selection before attributing student improvements to
CAM. Confidence filtering remains an exploratory arm; the negative-inclusive
objective, fixed-epoch selection, and additional teachers remain unapproved.
