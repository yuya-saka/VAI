# weak/test_v1: learning-result analysis

Date: 2026-09-11. Run: `fracture_detection/weak/outputs/09_10/test_v1/outer0`.
Saved CSVs and current implementation were inspected; no training was run.

## Evaluation populations and corrected results

`fold_metrics.json.best_metrics` contains **inner validation** metrics, not outer
test metrics. The trainer assigns `best_metrics = validation_metrics` and saves
that dictionary after outer inference. The earlier training work log mixed these
inner values with Baseline 0 outer values. Recomputed outer metrics below join
both prediction CSVs one-to-one by `(study_id, level)` and verify identical labels.

| Metric | Inner at selected pass 35 | Outer at selected pass 35 |
|---|---:|---:|
| Whole population / positives | 2,687 / 268 | 2,671 / 262 |
| Annotated-positive population / studies | 53 / 30 | 56 / 32 |
| Region macro AP, annotated positives only | 0.777960 | 0.776598 |
| Region BCE, mean over annotated cells | 0.444171 | 0.461642 |
| Whole AP | 0.703241 | 0.728427 |
| Whole AUROC | 0.910381 | 0.908413 |
| Baseline 0 whole AP, AUROC-selected checkpoint | 0.733349 | 0.772599 |
| Baseline 0 whole AUROC, same checkpoint | 0.919651 | 0.918605 |

The whole outer AP difference is **-0.044172**, not -0.069358.
Outer results are one fold, not five-fold OOF or a repeated-seed comparison.
Current manifest splits have no study overlap across train, inner, and outer.

## 1. Why a small amount of GT supports validation improvement

- Training contains 159 annotated-positive bags from 98 studies: 636 observed
  region cells, including 223 positive cells. All four cells receive direct BCE;
  zero is a known negative. These cells are correlated, not 636 independent cases.
- Sampling presents all 159 annotated bags every GT-pass, with four annotated bags
  in each full batch. At pass 35, each has been presented 35 times with augmentation.
  Negative bags also provide four reliable negative region targets. Learning is
  therefore not based solely on a handful of positive labels.
- With group means, the objective is `0.25 L_N + 0.25 L_A + 0.5 L_U`.
  N/A losses sum four cells, whereas U has one OR term. At pass 35, training
  contributions are N=0.115387, A=0.354868, U=0.095471: A accounts for 62.7% of
  the scalar objective despite occupying 25% of the batch. These are loss shares,
  not parameter-gradient shares; component gradient norms were not recorded.
- The encoder already learned fracture features in Baseline 0. FPN mask pooling
  provides an anatomical readout, and the region LSTM/head share weights across
  all four regions. This reduces the burden on scarce region GT. The encoder is
  fine-tuned at one tenth of the new-module learning rate, with frozen BN running
  statistics; new FPN normalization uses GroupNorm. These are plausible stabilizers,
  not independently established causes from this run.
- There are no CAM pseudo targets competing with GT, and no positive-only second
  encoder forward updating BN buffers. Earlier region_branch BN interventions
  established that such buffer updates could harm whole validation; that result
  does not quantify their contribution to this run's region improvement.

Inner GT BCE changes from 0.561868 at pass 1 to 0.459764 at pass 5, reaches its
minimum 0.421202 at pass 21, and is 0.444171 at selected pass 35. Region macro AP
increases from 0.631243 to 0.777960. Most BCE improvement is early, followed by a
plateau/fluctuations, not continuous improvement through all 45 passes.

A constant prediction equal to each region's training-GT prevalence has inner
BCE 0.589413 and macro AP 0.339623; on outer these are 0.589863 and 0.321429.
Observed outer BCE 0.461642 and AP 0.776598 support useful held-out region ranking,
rather than merely lowering loss by predicting a region's prevalence. This still
does not identify whether U supervision contributed to that improvement.

## 2. Why whole classification does not surpass Baseline 0

The baseline comparison changes several things simultaneously. Only `encoder.*`
is transferred; Baseline 0's trained whole LSTM/head is discarded. The new
FPN/region LSTM/head starts randomly, and whole scores are restricted to fixed
noisy-OR aggregation. Region validation AP selects the checkpoint. Whole inner
AP improves from 0.649576 at pass 1 to 0.703241 at pass 35, so whole learning is
occurring, but its maximum across recorded passes is 0.716699 at pass 42, below
the Baseline 0 inner AP 0.733349. Selection mismatch alone cannot explain the gap.

The 4/4/8 batch has 75% positive bags versus 9.9% in the training population.
This is consistent with inflated whole scores: outer-negative mean is 0.291312
versus Baseline 0's 0.067829, and its 90th percentile is 0.703248 versus 0.173828.
However, **a strictly increasing score transformation preserves ranking and AP**.
Thus probability inflation/prior shift alone does not establish why AP fell;
training may also have changed positive-negative ordering. Indeed precision
among the highest-scoring 262 bags is 0.656489 versus 0.706107 (172 versus 185
positives). Threshold adjustment alone cannot repair that ranking difference.

There is also a strong exposure imbalance in the newly initialized readout:
by pass 35, A has 35 passes, U has 17.418 passes, and N only 0.770 passes
(5,600 of 7,272 negative bags presented). Even at stopping pass 45, N has only
0.990 passes. Baseline's transferred encoder previously saw all training
negatives, so this is a statement about the new training stage/readout, not a
claim that the model has never encountered those negatives in any form.

Noisy-OR accumulates moderate regional false positives: four q=0.2 values give
p_whole=0.5904. On outer negatives, 464/2,409 have p_whole>=0.5; 197 of those
have every q<0.5. This is a score diagnostic, not an inner-selected operating
threshold. An exploratory reaggregation of the same outer q as max(q) gives
AP 0.750713, versus noisy-OR's 0.728427. This supports aggregation affecting
ranking, but it is a post-hoc outer diagnostic, not a validated replacement:
any aggregation choice must be tested/selected on inner data in a new experiment.

## 3. Why the incremental benefit from weak labels may be small

For U, with q_r=sigmoid(z_r) and p=1-product_r(1-q_r):

```text
L_U = -log(p)
dL_U/dz_r = -q_r * (1-p)/p
```

The expression matches CPU autograd from the implemented loss to maximum
absolute error 1.9e-8 on three illustrative logit vectors. It has two relevant
properties:

1. U supplies no information about which region is positive, nor which other
   regions must be negative. Every logit is pushed upward; the larger q values
   receive larger logit gradients within that bag. An incorrect region or several
   moderate scores can satisfy the weak label without learning correct localization.
   Region pooling does not strictly exclude outside-region information because
   upstream CNN receptive fields cross region boundaries.
2. Once p is near one, gradients to every region become small. For q_r=0.5 in
   all four regions, p=0.9375 and L_U=0.064539, despite no discrimination.
   The logit gradient magnitude is 0.033333 per region; a known positive region's
   direct BCE gradient at q=0.5 is 0.5, fifteen times larger. This is illustrative,
   not a measurement of initialization or full parameter-gradient ratios.

Outer U has median p=0.974947, and 59.7% have p>0.95. For this fraction, the
shared `(1-p)/p` factor is below 0.0527, consistent with many easy positives
providing little further signal. These are held-out predictions; actual
component parameter gradients during training were not measured.

Training U loss improves from 0.281408 at pass 1 to 0.190942 at pass 35, whereas
inner U loss changes from 0.262599 to 0.296947 and ends at 0.381963. This is
consistent with limited improvement on unseen U, not evidence of a disconnected
or always-zero loss. U-positive loss alone omits negative discrimination and
cannot establish overall whole performance. Outer U versus N has AP 0.697152
and AUROC 0.903491; Baseline pretraining and A/N supervision can also explain
this separation without any incremental benefit from U.

Gross all-region activation collapse is not observed: inner all-q>=0.8 rate
is 0.0744% at pass 35, and the first standardized-probability PC share falls
from 0.8288 at pass 1 to 0.5584. The PCA implementation uses q, not raw logits.
These whole-population diagnostics cannot rule out individual shortcut regions.
Only 325/28,755 presented training bags (1.13%) are excluded for missing observed
regions, across all groups; this is not evidence of wholesale U exclusion.

## Controlled follow-ups, not yet executed

1. Compare beta=0 and beta=1 with the same initialization, seed, 4/4/8 inputs,
   GT-pass budget, denominator, and selection rule. Keep U forward passes and
   denominator terms even at beta=0. This isolates the additional OR loss.
2. N/A/U=8/4/4 was selected for the protocol-v2 LSE run, with A=4 and batch
   size=16 unchanged. Because aggregation and sampling changed together, compare
   against matched single-change arms if causal attribution is required.
3. Measure N/A/U gradients separately on fixed training batches, and inspect
   localization and whole metrics on inner data. Loss magnitudes alone cannot
   determine gradient influence. Do not select aggregation or beta from outer.

No design change or new training run has been adopted from this analysis.
Reproduction: `UV_CACHE_DIR=/tmp/vai-uv-cache MPLCONFIGDIR=/tmp/vai-matplotlib uv run --no-sync python .claude/docs/experiments/2026-09-11-weak-test-v1/analyze.py`.
The script writes `metrics.json` and `learning_curves.png` in this directory.

## Follow-up proposal: improve whole detection while retaining localization

The user's explicit objective is to improve vertebra classification while
learning localization. The existing serial region-to-whole constraint remains
in force. The following are proposals, not adopted settings or authorization to
start new training.

The first pooling experiment should compare fixed noisy-OR with max pooling,
holding the model, N/A losses, sampler, beta, and training budget constant:

```text
whole_score = max_r sigmoid(z_r) = sigmoid(max_r z_r)
N: sum_r BCEWithLogits(z_r, 0)
A: sum_r BCEWithLogits(z_r, t_r)
U: softplus(-max_r z_r)
```

Both training's U term and inference aggregation must change consistently.
N retains all four negative targets; replacing its loss with only the negative
max-bag BCE would unnecessarily discard dense negative supervision. A retains
all four GT labels without an extra whole-positive loss. The N loss is no longer
algebraically identical to the whole-score negative BCE under max aggregation.

Max pooling removes the path where several moderate q values jointly produce
a high whole score. At four q=0.5, the U loss is 0.693147 rather than 0.064539.
It supplies a focused positive signal but can reinforce an incorrect current
maximum and, for a unique maximum, sends U gradients to only one region.
Multi-region A supervision remains essential. Max is a detection surrogate,
not an exact probability of the union of correlated regional events. Any
benefit over noisy-OR on these CT data is unproven.

In a separate sampling comparison, change N/A/U=4/4/8 to 8/4/4 while retaining
the same GT-pass budget. This preserves A's nominal 4/16 coefficient and
doubles negative presentations per pass. It also doubles N's coefficient and
halves U's coefficient, so it changes both exposure and objective weighting;
the effects cannot be described as exposure alone. Preserving A's coefficient
does not guarantee preserving its gradients or validation performance.

If these changes do not retain enough baseline discrimination, a later option
is whole-score knowledge distillation from the frozen, fold-matched Baseline 0
teacher to the serial student's whole score, alongside the true labels and
region GT. This can transfer the discarded whole readout's behavior without a
separate whole head at inference. Distillation must not generate regional
pseudo labels or override known GT. The teacher can be confidently wrong on
unseen examples, and its training scores can be overconfident; its coefficient
and temperature require inner validation. Distillation alone does not guarantee
surpassing the teacher.

Evaluate variants by both natural-inner whole AP and annotated-inner region
macro AP; a prespecified localization tolerance can constrain selection of the
best whole AP. Do not select from a weighted aggregate invented after seeing
outer results. The already inspected outer0 is exploratory for these proposals;
use untouched outer folds for confirmatory comparisons. Retain a matched beta=0
control when measuring the incremental benefit of weak labels.

Primary literature provides motivation rather than evidence of CT improvement:

- Wang et al., 2018, [Max versus noisy-OR](https://arxiv.org/abs/1804.01146):
  speech/sound sequence experiments and theoretical pooling comparison.
- Wang et al., 2018, [Five MIL pooling functions](https://arxiv.org/abs/1810.09050):
  max's limited gradient coverage and linear-softmax alternatives for sound
  tagging/localization. These longer sequences differ from four anatomical regions.
- Hinton et al., 2015, [Knowledge distillation](https://arxiv.org/abs/1503.02531):
  transfer of teacher predictions to a student; not a validation of this proposal.

## 2026-09-12: normalized logit-LSE decision

The user selected temperature-controlled normalized LSE of the four region
logits, followed by sigmoid, with tau=0.5. This is log-mean-exp, not plain
unnormalized log-sum-exp and not LSE applied to the sigmoid probabilities:

```text
whole_logit = tau * (logsumexp(region_logits / tau) - log(4))
whole_score = sigmoid(whole_logit)
U_loss = softplus(-whole_logit)
dU_loss/dz_r = (whole_score - 1) * softmax(region_logits / tau)_r
```

For tau>0, high logits receive higher weights, but all finite logits receive
a nonzero derivative in exact arithmetic. Very small tau or large logit gaps
can still concentrate virtually all gradient on one region. As tau tends to
zero the aggregation tends to max; as tau grows it tends to the mean logit,
not the mean region probability. Temperature does not identify the true region:
the U term still pushes every region upward and can reinforce an incorrect peak.
Full four-cell A supervision and all-negative N supervision remain necessary.

The normalization makes identical region logits map to the same whole logit.
Without it, the whole logit is shifted upward by tau*log(4); four q=0.5 values
would produce score 0.8 at tau=1, rather than 0.5. For fixed logits, four regions,
and fixed tau, this constant shift does not change AP/AUROC: normalization is a
score-scale/training-objective choice, not a post-hoc ranking repair.

CPU numerical checks gave:

| Region probabilities | Noisy-OR | Max | Normalized LSE tau=0.25 | tau=0.5 | tau=1 |
|---|---:|---:|---:|---:|---:|
| 0.5, 0.5, 0.5, 0.5 | 0.9375 | 0.5 | 0.5 | 0.5 | 0.5 |
| 0.2, 0.2, 0.2, 0.2 | 0.5904 | 0.2 | 0.2 | 0.2 | 0.2 |
| 0.9, 0.1, 0.1, 0.1 | 0.9271 | 0.9 | 0.864204 | 0.818216 | 0.7 |

At equal zero logits, U loss is 0.693147 and each logit gradient is -0.125,
versus noisy-OR's 0.064539 and -0.033333. This illustrative computation does
not measure real-data gradient norms or demonstrate better generalization.
Normalized LSE is at most the max logit; at large tau it can dilute a single
positive region. Its whole score is a detection surrogate, not an exact union
probability or a guarantee of regional/whole threshold equivalence.

Protocol v2 implements the same normalized LSE in U training and whole inference.
N/A retain dense region BCE; N's BCE sum no longer equals negative whole BCE
under LSE, and no additional negative whole term is used. The selected config
sets tau=0.5 and N/A/U=8/4/4 in the same run, so this experiment measures their
combined effect rather than isolating aggregation from sampling. No training
result exists yet for protocol v2.

Primary source: Pinheiro and Collobert, CVPR 2015,
[From Image-level to Pixel-level Labeling with Convolutional Networks, section 3.1](https://arxiv.org/html/1411.6228v3#S3.SS1),
uses normalized score-LSE for weakly supervised image segmentation; its inverse
temperature r corresponds to 1/tau here. Application to four CT regions remains
an untested adaptation.
