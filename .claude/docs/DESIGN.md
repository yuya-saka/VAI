# Project Design Document

> Active design decisions only. Historical designs are available from Git history.

The current new-model design is `fracture_detection/REGION_MIL_DESIGN.md`.
Its 2026-09-12 v2 amendment supersedes the original fixed noisy-OR and 4/4/8
starting settings: current defaults are normalized logit-LSE with tau=0.5 and
N/A/U=8/4/4. Both settings are explicit in config, and legacy v1 artifacts keep
their original noisy-OR interpretation. The user accepted the original serial
local-region design on 2026-09-10 and authorized implementation later that day;
the v2 aggregation and sampling revision was authorized and implemented on
2026-09-12. Numerical settings remain experimental starting values, not validated
optima. The implementation is at `fracture_detection/weak/`.
`patience_gt_passes` is config-controlled with positive-integer validation.
The editable config uses 15 GT-passes; completed test_v2 used 10 according to
its saved effective config. Other frozen training settings remain unchanged.
On 2026-09-10 the user specified end-to-end CNN fine-tuning, four local region
fracture outputs, and whole prediction derived only from those outputs. There
is no parallel whole classifier. Negative vertebrae now also train the region
path. This supersedes the positive-only and frozen-CNN proposals retained as
historical discussion in `fracture_detection/CONDITIONAL_MIL_DESIGN.md`.

The current aggregation is configurable and parameter-free. Protocol v2 defaults
to normalized logit-LSE, while protocol v1 and the v2 `noisy_or` option retain
the original aggregation. On the 2026-09-10 review, the user relaxed Baseline 0
loss parity and requested the most appropriate fine-tuning design. The resulting
observation-specific mixed
supervision: negative bags receive the sum of four negative region BCE terms;
annotated positive bags receive BCE summed over all four region cells; positive
bags without region GT receive positive whole-aggregation loss. The user explicitly
clarified that every region in a GT-bearing vertebra is annotated: zero means
no fracture, never an unknown cell. Annotation availability is vertebra-level.
Annotated bags receive no redundant whole-positive OR term. Remove partial-cell
loss branches and include every annotated bag's four cells in evaluation.
The old 235-complete/33-partial and 983-known/89-unknown counts describe legacy
validity logic, not this label contract. Do not reuse annotation-run-based zero
invalidation from baseline0/data/region_validity.py for the new model. With the
previously counted 268 annotated bags, all 1,072 cells are supervised. Recheck
versioned inputs before implementation; existing artifacts are unchanged.

The user's follow-up asks whether annotated positives should classify the
vertebra using only GT-positive regions. A singleton GT-positive restricted
OR is exactly that region's positive BCE. For multiple known-positive regions,
restricted OR requires only one to fire; per-positive-region BCE additionally
uses the fact that each is positive. The user selected per-positive-region
supervision, not restricted-OR-only supervision. Confirmed negative regions
likewise receive local BCE with target zero even inside a positive vertebra;
do not overwrite them with the positive whole label or omit their local loss.
All zeros in GT-bearing vertebrae are valid negative targets.
GT selects training losses only;
inference aggregates all four regions. Local feature readout does not guarantee
strict exclusion of outside-region image information through the CNN.

The batch objective averages these per-bag losses over the actual number of
bags, with weak-loss coefficient beta=1. There is no extra pos_weight, source
importance correction, or separately normalized GT term. Four observed region
labels supply four BCE observations, not a per-bag cell mean. Under v1 noisy-OR,
negative whole BCE equals the four-negative-region BCE sum. Under v2 LSE they
are not equal; the design keeps the dense four-region negative BCE and adds no
whole-negative term. This is not natural-population risk matching or a guarantee
of calibrated region/whole probabilities.

The accepted design has no pre-training calibration phase: no CAM probability
calibration, pseudo-label generation, gradient-norm lambda calibration, or
calibration artifacts/CLI. Beta=1 is a fixed starting hyperparameter; any later
beta comparison is an inner-data hyperparameter experiment. Inner-only threshold
selection for thresholded metrics remains an evaluation step, not model
calibration. AP/AUROC require no threshold, and Brier/ECE remain diagnostics only.

The current batch of 16 contains 8 negative, 4 annotated-positive, and 4
unannotated-positive bags. One GT-pass exposes each annotated bag once;
negative and weak-positive shuffled queues rotate across passes. Exact ratios,
beta, and schedule remain provisional and require inner-validation comparison.
A matched no-weak-loss arm keeps identical inputs and batch denominators to
test whether direct weak supervision actually improves held-out localization.

All losses update the CNN and the same local region path. The proposal retains
pretrained BN running statistics while fine-tuning CNN weights and BN affine
parameters, to limit statistic drift under positive-heavy sampling; this is
not CNN freezing or a Baseline 0 loss-parity constraint. One checkpoint selected
by inner annotated-positive region macro AP produces both region and whole
outputs, with whole metrics reported to expose tradeoffs.

Region scores are unconditional fracture scores and are not multiplied by whole
probability at inference. LSE's surrogate-score meaning and legacy noisy-OR's
independence assumption are documented. Architecture direction, inclusion of
negatives, and three-source sampling are user requirements; Baseline 0 label-weight
parity is no longer required. Exact ratios and numerical hyperparameters remain
experimental starting values. Plane/mask
fracture coverage was confirmed by the user. This design-only task did not
itself authorize implementation or training; a later 2026-09-10 request did
(Arm B implemented at `fracture_detection/weak/`; outer0 training completed,
with the result re-audited on 2026-09-11).

The following existing-system sections and `.claude/docs/REGION_MODEL_DESIGN_JA.md`
describe the separate Baseline 0 / CAM project. Their parallel whole path and
pseudo-label procedures are not requirements for the new region-MIL design.

## Overview

The active fracture-detection project consists of the Baseline 0 teacher and
its pseudo-label generation pipeline. Previous MTL, mask-guided Proposed, and
Type2 approaches failed or were discontinued. Pseudo-label generation and its
CAM audit remain first-class components and are expected to be used heavily.

## Architecture

```text
15 planes x (5-channel 2.5D CT + whole-vertebra mask)
    -> EfficientNetV2-S encoder
    -> bidirectional LSTM across planes
    -> one fracture logit per plane
    -> mean sigmoid
    -> one vertebra-level fracture score
```

The active dataset contains 13,432 quality-filtered bags. Evaluation uses the
existing patient-grouped nested five-fold protocol.

Baseline 0 retains the patient-grouped nested five-fold protocol: each outer run
uses three training folds (60%), one inner-validation fold, and one outer-test
fold. The same fold-matched Baseline 0 checkpoints serve as the pseudo-label
teachers, the whole-vertebra classification baseline, and the initialization for
region fine-tuning. The proposed separate 80%-trained rerun is cancelled because
maximizing absolute classification accuracy is not the region experiment's goal.

The next region-localization model keeps the Baseline 0 whole path intact and
adds a separate region path. The shared EfficientNetV2-S trunk feeds an FPN at
stride 4, mask-normalized pooling for four anatomical regions, a dedicated
region BiLSTM, and the four-region output heads. The whole and region paths
therefore use two BiLSTMs and share only the CNN trunk.

Each student outer fold starts from the fold-matched Baseline 0 checkpoint.
The encoder, whole BiLSTM, and whole head are transferred; the FPN, region
BiLSTM, and region heads are randomly initialized. All parameters are then
fine-tuned without a freeze or warmup stage. Baseline-derived parameters use
an initial learning rate of `2.3e-5`, while the new region path uses `2.3e-4`.

The outer-0 pilot using cross-case CAM-density ranking failed and that objective
is retired. No remaining folds are trained with
`L_exact + alpha * L_rank`. Exact-only teacher pretraining is also rejected:
the same scarce-label objective already overfits and therefore cannot be used
as an independent pseudo-label authority. The fold-matched Baseline 0 CAM remains
the leading source candidate, but the final rule for constructing pseudo labels
from it is not decided. The per-region logistic calibrator is retired because its
intercept absorbed each region's base rate and systematically redirected targets
toward R4. Within-bag normalised CAM shares and four-view TTA produced promising
retrospective measurements, but they are candidate ingredients rather than a
frozen generation rule.

The four-view probability artifact and acceptance gate must be completed before
the replacement region objective is implemented. The pseudo term has no
independent coefficient: when enabled it is the masked mean soft BCE
`BCEWithLogits(z,q)`, and when disabled it is absent. No remaining-fold training
or pseudo-label regeneration should assume that the provisional calibration
coefficients are final.

The next region experiment uses a completely external human-GT holdout. First
identify the 160 studies containing the 268 human-labelled bags and remove all
1,111 bags from those studies before pseudo-label generation, splitting, or
student training. The remaining pseudo corpus contains 12,321 bags from 1,849
studies, including 982 whole-positive and 11,339 whole-negative bags.

Only that remaining corpus enters patient-grouped nested five-fold training.
Each outer run uses three folds for training, one for pseudo-target validation,
and one for pseudo-target test. The region student is optimized only against
frozen pseudo targets and selected only by pseudo-validation excess
cross-entropy `BCE(p,q)-H(q)=KL(Bern(q)||Bern(p))`. Its pseudo-test result measures
pseudo-teacher reproduction, not human-GT accuracy.

Whole-vertebra fracture classification remains a separate true-label endpoint.
The whole path uses the observed `vertebra_target` on the same non-GT-patient
train/validation/test folds, selects its checkpoint by whole-validation loss, and
reports patient-grouped five-fold OOF AUROC/AP on all 12,321 pseudo-corpus bags.
Thresholded precision, recall, specificity, and F1 use thresholds fixed from each
validation fold before its test fold is scored. Region and whole checkpoints may
therefore come from different epochs and are ensembled independently.

After all five pseudo folds finish, run the five selected models on the same
external 268-bag GT set, average their probabilities per bag and region, and
compute the human-GT endpoint once. Per-model GT metrics are diagnostic only and
must not select a model or configuration. Any upstream teacher, calibration, or
pseudo-label transformation used by this experiment must also exclude all 160
GT studies; otherwise the external holdout claim is invalid.

The external holdout supports two fixed endpoints. Conditional localization uses
the 268 whole-positive bags with human region labels. End-to-end region detection
adds the 761 whole-negative bags from the same held-out studies as four logical
zeros, for 1,029 evaluable bags; the 82 held-out whole-positive bags without human
region labels are excluded from region metrics. Whole-fracture evaluation may use
all 1,111 held-out bags.

For the external whole-fracture endpoint, average the five selected whole-path
probabilities before computing metrics against the 1,111 observed
`vertebra_target` labels. For end-to-end region detection, combine the independently
ensembled whole and conditional-region probabilities using the predeclared rule;
no threshold or checkpoint may be chosen from the external holdout.

## Approved Pseudo-Label Implementation Direction

Every unannotated whole-positive bag must receive fold-matched CAM supervision
during training, and that supervision must reach the four endpoint binary region
heads directly. The previous auxiliary-location-only proposal is rejected as too
indirect for the intended experiment.

The provisional transformation is a fold-specific, all-region-shared probability
calibration of four-view within-bag CAM shares. For view `v`, compute
`s_vr=e_vr/sum_j(e_vj)` from the region density enrichments, then average
`s_r=mean_v(s_vr)` over identity, horizontal flip, and rotations by plus/minus
10 degrees. The previous plan fitted
`q*_r=sigmoid(a_k*logit(clip(s_r,0.01,0.99))+b_k)` using human region labels.
That calibration is incompatible with the external-GT protocol and cannot be
reused. A GT-free pseudo-target transformation must be frozen before the new
five-fold run. If the same functional form is retained, its coefficients require
a non-GT source. Region identifiers, per-region intercepts, bag probability, CAM
total, and estimated cardinality remain excluded. If
`sum_r(q*_r)<1`, rescale the vector to sum to one as a whole-positive logical
guard. This map remains provisional until its actual four-view probability
distribution is regenerated and audited; the current identity-CAM diagnostic
has pooled pseudo medians R1/R2/R3/R4 = 0.350/0.263/0.320/0.289 and median
`sum(q)=1.396`.

Human targets never enter the new pseudo corpus. All bags belonging to a GT study
are excluded before pseudo-target attachment, so there is no human-over-CAM cell
override in this experiment.
Hard top-1 conversion remains rejected: existing audit precision is 0.648
overall, 0.503 for R1, and 0.522 for R2.

The pseudo loss is plain soft BCE:

```text
L_CAM_cell = BCEWithLogits(z, q)
L_CAM = sum(valid * L_CAM_cell) / sum(valid)
```

There is no `mu`, ramp, or pseudo-specific coefficient. Native
`BCEWithLogitsLoss(pos_weight=2)` is not used for `0<q<1`, because its optimum is
`p=2q/(1+q)` rather than `p=q`. The Baseline 0 `pos_weight=2.0` contract remains
unchanged for the hard exact region targets and whole path only.

The first controlled comparison keeps the endpoint architecture identical across
`no_pseudo`, `cam_soft`, and `cam_soft_shuffled`; only the CAM target association
and presence or case association differ. The pseudo-label formula is
approved for implementation, while its four-view coefficients and probability
artifact remain unresolved and must be fixed before training. The phased handoff
plan is `.claude/docs/work-logs/2026-09/2026-09-01-cam-soft-bce-implementation-plan.md`.

## Open Questions

- Proposed follow-up after the test_v2 AP diagnosis (not yet approved for
  implementation or training): keep normalized LSE tau=0.5, N/A/U=8/4/4,
  initialization, and the existing 60-pass cosine horizon. Run the full 60
  GT-passes without region-only early stopping to observe both inner endpoints;
  this supplies approximately 2.64 negative sampler cycles but may also
  increase GT overfitting. Save candidate checkpoints sufficient to apply an
  inner-only joint selection rule: maximize whole AP subject to a preregistered
  acceptable region-macro-AP floor. A 0.01 decrease from the completed v2 inner
  reference is an illustrative tolerance, not an accepted requirement or an
  outer-tuned threshold. If no candidate satisfies the floor, report that the
  candidate failed the localization constraint. Report both endpoints for the
  same selected model, without mixing whole and region outputs across models.
  Compare matched beta=1 and beta=0 runs using identical U inputs, denominators,
  schedules, and the same selection rule to isolate the direct U-loss benefit
  under this revised protocol. Longer training and this selection rule are
  hypotheses, not guarantees of better AP. The previously inspected outer0
  results are exploratory; use untouched outer folds for confirmation. Defer
  distillation or another loss change until this comparison identifies the
  remaining tradeoff. AUROC improvement alone does not guarantee PR-AUC
  improvement, as discussed by
  [Davis and Goadrich (2006)](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf).

- The user's 2026-09-11 objective is improved whole classification together with
  localization. Proposed next comparisons preserve the serial region-to-whole
  path: change only noisy-OR to max for U training and whole inference while
  retaining full N/A region BCE; separately test N/A/U=8/4/4 with A's nominal
  coefficient preserved. Max can miss additional positive regions and reinforce
  an incorrect maximum, so it is not yet selected. Whole-score distillation from
  the fold-matched Baseline 0 is a later option, not regional pseudo-labeling or
  a parallel inference head. Select using both inner endpoints, and reserve
  untouched outer folds for confirmation. No implementation or new training is
  authorized by this design discussion; details and primary sources are in
  `.claude/docs/experiments/2026-09-11-weak-test-v1/analysis.md`.
- For weak/test_v1, isolate the incremental noisy-OR contribution using matched
  beta=0/1 runs that retain identical U inputs and batch denominators. Outer0
  region macro AP is 0.776598 on 56 annotated-positive bags; whole AP is 0.728427
  versus Baseline 0's 0.772599 on the same 2,671 bags. The saved
  `fold_metrics.json.best_metrics` are inner metrics, not outer results. Positive
  oversampling and score inflation alone do not establish the AP regression's
  cause, because a strictly increasing calibration transform preserves ranking.
  Negative exposure in the new region path and noisy-OR aggregation remain
  candidates. No new training protocol is selected by this analysis; see
  `.claude/docs/experiments/2026-09-11-weak-test-v1/analysis.md`.
- Test true from-start joint learning before adopting staged freezing as the final
  training protocol. The current v3 run is not a from-start joint experiment: it
  initializes the whole path from an already trained Baseline 0 checkpoint and
  then jointly fine-tunes it past its validation-loss optimum. Compare a
  whole-only arm and a joint `L_whole + lambda * L_region` arm from the same
  pre-Baseline initialization, split, seed, batches, and schedule, using one
  natural-batch encoder forward so BatchNorm updates are matched. Select the
  joint arm by a consistently pooled validation total loss while retaining
  separate whole and region checkpoints for diagnosis. Adopt joint-from-start
  only if it lowers total loss without a material whole-loss regression; if the
  noisy region target harms whole learning, test delayed region activation or
  shared-gradient isolation rather than assuming freezing is universally best.
- Do not treat shared whole-region gradient conflict as the primary explanation
  for the natural-BN whole-loss plateau. On v3 outer-0, corrected epoch-5 to
  epoch-71 whole BCE is flat, but positive BCE significantly worsens while
  negative BCE significantly improves. A fixed 64-batch shared-trunk probe found
  aggregate whole-region gradients aligned and the weighted region norm only
  about 10%-11% of the whole norm. Both task gradients point against held-out
  positive whole loss and toward held-out negative whole loss, with the whole
  gradient dominant. Before choosing gradient surgery, run a matched single-pass
  BN ablation from the same initialization with `lambda=0` versus calibrated
  region training. If both reproduce the positive failure, investigate whole
  weighting/sampling and positive-loss-aware stopping; if only the region arm
  does, constrain its shared-trunk update. See the validation/BatchNorm work log.
- The epoch-71 class-decomposed probe shows that the training-positive whole
  gradient is useful for held-out positives (cosine +0.735), but its shared norm
  is 0.359 after seen-positive BCE has fallen to 0.011. The training-negative
  gradient has norm 1.303 and is opposed to held-out-positive improvement
  (cosine -0.966), so it dominates the current update. Do not assume the 9:1
  count-ratio `pos_weight` alone is sufficient or adopt the much larger local
  norm-balancing ratio directly. Baseline 0 learns under the same
  `pos_weight=2`; the region run starts from its already-trained epoch-44
  AUROC-best weights, and corrected region epoch 5 already beats Baseline 0's
  observed validation-loss minimum. Treat the later behavior as
  post-convergence continuation first: remove the BN double update, select the
  whole endpoint by whole validation loss, then freeze encoder/whole parameters
  while region-only parameters continue. Run a matched short `lambda=0` control
  before any positive-weight sweep.
- The user's immediate design priority is reducing the total objective, explicitly
  correcting the preceding whole-loss-only request. The current objective is
  `L_whole + lambda * L_region_conditional`; v3 already reduces logged training
  total loss from 0.366203 at epoch 1 to 0.291344 at epoch 71. Reconstructing
  validation `whole + 0.3156107231711445 * region_loss` gives 0.618893 at epoch 1,
  a minimum of 0.488207 at epoch 60, and 0.605251 at epoch 71. These are diagnostics
  from the existing contaminated inner pseudo targets, not clean validation
  evidence. Training averages batch losses with region-skipped steps as zero,
  while validation pools region cells over the fold; their absolute totals are
  not directly comparable. Define fixed targets, coefficients, and consistent
  reductions before deciding how total validation loss should govern selection
  and stopping. Current selection remains separate for region and whole; no
  runtime rule is changed by this priority clarification.
- The 2026-09-07 negative-inclusive CAM review reproduced the main AP values but
  identified unsupported causal conclusions and incorrect teacher-fold accounting.
  Resolve the inner-target selection leak before treating v3's conditional AP
  increase as clean generalization evidence. Confidence filtering remains
  exploratory, and excluding negatives from a conditional region objective is
  not itself a defect. See
  `.claude/docs/work-logs/2026-09/2026-09-07-negative-inclusive-cam-review.md`;
  no replacement architecture or checkpoint rule is adopted by this review.
- Before replacing the current mixed-supervision experiment, regenerate
  leakage-safe pseudo targets for its inner validation fold and rerun checkpoint
  selection with the validation objective matched to training. The completed
  outer-0 `cam_soft` run had `val_pseudo_loss=0` for all 29 epochs because eval
  loaders never attached pseudo targets, and selected epoch 9 by human-GT macro
  AP; it is not evidence from mixed validation monitoring. If GT and pseudo
  cells are unified rather than separately normalized, define one cell target
  `t=y` for valid GT and `t=q` otherwise, then monitor masked mean
  `BCE(p,t)-H(t)`. Hard GT has `H(t)=0`, so this equals GT BCE on hard cells and
  Bernoulli KL on soft cells. This single-reduction candidate must be changed in
  training and validation together; validation-only adoption would create a new
  objective mismatch.
- For any resumed mixed-supervision experiment, the leading researched
  candidate is a hierarchical conditional region objective. Build one target
  tensor with GT precedence, `t=where(gt_valid, y_gt, q_pseudo)`, but calculate
  plain masked BCE for the region head only on whole-positive bags and define
  the end-to-end score as `p(whole)*p(region|whole-positive)`. In the inspected
  outer-0 training data this reduces aggregate negative-to-positive target mass
  from 27.420:1 over unconditional cells to 1.823:1 over whole-positive cells,
  without source weighting or discarding any case from the whole task. Reduce
  each region over its valid cells and macro-average the four region losses; use
  a shuffled, without-replacement whole-positive region stream in which GT and
  pseudo cases remain mixed. Validation must attach leakage-safe pseudo targets
  and monitor the same macro objective as `BCE-H(t)` while hard-GT metrics remain
  separate. Native `pos_weight`, focal, asymmetric, and distribution-balanced
  losses are not leading candidates because they alter the probability-fitting
  objective or were derived for hard-label long tails. This is a researched
  candidate, not an accepted architecture: five-fold stability and hard-GT test
  accuracy still require direct verification against unconditional plain BCE.
- What GT-free transformation should replace the current human-calibrated CAM
  probability map before the external-holdout pseudo-only five-fold run?
- Do the regenerated four-view shares retain the identity-CAM calibration,
  positive slope, probability reliability, and observed target range?
- Which one pseudo source, Stage2 region-only or Baseline 0 CAM, must be frozen
  before the external GT set is evaluated? The held-out GT set cannot choose
  between sources without becoming a validation set.
- Which target-free checks and outer-fold-0 controls must pass before the chosen
  pseudo labels are accepted for the remaining folds?
- What pre-registered patient-grouped acceptance threshold should distinguish a
  useful CAM signal from ordinary regularization in the shuffled control?
- If the completed outer-0 controls show that calibrated soft targets are not
  useful, should a separately pre-registered confidence-gated hard or sharpened
  target arm be tested? Naive `q >= 0.5` hardening is not approved: an exploratory
  audit of the frozen artifact maps about 45% of human-positive cells to zero, so
  it may amplify CAM errors rather than fix weak probability fitting.

## Repository Structure

All active implementation lives in `fracture_detection/baseline0/`. Code is
grouped by responsibility into `cli/`, `config/`, `data/`, `modeling/`,
`training/`, `evaluation/`, and `pseudo_labeling/`; reproducibility inputs and
tests remain in `resources/` and `tests/`. The package root contains only documentation and its
package marker. A new top-level experiment directory is created only after the
approach is accepted as an active project.

Generated outputs are not mixed with source responsibilities. The current
five-fold Baseline 0 artifacts are retained locally. Pseudo-label outputs are
regenerated under the Baseline 0 output tree when needed; failed-arm outputs and
old diagnostic runs are removed.

## Key Decisions

| Decision | Rationale | Date |
|---|---|---|
| Prefer observation-specific mixed supervision without Baseline 0 risk matching | User relaxed Baseline 0 parity. Proposed 4/4/8 sampling intentionally emphasizes positives: sum four all-negative BCE terms for negatives, sum all four GT-cell BCE terms for annotated positives, and noisy-OR positive loss for unannotated positives; average over actual bags with beta=1 initially. Omit source correction, extra pos_weight, and redundant OR on GT-positive bags. Fine-tune the shared local region path and CNN; verify benefit against a matched no-weak arm | 2026-09-10 |
| ~~Use three-source positive-heavy sampling with Baseline 0 whole-loss weighting~~ (superseded by latest 2026-09-10 review) | User requires few negative bags and reliable exposure to annotated/unannotated positives. Proposed 4/4/8 batches need source importance correction before positive weight 2 and normalization by corrected weight sum; applying pos_weight alone would also change the whole class prior | 2026-09-10 |
| Distinguish Baseline 0 class-weight parity from identical training behavior | Inspected broadcast_bce_loss: it weights 15 plane BCE terms by positive 2/negative 1 and divides by weight sum. A four-region bag-OR model can preserve target population label weights but cannot claim identical plane supervision, stochastic batch normalization, or optimizer trajectories | 2026-09-10 |
| Fine-tune a single region-to-whole model with negative vertebrae included | Explicit user requirement: CNN learns four local fracture scores and whole is derived only from those scores. Remove the parallel whole classifier and CNN-freeze proposal. Negative whole labels supervise all four regions | 2026-09-10 |
| ~~Propose noisy-OR whole BCE plus observed-region BCE for the serial model~~ (superseded by observation-specific loss, 2026-09-10) | Negative noisy-OR BCE already equals four negative region BCE terms. Add direct GT only for annotated positives, let both objectives update the CNN, and derive whole and region outputs from the same checkpoint without probability gating | 2026-09-10 |
| Propose ordinary GT+OR as the first conditional-MIL comparison and defer head detachment | Regularization does not require a gradient stop; fixed head weights still permit shared feature shifts to increase every logit. A matched GT-only comparison should establish the weak-label effect before introducing unsupported gradient restrictions | 2026-09-10 |
| Specify conditional-MIL pilot reductions and comparison controls before implementation | Normalize batch sums with fixed train GT-cell/weak-bag counts, use each positive bag once per epoch, keep identical exposure across arms, and select checkpoints/coefficients only with each outer run's inner data. A shared-head global-pool ablation would produce identical region outputs, so locality comparisons require matched distinct heads | 2026-09-10 |
| ~~Treat positive-bag OR as a ramped feature-only regularizer~~ (superseded 2026-09-10 review) | The proposed head detach blocks direct bias gradients but does not fix output semantics or prevent collapse through feature changes; retain it only as an additional experimental arm | 2026-09-09 |
| ~~Start a frozen-CNN, positive-only conditional-MIL pilot~~ (superseded 2026-09-10) | Replaced by the user's requirement to fine-tune the CNN and derive whole from four region outputs using positive and negative vertebrae | 2026-09-09 |
| ~~Research a positive-only local-region model~~ (scope superseded 2026-09-10) | Local masks, mixed GT/weak supervision, and design-before-implementation remain required; positive-only training and a parallel whole path are no longer the target design | 2026-09-09 |
| Keep Baseline 0 as the active teacher and reference implementation | It is the current reliable model; previous MTL, Proposed, and Type2 approaches failed | 2026-08-24 |
| Keep pseudo-label generation and CAM audit as active first-class components | Pseudo-labels are a core upcoming workflow and require auditable Grad-CAM generation rather than historical deletion | 2026-08-24 |
| Remove MTL, Proposed, Type2, frozen multi-arm infrastructure, and local archives | Keeping failed approaches in the active tree obscured the current system and created excessive files and directories | 2026-08-24 |
| Group Baseline 0 code into seven responsibility directories | A completely flat package made the root hard to scan; responsibility directories provide useful navigation without reviving per-experiment package sprawl | 2026-08-24 |
| Use Git history instead of in-tree archives | Historical recovery remains possible without burdening the active repository structure | 2026-08-24 |
| Use separate whole and region BiLSTMs in the next region model | This keeps the Baseline 0 whole path unchanged and limits sharing to the CNN trunk | 2026-08-25 |
| Retire cross-case CAM-density ranking after the outer-0 pilot | Human AP degraded while student-teacher rank agreement increased; the replacement loss must be validated against an otherwise identical no-CAM arm before any remaining folds run | 2026-08-28 |
| Keep fold-matched Baseline 0 CAM as the leading pseudo-label source candidate | CAM discriminates regions, but the final transformation from CAM evidence to a regional training target remains open | 2026-09-01 |
| Reject exact-only region models as pseudo-label teachers | The exact-only objective is the overfitting regime being replaced; CAM comes instead from the independently trained whole-fracture Baseline 0 teacher and is calibrated with scarce regional GT | 2026-08-28 |
| Defer logical-OR loss from the initial CAM-soft experiment | Calibrated CAM targets already provide per-case supervision for whole-positive bags; OR behavior is monitored as a non-gradient diagnostic so the first ablation isolates the CAM contribution | 2026-08-28 |
| Limit each region epoch to one human-pool pass | Cycling 159 human bags to match large streams repeatedly reuses scarce GT and contributed to the observed training/validation divergence; pseudo and negative streams rotate around fixed human exposure | 2026-08-28 |
| ~~Keep `pos_weight=2.0` only for `L_whole`~~ (superseded 2026-08-31) | Baseline 0 whole-risk parity is preserved, while source-balanced sampling already corrects the region exact-label imbalance | 2026-08-25 |
| Compare against four fully independent single-region models | Each comparator retains its own CNN trunk, Baseline 0 whole path, FPN, region BiLSTM, and one target-region head so the experiment measures cross-region representation sharing rather than head count alone | 2026-08-25 |
| Initialize each region student from its fold-matched Baseline 0 checkpoint | This follows the intended single-task-to-multi-task fine-tuning design and avoids restarting the whole path from ImageNet weights | 2026-08-26 |
| Fine-tune all parameters with discriminative learning rates and no freeze/warmup | The pretrained whole model remains adaptable while the new region path learns faster; gradient calibration limits shared-trunk perturbation | 2026-08-26 |
| Force region CLI multiprocessing temporary files onto local `/tmp` | Project-level `TMPDIR` points to NFS, where interrupted DataLoader workers leave thousands of busy `pymp-*` directories and emit misleading cleanup tracebacks | 2026-08-26 |
| Version calibration artifacts independently from experiment outputs | `calibration.version` selects `outputs/calibration/<version>`; a config fingerprint prevents reuse after scientific settings change while allowing output name, GPU, and active-region comparator changes | 2026-08-26 |
| Backpropagate whole and region graphs sequentially within each optimizer step | Gradient accumulation is equivalent to the combined objective, while releasing the 16-bag whole-path graph before constructing the 16-bag FPN region graph substantially reduces peak VRAM | 2026-08-26 |
| Compile CUDA region training with TorchInductor `default` mode | RTX A6000 measurement showed 1.70x steady-state speed and lower peak VRAM; channels-last added no benefit, fused AdamW/TF32 were neutral, CUDA Graph mode was incompatible, and max-autotune was impractically slow | 2026-08-26 |
| Name the region model's source package `data_pipeline/` instead of `data/` | The repository-wide `data/` ignore rule is reserved for dataset artifacts and otherwise hides the region dataset/loader Python sources from Git | 2026-08-27 |
| Emit timed progress logs for every region training startup phase | Manifest loading, calibration validation, loader construction, model initialization, and first-batch compilation can each appear stalled; start/completion/failure logs make the active phase and elapsed time observable | 2026-08-27 |
| Aggregate region bag probabilities in float32 outside BF16 autocast | BF16 rounds high sigmoid probabilities and the `1 - 1e-6` clamp bound to exactly one, producing infinite bag logits after several epochs; float32 preserves the same objective while keeping the logit transform finite | 2026-08-27 |
| Apply Baseline 0 `pos_weight=2.0` to the region exact loss as well | Measured region imbalance is only 2.14:1-3.12:1 (R4 is majority-positive at 0.59:1), so no elaborate imbalance handling is warranted; reusing the existing `training.pos_weight` keeps the mechanism identical to Baseline 0 and needs no new config key, since `region_pos_weight` stays forbidden. Supersedes the 2026-08-25 whole-only decision | 2026-08-31 |
| Keep human GT targets hard 0/1 while pseudo targets become soft | The pilot failure was a semantic split between a scale-invariant ranking objective and an absolute-probability objective, not incorrect pseudo labels; soft per-cell targets put both terms on one probability scale without softening the scarce exact supervision | 2026-08-31 |
| Mix every component into the observed validation loss | The region objective is a sum of GT, negative, and pseudo terms; observing them combined keeps the monitored quantity aligned with what is optimized. The pseudo component's floor is the non-zero entropy H(q), so its absolute value is not comparable across arms with different targets | 2026-08-31 |
| Judge GT learning behavior by validation AP/AUROC, not validation BCE | Hard 0/1 BCE has no finite optimum, so validation BCE rises as the model extremizes on the 159-bag human pool even while ranking holds; the pilot showed val_human 0.4909->1.0303 with region AUROC flat at 0.8305->0.8155 | 2026-08-31 |
| Select whole and region checkpoints separately | The endpoint for the region path is region macro AP while the whole path is a guardrail; a single mixed-loss criterion selected pilot epoch 4 over epoch 1 despite epoch 1 having the best region AP (0.8420 vs 0.7913), and the best whole epoch was 24 | 2026-08-31 |
| Treat individual pseudo-label errors as acceptable | Pseudo labels are auxiliary supervision toward human-GT accuracy, not an endpoint; the acceptance criterion stays improvement over a human-only arm rather than agreement with the teacher | 2026-08-31 |
| Retire the per-region CAM logistic calibrator | Its intercept absorbed each region's base rate, so the posterior target was the argmax on bags whose fracture was elsewhere in 0 of 5 folds for R1, R2 and R3 | 2026-09-01 |
| Use shared logit-share calibration as the provisional CAM probability map | Complete-only patient-grouped OOF showed lower log loss and ECE than constant-scaled share, while raw logistic, isotonic, and Beta calibration did not improve it. Sharing one positive-slope map across regions preserves CAM direction and prevents regional base-rate shortcuts. Coefficients stay unfrozen until four-view probabilities are regenerated | 2026-09-01 |
| Use plain soft BCE for CAM targets with no pseudo-loss coefficient | A separate `mu`, ramp, confidence weight, or soft-label `pos_weight` is not part of the experiment. Human and logical-zero exact targets retain Baseline 0 `pos_weight=2.0`; CAM targets use the masked mean `BCEWithLogits(z,q)` so their optimum remains `p=q` | 2026-09-01 |
| Keep the area-correction exponent at one | Sweeping mass/area^gamma is monotone in gamma: macro AP 0.580 at gamma 0 versus 0.649 at gamma 1, and own-region argmax 5/20 versus 19/20. Posterior elements occupy 62 percent of the vertebra, so an uncorrected share is dominated by them. The bbox pipeline's gamma=0.5 finding does not transfer | 2026-09-01 |
| Stop pursuing per-bag cardinality in the pseudo target | Predicted cardinality saturates above K=2 and its Spearman correlation with the truth stays between 0.24 and 0.42 across folds, missing the pre-registered bar of 0.4 in every fold; it adds only 0.017 macro AP with a larger standard deviation. Oracle cardinality would be worth 0.18, so the information is valuable but not obtainable from the available teacher signals | 2026-09-01 |
| Do not weight pseudo cells by teacher disagreement | Splitting cells at the median teacher-to-teacher spread leaves AUROC unchanged (+0.008, -0.021, -0.002, +0.001), so disagreement does not identify wrong cells. TTA spread does separate them for R1 and R2 but reverses for R3, so it stays an explicit later ablation | 2026-09-01 |
| Keep the region model as close to Baseline 0 as possible | The only intended differences are that a region path exists and that most of its supervision is pseudo-labelled; every other question defaults to whatever Baseline 0 does, which is what retired the source-balanced sampler, the 0.5/0.5 source split and any imbalance handling beyond `pos_weight` | 2026-09-01 |
| Treat GT-driven training variance as a model-design problem rather than an evaluation-data objection | The fixed 268 human-labelled bags are the required endpoint. Before five-fold comparison, GT exposure must be deterministic and GT adaptation must avoid destabilizing the high-capacity weakly supervised representation | 2026-09-02 |
| Do not retry separated GT and soft-label optimization as the stabilization method | Source-separated sampling/losses already failed, and the current separately normalized `L_E + L_P` objective shows the same instability; another GT-specific phase, loader, or coefficient would repeat the failed mechanism | 2026-09-02 |
| Train, validate, and pseudo-test the next region student only on non-GT patients, then ensemble five models on one external GT holdout | This removes scarce hard GT from optimization and checkpoint selection. Pseudo test measures teacher reproduction only; the ensembled prediction on the untouched 268-bag human set is the sole region-accuracy endpoint | 2026-09-02 |
| Exclude every bag from each GT-labelled study from all upstream pseudo training and generation | Bag-only exclusion leaves other vertebrae from the same patient in training. The holdout is therefore 160 studies and 1,111 bags, leaving 12,321 bags from 1,849 studies for pseudo-only five-fold training | 2026-09-02 |
| Preserve whole-vertebra fracture classification as an independently selected true-label endpoint | Region supervision is pseudo-only, but `vertebra_target` remains observed for all bags. Each fold selects whole and region checkpoints separately, reports whole OOF performance on 12,321 bags, and ensembles whole predictions on all 1,111 external-holdout bags | 2026-09-02 |
| Do not attribute the current `cam_soft` instability to mixed supervision before matched mixed validation is run | Training optimized `L_exact + L_pseudo`, but inner eval attached no pseudo targets, logged `val_pseudo_loss=0` for every epoch, and selected by human-GT macro AP. The existing result therefore confounds training behavior with an unmatched noisy checkpoint criterion | 2026-09-02 |
| Adopt one GT-precedence region target tensor with conditional-positive macro BCE for the next mixed-supervision run | GT and pseudo cells share one reduction with no source weights; whole-negative bags remain in the whole task, inner validation receives held-out-teacher pseudo targets, and region checkpoints minimize the matched entropy-centered objective. End-to-end region scores multiply the best whole probability by the best region conditional probability | 2026-09-02 |
| ~~Retrain Baseline 0 with standard five-fold 80% training and complete OOF prediction~~ (superseded 2026-09-04) | The extra rerun would improve exposure but complicate the clean nested interpretation, while maximizing absolute accuracy is not the region experiment's goal | 2026-09-04 |
| ~~Initialize each region student from the 80%-trained Baseline 0 checkpoint that excludes the same outer fold~~ (superseded 2026-09-04) | Although the outer endpoint remains unseen, the inner fold has already contributed whole-fracture supervision to the initialization and is not an independent validation set | 2026-09-04 |
| Retain the fold-matched 60%-trained Baseline 0 checkpoint for region initialization | The region experiment prioritizes a simple, interpretable, leakage-safe nested comparison rather than maximizing absolute accuracy | 2026-09-04 |
| Use the existing 60%-trained nested Baseline 0 for all three upstream roles | One fold-matched checkpoint remains the pseudo-label teacher, whole-classification baseline, and region initialization, preserving a simple 60% train / 20% inner / 20% outer contract | 2026-09-04 |
| Restore vertical flip and transpose augmentation for Baseline 0 and region training | Stage1 parity used both transforms and achieved higher whole-classification OOF performance; CT, whole mask, and anatomical region mask are transformed together once, retaining region IDs while moving their pixels synchronously | 2026-09-04 |
| Restore fold-process GPU parallelism for Baseline 0 and region training | A single parent process assigns outer folds round-robin to configured GPUs with one fold per GPU process; this shortens wall time without changing global batch size, BatchNorm behavior, sampling, or the statistical protocol | 2026-09-04 |
| Generate CAM soft pseudo-labels from the completed orientation-augmented Baseline 0 teachers | Use each fold-matched `baseline0_aug追加/outer{k}/best_model.pt` selected by inner validation AUROC. Keep the leakage-safe four-view CAM and shared logit-share calibration pipeline, writing new artifacts to `outputs/09_04/pseudo_labels/` | 2026-09-05 |
| Use one natural-batch encoder forward for joint whole/region training | The former positive-only second encoder pass causally corrupted BatchNorm buffers. Protocol v7 computes encoder features once for the full natural batch, applies the whole path to all bags and FPN/region modules only to selected whole-positive bags, then backpropagates the summed objective once. Calibration v5 measures both shared-trunk norms from the same forward | 2026-09-08 |

## Changelog

- 2026-09-12: Recorded an unimplemented follow-up proposal: retain LSE and
  8/4/4 sampling, compare full 60-pass beta=1/0 runs, and select a single
  checkpoint by inner whole AP under a predeclared localization floor.
  Also distinguished the editable patience=15 config from completed test_v2's
  saved patience=10. No training configuration or code was changed.

- 2026-09-12: Qualified the test_v2 whole-AP diagnosis. Encoder-only transfer,
  less than one negative sampler cycle at the selected pass, and checkpoint
  selection on annotated-positive regions are concrete protocol facts; their
  causal contributions remain unseparated. Post-hoc pooling comparisons do
  not prove an information limit or establish distillation as necessary.

- 2026-09-12: Audited the completed `weak/outputs/09_12/test_v2/outer0` run.
  LSE tau=0.5 plus N/A/U=8/4/4 largely removed v1 whole-score inflation
  (outer BCE 0.4619 to 0.1740; negative mean score 0.2913 to 0.0670) and
  improved whole AP/AUROC to 0.7405/0.9160, while outer conditional region
  macro AP stayed approximately flat at 0.7717 versus v1 0.7766. Because both
  aggregation and sampling changed and no matched beta=0 arm exists, weak-label
  benefit remains unidentified. Full review is under
  `.claude/docs/experiments/2026-09-12-weak-test-v2/analysis.md`.

- 2026-09-12: Removed `training.patience_gt_passes` from the frozen-value
  contract after the user clarified that the v1 experiment had completed.
  Patience is now a positive config integer, and the current v2 run uses 15;
  all other frozen training settings remain unchanged.

- 2026-09-12: Implemented protocol v2 after user approval: configurable
  `loss.whole_aggregation` supports normalized logit-LSE and legacy noisy-OR,
  `loss.lse_temperature` controls LSE and defaults to 0.5, and the runnable
  config uses N/A/U=8/4/4 under a new `09_12/lse_tau05_n8` output directory.
  U training and whole inference use the same aggregation; N/A retain dense
  four-region BCE. Protocol v1 configs remain valid and map to noisy-OR.

- 2026-09-12: Recorded the user's LSE suggestion and a normalized logit-LSE
  comparison proposal in Open Questions and the test_v1 analysis. Checked toy
  probabilities and gradients on CPU; no model, config, or training changes.

- 2026-09-11: Recorded the user's joint whole-classification/localization goal
  and unadopted max-MIL, sampling, and later whole-distillation proposals in Open
  Questions and the test_v1 analysis. These are separate controlled comparisons,
  not an accepted replacement for the then-current noisy-OR design. Superseded
  by the user's 2026-09-12 LSE implementation decision.

- 2026-09-11: Audited weak/test_v1 outer0 learning results and corrected the earlier
  inner/outer metric mix-up in the training work log. Saved a reproducible CSV
  analysis and inner learning-curve figure under
  `.claude/docs/experiments/2026-09-11-weak-test-v1/`. GT validation improvement
  is supported, but the incremental weak-label benefit requires a matched beta
  control. Sampling, negative exposure, and aggregation explanations remain
  hypotheses; no model/config changes or new training were performed.

- 2026-09-10: Rework weak/'s per-GT-pass validation curves. The former `val_loss` averaged the N/A/U training loss over the natural inner fold (~90% whole-negative bags), so it matched neither the design's §8 split nor the train loss, and it was not written to `history.csv`; Brier/ECE were computed and discarded, and `training/monitoring.py` was never called. Now region metrics (per-region AP, BCE, Brier, ECE) use inner annotated-positive cells only, whole metrics (AP, AUROC, BCE, Brier, ECE) use every inner bag, and per-group per-bag losses (N, A, and weak-positive `-log p_whole`) are recorded for train and val. At the user's request a weak-positive-inclusive validation loss is also observed as `val_objective`: group means combined with the training 4/4/8 composition and beta, which equals the training batch loss for that composition and is computed identically for train. Annotated+negative pooled region BCE is not recorded because ~98% of its cells are negative bags and that part is identical to whole BCE by the noisy-OR identity. `diagnostics.csv` is written every pass and predictions now include raw region logits. Checkpoint selection is unchanged.

- 2026-09-10: Measured peak VRAM of one train step at real shapes (16 bags × 15 planes, 224², bf16, eager, RTX A6000, random inputs): baseline0 17.90 GiB, region_branch 19.00/20.14 GiB with 2/4 positive bags, weak with an unchunked FPN 26.73 GiB. The earlier "4-8x region_branch" note was wrong; it described only the FPN share, not total memory. The extra cost is the stride-4 FPN maps kept for backward (~0.57 GiB per bag), which region_branch pays only for whole-positive bags. weak now runs FPN + mask pooling per one-bag chunk under activation checkpointing (the encoder is not recomputed; outputs and gradients match the unchunked computation), measuring 18.42 GiB at ~20% longer steps. Batch composition, the one-encoder-forward-per-batch contract, and model math are unchanged.

- 2026-09-10: Implemented the accepted Region-MIL design as `fracture_detection/weak/` (the design doc's `region_mil/` placeholder is superseded by this user-specified directory name; no design content changed). Baseline0 modules (`data.splits`, `data.dataset`, `data.sampling`, `data.staging`, `training.trainer`, `training.parallel`, `evaluation.metrics`) are imported directly; everything else (N/A/U group resolution, the GT-pass batch sampler, the FPN/pooling/model/noisy-OR-loss stack, initialization, optimizer/scheduler, trainer, monitoring, evaluation metrics, config schema, CLIs) is implemented inside `weak/` with no dependency on `region_branch/`, keeping the rollback unit self-contained. Only Arm B (the proposed model) is implemented; no Arm A/C config switch exists. Augmentation reuses baseline0's frozen recipe unchanged per the user's instruction, except MixUp, which is structurally inapplicable to this model (no loss path exists to skip during a mixed step, unlike region_branch) and is enforced as a forbidden config key. 80 unit tests pass, including exact reproduction of the frozen manifest's N=12,100/A=268/U=1,064 counts and a genuine crash-and-resume determinism test. No training has been run; VRAM has not been measured against real data. Existing baseline0/region_branch tests (256/257, one pre-existing unrelated failure confirmed via git stash) show no regression.

- 2026-09-10: Clarify that the accepted Region-MIL workflow has no pre-training calibration phase or calibration CLI/artifacts. Preserve only inner-data threshold selection for thresholded evaluation metrics and probability-quality diagnostics; these do not alter the initial model outputs.

- 2026-09-10: Save the accepted region-MIL design, implementation order, validation gates, initial experiment, rollback policy, and dirty-worktree state to `.claude/docs/work-logs/2026-09/2026-09-10-region-mil-implementation-handoff.md` for direct implementation startup in the next session. No implementation or training performed.

- 2026-09-10: User accepts the basic region-MIL design and requests a consolidated summary. Record acceptance of the serial local-region/noisy-OR path and three-source supervision with all four GT cells valid. Retain numerical settings as initial experimental choices; no implementation or training is authorized by this summary request.

- 2026-09-10: User corrects annotation semantics: GT-bearing vertebrae have all four region labels and zero always means no fracture. Supersede earlier partial-cell assumptions and legacy-validity-derived counts for the new model; train/evaluate all four cells in all annotated bags. Update the active region-MIL contract and flag old validity logic for replacement only when implementation is authorized. No source/data edits or metric recomputation.

- 2026-09-10: User selects supervision of each confirmed positive region rather than positive-subset OR. Clarify that confirmed-zero regions receive local negative BCE, whereas unknown partial-annotation zeros remain excluded by validity. No implementation.

- 2026-09-10: Clarify the user's GT-region-only classification question: single-positive restricted OR equals local positive BCE, whereas multi-positive restricted OR is weaker than supervising each confirmed positive. Record the distinction without silently replacing the loss design or claiming strict pixel isolation. Design only.

- 2026-09-10: Latest user review removes Baseline 0 loss-parity constraint. Revise the serial region-MIL proposal to observation-specific mixed supervision with 4/4/8 positive-heavy sampling, per-bag summed observed-cell BCE or weak OR, initial beta=1, no source importance correction or extra pos_weight, and no redundant whole-positive term when region-positive GT is observed. Record partial-GT likelihood handling, rotating data queues, and matched weak/no-weak validation. Design only; no model implementation or training.

- 2026-09-10: Incorporated the user's three-source sampling and Baseline 0
  pos_weight requirements. Replaced class-balanced whole BCE with corrected
  positive-2/negative-1 weighted risk, proposed 4/4/8 batches and rotating queues
  around one GT-pass, specified a separate observed-cell GT reduction, and
  documented plane-vs-bag loss and BatchNorm limits on parity. CNN remains
  trainable; pretrained BN-stat retention is a proposal. No code was implemented.
- 2026-09-10: User changed the target architecture to end-to-end fine-tuned
  region-to-whole MIL with negative vertebrae. Added REGION_MIL_DESIGN.md as the
  current design, marked the conditional positive-only document historical,
  and specified noisy-OR aggregation, negative-loss equivalence, all-bag whole
  supervision plus positive observed GT, one-checkpoint inference, and matched
  comparisons. CNN freezing and the parallel whole classifier are superseded.
  Model code and training remain untouched.
- 2026-09-10: Recorded user confirmation of plane/mask fracture coverage and
  clarified that the frozen-CNN pilot is optional and not yet agreed. Its weak
  supervision affects only the trainable region branch; CNN-level regularization
  requires a matched comparison that also updates the CNN.
- 2026-09-10: Reviewed the design-only conditional MIL proposal. Replaced the
  unsupported feature-only OR default with ordinary region-path GT+OR, explained
  why head detachment cannot fix score semantics, and specified a small shared
  region head, frozen-encoder pilot, count-normalized step losses, ramp/weight
  candidates, matched GT exposure, and nested model selection. Corrected the
  degenerate shared-head/global-pool comparison. No runtime implementation or
  model training was performed; the detailed settings remain review proposals.
- 2026-09-09: Refined the conditional-MIL proposal after reviewing mixed weak/
  strong supervision evidence. The leading arm now shares the scalar region
  classifier, ramps noisy-OR from zero, and detaches classifier parameters on
  weak examples so OR regularizes local features rather than directly shifting
  head biases. The clean first pilot freezes the fold-matched Baseline 0 trunk,
  uses every positive training bag once with GT/weak stratification, and compares
  GT-only against feature-only OR before full-gradient OR or consistency.
- 2026-09-09: Added the design-only positive-region MIL review in
  `fracture_detection/CONDITIONAL_MIL_DESIGN.md`. Read the supplied Fang paper
  and primary mixed/partial-supervision literature; distinguished local mask
  readout from strict input isolation and noisy-OR scores from conditional
  marginals. Audited 235 complete / 33 partial / 1,064 unannotated positive bags
  and the differing vertebral-level distributions. No model code, configuration,
  or training run was created; implementation awaits completion of design review.
- 2026-09-08: Implemented protocol v7 single-pass joint training and calibration
  v5. Removed the positive-only second encoder forward while retaining
  conditional-positive FPN/region computation, and added a BatchNorm update-count
  regression test. Existing v4 calibrations and v6 checkpoints are not reusable.
- 2026-09-08: Reframed from-start whole/region joint learning as the next matched
  experiment rather than an established replacement. Required identical
  pre-Baseline initialization and training conditions, single-pass BatchNorm,
  and an explicit whole-only control before choosing joint training or staged
  freezing.
- 2026-09-08: Verified that natural-BN whole-loss stagnation is an opposing
  class trade-off: inner positive BCE worsens while negative BCE improves.
  Shared whole-region gradients are not dominantly conflicting; both updates
  instead show poor held-out-positive generalization, so a matched `lambda=0`
  control is required before selecting gradient surgery or reweighting.
- 2026-09-08: Decomposed the epoch-71 whole gradient by class. Seen-positive
  gradients still point toward held-out-positive improvement but are much
  smaller than opposed negative gradients after positive training loss has
  saturated; this identifies the local failure direction but does not by itself
  justify class reweighting or whole-region gradient surgery.
- 2026-09-08: Corrected the whole-loss interpretation against Baseline 0 parity.
  Baseline 0 learns with the same `pos_weight=2` and itself worsens after its
  epoch-30 loss optimum; v3 begins from epoch-44 trained weights and reaches a
  lower corrected whole loss by region epoch 5. Prioritized endpoint selection
  and freezing over class reweighting, pending a matched short `lambda=0` control.
- 2026-09-08: Verified with fixed weights and paired patient-cluster bootstrap
  that the positive-only second encoder pass materially worsens validation whole
  and total loss through BatchNorm buffer updates; selected a shared natural-batch
  encoder forward as the preferred repair direction without implementing it.
- 2026-09-07: Decomposed v3 validation-total loss and confirmed extra positive-only
  encoder BatchNorm updates in saved checkpoints; recorded a fixed-parameter
  diagnostic without changing the model or claiming a verified causal effect.
- 2026-09-07: Recorded the user's correction from whole-loss improvement to total
  loss improvement as the immediate design priority, with the existing v3 total
  loss diagnostics and unresolved train/validation aggregation differences.
- 2026-09-07: Recorded the negative-inclusive CAM review, including replicated
  AP and paired confidence comparisons, missing-label caveats, corrected nested
  teacher accounting, and unresolved leakage in region checkpoint selection.
- 2026-09-05: Selected the completed `baseline0_aug追加` five-fold AUROC-best
  checkpoints as the fold-matched teachers for CAM soft pseudo-label generation
  and aligned Baseline 0 and region configuration paths with the actual run.
- 2026-09-04: Assigned the standard five-fold, 80%-trained Baseline 0 rerun the
  dual role of OOF pseudo-label teacher and whole-vertebra classification
  baseline, while retaining the current nested 60% region fine-tuning split.
- 2026-09-04: Chose the fold-matched 80%-trained Baseline 0 checkpoint, rather
  than the existing 60%-trained checkpoint, to initialize each region student;
  only the outer fold remains fully unseen by initialization.
- 2026-09-04: Reversed the region initialization choice and retained the
  fold-matched 60%-trained Baseline 0 checkpoint. The experiment prioritizes
  clean nested interpretation over absolute region accuracy; the 80%-trained
  Baseline 0 remains the pseudo-label teacher and whole-classification baseline.
- 2026-09-04: Cancelled the separate 80%-trained Baseline 0 rerun. The existing
  nested 60%-trained checkpoints now remain the pseudo-label teachers,
  whole-classification baseline, and region initialization.
- 2026-09-04: Restored Stage1-parity vertical flip and transpose augmentation
  for the next Baseline 0 and region runs, with one synchronized transform for
  CT, whole mask, and the anatomical region mask.
- 2026-09-04: Restored the former `parallel.mode: fold` launcher for Baseline 0
  and region training, configured for GPU 0 and 1 with two concurrent folds.
- 2026-09-02: Required a low-variance GT integration protocol before the
  controlled comparison; retained the fixed 268-bag human set as the mandatory
  endpoint rather than deferring evaluation because it is small.
- 2026-09-02: Rejected separate GT/soft loaders, phases, and normalized loss
  terms as a stabilization remedy after confirming that both the old separated
  pilot and current `L_E + L_P` design share the observed instability.
- 2026-09-02: Clarified that the conditional-positive unified-target candidate
  only has a gradient-variance rationale; validation-loss and validation-metric
  stability remain unverified empirical requirements.
- 2026-09-02: Researched imbalance handling after unifying GT and pseudo targets;
  selected conditional-positive macro BCE as the leading controlled candidate
  because it preserves soft-target semantics and reduced measured target-mass
  imbalance from 27.420:1 to 1.823:1. Kept its stability and accuracy explicitly
  empirical rather than guaranteed.
- 2026-09-02: Implemented the conditional-positive unified-target candidate as
  protocol v5, including leakage-safe inner pseudo targets, matched centered-loss
  checkpoint selection, and best-whole/best-region probability composition.
- 2026-09-02: Added frozen region features plus a deterministic L2-regularized
  shared linear probe as the candidate with a direct optimization-stability
  rationale; kept its accuracy benefit explicitly unverified.
- 2026-09-02: Switched the next stabilization experiment to pseudo-only region
  optimization and pseudo-loss checkpoint selection, with human regional GT
  reserved for locked outer-fold evaluation.
- 2026-09-02: Replaced the nested human-GT endpoint with a completely external
  160-study holdout. The remaining 12,321 bags undergo pseudo-only nested
  five-fold train/validation/test, and the five selected models are ensembled
  before the 268 human-labelled bags are evaluated once.
- 2026-09-02: Preserved whole-vertebra fracture classification as a separate
  true-label endpoint with its own validation checkpoint, five-fold OOF metrics,
  and five-model external-holdout ensemble.
- 2026-09-02: Identified that the completed `cam_soft` run never monitored mixed
  validation supervision: pseudo targets were absent from inner eval, all
  `val_pseudo_loss` values were zero, and region checkpoints used GT macro AP.
  Required a matched mixed-validation rerun before interpreting instability.
- 2026-09-02: Kept naive 0.5 hard pseudo-labeling unapproved after the first
  completed outer-0 CAM-soft run; recorded confidence-gated hardening as a
  possible future controlled arm only if the planned soft/no-pseudo/shuffled
  comparison fails.
- 2026-09-02: Added leakage-free fold-matched Stage2 region-only evidence as an
  unresolved alternative pseudo-label source; CAM remains the approved leading
  arm until a controlled same-target audit supports replacement.
- 2026-08-25: Added the canonical Japanese end-to-end design document for the four-region model.
- 2026-08-24: Reset the active design to Baseline 0 and removed discontinued experiment families.
- 2026-08-24: Reorganized Baseline 0 into responsibility-based directories.
- 2026-08-24: Corrected the scope: restored pseudo-label generation and CAM audit as active core functionality under `baseline0/pseudo_labeling/`.
- 2026-08-25: Fixed the two-BiLSTM region architecture and the combined exact-plus-ranking loss structure.
- 2026-08-25: Clarified that Baseline 0 `pos_weight=2.0` remains on `L_whole` only; region losses use their own balanced reductions without `pos_weight`.
- 2026-08-25: Defined the four single-region comparators as fully independent two-path models with matched per-region supervision.
- 2026-08-26: Fixed fold-matched Baseline 0 initialization and all-parameter fine-tuning with a 10x learning-rate difference between transferred and new parameters.
- 2026-08-26: Moved region calibration/training multiprocessing temporary files off NFS and added explicit calibration progress bars.
- 2026-08-26: Added config-selected calibration versions and compatibility fingerprints; migrated the completed outer-0 artifact to `calibration/v1`.
- 2026-08-26: Changed region training to sequential whole/region backward with one optimizer update per step, preserving the calibrated objective while reducing peak VRAM.
- 2026-08-26: Enabled `torch.compile(mode="default", dynamic=False)` for CUDA region training and placed its cache on local `/tmp`; calibration and CPU paths remain eager.
- 2026-08-27: Renamed the region source package from `data/` to `data_pipeline/` so its Python modules are not swallowed by the repository-wide dataset ignore rule.
- 2026-08-27: Added elapsed-time startup logs and an explicit first-epoch warning for DataLoader worker startup and TorchInductor compilation.
- 2026-08-27: Fixed the epoch-4 non-finite region loss by performing sigmoid, masked averaging, clamping, and logit conversion for region bags in float32.
- 2026-08-28: Retired cross-case CAM-density ranking after its failed outer-0 pilot and reopened the region-loss design; CAM remains an audited optional signal rather than an active target.
- 2026-08-28: Narrowed the replacement loss to a partial-label protocol informed by primary literature; pseudo supervision is permitted only after target-free whole-positive constraints improve on the exact-only reference.
- 2026-08-28: Corrected the replacement protocol after rejecting its circular exact-only-teacher premise; whole-positive data now enters from the first region epoch through target-free constraints, with pseudo labels treated as later joint latent estimates.
- 2026-08-28: Reopened CAM pseudo-labeling while keeping cross-case ranking retired; CAM density is now converted to independent per-case soft targets by train-fold-only human calibration and trained with an auxiliary soft-BCE term.
- 2026-08-28: Removed logical-OR from the initial CAM-soft objective; whole-positive any-region probability remains a monitoring metric only.
- 2026-08-31: Fixed the hard-GT / soft-pseudo target split and retired the cross-case ranking term from the region objective.
- 2026-08-31: Extended Baseline 0 `pos_weight=2.0` to the region exact loss and ruled out focal loss, class weighting, and balanced samplers, after measuring region imbalance at 2.14:1-3.12:1 with R4 majority-positive.
- 2026-08-31: Separated whole and region checkpoint selection and moved GT health monitoring from validation BCE to validation AP/AUROC.
- 2026-09-01: Retired the per-region CAM calibrator in favour of the within-bag normalised share scaled by a constant, after measuring a systematic posterior misdirection in every fold.
- 2026-09-01: Added four-view test-time augmentation to pseudo-label generation using only the fold-matched teacher, recovering most of a multi-teacher ensemble without breaking fold matching.
- 2026-09-01: Closed the cardinality question against a pre-registered threshold and confirmed the area-correction exponent stays at one.
- 2026-09-01: Reopened the final pseudo-label construction decision; within-bag share and four-view TTA are empirical candidates rather than a frozen generation contract.
- 2026-09-01: Added an unapproved factorized pseudo-label candidate: CAM supervises a shared conditional-location auxiliary head, while binary region heads remain supervised only by hard human labels and logical whole-negative zeros.
- 2026-09-01: Rejected the auxiliary-location-only candidate. Unannotated whole-positive CAM targets must directly supervise the endpoint region heads, using a target-preserving soft-label extension of `pos_weight=2.0` rather than naive native weighted BCE.
- 2026-09-01: Audited actual identity-CAM probabilities with complete-only patient-grouped calibration; selected shared logit-share Platt calibration provisionally, extended CAM supervision to partial-label unknown cells, and kept four-view coefficient freezing blocked until GPU regeneration.
- 2026-09-01: Removed the pseudo-loss coefficient and target-preserving weighting candidate; CAM supervision now uses plain soft BCE, while `pos_weight=2.0` remains confined to hard exact region targets and the whole path.
- 2026-09-01: Approved the CAM-soft implementation direction for the next session and recorded the phased implementation, validation, artifact-gating, and rollback plan; only the GPU-derived four-view coefficients remain unfrozen.
