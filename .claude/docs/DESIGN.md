# Project Design Document

> Active design decisions only. Historical designs are available from Git history.

The canonical Japanese overview for the next four-region model is
`.claude/docs/REGION_MODEL_DESIGN_JA.md`.

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
