# Validation total loss and BatchNorm diagnosis

Date: 2026-09-07

Status: History and implementation inspected, followed by fixed-weight BN
intervention experiments on physical GPU 0 only. No training or model edits.

## Which component accounts for the validation behavior?

For v3 outer 0, reconstruct validation total as
`val_whole + 0.3156107231711445 * val_region_loss`.

| Epoch | Training total | Validation whole | Weighted validation region | Validation total |
|---|---:|---:|---:|---:|
| 5 | 0.349396 | 0.314235 | 0.181509 | 0.495744 |
| 51 | 0.289076 | 0.358394 | 0.169325 | 0.527719 |
| 60 | 0.299804 | 0.316332 | 0.171875 | 0.488207 |
| 71 | 0.291344 | 0.433346 | 0.171905 | 0.605251 |

From epoch 5 to 71, whole increases by 0.119111, weighted region decreases by
0.009604, and total increases by 0.109507. Whole dominates the observed total
fluctuations (Pearson correlation 0.9874). This is a decomposition, not evidence
of a particular cause. Region target entropy remains 0.404084; its fixed floor
cannot explain epoch-to-epoch fluctuations. Validation total is not uniformly
nondecreasing: its observed minimum is epoch 60.

Existing inner pseudo targets have the previously documented outer-access
problem. Training and validation also use different reductions and augmentation,
so their absolute totals are not directly comparable.

## Confirmed additional BatchNorm updates

`region_branch/training/trainer.py:529` runs the whole path on the natural batch
in training mode. On non-mixup steps containing positive bags, line 556 runs the
encoder again on just those positive bags. The backbone is not frozen in this
run, so both passes update its BatchNorm running statistics.

In addition, `region_branch/modeling/model.py:183` requests
`forward_intermediates(..., intermediates_only=False)` even for a region-only
forward. The installed timm EfficientNet implementation consequently executes
`conv_head` and `bn2`, although their final output is unused by the region path.
Thus even the whole-specific encoder `bn2` receives the positive-only update.
Loss weighting does not scale these forward-time buffer updates.

Actual `num_batches_tracked` values from the local checkpoints:

| Checkpoint | Epoch | Encoder bn1 | Encoder bn2 | Whole head BN | Region head 0 BN |
|---|---:|---:|---:|---:|---:|
| Initial teacher | 44 | 22220 | 22220 | 22220 | Not present |
| v3 best whole | 5 | 26414 | 26414 | 24745 | 1669 |
| v3 last | 71 | 81469 | 81469 | 58075 | 23394 |

At epoch 71 there were 35855 optimizer steps. Each encoder BN has advanced by
59249 = 35855 + 23394 updates relative to initialization; the extra 23394 match
region head forward counts. The whole-head BN advanced by only 35855 updates.
These counters confirm that the extra-update path occurred in the saved run,
not merely that the current source permits it.

## Interpretation and next diagnostic

Training normalizes with current-batch statistics, whereas validation uses
running statistics. Repeated small positive-only passes change the population
represented by those running statistics and are a concrete candidate for
validation instability. The counters prove the mechanism exists, not that it
explains the measured loss increase or how large its effect is.

The targeted diagnostic keeps one saved model's parameters fixed and compares
encoder BN buffers recalibrated on a fixed natural-distribution training subset.
Dropout and non-encoder modules remain in evaluation mode. All comparisons use
the same inner set; the outer test is not used. This cannot repair the separate
inner-target leakage.

If this intervention explains a material part of the instability, consider
sharing one encoder forward between the tasks or preventing positive-only
recomputation from changing encoder BN buffers. Verify memory/compile behavior
and both endpoints before adopting a change. Ordinary overfitting, interference
between task gradients, and teacher-target generalization remain possible;
none had been isolated as the dominant cause at the inspection stage.

## Fixed-weight intervention results

The diagnostic used v3 outer-0, physical GPU 0, one process, and at most 35% of
that GPU's memory. Model parameters were protected by an SHA-256 digest before
and after every condition. All conditions used the same 1,024 training bags
(92 whole-positive), the same 2,687 inner-validation bags from 401 studies, the
same targets and masks, and no outer-test data. Encoder BN buffers were reset and
estimated with cumulative averages for this controlled probe.

At epoch 71:

| BN condition | Whole loss | Region loss | Total loss | Whole AUROC | Whole AP |
|---|---:|---:|---:|---:|---:|
| Saved buffers | 0.433338 | 0.544688 | 0.605248 | 0.897131 | 0.716727 |
| Natural batches, one encoder pass | 0.266002 | 0.548672 | 0.439169 | 0.899040 | 0.719157 |
| Positive bags only | 0.566799 | 0.555350 | 0.742073 | 0.883768 | 0.678030 |

Natural recalibration also improved the epoch-5 best-whole checkpoint's total
loss from 0.495807 to 0.447178. The paired patient-cluster bootstrap total-loss
difference was -0.05735, 95% CI [-0.08481, -0.03075]. At epoch 71 it was
-0.18212, 95% CI [-0.23109, -0.13537]. Thus the finding is present early and is
larger late in training.

### Direct positive-second-pass control

To isolate the suspected mechanism from generic BN recalibration, the epoch-71
weights and the same natural batches were used in two conditions:

1. one full-batch encoder pass per batch;
2. the same full-batch pass followed by a second encoder pass over that batch's
   whole-positive bags.

Only BN buffers could change. Adding the positive second pass changed pooled
whole loss by +0.13289, 95% CI [+0.09463, +0.17189], and pooled total loss by
+0.13292, 95% CI [+0.09441, +0.17209]. Region loss changed by +0.000079,
95% CI [-0.00420, +0.00439]. The historical batch-averaged evaluator similarly
gave total loss 0.439169 versus 0.559526.

This directly verifies that positive-only second-pass BN updates can cause a
large validation-total degradation with weights, examples, and targets fixed.
The probe uses cumulative BN averages and performs the second pass whenever a
sampled batch contains a positive bag; the training run used momentum 0.1 and
skipped region on mixup steps. Therefore the probe establishes the direction and
material capability of the mechanism, not the exact fraction of the historical
epoch-71 loss attributable to it.

Recalibrating only `encoder.bn2` reduced epoch-71 total loss by 0.04919,
95% CI [-0.09486, -0.00450], but reduced ranking metrics. This confirms that
`bn2` is affected while also showing that changing it alone is not the proper
repair. The structural fix should prevent the second encoder pass from updating
encoder BN statistics, preferably by sharing one natural-batch encoder forward.

## Why natural-BN whole loss still does not improve

Natural-BN recalibration fixes a large measurement/runtime-buffer distortion but
does not make the whole objective improve from epoch 5 to epoch 71. On the same
2,687 inner-validation bags, pooled `pos_weight=2` whole BCE changed from
0.274383 to 0.278497, a paired patient-cluster delta of +0.004114 with 95% CI
[-0.009129, +0.018590]. The historical batch-averaged evaluator similarly gave
0.264452 and 0.266002. The corrected whole loss is therefore statistically flat,
not decreasing.

The pooled endpoint difference decomposes into a significant class trade-off:

| Whole target | Epoch 5 BCE | Epoch 71 BCE | Delta | Patient-cluster 95% CI |
|---|---:|---:|---:|---:|
| Positive | 0.873297 | 1.023179 | +0.149882 | [+0.082728, +0.222101] |
| Negative | 0.141676 | 0.113491 | -0.028185 | [-0.035128, -0.021257] |

With 268 positive and 2,419 negative bags, `pos_weight=2` assigns only 18.14% of
the pooled denominator mass to positives. The positive deterioration contributes
+0.027187 to the total delta, while the negative improvement contributes
-0.023072, leaving the observed +0.004114. Mean positive bag score is essentially
unchanged (0.65667 to 0.65566), but the difficult positive tail becomes more
confidently wrong: among the 45 epoch-5 positive bags with score 0.1-0.5, mean
positive BCE rises by 0.48738. Across all positive bags, 57.5% have higher BCE at
epoch 71.

### Shared-gradient probe

A separate fixed-weight probe tested whether whole-region gradient conflict is
the main explanation. It used only physical GPU 0, one process, at most 35% GPU
memory, no optimizer updates, natural-BN recalibration on the same 1,024 bags,
and 64 fixed no-augmentation training batches of eight unique-study bags each.
Each batch contained one whole-positive and seven whole-negative bags, matching
the approximately 12.5% positive share of natural batches conditional on the
region path running. Gradients were measured over the actually shared encoder
from the stem through `blocks[4]`. Separate 256-bag positive and negative inner
validation subsets provided class-conditional whole gradients; outer data was
not used.

Whole and weighted-region training gradients were aligned in aggregate rather
than conflicting: cosine was +0.243 at epoch 5, +0.193 at epoch 51, and +0.326 at
epoch 71. The weighted region-gradient norm was only 10.3%-11.3% of the whole
gradient norm. Against the ordinary mixed validation gradient, the region
component was helpful at epoch 5 and weakly harmful at epochs 51/71, but its
batch-bootstrap directional-effect CI included zero late. The combined
first-order direction still reduced mixed validation whole loss at every
checkpoint. This does not support whole-region conflict as the primary cause of
the plateau.

The class-conditional result does identify the immediate failure mode. At epochs
5/51/71, the whole training gradient cosine with held-out positive whole loss was
-0.975/-0.958/-0.960, while its cosine with held-out negative whole loss was
+0.952/+0.924/+0.904. The region gradient had the same, smaller pattern:
-0.289/-0.217/-0.335 for positives and +0.307/+0.281/+0.398 for negatives. Thus
the observed updates are strongly oriented toward easier negative improvement
and away from held-out positive generalization. Region training can reinforce
this failure, but it is not its dominant gradient component; the whole objective
itself already has the adverse positive-generalization direction.

An epoch-71 decomposition of the whole gradient separates underweighting from a
wrong positive signal. The mean training-positive whole gradient is aligned with
held-out positive loss (cosine +0.735), whereas the training-negative gradient is
almost exactly opposed to it (cosine -0.966). Training-positive BCE on the probe
is already 0.0110 versus 1.0204 on held-out positives, and its shared-gradient
norm is 0.359 versus 1.303 for the training-negative mean. The positive signal
therefore points in a useful direction but is small after the seen positives have
become easy; the negative update dominates and damages unseen positives. Merely
setting `pos_weight` to the approximately 9:1 count ratio is not guaranteed to
balance these gradient magnitudes, and the much larger local coefficient implied
by this one checkpoint must not be adopted without a controlled sweep.

### Baseline 0 parity correction

The class decomposition does not establish that `pos_weight=2` is the root
design defect. Baseline 0 uses the same whole architecture, natural batch size,
mixup probability, `pos_weight=2`, train/inner split, and whole loss, and it does
learn: validation loss falls from 0.447470 at epoch 1 to its minimum 0.269971 at
epoch 30. It also exhibits the same post-optimum pattern, rising to 0.288257 at
the AUROC-best epoch 44 and 0.338194 at epoch 50 while training loss continues
to fall.

The v3 region run is not a fresh whole-model training run. It initializes the
entire encoder/whole path from that epoch-44 `best_val_auroc` Baseline 0
checkpoint, resets the optimizer, uses a 10x lower transferred-parameter learning
rate, and then continues for up to another 71 epochs under region-loss stopping.
After natural-BN correction, its epoch-5 whole loss is 0.264452, already below
Baseline 0's observed 0.269971 minimum. Whole learning therefore occurred; the
epoch-5 to epoch-71 comparison is post-convergence continuation, not evidence
that the Baseline-compatible whole objective cannot learn.

This is a first-order fixed-checkpoint diagnostic, not a replacement for a
training ablation: it freezes BN and dropout, omits augmentation and mixup, and
does not model AdamW moments. Do not change class weighting first. The direct
repair candidate is to remove the BN double update, choose or briefly fine-tune
the whole endpoint by whole validation loss, then freeze the encoder and whole
path while region-only modules continue. A matched short `lambda=0` continuation
from the same Baseline checkpoint remains the control for any claim that region
gradients, rather than ordinary post-convergence fine-tuning, damage whole
generalization. Only if that control diverges should gradient constraints or
class-weight changes be promoted.

Artifacts:

- `.tmp/bn_validation_probe_20260907.py`
- `.tmp/bn_validation_probe_summary_20260907.py`
- `.tmp/bn_validation_probe_20260908_direct/results.json`
- `.tmp/bn_validation_probe_20260908_direct/paired_bootstrap.json`
- `.tmp/bn_validation_probe_20260908_best_whole/results.json`
- `.tmp/bn_validation_probe_20260908_best_whole/paired_bootstrap.json`
- `.tmp/bn_validation_probe_epoch5_vs_71_paired.json`
- `.tmp/bn_validation_positive_plane_dispersion_epoch5_vs_71.json`
- `.tmp/gradient_conflict_probe_20260908.py`
- `.tmp/gradient_conflict_probe_20260908_b8_v2/results.json`
- `.tmp/gradient_conflict_probe_20260908_epoch5_51_classwise/results.json`
- `.tmp/gradient_conflict_probe_20260908_epoch71_classwise/results.json`
- `.tmp/gradient_conflict_probe_20260908_epoch71_train_classwise/results.json`

## 2026-09-08 implementation

Protocol v7 removes the confirmed positive-only second encoder pass. On every
non-mixup step, the full natural batch now passes through the encoder once; the
whole path uses every bag, while only selected whole-positive intermediate
features enter the FPN and region modules. The two losses are summed and
backpropagated once. Lambda calibration v5 measures both shared-trunk gradient
norms from that same forward, so older calibration artifacts must not be reused.
Regression tests assert that an encoder BatchNorm counter advances exactly once
per non-mixup natural step. This implementation fixes the verified normalization
defect; it does not by itself establish that from-start joint training beats
staged training or that validation whole loss will continue decreasing after its
optimum.
