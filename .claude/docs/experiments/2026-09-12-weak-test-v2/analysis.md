# weak/test_v2 outer0 result review

## Run status

- Output: `fracture_detection/weak/outputs/09_12/test_v2/outer0`
- Aggregation: normalized logit-LSE, tau=0.5
- Batch N/A/U: 8/4/4
- Effective early-stopping patience: 10 GT-passes
- Selected pass: 21; training stopped at pass 31
- Checkpoint selection: inner annotated-positive region macro AP

`fold_metrics.json.best_metrics` contains inner-validation metrics, not outer-test
metrics. Outer metrics below were recomputed from `outer_predictions.csv` using
the same definitions as the v1 audit.

## Main comparison

| Endpoint | Baseline 0 | v1 noisy-OR, 4/4/8 | v2 LSE, 8/4/4 |
|---|---:|---:|---:|
| Outer whole AP | 0.7726 | 0.7284 | 0.7405 |
| Outer whole AUROC | 0.9186 | 0.9084 | 0.9160 |
| Outer whole BCE | 0.1733 | 0.4619 | 0.1740 |
| Outer precision at top 262 | 0.7061 | 0.6565 | 0.6832 |
| Outer conditional region macro AP | - | 0.7766 | 0.7717 |
| Outer conditional region mean BCE | - | 0.4616 | 0.4966 |

The outer set contains 2,671 vertebra bags with 262 whole-positive bags. The
conditional localization endpoint contains only 56 annotated-positive bags.

## Region breakdown

| Region | v1 AP | v2 AP | v2 - v1 |
|---|---:|---:|---:|
| R1 | 0.8362 | 0.8682 | +0.0319 |
| R2 | 0.7368 | 0.7735 | +0.0368 |
| R3 | 0.6528 | 0.5899 | -0.0629 |
| R4 | 0.8806 | 0.8551 | -0.0255 |
| Macro | 0.7766 | 0.7717 | -0.0049 |

The macro change is small relative to the 56-bag endpoint and is mixed across
regions. It does not support a general localization improvement.

## Probability behavior

- Negative outer mean whole score fell from 0.2913 to 0.0670.
- Negative outer 90th percentile fell from 0.7032 to 0.1717.
- Negative bags with `p_whole >= 0.5` fell from 464 to 75.
- Whole BCE recovered from 0.4619 to 0.1740, essentially matching Baseline 0
  at 0.1733.
- Whole ranking improved over v1 (AP +0.0121, AUROC +0.0076), but AP remains
  0.0321 below Baseline 0. AUROC is only 0.0026 below Baseline 0.

Thus the combined LSE plus N/A/U=8/4/4 change primarily fixed v1's whole-score
inflation and calibration. It improved ranking partially but did not surpass
the dedicated Baseline 0 whole classifier.

## Why whole AP remains below Baseline 0

AP depends on the ordering of positive and negative bags, not absolute
calibration. The v2 change greatly reduced score magnitude without fully fixing
the top-ranked ordering. Baseline 0 has 99 positives in the top 100 predictions,
whereas v2 has 94; at the top 262, the counts are 185 and 179. These relatively
few high-ranked false positives have a large AP cost at 9.8% prevalence while
having a much smaller effect on AUROC over all positive-negative pairs.

Re-aggregating the same saved v2 region logits does not close the gap: outer AP
is 0.7387 with max pooling, 0.7341 with noisy-OR, 0.7405 with LSE tau=0.5, and
0.7426 with LSE tau=1.0, versus Baseline 0 at 0.7726. These exploratory
post-hoc aggregations do not close the gap. They neither establish an information
limit of the logits nor test what retraining at another tau would achieve.
Outer results must not be used to select tau for a confirmatory comparison.

The training and selection protocol also prioritizes localization rather than
whole AP. The selected pass 21 has inner whole AP 0.6992, while inner whole AP
peaks at 0.7179 on pass 28 after region macro AP falls from 0.7641 to 0.7220.
In addition, v2 presents 160 U bags per pass and selected after 21 passes (3,360
U presentations), compared with 320 per pass and pass 35 in v1 (11,200 U
presentations). This is a plausible reason for limited whole-ranking learning,
but it is confounded with aggregation, negative exposure, and stopping time.

### Mechanistic hypotheses and limits of the evidence

Only Baseline 0's encoder is transferred. Its trained whole BiLSTM and classifier
are discarded, and the FPN, region BiLSTM, and region classifier start randomly.
Consequently, Baseline 0's whole ranking is not preserved by initialization.
At the selected pass 21, the new path has received 21 passes over the 159
annotated training bags, 5.23 passes over weak positives, and only 0.924 passes
over negatives. These are sampler presentations before augmentation-related
drops, not counts of successfully optimized examples. A lack of repeated
negative exposure is a plausible contributor to high-ranked false positives;
the transferred encoder has already seen training negatives in Baseline 0.

Checkpoint selection is based exclusively on annotated-positive region macro
AP. False positives on whole-negative vertebrae cannot directly affect that
selection metric. Thus improving it does not ensure that whole AP improves.
The observed pass-21/pass-28 tradeoff supports this mismatch but does not show
that selecting pass 28 would improve outer AP.

For weak-positive bags, the logit derivative is
`(p_whole - 1) * softmax(region_logits / tau)`. A currently high-scoring region
receives the largest upward update even when it is not a true fracture region;
the weak label cannot specify the correct location. Local GT and negatives
can oppose this effect. An exploratory check on the 56 outer annotated bags
finds that the highest-scoring region is GT-negative in 12 bags. If these bags
were supervised only by the U loss, the mean fraction of logit-gradient weight
on GT-negative regions would be 0.2333. Actual A training uses local BCE, and
this diagnostic cannot measure erroneous localization in U bags without GT.

Train weak loss falls from 1.2319 at pass 1 to 0.8257 at pass 21, while inner
weak loss is 1.1135 and 1.1199 respectively. The training improvement therefore
does not transfer comparably to held-out weak positives. This is consistent
with limited generalization, not proof of one architectural or sampling cause.
Neither a representation plateau nor the necessity of distillation has been
established by this single run.

## Inner selection behavior

- v2 selected pass 21 at inner region macro AP 0.7641; v1 selected pass 35 at
  0.7780.
- v2 inner whole AP at the selected pass was 0.6992. Its best whole AP was
  0.7179 at pass 28, when region macro AP had fallen to 0.7220.
- No region-macro-AP improvement occurred for the next 10 passes, so the run
  stopped at pass 31 under its effective patience of 10.

This confirms a checkpoint-selection tradeoff: optimizing the required region
endpoint does not select the pass with the best inner whole classification.

## Weak-label interpretation

On outer weak-positive bags, median `p_whole` fell from 0.9749 to 0.8017 and the
fraction above 0.95 fell from 59.7% to 21.8%. This is consistent with the desired
removal of noisy-OR score inflation, not by itself evidence that weak supervision
failed. However, v2 also halves U exposure and doubles N exposure, so aggregation
and sampling effects are confounded. The run cannot identify the incremental
benefit of weak labels. A matched `beta=0` arm with the same LSE, tau, 8/4/4
batches, inputs, denominator, and checkpoint selection remains the required
test.

## Conclusion

The v2 run substantially improves whole-score behavior and nearly restores
Baseline 0 AUROC/BCE while preserving approximately the same outer localization
macro AP as v1. It does not improve localization overall, and whole AP still
lags Baseline 0. The next highest-information experiment is the matched beta=0
control; changing more factors before that would leave the weak-label effect
unresolved.
