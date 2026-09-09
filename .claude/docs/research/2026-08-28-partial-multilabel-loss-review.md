# Partial Multi-Label Loss Review for the Region Branch

Date: 2026-08-28

## Research Status

Gemini CLI was requested first, but the configured organization account requires
`GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_PROJECT_ID` and produced no research
output. The review below was therefore completed by directly reading primary
conference papers and official proceedings.

## Project-Specific Label Structure

For outer fold 0, the region-training pool is not an ordinary missing-label
dataset:

- 159 human-annotated bags are all whole-positive; 596 of 636 region cells are
  observed.
- 144 bags have all four region labels. Their mean positive cardinality is
  1.410, and their conditional positive rates are R1 0.299, R2 0.215, R3 0.285,
  and R4 0.611.
- 7,272 whole-negative bags have a logically exact all-zero region vector.
- 643 whole-positive bags have an unknown region vector but satisfy
  `OR(region_1, ..., region_4) = 1`.

This is a hybrid of semi-supervised multi-label learning and constrained partial
multi-label learning. It is not pure positive-unlabeled learning, because exact
positive and negative region cells are both available. It is also not the usual
missing-at-random setting, because the unlabeled pool is conditioned on whole
positivity.

## Primary Literature

| Work | Main idea | Applicability here |
|---|---|---|
| [Durand et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.html) | Partial-BCE masks unknown cells and normalizes by the known-label proportion. | The current human exact term already performs the important masking. Only 6.3% of human cells are missing, so normalization cannot explain or fix the observed failure. |
| [Cole et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html) | Expected-positive regularization and ROLE constrain or estimate missing labels when only one positive is known. | The expected-cardinality idea is useful, but the single-positive/no-confirmed-negative assumption does not match this dataset. |
| [Ridnik et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) | ASL suppresses easy-negative gradients and treats positive and negative errors asymmetrically. | Useful only as a secondary imbalance ablation. It does not solve unknown labels, excessive human reuse, or plane-level MIL mismatch. |
| [Ben-Baruch et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.html) | Class-aware selective loss chooses Ignore or Negative treatment for each unknown class using estimated class distributions and asymmetric weighting. | Supports class-specific handling and rejects blindly treating every unknown as negative. Directly copying its global prior assumptions would be unsafe because the unlabeled pool is whole-positive only. |
| [Kim et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Large_Loss_Matters_in_Weakly_Supervised_Multi-Label_Classification_CVPR_2022_paper.html) | Assume unknown labels are negative, then reject or correct large-loss cells before memorization. | Its memorization diagnosis matches the pilot. The training assumption is a poor default here because every whole-positive bag is guaranteed to contain at least one false negative under Assume-Negative. |
| [Zhou et al., ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840418.pdf) | Treat unknown labels as unknown, then use entropy maximization and self-paced asymmetric pseudo-labeling. | The asymmetric, self-paced treatment is relevant; the single-observed-positive setup is not. |
| [Xie et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/751ef1e7f557a8a88f0837b61bf5070f-Abstract-Conference.html) | Label-aware global consistency recovers potential positives without assuming a known positive count. | Supports per-case consistency rather than cross-case CAM ranking, but its graph machinery is unnecessary for four region labels. |
| [Xie et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5195825ee60d7efc1e42b7f3f3137040-Abstract-Conference.html) | CAP assigns positive and negative pseudo-labels with class-aware thresholds so their distribution follows estimates from the labeled subset. | The closest general method. It should be applied within the whole-positive stratum, using train-fold-only conditional priors and confidence filtering. |
| [Zhang et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.html) | COMIC combines class-aware correction, multi-focal reweighting, and head-tail balancing. | Over-complex for four classes whose conditional rates are 0.215--0.611. It is not an appropriate first repair. |
| [Ihler et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Ihler_Distribution-Aware_Multi-Label_FixMatch_for_Semi-Supervised_Learning_on_CheXpert._CVPRW_2024_paper.html) | Medical multi-label FixMatch masks missing supervised labels, uses weak/strong consistency, selects both confident positive and negative pseudo-labels, and applies distribution alignment. | Strong medical precedent for an EMA/teacher per-case consistency loss. Alignment must target whole-positive conditional priors, not a uniform population distribution. |
| [Xie and Huang, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11644) | Partial multi-label learning assumes a candidate label set contains at least one true label and disambiguates candidate confidences. | The whole-positive pool supplies exactly an at-least-one constraint, but the candidate set is always all four regions, so the logical constraint alone cannot identify a region. |
| [Dong et al., CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Dong_Revisiting_Vicinal_Risk_Minimization_for_Partially_Supervised_Multi-Label_Classification_Under_CVPRW_2022_paper.html) | MixUp-PME adapts vicinal risk minimization to scarce, partially labeled chest radiographs. | Relevant as a later regularization experiment, not as the replacement supervision mechanism. The reported behavior is class- and hyperparameter-sensitive. |

## Recommended Objective

Do not pretrain an exact-only region teacher. That objective is already known
to overfit, so using its confident predictions as pseudo labels is circular.
Use the independently trained whole-fracture Baseline 0 CAM as the pseudo-label
source, but calibrate its raw density against train-fold human regional labels:

```text
L_region = L_GT + beta_N * L_N
         + mu(t) * L_CAM
```

- `L_GT`: masked BCE on observed human cells, normalized by observed cells and
  balanced across regions. Unknown cells contribute zero loss.
- `L_N`: exact all-zero BCE on sampled whole-negative bags. It is a supporting
  regularizer, not half of the objective; start with `beta_N` around 0.1--0.25.
- `L_CAM`: region-wise normalized soft BCE. For each region, a regularized
  monotone calibrator maps `log(CAM density)` to a probability using only the
  student's train-fold human labels; the target is then applied to
  region-unlabeled whole-positive bags.
The pseudo-label loss must be normalized by valid cells within each region and
its contribution must remain auxiliary to the exact human term. The primary
target is a calibrated soft probability because hard thresholds produced
almost no confident R2/R3 positives in the outer-0 train-fold probe. Do not
force an argmax hard positive. In the initial experiment, logical OR is not a
loss; monitor `1 - product_r(1 - p_r)` on whole-positive bags without
backpropagating it.

## Required Training Sequence

1. Fit four fold-specific CAM calibrators using only human labels from the
   student's three training folds; audit them with patient-grouped
   cross-validation.
2. Generate independent soft probabilities for the region-unlabeled
   whole-positive bags; retain provenance and let human labels take precedence.
3. Train masked GT, sampled exact negatives, and an epoch-1 auxiliary CAM
   soft-BCE term with a gradual ramp from a fixed one-human-pass region epoch.
4. Keep cross-case CAM-density ranking disabled. The CAM signal is now used as
   a per-case calibrated target, which is the form supported by the closest
   literature.

Plane-to-bag aggregation is an independent MIL problem. Missing-label methods
cannot repair the current mean-over-planes positive-bag pressure, so top-k or
another localized pooling rule must be evaluated separately.

## Acceptance Criteria

- Calibrator AUROC, AP, Brier score, slope/intercept, and soft-target
  distribution are reported per region using train-fold grouped validation.
- CAM soft-BCE improves inner-human macro AP over an otherwise identical
  `mu=0` arm, not merely whole loss or teacher agreement.
- A within-region shuffled-CAM negative control does not reproduce the gain.
- Fold-specific priors are estimated from training folds only.
- Outer folds remain untouched until the inner-fold design is fixed.
