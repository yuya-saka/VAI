# Mixing CAM-Derived Pseudo Labels with Exact Labels

Date: 2026-08-27 (updated 2026-08-28)  
Scope: Evidence relevant to `fracture_detection/region_branch/outputs/08_26_region_branch_all/test_v1`

## Executive conclusion

There are successful precedents in which pseudo-labeled or weakly labeled samples greatly
outnumber exact labels. The closest CAM-derived example is PseudoSeg: 92 pixel-labeled VOC
images were combined with approximately 10,490 additional images, and mIoU improved from
56.03 to 71.22. The closest fracture examples are a thighbone detector using about 1% box
labels and 99% teacher-pseudo-labeled images (mAP 6.1 to 22.2), and a chest X-ray detector
using 808 expert-positive boxes and 5,984 teacher-pseudo-positive maps (rib AUROC 0.9025 to
0.9318 and FROC 0.7267 to 0.8914 versus supervised pre-training).

The important distinction is that pseudo labels dominate the *corpus size* in these studies,
but usually do not receive uncontrolled dominance in every optimizer step. PseudoSeg sampled
equal labeled and additional-data batches despite an approximately 114:1 corpus ratio;
ComWin likewise used balanced labeled/unlabeled mini-batches despite 19:1 to 22:1 corpus
ratios. Successful methods also refined, calibrated, filtered, or dynamically updated the
pseudo targets.

The successful literature consistently does four things:

1. retains an exact-label-only baseline;
2. repeatedly oversamples or explicitly weights the scarce exact labels;
3. refines, calibrates, or confidence-controls CAM-derived targets; and
4. evaluates the pseudo-label increment on held-out exact labels.

No directly analogous primary study was found that distils *inter-case rankings of regional
CAM density* into the same multi-label heads that receive exact binary regional labels. The
current rank-distillation design is therefore an experimental extrapolation from CAM-mask
semi-supervision, not an established recipe.

## What the current run shows

The run completed only outer fold 0, so it cannot establish a five-fold performance result.
Its best validation checkpoint was epoch 4. From epoch 4 to epoch 24:

- training human-label loss fell from 0.349 to 0.101;
- validation human-label loss rose from 0.542 to 1.030;
- student-teacher Spearman correlations increased from 0.615-0.754 to 0.826-0.890;
- the collapse alarm never fired.

This is more consistent with exact-label overfitting plus increasing imitation of the CAM
teacher than with four-head collapse. It does not show that CAM supervision helped the
held-out human endpoint.

The fixed pseudo coefficient also changes its *effective* role during training. With
`alpha=0.5202`, the pseudo term accounts for approximately:

- epoch 1: `alpha * L_rank / (L_exact + alpha * L_rank) = 55%`;
- epoch 4: 66%;
- epoch 24: 85%.

These are loss-value shares, not gradient shares, but they show why a coefficient calibrated
only at initialization does not guarantee that pseudo supervision remains auxiliary later.

## Pseudo-majority success cases

| Study | Supervision mixed | Main result | Relevance |
|---|---|---|---|
| Zou et al., PseudoSeg, ICLR 2021 | 92 pixel masks plus approximately 10,490 unlabeled or image-level-labeled images (about 114:1 additional:GT); self-attention Grad-CAM fused with decoder predictions | Supervised mIoU was 56.03, self-training 64.20, calibrated PseudoSeg 67.06, and PseudoSeg with image labels 71.22. | Closest CAM-derived pseudo-majority example. CAM was not used raw: calibration, soft labels, fusion, and strong augmentation were central. Labeled and additional samples were nevertheless balanced 1:1 per batch. |
| Lee et al., AdvCAM, CVPR 2021 | 1.5K exact masks plus 9.1K image labels converted to refined CAM masks | Semi-supervised mIoU was 77.8 versus 73.2 for the CCT baseline. Raw CAM seed quality was 48.0; AdvCAM increased it to 55.6 and the refined pseudo mask reached 68.0. | Shows that pseudo-label quality/refinement can be more important than merely adding more CAM-labeled samples. |
| Wei et al., thighbone fracture localization, 2022 | At the 1% split, about 35 box-labeled radiographs plus about 3,449 unlabeled radiographs receiving online teacher pseudo boxes (about 99:1) | mAP was 22.2 versus 6.1 for the same YOLOF supervised baseline; AP50 was 53.9 versus 22.6. Gains also held at 5% and 10% labels. | Closest pseudo-majority fracture localization example. It used confidence filtering, EMA teacher updates, pseudo-box fusion, and a weighted unsupervised loss rather than static raw targets. |
| Wang et al., IPMI 2021 | 808 expert-positive fracture boxes, 5,984 image-positive teacher pseudo maps, and 59,051 image-negative CXRs | Rib AUROC/FROC rose from 0.9025/0.7267 for supervised pre-training to 0.9318/0.8914; it also beat Mean Teacher by 1.63/3.74 points. | Direct fracture precedent with 7.4 times more pseudo-positive than expert-positive cases. The large exact-negative pool resembles the current logical-negative setting. Adaptive asymmetric sharpening was necessary; excessive sharpening degraded performance. |
| Lin et al., intracranial hemorrhage CT, 2021/2024 | 457 pixel-labeled head CT examinations plus 25,000 teacher-pseudo-labeled examinations (54.7:1) | On external CQ500, exam AUC improved from 0.907 to 0.939, Dice from 0.809 to 0.829, and pixel AP from 0.828 to 0.848. | Closest high-volume CT example. It used a ranker to suppress false-positive pseudo labels; removing it reduced validation examination AUC from 0.948 to 0.918. |
| Wu et al., ComWin, TMI 2023 | Pancreas CT: 3 labeled plus 57 online-pseudo-labeled volumes (19:1); ACDC: 6 labeled plus 134 additional volumes (22.3:1) | Pancreas Dice improved from 26.39 exact-only to 74.03; ACDC mean Dice improved from 54.55 to 79.53. | Strong controlled medical segmentation evidence. Multiple competing networks and adaptive confidence selection prevented low-quality pseudo labels; mini-batches remained balanced. |
| Gadgil et al., CheXseg, MIDL 2021 | Expert pixel masks plus Grad-CAM or IRNet pseudo masks for chest X-ray pathology segmentation | Grad-CAM mixture reached mIoU 0.270 versus 0.246 exact-only and 0.142 CAM-only. Best sampling used 90% expert examples and 10% CAM examples. | Positive CAM/medical evidence, but not a pseudo-majority success; retained as a counterexample showing that the best sampling ratio is task dependent. |
| Papandreou et al., ICCV 2015 | Exact pixel masks plus weak image-level or bounding-box annotations handled by EM/latent constraints | With 1,464 exact masks, adding 9,118 weak bounding-box labels raised validation IoU from 62.5 to 65.1. | Foundational evidence that strong and weak localization labels can be complementary when the weak labels are handled by a latent-label model. |

## Evidence against naive CAM mixing

- CheXseg's pseudo-only model was much worse than exact-only, and the best mixture was
  heavily exact-label dominated. Some individual pathologies were not improved by mixing.
- PseudoSeg reported that a decoder-only or CAM-only pseudo-label source was inferior to
  calibrated fusion. Well-calibrated soft labels outperformed simple hard confidence
  selection.
- AdvCAM starts from the known failure mode that ordinary CAM covers only a small
  discriminative portion of the object. Without regularization, iterative CAM expansion
  increasingly activated background.
- Arazo et al. showed that naive pseudo-labeling overfits incorrect pseudo labels through
  confirmation bias; maintaining a minimum labeled count per mini-batch and MixUp reduced
  the effect.
- Arun et al. found that eight medical saliency methods each failed at least one of
  localization utility, model sensitivity, repeatability, or reproducibility.
- Venkatesh et al. found that six gradient-based saliency methods on musculoskeletal X-rays
  were inferior to radiologists for localization, repeatability, and reproducibility; no
  method passed all trustworthiness criteria.
- Choe et al. found that several purported WSOL improvements did not materially outperform
  vanilla CAM under a corrected protocol and did not reach a few-shot fully supervised
  baseline. This supports spending scarce exact labels on a controlled strong-label arm.
- Oliver et al. showed that semi-supervised gains can disappear or reverse under data or
  class-distribution mismatch and emphasized reporting a strong supervised-only baseline.

## Applicability to the region branch

The current method differs from the positive precedents in three important ways.

First, the teacher CAM is produced by a whole-fracture classifier that never received
regional labels. CheXseg, PseudoSeg, and AdvCAM all use CAM as a spatial pseudo mask for the
same image; the region branch instead derives a dimensionless regional density score and
then compares that score across different bags.

Second, the current pseudo objective preserves only teacher ordering. It cannot correct a
systematic teacher bias shared by many cases, and no exact label is used to calibrate the
ordering-to-probability relationship. The internal CAM audit shows useful but imperfect
regional discrimination (AUROC 0.736-0.798), so the teacher ceiling is material.

Third, the current experiment has no exact-only arm. Without an otherwise identical
`alpha=0` model, neither the outer-fold metrics nor the observed training dynamics can
identify the causal contribution of CAM supervision.

## Recommended next experiment

Keep cross-case ranking retired. Map each region's `log(CAM density)` to a
per-case soft probability with a regularized monotone calibrator fitted only on
the student's train-fold human labels. Compare three otherwise identical
outer-0 arms: `no_cam`, `cam_soft`, and a within-region `cam_shuffled` negative
control.

Define the region epoch by one pass through the scarce human pool and rotate CAM
and negative samples across epochs. Corpus count must not determine optimizer
exposure. Ramp the CAM soft-BCE gradually, log its gradient ratio against human
BCE, and stop or decay it if it becomes dominant. Evaluate only on held-out
human regional endpoints; teacher agreement is diagnostic, not success.

## References

- Gadgil et al. [CheXseg: Combining Expert Annotations with DNN-generated Saliency Maps for X-ray Segmentation](https://proceedings.mlr.press/v143/gadgil21a.html), MIDL 2021.
- Zou et al. [PseudoSeg: Designing Pseudo Labels for Semantic Segmentation](https://arxiv.org/abs/2010.09713), ICLR 2021.
- Lee et al. [Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2103.08896), CVPR 2021.
- Wang et al. [Knowledge Distillation with Adaptive Asymmetric Label Sharpening for Semi-supervised Fracture Detection in Chest X-rays](https://arxiv.org/abs/2012.15359), IPMI 2021.
- Wei et al. [Semi-supervised object detection based on single-stage detector for thighbone fracture localization](https://arxiv.org/abs/2210.10998), 2022.
- Lin and Yu. [Semi-supervised learning for generalizable intracranial hemorrhage detection and segmentation](https://arxiv.org/abs/2105.00582), 2021; journal version 2024.
- Wu et al. [Compete to Win: Enhancing Pseudo Labels for Barely Supervised Medical Image Segmentation](https://arxiv.org/abs/2304.07519), IEEE TMI 2023.
- Papandreou et al. [Weakly- and Semi-Supervised Learning of a Deep Convolutional Network for Semantic Image Segmentation](https://openaccess.thecvf.com/content_iccv_2015/html/Papandreou_Weakly-_and_Semi-Supervised_ICCV_2015_paper.html), ICCV 2015.
- Arazo et al. [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/abs/1908.02983), IJCNN 2020.
- Arun et al. [Assessing the (Un)Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging](https://arxiv.org/abs/2008.02766), 2020.
- Venkatesh et al. [Gradient-Based Saliency Maps Are Not Trustworthy Visual Explanations of Automated AI Musculoskeletal Diagnoses](https://pubmed.ncbi.nlm.nih.gov/38710971/), 2024.
- Choe et al. [Evaluating Weakly Supervised Object Localization Methods Right](https://arxiv.org/abs/2001.07437), CVPR 2020.
- Oliver et al. [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html), NeurIPS 2018.
