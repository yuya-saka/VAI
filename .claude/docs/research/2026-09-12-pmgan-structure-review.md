# PMGAN architecture review for fracture localization

Date: 2026-09-12. Scope: primary-paper review only; no implementation decision.

Source: Zhang et al., *Part-Aware Mask-Guided Attention for Thorax Disease
Classification*, Entropy 2021, 23, 653, DOI: 10.3390/e23060653.
Local PDF: `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`.
Page numbers below refer to the PDF's printed pages (1–20). The local text was
extracted with `pdftotext -layout`; Figure 2 was also inspected as a rendered image.

## Evidence and exact architecture

- **Shared stem and independent tails:** Figure 2, p. 6, shows Conv1 and residual
  Blocks I–III shared by all branches. Four separate MA modules follow Block III;
  each is followed by its own Block IV, SA, GAP and FC. Section 3.3, p. 9,
  explicitly specifies independent branch parameters. Section 4.4.1, pp. 14–15,
  further states that sharing ends before MA: MA and following layers are not
  shared. Table 5 studies alternative split depths, with Block III best in this
  paper's experiment. Table 7 discussion, p. 16, confirms Conv1 and Blocks I–III
  are shared. Copying pretrained values into tails would be initialization, not
  ongoing weight tying; the paper does not specify a branch-by-branch copy policy.
- **Two stages are feature-processing stages:** Section 4.4.5, p. 17, calls the
  initial attentive CNN and subsequent multi-branch network two stages. Figure 2
  shows a single forward graph; the paper calls it end-to-end trainable. Section
  4.1.3, p. 12, reports ImageNet initialization followed by ChestX-ray14 fine-tuning.
  It does not establish a separate Stage-1 training/freeze/Stage-2 training schedule.
- **Branches:** One all-organ global branch plus left-lung, right-lung and heart
  branches. Thus the paper's four branches are not four local parts; applying its
  global-plus-local design to four vertebral regions would require five branches.
- **No implemented ROI crop:** Despite introductory descriptions of organ images,
  Figure 2 and Section 3.3 implement feature attention on shared feature maps.
  The MA output feeds full-spatial-size Block IV features, followed by GAP. No
  ROIAlign, crop-and-resize, bounding-box selection, or masked pooling operation is
  specified. Cropped-input methods in Section 2.2 describe earlier work.
- **SA locations:** Figure 2, p. 6, places generic SA after shared Blocks I and II
  and after each independent Block IV. The after-Block-III attention is MA.
- **MA is still learned soft attention:** Sections 3.2–3.3, pp. 7–9, specify the
  same architecture as SA, with an added mask-supervision constraint. Figure 3,
  p. 8, and Equations 4–7 describe a spatial encoder-decoder attention branch,
  channel attention from GAP and two convolutions, their product, a 1x1 conv and
  sigmoid. Equation 8 supervises the spatial attention map against an anatomical
  mask with an RMSE-style loss. Equation 9 combines all-organ and single-organ
  mask losses. These masks are anatomical support, not lesion segmentations.
- **Residual attention does not exclude other anatomy:** Equation 3, p. 7, uses
  `adjusted_features = (1 + attention) * features`, with final attention in [0,1].
  Every location is retained with multiplier in [1,2]; outside-mask information is
  not zeroed. The mask loss supervises the spatial sub-map, not directly the final
  spatial-by-channel attention tensor. Finite mask supervision also permits errors.
- **Pooling/head:** GAP follows Block IV and its SA, then a q-dimensional FC and
  sigmoid, where q=14 diseases (Figure 2; Section 3.1, p. 6; Table 1, p. 7).
  The PMGAN classifier contains no LSTM. Any Baseline0 sequence module is therefore
  a task-specific adaptation, not a component copied from this paper.
- **Classification supervision:** Section 3.4, p. 10, and Equations 2 and 10–12
  give a BCE on the global prediction and a BCE on the per-disease maximum over
  the three local predictions. Both use the image-level disease label. Each local
  branch is not separately trained to be positive for every positive image.
  Classification loss is global BCE + alpha * aggregated-local BCE, with alpha=0.5
  in the reported implementation, plus anatomical attention supervision.
- **Inference fusion:** Section 4.1.3, p. 12, takes the maximum across global and
  local classification scores. Organ masks and the segmentation model are only
  needed during training; the learned MA modules remain active at inference.

## Implications and unresolved choices for this project

1. Removing only modules labeled SA preserves learned MA, including its spatial
   and channel attention internals. Removing every learned soft-attention operation
   also removes the paper's MA and requires a different explicit-mask operation.
   These are materially different interpretations of "without softattention".
2. Paper-faithful independent tails differ from tying a single pretrained LSTM/head
   across all branches. Both can initialize from Baseline0, but only tied weights
   share subsequent updates. The paper is evidence for shared early layers and
   independent high-level branches, not maximal downstream parameter tying.
3. A Baseline0-compatible late CNN tail can retain its original feature width and
   accept copied pretrained LSTM and nonlinear head weights. Exact split depth,
   global-branch presence, weight tying, and fusion remain project design choices.
4. In this repository, each GT-bearing vertebra has four valid binary regional
   targets. A zero is a known negative even when the whole vertebra is positive.
   Never broadcast whole-positive labels onto those negative regions. Use observed
   regional labels directly; reserve bag-level aggregation for unannotated positives
   if adopting the existing mixed-supervision contract.
5. PMGAN's local outputs are branch-specific image-disease evidence, not verified
   organ-level lesion probabilities. Anatomical attention plus image-level BCE
   alone does not establish fracture localization. Existing region GT is still
   essential to the intended interpretation and its validation.
6. The paper's max global/local fusion can preserve or increase false positives;
   its ROC-AUC results do not guarantee PR-AUC gains here. Exact fusion and
   checkpoint-selection rules must be fixed on inner data before outer evaluation.
7. The paper's RMSE normalization notation is underspecified: Equation 8 divides
   spatial squared errors by N described as number of images, without a displayed
   image sum or spatial averaging factor. Do not present an implementation-specific
   normalization as an unambiguous reproduction of that equation.

## Gemini consultation status

The required Gemini consultation was attempted with only the extracted text of
this user-requested public paper, without clinical data or repository code.
The first CLI attempt failed authentication with a DNS `EAI_AGAIN` error.
The required escalated retry loaded cached credentials but failed because this
account requires `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_PROJECT_ID`; it also
reported an unrelated IDE workspace mismatch. No Gemini analysis was returned.
The findings above therefore derive from direct inspection of the local primary
PDF, not from a claimed Gemini result. No credential/configuration changes were
made to bypass this prerequisite.
