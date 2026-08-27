# Region-head Loss Balancing Research

Date: 2026-08-25

## Scope

The region branch uses a dedicated BiLSTM that is separate from the Baseline 0
whole-vertebra BiLSTM. The same four-region student output is supervised by:

- exact human labels on 268 fracture-positive bags, with invalid cells masked;
- exact logical zeros from 12,100 whole-negative bags;
- pairwise CAM-ranking pseudo supervision on 1,064 non-annotated
  fracture-positive bags.

The fixed loss structure is:

\[
L=L_{\mathrm{whole}}+\lambda\left(L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}\right).
\]

This note investigates (1) imbalance inside the exact-label population,
(2) combining clean and pseudo supervision in one head, and (3) setting
\(\lambda\) and \(\alpha\).

## Primary Recommendation

### Exact-label reduction

Keep human labels and logical-zero labels inside one semantic term, but use a
source-balanced reduction so that the 12,100 whole-negative bags do not define
the objective merely through their count.

For active regions \(\mathcal R\):

\[
L_H=\frac{1}{|\mathcal R|}\sum_{r\in\mathcal R}
\frac{\sum_{i\in H}v_{ir}\,\mathrm{BCE}(z_{ir},y_{ir})}
     {\sum_{i\in H}v_{ir}},
\]

\[
L_N=\frac{1}{|\mathcal R|}\sum_{r\in\mathcal R}
\frac{1}{|N|}\sum_{i\in N}\mathrm{BCE}(z_{ir},0),
\]

\[
L_{\mathrm{exact}}=\tfrac12 L_H+\tfrac12 L_N.
\]

Here, \(v_{ir}\) is the existing human-label validity mask. The two subgroup
means are an internal imbalance correction; both remain exact ground truth.

Use a persistent, shuffle-without-replacement queue for each subgroup. For a
16-bag auxiliary region batch, the recommended split is:

| Source | Bags | Role |
|---|---:|---|
| Human-annotated | 4 | \(L_H\) |
| Whole-negative | 4 | \(L_N\) |
| Pseudo-positive endpoints | 8 | \(L_{\mathrm{rank}}\) |

Every auxiliary region update must contain all three sources. Do not run
negative-only or pseudo-only region updates. The batch ratio controls exposure
and gradient-estimator variance; the loss coefficients control scientific
weighting. Log source visits and unique coverage separately.

Uniformly cycle through human bags rather than oversampling bags according to
their rarest positive region. Multi-label class-aware oversampling changes the
exposure of all co-occurring labels and can over-represent multi-region cases.
With the 1:1 human/whole-negative split, the expected exact-label positive
rates become approximately 15.9% / 12.1% / 14.8% / 31.5% for R1-R4, which is
already a manageable range.

Do not combine this sampler with corpus-derived `pos_weight`. Using the full
hard-label counts would produce approximate per-region positive weights of
157 / 208 / 170 / 77, causing a second and much stronger correction on top of
sampling. It would also make the sigmoid output unsuitable as a population
probability without separate prior correction.

### Clean and pseudo supervision in one head

Continue using the same four logits for both losses:

\[
L_{\mathrm{region}}=L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}.
\]

The sources are complementary. Exact BCE anchors the sign, scale, and bias of
each logit. Pairwise ranking transfers only the teacher-supported ordering
among fracture-positive bags and is invariant to a region-wise common logit
shift. Keeping both losses on the same logits lets exact labels anchor the
otherwise unidentifiable ranking solution.

Implementation contracts:

- A valid human cell always takes precedence and is excluded from pseudo
  pairs. The existing `build_region_pair_batch` already supports this.
- `L_exact` and `L_rank` are reduced independently, first within each region
  and then across active regions. Never pool exact cells and pseudo pairs into
  one denominator.
- Do not add CAM-magnitude confidence weights. The soft rank target already
  expresses score-gap uncertainty, and CAM magnitude is confounded with easy
  or severe cases. Only independently defined quality exclusions, such as an
  undefined CAM score, may remove a pair.
- Keep hard and pseudo streams on the same fixed update cadence so that
  `alpha` is not silently changed by update frequency.

### Setting alpha

Use a one-time, training-only gradient calibration rather than sample counts,
grid search on the 268 labels, or dynamic GradNorm.

On 64 deterministic calibration batches from the outer-training folds, before
any optimizer update, measure unweighted gradient norms on parameters touched
by both losses, preferably the region BiLSTM parameters:

\[
g_H=\|\nabla L_{\mathrm{exact}}\|_2,\qquad
g_P=\|\nabla L_{\mathrm{rank}}\|_2.
\]

Set:

\[
\alpha_k=
\operatorname{clip}_{[0.01,1]}
\left[
0.25\exp\left\{
\operatorname{median}_b\log\frac{g_{H,b}+\epsilon}{g_{P,b}+\epsilon}
\right\}
\right].
\]

The target `0.25` makes pseudo supervision an auxiliary signal whose initial
gradient norm is about one quarter of the exact-label gradient. The cap at 1
prevents a numerically small pseudo loss from receiving a coefficient larger
than the clean loss. This is a trust constraint, not an estimate of a
statistically optimal value.

Compute `alpha_k` once with the integrated four-region model as the reference
and reuse the same numerical value for the integrated model and all four
single-region models in outer fold `k`. Per-model or per-region calibration
would confound the sharing comparison.

### Setting lambda

After fixing `alpha_k`, measure gradients at the last CNN block shared by the
whole and region paths:

\[
g_W=\|\nabla L_{\mathrm{whole}}\|_2,\qquad
g_R=\|\nabla (L_{\mathrm{exact}}+\alpha_kL_{\mathrm{rank}})\|_2.
\]

Set:

\[
\lambda_k=
\operatorname{clip}_{[0.01,10]}
\left[
0.25\exp\left\{
\operatorname{median}_b\log\frac{g_{W,b}+\epsilon}{g_{R,b}+\epsilon}
\right\}
\right].
\]

This targets a region-to-whole shared-trunk gradient norm ratio of about 1:4
at initialization. Use the same `lambda_k` for the integrated and single-region
models in that outer fold. Record all raw norms, calibrated values, clipping,
RNG state, and the reference-model hash. Any non-finite norm is an
implementation failure.

If one-time calibration is not implemented, `lambda=0.25` and `alpha=0.25`
are acceptable only after a label-blind preflight confirms that both raw
gradient ratios are within a factor of two of one. Otherwise the numerical
coefficients do not represent the intended 1:4 budgets.

Do not dynamically equalize task learning rates. GradNorm treats tasks as
peers and can increase a noisy pseudo task when it learns slowly. Learned
homoscedastic-uncertainty weighting similarly estimates task noise within a
different probabilistic model and does not encode the required clean-over-
pseudo trust order. A fixed, auditable calibration is preferable here.

The teacher and pseudo targets are frozen, so a Mean-Teacher-style pseudo-loss
ramp-up is not required. If the pre-trained whole trunk needs protection,
freeze it for a short, predeclared head warm-up rather than introducing another
adaptive coefficient schedule.

## Alternatives Considered

| Method | Strength | Decision for this project |
|---|---|---|
| Full-corpus BCE | Preserves natural prevalence | Reject: 12,100 all-zero bags dominate optimization |
| Full-corpus `pos_weight` | Simple class correction | Reject: weights of roughly 77-208 are too aggressive and duplicate sampling correction |
| Per-label positive oversampling | Raises rare-label exposure | Reject: distorts multi-region co-occurrence and over-samples multi-positive bags |
| Asymmetric Loss | Suppresses easy negatives | Keep only as a pre-registered sensitivity method; it adds focusing/shift hyperparameters and weakens the meaning of exact logical zeros |
| Distribution-Balanced Loss | Handles co-occurrence and negative dominance | Reject for the primary model: designed for many-class long-tailed recognition and unnecessary after explicit source balancing |
| Equal exact/pseudo coefficient | Simple | Reject: loss scale and pair/cell estimands differ |
| Sample-count weighting | Appears data-driven | Reject: pair endpoints and exact cells are not comparable independent observations |
| Dynamic GradNorm | Adapts during training | Reject: may promote the noisy task and changes the objective during the run |
| Learned uncertainty weighting | Removes manual coefficients | Reject: does not enforce hard-label precedence and adds trainable task-weight parameters |

## Monitoring and Failure Criteria

Log every epoch on a fixed fracture-positive diagnostic subset that excludes
human labels from model selection:

- six inter-region Spearman correlations;
- four region-versus-whole Spearman correlations;
- first-PC explained variance of standardized four-region logits;
- student-teacher rank correlation by region;
- raw `g_rank/g_exact` and weighted `alpha*g_rank/g_exact`;
- raw and weighted region/whole shared-trunk gradient ratios;
- exact-source and pseudo-pair unique coverage;
- per-region valid-cell counts and effective positive rate.

Keep the existing collapse alarm: median inter-region Spearman at least 0.95
and median region-whole Spearman at least 0.95 for three consecutive checks.
The alarm invalidates the run; it must not trigger post-hoc coefficient tuning
against the 268 human labels.

Because balanced sampling and ranking supervision alter the score intercept,
the region sigmoid must be described as a score, not a calibrated unconditional
fracture probability. Evaluate locked OOF logits with per-region AP/AUROC. A
population-probability claim would require a separate representative
calibration set, which is not available here.

## Evidence Base

- Durand et al., [Learning a Deep ConvNet for Multi-Label Classification with
  Partial Labels](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf):
  known cells contribute to a partial BCE while unknown cells are excluded;
  normalization must not depend naively on the number of observed labels.
- Ben-Baruch et al., [Multi-Label Classification with Partial Annotations using
  Class-Aware Selective Loss](https://openaccess.thecvf.com/content/CVPR2022/html/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.html):
  annotated labels are emphasized over inferred/unannotated labels, and
  asymmetric treatment is used for positive-negative imbalance.
- Wu et al., [Distribution-Balanced Loss for Multi-Label Classification in
  Long-Tailed Datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf):
  negative-label dominance is a distinct multi-label failure mode, and
  resampling one label changes the exposure of co-occurring labels.
- Ridnik et al., [Asymmetric Loss for Multi-Label
  Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf):
  asymmetric focusing and probability shifting can suppress abundant easy
  negatives without symmetrically weakening positives.
- Radhakrishnan et al., [Design Choices for Enhancing Noisy Student
  Self-Training](https://openaccess.thecvf.com/content/WACV2024/papers/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.pdf):
  a split batch and separately averaged clean/pseudo losses prevent a large
  pseudo pool from dominating a uniform batch.
- Sohn et al., [FixMatch](https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f73-Paper.pdf),
  Xie et al., [Noisy Student](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper):
  clean and pseudo examples can train the same student output, but pseudo
  contribution is controlled through a separate loss, selection, batching,
  and student noise rather than raw dataset size.
- Hinton et al., [Distilling the Knowledge in a Neural
  Network](https://research.google.com/pubs/archive/44873.pdf): hard and soft
  targets are combined as separate weighted objectives when both are
  available.
- Tarvainen and Valpola, [Mean Teacher](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html):
  consistency supervision is a separate weighted regularizer, commonly with
  an early ramp-up when the teacher target itself evolves.
- Chen et al., [GradNorm](https://proceedings.mlr.press/v80/chen18a.html):
  task imbalance is observable in back-propagated gradient norms; the present
  proposal uses only a one-time, trust-constrained calibration rather than
  dynamic equalization.
- Kendall et al., [Multi-Task Learning Using Uncertainty to Weigh
  Losses](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html):
  relative task weights materially affect shared-representation training, but
  their uncertainty model does not encode the required exact-versus-pseudo
  ordering.
- Telesco et al., [Semi-Supervised Multi-Task Learning for Interpretable
  Quality Assessment of Fundus Images](https://arxiv.org/abs/2511.13353): a
  recent medical-imaging precedent combines a manual primary task with a
  pseudo-labeled auxiliary task in one student. Its teacher was directly
  trained on detailed labels, so it is an architectural precedent rather than
  validation of CAM-ranking pseudo labels.

