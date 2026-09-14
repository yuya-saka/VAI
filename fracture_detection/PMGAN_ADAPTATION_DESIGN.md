# PMGAN-inspired fracture model without learned attention

Status: design proposal, 2026-09-12. Source code and runnable configuration are
unchanged. The user requests a structure inspired by PMGAN without soft
attention, following their request to reuse Baseline 0's learned classification
machinery as much as possible. The user subsequently clarified that neither SA
nor MA should be used. The whole-branch design below remains a proposal.

## Source and verified interpretation

Zhang et al., *Part-Aware Mask-Guided Attention for Thorax Disease
Classification*, Entropy 2021, 23, 653, DOI: 10.3390/e23060653.
Local source: `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`.

| Source | What the paper specifies |
|---|---|
| Page 6, Figure 2 | Shared Conv1 and residual Blocks I–III; separate global/all-organ and three organ branches after MA |
| Pages 7–9, Sections 3.2–3.3 | Standalone SA and mask-supervised MA use spatial/channel attention; MA additionally receives organ-mask supervision |
| Page 7, Equation 3 | Residual feature weighting `(1 + attention) * features` |
| Pages 9–10, Equations 8–9 | Organ masks supervise the MA spatial map using an RMSE-style constraint |
| Page 10, Equations 10–12 | Global classification BCE plus BCE of the maximum local-branch scores, plus mask-guidance losses |
| Page 12, Section 4.1.3 | At inference, maximum fusion of global and local scores |

The architecture branches feature maps; it is not a requirement to crop and
re-encode four separate input images. The paper has no CT-plane LSTM. Adding a
Baseline 0 LSTM is an adaptation to the existing 15-plane vertebra input.
The local classification loss follows aggregation: the paper does not assign
the whole-positive label independently to every local branch.

MA is itself learned attention. Removing the standalone SA blocks while keeping
MA does not remove all soft feature weighting. Also, `(1 + attention) * features`
preserves the original features outside attended locations; it does not strictly
isolate a branch to its anatomical region. Organ masks are anatomy supervision,
not fracture-segmentation targets.

## Common adaptation to the current repository

- Keep the existing 15-plane, 6-channel input and patient-grouped nested folds.
- Reuse the fold-matched Baseline 0 encoder, LSTM, and nonlinear classification
  head.
- Branch after EfficientNetV2-S `blocks[4]`, before the final semantic block.
- Keep the original post-branch feature dimensions so the trained LSTM and head
  can be loaded without a randomly initialized dimensional adapter.
- Apply the head per plane, then sigmoid and plane averaging as in Baseline 0.
- Maintain four outputs with anatomical identities and valid local fracture GT.
  Do not overwrite GT-negative regions of a positive vertebra with whole-positive
  labels.
- Omit both the paper's standalone SA and learned MA blocks. Use the supplied
  four-region masks as deterministic spatial guidance. The existing EfficientNet
  architecture is retained; this request is not interpreted as deleting
  pretrained backbone components such as its built-in squeeze-excitation modules.

Local timm CPU inspection confirmed these interfaces at input size 224:

| Stage | Per-plane feature shape |
|---|---|
| Shared stem and `blocks[0:5]` | `160 x 14 x 14` |
| `blocks[5]` | `256 x 7 x 7` |
| `conv_head` and `bn2` | `1280 x 7 x 7` |
| Global average pooling | `1280` |
| Baseline 0 BiLSTM | input 1280, hidden 256, two bidirectional layers |
| Baseline 0 head | Linear 512→256, BN, dropout, LeakyReLU, Linear 256→1 |

Applying anatomical guidance at 14x14 before the semantic block is different
from the earlier proposal of pooling regions directly from the final 7x7 map.
It preserves the pretrained downstream feature-processing interface but does
not create new spatial resolution beyond that intermediate feature map.

## Parameter reuse and sharing

The proposed starting adaptation shares the CNN prefix and reuses the same
learned LSTM/head across all regional streams, with independent recurrent states
for each stream. The streams are not concatenated into one region-mixing temporal
sequence. The same late CNN can also be applied to each guided stream to maximize
parameter sharing. This is a deliberate difference from the paper's independent
late branches, not a claim of exact reproduction.

If independent late CNNs are chosen later, initialize every copy from the same
Baseline 0 weights. Copying initial weights and tying parameters during training
are different operations. In the local model, `blocks[5]` has 14,561,832
parameters and `conv_head` plus `bn2` has 330,240; four independent copies would
therefore substantially increase parameter and optimizer-state memory.
Shared weights still require downstream processing for each regional stream;
they do not eliminate branch compute or activation memory.

BN running statistics in the reused CNN and head require explicit handling.
Retaining the trained statistics while fine-tuning weights and BN affine
parameters is the proposed starting policy, consistent with the current effort
to avoid sampling-driven BN drift. It is not a guarantee of transfer accuracy.

## Proposed whole output

Add one global stream and retain four regional streams. The global stream follows
the unchanged Baseline 0 path and produces the primary whole probability. The
regional probabilities remain localization endpoints; their normalized LSE is
an auxiliary consistency/supervision path and a reported diagnostic. Do not use
PMGAN's maximum of global and local probabilities as the initial endpoint: one
false-high region can only increase the fused score, which is undesirable given
the observed high-ranking-negative AP error.

At the 14x14 split, form one unchanged global tensor and four deterministically
masked regional tensors. Apply the same transferred `blocks[5]`, `conv_head`,
`bn2`, BiLSTM and head weights to all five streams, while keeping independent
LSTM states. The exact deterministic mask operator remains to be fixed. Strict
multiplication enforces local support but changes feature statistics strongly;
fixed residual weighting preserves outside-region context but is a weaker
locality constraint. Neither choice introduces learned attention.

## Proposed batch and objective

Keep one stratified batch with N/A/U=8/4/4. It simultaneously contains eight
whole-negative and eight whole-positive bags:

| Group | Count | Whole target | Regional supervision |
|---|---:|---:|---|
| N: whole negative | 8 | direct 0 | four direct zeros |
| A: annotated positive | 4 | direct 1 | four observed regional GT values |
| U: unannotated positive | 4 | direct 1 | positive LSE bag constraint |

Use the same 16 bags and one shared encoder-prefix forward for both tasks. A
separate natural batch for the whole path would increase compute and recreate
the multiple-forward/BN-distribution problem encountered in earlier designs.

Compute two independently normalized objectives:

```text
L_total = L_whole + lambda_region * L_region
```

- `L_whole` directly supervises the global Baseline 0 stream on all 16 bags.
  To preserve the meaning of the Baseline 0 objective under stratified sampling,
  apply fold-specific source importance weights `natural_group_fraction /
  sampled_group_fraction`, then the existing positive weight 2, and normalize by
  the resulting weight sum. Retain the Baseline 0 per-plane head, broadcast
  target loss, and mean-sigmoid bag prediction.
- `L_region` keeps the N/A/U observation rules. For task-scale comparability,
  average the four N/A cell losses within each bag, keep one U LSE loss per bag,
  then average across the 8/4/4 batch. `beta` continues to scale only the U term.
- `lambda_region=1` is a transparent starting value after per-bag normalization,
  but it is not established as optimal. Log whole and region gradient norms on
  shared parameters before deciding whether it needs adjustment.

The transferred encoder tail, LSTM and head belong to the low-learning-rate
parameter group. Any new regional-only parameters belong to the new group. The
global branch exactly reproducing Baseline 0 in eval mode before fine-tuning is
a required initialization test.

## Implementation and validation plan

1. Add a versioned model variant and explicit config fields after choosing the
   deterministic mask operator and accepting the proposed global endpoint/loss.
   Preserve loading and interpretation of v1/v2 artifacts.
2. Implement the shared prefix, guided streams, pretrained late CNN and shared
   temporal classifier. Audit every loaded parameter and any genuinely new
   parameter; include the LSTM and head in the transferred optimizer group.
3. Implement deterministic mask guidance. Anatomy masks must use synchronized
   geometric augmentation and area downsampling with explicit region IDs.
4. Connect whole and regional loss diagnostics to training and
   validation, preserving their meanings and reporting whole/local endpoints
   separately. An absent anatomical region must not be silently replaced by a
   high-confidence negative prediction; define temporal support for the chosen
   guidance mechanism explicitly.
5. Verify eval-mode Baseline 0 parity with a neutral/bypassed guide. Verify
   parameter sharing, independent recurrent states, valid GT handling, mask
   alignment, finite losses/gradients, optimizer coverage, checkpoint loading,
   and resume behavior. The unmodified global stream, rather than a regional
   stream, is the required exact initialization-parity path.
6. Run a bounded shape/gradient/memory smoke test before proposing a full GPU
   experiment. Save new experiments under a new name and do not resume old FPN
   checkpoints as this architecture.

The old `memo/計画書/提案手法.md` contains a historical PMGAN-based proposal, but its
10-channel input, former protocols, loss settings, and architectural restrictions
are not automatically reactivated by this request.
