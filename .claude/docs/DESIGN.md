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

The same four-region logits receive both exact and pseudo supervision:

\[
L=L_{\mathrm{whole}}+\lambda\left(L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}\right).
\]

`L_exact` includes valid human four-region labels and the logical
`[0, 0, 0, 0]` target from whole-negative bags. `L_rank` is the existing
fold-matched CAM pairwise-ranking supervision on fracture-positive bags.
Exact-label imbalance handling and deterministic loss-weight calibration are
specified in `.claude/docs/research/20260825-region-loss-balancing.md`.
`L_whole` retains the Baseline 0 weighted BCE with `pos_weight=2.0` and
weight-sum normalization. The region losses do not inherit this weight:
`L_exact` uses plain BCE after source-balanced human/whole-negative reduction,
and `L_rank` keeps its existing region-balanced pairwise BCE.

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
| Train one four-region output with exact and CAM-ranking supervision | Human labels and logical whole-negative zeros are exact targets in `L_exact`; pseudo supervision remains a separate ranking term on the same logits | 2026-08-25 |
| Keep `pos_weight=2.0` only for `L_whole` | Baseline 0 whole-risk parity is preserved, while source-balanced sampling already corrects the region exact-label imbalance | 2026-08-25 |
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

## Changelog

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
