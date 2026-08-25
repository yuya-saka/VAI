# Project Design Document

> Active design decisions only. Historical designs are available from Git history.

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

## Changelog

- 2026-08-24: Reset the active design to Baseline 0 and removed discontinued experiment families.
- 2026-08-24: Reorganized Baseline 0 into responsibility-based directories.
- 2026-08-24: Corrected the scope: restored pseudo-label generation and CAM audit as active core functionality under `baseline0/pseudo_labeling/`.
