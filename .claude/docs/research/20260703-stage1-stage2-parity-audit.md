# Stage1 vs Stage2 Training Parity Audit

Date: 2026-07-03

Scope: compare `train_models/stage1` (config `config/config.yaml`, currently
named experiment `baseline/v1_gpu2`) against `train_models/stage2` (config
`config/parity.yaml`, currently running as `baseline/v1`) to determine what
needs to change in stage1 before it can be retrained as a fair comparison
baseline against stage2's region-auxiliary-head model. Investigation only,
no code changes made.

Files read: `.claude/docs/work-logs/2026-07-01.md`, `2026-07-02.md`;
`train_models/stage1/{train.py,src/trainer.py,src/dataset.py,
src/data_utils.py,src/experiment.py,utils/losses.py,src/evaluation.py,
config/config.yaml}`; `train_models/stage2/{train.py,src/trainer.py,
src/dataset.py,src/staging.py,src/data_utils.py,src/evaluation.py,
utils/losses.py,config/parity.yaml}`; `train_models/stage1/outputs/baseline/
v1_gpu2/{run_fold0_4.log,fold0/training.log}`; `data/rsna_data/
excluded_studies.csv`, `excluded_levels.csv`.

---

## 1. Config diff table

Comparing `train_models/stage1/config/config.yaml` (experiment `v1_gpu2`,
already-run) against `train_models/stage2/config/parity.yaml` (experiment
`v1`, currently running). "=" means the value is functionally identical.

### `data:`

| Field | Stage1 | Stage2 (parity) | Status |
|---|---|---|---|
| `dataset_dir` | `data/rsna_data/fracture_dataset` | same | = (both resolve to the same NFS4 mount `172.26.68.101:/share`, confirmed via `df -h`) |
| `csv_path` | `data/rsna_data/train.csv` | same | = |
| `excluded_studies_path` | **not present** | `data/rsna_data/excluded_studies.csv` | **DIFFERS** — stage1 has no exclusion mechanism at all |
| `excluded_levels_path` | **not present** | `data/rsna_data/excluded_levels.csv` | **DIFFERS** |
| `stage_to_local` / `stage_root` | **not present** | `true` / `/dev/shm` | **DIFFERS** (Fix #1, see §2) |
| `n_slices` | 15 | 15 | = |
| `in_channels` | 6 | 6 | = |
| `image_size` | 224 | 224 | = |
| `n_regions` | n/a | 4 | stage2-only (architecture) |
| `n_folds` | 5 | 5 | = |
| `random_seed` | 42 | 42 | = |
| `start_fold`/`end_fold` | 0/4 | 0/4 | = |
| `test_holdout_size` | 0.2 (code default; not in YAML) | 0.2 (code default) | = |

### `model:`

| Field | Stage1 | Stage2 (parity) | Status |
|---|---|---|---|
| `backbone` | `tf_efficientnetv2_s` | same | = |
| `pretrained` | true | true | = |
| `drop_rate` | 0.0 | 0.0 | = |
| `drop_path_rate` | 0.0 | 0.0 | = |
| `drop_rate_last` | 0.3 | 0.3 | = |
| `lstm_hidden` | 256 | 256 | = |
| `lstm_layers` | 2 | 2 | = |
| `out_dim` | 1 | n/a (implicit) | stage1-only field, unused elsewhere |
| `use_patient_head` | false | n/a | stage1-only (auxiliary patient head unused by default) |
| `fpn_channels`/`region_hidden`/`region_layers`/`region_dropout` | n/a | 256/256/2/0.3 | stage2-only (region branch) |

The shared primary-path architecture hyperparameters match exactly.

### `training:`

| Field | Stage1 | Stage2 (parity) | Status |
|---|---|---|---|
| `n_gpu` | 2 | 2 | = |
| `gpu_ids` | `[0, 2]` | `[0, 1]` | different physical GPUs used for the past run — cosmetic, but should be set explicitly for the next comparison run |
| `batch_size` | **4** (per GPU, comment says "実効 16×n_gpu" which is now wrong — 4×2=8, not 16; looks like a stale comment left over from when batch_size was 8) | **8** (per GPU) | **DIFFERS — effective global batch 8 vs 16** |
| LR | `learning_rate: 2.3e-4` (single group, all params) | `backbone_learning_rate: 2.3e-4` + `head_learning_rate: 2.3e-4` (two param groups, same value) | = numerically (both param groups get 2.3e-4 in parity.yaml), different code path (stage2 supports differential LR, stage1 doesn't) |
| `eta_min` | 2.3e-5 | 2.3e-5 | = |
| `weight_decay` | not set (code default `1e-4`) | `1.0e-4` (explicit) | = (same effective value) |
| `epochs` | 75 | 75 | = |
| `early_stopping_patience` | 15 | 15 | = |
| `num_workers` | 8 (per process) | 4 (per process) | DIFFERS — total DataLoader workers 16 vs 8. Throughput-only, not correctness, but relevant given fix #1 |
| `persistent_workers` | true | true | = |
| `prefetch_factor` | 4 | 4 | = |
| `use_amp` | true | true | = |
| `amp_dtype` | n/a (implicit FP16 via `autocast(device_type=...)` with no explicit dtype → torch default FP16, real `GradScaler`) | `bfloat16` (explicit; `GradScaler` effectively disabled since `enabled=amp_dtype==float16`) | **DIFFERS**, but intentionally per parity.yaml's own header comment: BF16 was required because the region/FPN path is FP16-unstable; stage1 never exercises that path so this is accepted, not a bug to port |
| `amp_initial_scale` / `max_consecutive_amp_skips` | n/a | 4096.0 / 8 | stage2-only knobs, moot for stage1 (BF16 path has GradScaler disabled anyway; stage1 uses real GradScaler with library defaults) |
| `gradient_clip_norm` | **no clipping mechanism exists in stage1 trainer at all** | `null` (→ effectively no clipping either, `float("inf")`) | = in practice today (both effectively unclipped), but stage1 has no code path to enable clipping if ever needed |
| `p_mixup` | 0.2 | 0.2 | = |
| `p_rand_order` | **0.2** | **0.0** | **DIFFERS** — stage2 parity.yaml disabled slice-order randomization, stage1 has it on |
| `positive_weight` | 2.0 | 2.0 | = |
| `slice_loss_weight` / `patient_loss_weight` | 15.0 / 1.0 | n/a | stage1-only, inert since `use_patient_head: false` |
| `region_loss_weight` | n/a | 0.5 | stage2-only |

### `augmentation:`

All 14 fields (`horizontal_flip_p`, `vertical_flip_p`, `transpose_p`,
`brightness_limit`, `brightness_p`, `shift_limit`, `scale_limit`,
`rotate_limit`, `ssr_border_mode`, `ssr_p`, `blur_noise_p`, `distortion_p`,
`cutout_p`, `cutout_ratio`) are **byte-for-byte identical** between the two
configs. Only the *mechanism* that applies them differs (see Fix #2 below).

### `wandb:`

Identical structure (`enabled: true`, `project: null`, `run_name: null`).

---

## 2. Fix-by-fix status

### Fix #1 — NFS staging (`stage_to_local`)

**Status: MISSING in stage1 (needs full port).**

- Stage1's `dataset_dir` (`data/rsna_data/fracture_dataset`) resolves to
  `172.26.68.101:/share` — the same slow NFS4 mount identified as the I/O
  bottleneck (confirmed via `df -h`), and stage1 reads directly from it with
  no staging step anywhere in `train_models/stage1/train.py` or
  `src/data_utils.py`.
- There is no `stage_to_local`/`stage_root` key in `train_models/stage1/
  config/config.yaml`'s `data:` section, and no equivalent of
  `train_models/stage2/src/staging.py` exists under `train_models/stage1/`.
- `train_models/stage1/train.py::_do_training` (line 156) builds
  `dataset_dir = ROOT / str(data_cfg.get("dataset_dir", ...))` and passes it
  straight into `collect_items(dataset_dir, csv_path)` — no local-copy step,
  no `try/finally` cleanup, no SIGTERM handling.
- Stage1's own `n_slices=15, in_channels=6` per-item files (`ct.npy` +
  `vertebra_mask.npy`, no `region_4class.npy`) are ~4.5MB/item — smaller
  than stage2's ~5.27MB/item (which also stages `region_4class.npy`), but
  it's the same NFS mount and the same I/O-bound symptom would apply.

### Fix #2 — Augmentation aggregation (15x loop → 1x stacked call)

**Status: MISSING in stage1, and stage1's current pattern has a correctness
issue that is arguably worse than stage2's old bug.**

- `train_models/stage1/src/dataset.py::get_train_transforms` returns a plain
  `A.Compose` (line 36) — **not** `A.ReplayCompose`. There is no `replay`
  mechanism, no `ReplayCompose`, no `.replay` anywhere in stage1's
  `dataset.py` (grep confirmed zero occurrences).
- `RSNAFractureDataset.__getitem__` (lines 134–145) loops over all 15 planes
  and calls `self.transform(image=ct_slice, mask=mask_slice)` **fresh, once
  per plane**:
  ```python
  for s in range(n_slices):
      ct_slice = ct[s].transpose(1, 2, 0)
      mask_slice = mask[s]
      augmented = self.transform(image=ct_slice, mask=mask_slice)
      ...
  ```
- Because this is a plain `Compose` (no replay), **each of the 15 planes
  draws independent random parameters** — different rotation/shift/scale/
  flip per plane, not just different noise. This is the *opposite* problem
  from stage2's pre-fix bug (which over-shared spatial params via replay but
  accidentally also over-shared noise): stage1 currently applies
  spatially-inconsistent geometric transforms across the 15 anatomically
  related planes of the same vertebra, which likely undermines whatever the
  BiLSTM-over-planes head assumes about slice-to-slice spatial
  correspondence. This looks like a pre-existing, never-fixed bug in
  stage1, unrelated to and predating the stage2 redesign.
- It is also the same performance pattern stage2 fixed (15 separate
  Python-level `transform()` calls per `__getitem__`, ~125ms/sample class of
  cost per the 07-02 benchmark), so stage1 has neither stage2's speed fix
  nor its later correctness guarantee.

### Fix #3 — DDP freeze (`broadcast_buffers=False` + unwrap-before-eval)

**Status: not applicable today — stage1's validation loop is not
susceptible to the specific deadlock stage2 hit, but this is incidental,
not a designed safeguard, and stage1 pays for it with 2x redundant
validation compute under DDP.**

- Root cause in stage2 was **asymmetry**: `evaluate()` was called only
  `if is_main:` while every rank still executed the DDP-wrapped model
  (`broadcast_buffers=True` issuing a BatchNorm-buffer collective every
  forward), so rank0's per-batch validation broadcasts and rank1's
  fold-end `dist.broadcast(stop_tensor, ...)` interleaved with mismatched
  counts and deadlocked (`train_models/stage2/src/trainer.py` fix at
  `DistributedDataParallel(..., broadcast_buffers=False)` and the two
  `_base_model(model)` unwraps before `evaluate(...)`).
- In `train_models/stage1/src/trainer.py::run_training_loop` (lines
  316–336), `_validate(model, val_loader, ...)` is called **unconditionally
  by every rank**, with no `if is_main:` guard around it — only checkpoint
  saving, logging, and wandb logging are rank0-gated afterward (lines
  354–394). Every rank runs the identical full validation set (the
  `val_loader` in `create_data_loaders` has no `DistributedSampler`, so all
  ranks iterate the same 100% of `val_items`), in lockstep, so the
  `broadcast_buffers=True` collective calls are issued identically and
  synchronously by all ranks — no desync, no deadlock. The subsequent
  `is_best`/`no_improve`/`break` early-stopping logic is computed
  identically and deterministically on every rank from that redundant
  validation, so all ranks reach `break` together without needing stage2's
  explicit `dist.broadcast(stop_tensor, src=0)` mechanism.
- `train_models/stage1/src/trainer.py::train_one_fold` (line 462) builds
  DDP with `model = DDP(model, device_ids=[device.index])` — **no
  `broadcast_buffers=False`**, so it pays the default per-forward
  BatchNorm-buffer broadcast cost on every train *and* eval step, on top of
  running the full validation set on every rank (2x compute for `n_gpu=2`).
  This mirrors the *pre-fix* stage2 behavior flagged as "known but
  unaddressed" duplicate-validation waste in the 2026-07-01 log, just
  without the fatal asymmetry.
- **Conclusion:** stage1 is currently safe from the deadlock only because
  it never introduced the rank0-only branch that caused it. If stage1 is
  ever changed to validate rank0-only (e.g. as a speed optimization to
  match stage2's now-fixed pattern), it must also adopt
  `broadcast_buffers=False` + unwrap-before-eval, or it will reproduce the
  exact same freeze stage2 hit.

---

## 3. Stage1 `v1_gpu2` log findings

`train_models/stage1/outputs/baseline/v1_gpu2/fold0/training.log` (51
lines = header + 48 epoch rows) and `run_fold0_4.log` (tail) show:

- Fold0 ran **48 consecutive epochs** with per-epoch wall time consistently
  ~165–169s (epoch1 was 290s, presumably first-epoch cudnn-autotune/warm
  cache overhead) — **no timestamp gaps, no stalls, no abrupt cutoff mid-
  epoch**.
- Best val AUROC 0.9291 at epoch 33; training correctly triggered
  `[EARLY STOP] 15 epoch 改善なし (fold=0, epoch=48)` at epoch 48
  (33 + 15 = 48), i.e. early stopping fired exactly as designed, not a
  crash.
- The run log ends cleanly right after the wandb sync for fold0
  ("Synced 5 W&B file(s)..."). No Python traceback, no CUDA OOM, no
  NCCL error anywhere in the log. Only `fold0/` exists under
  `outputs/baseline/v1_gpu2/` — folds 1–4 never started.
- This is consistent with the process being intentionally stopped between
  sessions (the repo's subsequent work moved to the stage2 redesign
  starting 07-01) rather than with any DDP freeze. Combined with the §2
  analysis (stage1's validation is symmetric across ranks), there is no
  evidence stage1 has ever hit — or is likely to hit — the deadlock stage2
  had.

---

## 4. Other comparability risks

1. **Exclusion lists only applied in stage2** (bigger than a footnote —
   flagging prominently). `train_models/stage1/src/data_utils.py::
   collect_items(dataset_dir, csv_path)` has no `excluded_studies_path`/
   `excluded_levels_path` parameters at all, while stage2's `collect_items`
   filters both. Contents:
   - `data/rsna_data/excluded_studies.csv`: 1 study, reason
     `ct_mask_geometry_mismatch` ("DICOM series has varying slice
     orientation; CT and mask physical coordinates are misaligned") — a
     **general CT/vertebra-mask data-quality issue**, not specific to
     region masks. This one is plausibly a real defect stage1 should also
     exclude, since it's a claimed CT/mask misalignment stage1 would train
     on directly.
   - `data/rsna_data/excluded_levels.csv`: 645 levels, reasons like
     `fragmented_mask_comp*`, `oversized_vol_z*`, `insufficient_line_
     anchors`, `sdf_region_qc_outlier_no_bbox` — these are almost all
     **region-mask (SDF boundary) generation QC failures** specific to
     stage2's region head, per `.claude/docs/DESIGN.md` (2026-06-28
     entries). They don't obviously indicate the underlying `vertebra_mask.
     npy`/`ct.npy` is bad, so blindly porting all 645 exclusions into
     stage1 would shrink/bias its training set for no stage1-relevant
     reason.
   - Net effect today: stage1's `v1_gpu2` log shows `train_val=11263`
     items (from 14,084 total minus the 20% holdout, note the 7 "missing
     study" items are unrelated to exclusions), while stage2's item pool is
     filtered by both exclusion lists **and** requires `region_4class.npy`
     to exist, so the two models are not currently trained/evaluated on
     identical item sets — a real confound for a "same data, different
     architecture" comparison.
2. **`p_rand_order` mismatch** (already in §1 table, restated because it's
   a training-dynamics knob, not just a hyperparameter typo): stage1 = 0.2,
   stage2 parity = 0.0. This changes how often slice order is shuffled
   during training and should be reconciled explicitly (pick one value for
   both) before comparing results.
3. **Effective batch size mismatch**: stage1 batch_size=4/GPU (effective 8
   across `n_gpu=2`) vs stage2 batch_size=8/GPU (effective 16). Different
   effective batch size at the same nominal LR changes gradient noise and
   optimization dynamics, independent of any architecture difference.
4. **`num_workers` mismatch** (8/process vs 4/process) — throughput/runtime
   only, not a correctness issue, but worth aligning once stage1 also gets
   NFS staging (fix #1), since staging removes the original motivation for
   keeping `num_workers` low.
5. **Loss/metric definitions are consistent** — good news, not a risk:
   `train_models/stage2/src/evaluation.py::compute_metrics` explicitly
   imports and reuses `train_models/stage1/utils/metrics.py::
   compute_oof_metrics`/`compute_level_metrics`, and stage2's `weighted_bce`
   in `utils/losses.py` is mathematically the same positive-weighted-mean
   BCE as stage1's `criterion()` in `utils/losses.py`. AUROC/AUPRC/F1
   numbers from the two pipelines are computed the same way and are
   comparable once the dataset composition and hyperparameters above are
   reconciled.
6. **No `stdout` log capture in stage2's `train.py`**, unlike stage1's
   `_Tee`-based `run_fold{N}_{M}.log`. Not a fairness issue, just means
   stage2's console history currently only lives in wandb/terminal
   scrollback, worth noting if debugging is needed later.
7. **`gpu_ids` differ** (`[0, 2]` vs `[0, 1]`) — purely operational (which
   physical GPUs happened to be free), not a fairness concern by itself,
   but should be set to whatever's actually free for the next stage1 run
   and recorded for reproducibility.

---

## 5. Prioritized punch-list for stage1

1. **[High] Port NFS staging (Fix #1).** Copy `train_models/stage2/src/
   staging.py` pattern into `train_models/stage1/src/staging.py` (or share
   a common module) and wire it into `train_models/stage1/train.py::main`/
   `_do_training` the same way: stage once in the parent process before
   `mp.spawn`, pass `dataset_root` through as a separate argument (not into
   `cfg["data"]["dataset_dir"]`, for the same resume/reproducibility reason
   documented in the 07-02 log), `try/finally` cleanup, SIGTERM→SystemExit
   handling. Add `stage_to_local`/`stage_root` keys to `train_models/
   stage1/config/config.yaml`.
2. **[High] Port augmentation aggregation (Fix #2), and treat it as a
   correctness fix, not just a speed fix.** Replace stage1's per-plane loop
   in `train_models/stage1/src/dataset.py::RSNAFractureDataset.__getitem__`
   with the stack-once-transform-once approach from `train_models/stage2/
   src/dataset.py::_augment_volume` (adapted for stage1's `(15,6,224,224)`
   layout without the region mask / horizontal-flip remap parts, which are
   stage2-only). This both speeds up loading and fixes the currently
   inconsistent per-plane geometric augmentation in stage1.
3. **[Medium] Reconcile `p_rand_order` and `batch_size`/effective batch
   size** between the two configs before running the comparison — pick one
   value for each and set both configs to match (or explicitly justify the
   difference in a comment, as parity.yaml already does for `amp_dtype`).
4. **[Medium] Decide on exclusion-list parity.** At minimum, port
   `excluded_studies_path` filtering (the single `ct_mask_geometry_mismatch`
   entry) into `train_models/stage1/src/data_utils.py::collect_items` since
   it's a general CT/mask defect. Decide deliberately (not by omission)
   whether to also apply `excluded_levels.csv` to stage1, given most of its
   645 entries are region-mask-specific QC failures that may not indicate
   bad `ct.npy`/`vertebra_mask.npy` data for stage1's purposes.
5. **[Low, defer] `broadcast_buffers=False`.** Not required for correctness
   today (see §2, Fix #3 — stage1's symmetric all-ranks validation avoids
   the deadlock), but if stage1's validation is ever changed to rank0-only
   (e.g. while porting other DDP efficiency improvements), it must adopt
   both `broadcast_buffers=False` and `_base_model(model)`-style
   unwrap-before-eval at the same time, exactly as stage2's fix did in
   `train_models/stage2/src/trainer.py`, or it will reproduce the same
   freeze.
6. **[Low] Align `num_workers`** once staging (item 1) is in place, since
   the original NFS-load rationale for stage1's higher `num_workers: 8`
   goes away after staging to `/dev/shm`.
