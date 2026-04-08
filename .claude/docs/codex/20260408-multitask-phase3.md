# Codex Implementation: Multitask Phase 3

Date: 2026-04-08

## Task
Create Phase 3 files for Unet/multitask/src/:
- dataset.py
- data_utils.py

## Files Created

### Unet/multitask/src/dataset.py
- Verbatim copy of Unet/line_only/src/dataset.py
- No changes needed (already returns gt_region_mask and has_gt_region_mask)

### Unet/multitask/src/data_utils.py
Based on Unet/line_only/src/data_utils.py with 4 changes:

1. Import: `from torch.utils.data import DataLoader, WeightedRandomSampler`
2. Import: `from .model import ResUNet` (instead of TinyUNet)
3. Added `_make_weighted_sampler()` function - weights seg GT samples 3x higher
4. `create_data_loaders` uses `sampler=_make_weighted_sampler(train_ds)` for train
5. `create_model_optimizer_scheduler` creates ResUNet with seg_classes, line_channels, norm_groups params

## Verification
Import check passed:
```
dataset import OK
data_utils import OK
```
