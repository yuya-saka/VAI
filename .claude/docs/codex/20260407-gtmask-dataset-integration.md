# Codex Analysis: gt_masks loading integration in dataset.py
Date: 2026-04-07

## Question
How to integrate gt_masks loading into PngLineDataset in dataset.py, covering:
1. `_build_index`: optional gt_mask_path per item
2. `__getitem__`: load as np.uint8 (values 0-4), resize with INTER_NEAREST
3. Augmentation: treat gt_mask as additional_target 'gt_mask': 'mask' (nearest-neighbor)
4. Flip: swap right/left labels (2 <-> 3) when horizontal flip applied
5. Return 'gt_region_mask': torch.LongTensor(H,W) with 0-4
6. Graceful handling of missing gt_masks

## Codex Response (fully implemented)

Codex implemented all changes directly into dataset.py. Syntax verified with py_compile.

### Design Decisions Made

1. **Missing gt_mask slices**: Include them with zero mask (do NOT skip).
   - `gt_mask_path=None` in index entry
   - Return `gt_region_mask` as zero tensor, `has_gt_region_mask=False`
   - Rationale: Preserves all training data; caller can filter by flag if needed
   - Collate-safe: always returns a tensor (not None)

2. **Augmentation integration**: `additional_targets["gt_mask"] = "mask"`
   - Nearest-neighbor interpolation auto-applied by albumentations
   - Works for both `A.Compose` and `A.ReplayCompose`

3. **Flip label swap**: Applied after augmentation via lookup table
   - `label_map = np.array([0, 1, 3, 2, 4], dtype=np.uint8)` — swaps 2 (right) <-> 3 (left)
   - Only triggered when `has_gt_mask=True` is not gated — swap is harmless on zero mask

4. **Return type**: `torch.LongTensor` (int64) for `gt_region_mask`
   - Compatible with `nn.CrossEntropyLoss` which expects LongTensor targets

### Changes Made to dataset.py

**Imports**: Added `from typing import Any`

**`get_transforms` (line ~121)**:
- Added `"gt_mask": "mask"` to `additional_targets` dict
- Added type hints to signature

**`_build_index` (line ~261)**:
- Moved `qc_excludes` load outside inner loop (minor optimization)
- Added gt_mask_path detection:
  ```python
  gp = vd / "gt_masks" / f"slice_{slice_idx:03d}.png"
  gt_mask_path = gp if gp.exists() else None
  ```
- Added `"gt_mask_path": gt_mask_path` to item dict

**New `_load_gt_mask` method (line ~319)**:
- Returns `(np.ndarray[uint8], bool)` tuple
- Handles None path, file read errors, multi-channel PNGs
- Clips values to [0,4]

**New `_did_apply_horizontal_flip` method (line ~344)**:
- Extracted from inline code in `__getitem__`
- Cleaner, reusable

**New `_swap_gt_mask_left_right` method (line ~357)**:
- Lookup-table based swap: `label_map[gt_mask]`

**`__getitem__` (line ~390)**:
- Loads `gt_mask, has_gt_mask` via `_load_gt_mask`
- Passes `gt_mask=gt_mask` to transform
- Retrieves `out["gt_mask"]` after transform
- Calls `_swap_gt_mask_left_right` on flip
- Returns `gt_region_mask: torch.LongTensor` and `has_gt_region_mask: torch.BoolTensor`

### Return Dict (new fields)
```python
{
    "image": torch.FloatTensor,        # (2, H, W) - CT + vertebra mask
    "heatmaps": torch.FloatTensor,     # (4, H, W)
    "line_params_gt": torch.FloatTensor,  # (4, 2)
    "gt_region_mask": torch.LongTensor,   # (H, W), values 0-4  ← NEW
    "has_gt_region_mask": torch.BoolTensor,  # scalar            ← NEW
    "sample": str,
    "vertebra": str,
    "slice_idx": int,
}
```

## Status
- Implementation: COMPLETE (Codex implemented directly)
- Syntax check: PASS (`python3 -m py_compile` succeeded)
- Runtime test: NOT YET (DataLoader with real data not verified)
