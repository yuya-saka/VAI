# Codex Analysis: mask resample fix for convert_to_png.py
Date: 2026-04-06

## Problem
CT-mask voxel grid mismatch not detected.

When CT was re-cropped in 3D Slicer (new affine), but the mask was NOT re-saved (old affine),
the two NIfTIs have different voxel grids. Using the same slice index `mask_data[:, :, slice_idx]`
on a misaligned mask causes CT-mask misalignment in the output PNGs.

## Fix Applied

Codex added two changes to `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/preprocessing/convert_to_png.py`:

1. Added `import nibabel.processing` at line 17 (top of file, after `import nibabel as nib`)
2. Added alignment check + resample block in `convert_vertebra_to_png()` at lines 309-311,
   immediately after `load_vertebra_data()` and before `get_fdata()` calls

## Key lines modified

**Line 17 (new import):**
```python
import nibabel.processing
```

**Lines 309-311 (new alignment check):**
```python
if not np.allclose(ct_nii.affine, mask_nii.affine) or ct_nii.shape != mask_nii.shape:
    print('  警告: CTとマスクのボクセルグリッドが異なります。マスクをCT空間にリサンプリングします。')
    mask_nii = nibabel.processing.resample_from_to(mask_nii, ct_nii, order=0)
```

After this block, `ct_data` and `mask_data` are obtained via `get_fdata()` on the (possibly resampled) objects,
guaranteeing they share the same voxel grid and slice indices are consistent.
