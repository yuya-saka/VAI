# Codex Implementation: Phase 5 preprocess_all.py
Date: 2026-04-06

## Task
Batch GT region mask generation script for all samples/vertebrae/slices

## Output format
Single-channel label PNG (argmax of 5-channel one-hot), pixel values 0-4:
- 0 = background
- 1 = body
- 2 = right_foramen
- 3 = left_foramen
- 4 = posterior

## File created
Unet/preprocessing/preprocess_all.py

## Key design

- VERTEBRAE = ['C1','C2','C3','C4','C5','C6','C7'] (all cervical vertebrae)
- DATASET_ROOT = Path('/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset')
- BAD_SLICES_OUTPUT = DATASET_ROOT / 'bad_slices_all.json'

### New function: process_and_save_slice(candidate, dataset_root) -> SliceProcessResult
- Loads vertebra_mask via cv2.imread(..., IMREAD_GRAYSCALE)
- Calls generate_region_mask() and validate_region_mask()
- Extra shape checks: ndim==3, channels==5 (guards added vs pilot)
- cv2.imwrite() return value checked; failure treated as hard_fail
- label_img = np.argmax(seg_mask, axis=0).astype(uint8)
- Saves to: dataset_root / sample / vertebra / 'gt_masks' / 'slice_NNN.png'

### main()
- Collects all slices (C1-C7 full dataset)
- Prints total candidate count
- Progress log every 100 slices
- Saves failures to bad_slices_all.json
- Prints full summary

### Additional constants vs pilot
- PROGRESS_INTERVAL = 100
- SEG_MASK_NDIM = 3
- SEG_MASK_CHANNELS = 5
- GT_MASK_DIRNAME = 'gt_masks'
- LABEL_IMAGE_DTYPE = np.uint8

### Differences from pilot_region_mask.py
- No random sampling (processes all candidates)
- Adds actual file saving logic (process_and_save_slice vs process_one_slice)
- Additional validation of seg_mask shape before argmax
- cv2.imwrite success check
- build_slice_filename() helper extracted as named function
