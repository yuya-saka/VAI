# Fix C Implementation: Side-gating for _classify_by_half_planes()

Date: 2026-04-07

## Problem

Fix B (implemented earlier) incorrectly reversed the normal direction for C1 by changing n2 from `centroid+[0,-1]` to `j_r`. This caused C1 body pixels to drop from 24% to 3%.

For C2/C7, Fix B provided no improvement because it did not address the root cause.

## Root Cause

The original body condition `s2>=0 AND s4>=0` used the full half-plane for both foramen walls simultaneously. When line_2 (right foramen posterior wall) is tilted, its s2=0 boundary runs diagonally. Left-side body pixels at certain y values fail s2>=0 even though they are far from the right foramen. Similarly for s4 on the right side.

## Fix C: Revert n2/n4 + Side-gating

### Changes to `_classify_by_half_planes()` in `generate_region_mask.py`:

1. **Reverted n2 and n4** back to `centroid+[0,-1]` reference (Fix B was wrong):
   - `n2 = _body_side_normal(l2, l2.centroid + np.array([0.0, -1.0], dtype=np.float64))`
   - `n4 = _body_side_normal(l4, l4.centroid + np.array([0.0, -1.0], dtype=np.float64))`

2. **Added x_mid side-gating**:
   - `x_mid = (float(j_r[0]) + float(j_l[0])) / 2.0`
   - `right_side = xs >= x_mid`

3. **Side-gated body condition**:
   - `body = mask & (s1 >= 0.0) & (s3 >= 0.0) & (~right_side | (s2 >= 0.0)) & (right_side | (s4 >= 0.0))`
   - Right-side pixels (x >= x_mid): must satisfy s2>=0 (right foramen's posterior wall)
   - Left-side pixels (x < x_mid): must satisfy s4>=0 (left foramen's posterior wall)

### Also fixed n3 reference:
- n3 correctly uses `j_r` as reference (Codex had incorrectly changed it to `j_l`)
- `n3 = _body_side_normal(l3, np.asarray(j_r, dtype=np.float64))`

## Test Results

### Unit tests: 7/7 passed

### Real data body percentages (C1/C2/C7):

| Vertebra | Before Fix C (Fix B) | After Fix C |
|----------|---------------------|-------------|
| C1/slice_034 | ~3% (broken by Fix B) | **24.7%** |
| C2/slice_049 | ~22% | **27.4%** |
| C7/slice_067 | ~27% | **30.7%** |

## Files Changed

- `Unet/preprocessing/generate_region_mask.py`: `_classify_by_half_planes()` function
- `Unet/preprocessing/tests/test_generate_region_mask.py`: `test_tilted_line2_body_not_posterior` test updated to validate Fix C behavior
