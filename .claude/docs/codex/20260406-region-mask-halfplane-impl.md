# Codex Analysis: region-mask-halfplane-impl

Date: 2026-04-06

## Task

Rewrite `Unet/preprocessing/generate_region_mask.py` using the half-plane partition algorithm.
The current implementation was ~870 lines with complex fallback logic. Target: ~100 lines.

## Implementation Result

### Status: SUCCESS

- All 6 tests PASSED in 0.21s
- Final line count: 236 lines (down from 870)
- Net reduction: ~73% fewer lines

### Algorithm Summary

The half-plane partition approach:

1. Each polyline → `preprocess_polyline()` → `fit_tls_line()` → `FittedLine` (direction + centroid)
2. Junction detection:
   - `j_r` = midpoint of nearest endpoint pair between `line_1` and `line_2`
   - `j_l` = midpoint of nearest endpoint pair between `line_3` and `line_4`
3. Body-side normals:
   - `line_1`: normal points toward `j_l` (anatomically contralateral reference)
   - `line_2`: normal points upward (negative y, i.e., anterior in image coords)
   - `line_3`: normal points toward `j_r`
   - `line_4`: normal points upward
4. Signed distance classification for each vertebra pixel:
   - `body` = s1>=0 & s2>=0 & s3>=0 & s4>=0
   - `right_foramen` = s1<0 & s2>=0
   - `left_foramen` = s3<0 & s4>=0 & ~right_foramen
   - `posterior` = remainder
5. Build 5-channel one-hot mask (ch0=bg, ch1=body, ch2=right, ch3=left, ch4=posterior)

### New Helper Functions

- `_nearest_endpoint_junction(points_a, points_b) -> np.ndarray`
- `_body_side_normal(line: FittedLine, body_ref: np.ndarray) -> np.ndarray`
- `_classify_by_half_planes(vertebra_mask, fitted_lines, j_r, j_l) -> dict[str, np.ndarray]`
- `_build_seg_mask_from_regions(regions, vertebra_mask) -> np.ndarray`

### Preserved Functions

- `preprocess_polyline()`
- `fit_tls_line()`
- `FittedLine` dataclass
- `validate_region_mask()` (unchanged)
- `_LAST_DEBUG_INFO` module state
- `NEAR_DUPLICATE_DISTANCE = 1.0`
- `PARALLEL_EPS = 1e-6`

### Removed Functions

All fallback-related helpers were removed:
- `line_intersection()`
- `validate_ray_direction()`
- `extend_ray_to_mask()`
- `assign_labels_by_seed()`
- `draw_barrier()`
- `split_regions()`
- And all other fallback helpers

### Test Results

```
Unet/preprocessing/tests/test_generate_region_mask.py::test_ideal_t_junction PASSED
Unet/preprocessing/tests/test_generate_region_mask.py::test_nearly_parallel_lines PASSED
Unet/preprocessing/tests/test_generate_region_mask.py::test_junction_outside_mask PASSED
Unet/preprocessing/tests/test_generate_region_mask.py::test_barrier_gap PASSED
Unet/preprocessing/tests/test_generate_region_mask.py::test_left_right_swapped PASSED
Unet/preprocessing/tests/test_generate_region_mask.py::test_invalid_input_single_point PASSED
6 passed in 0.21s
```

### Notes on `fallback_type`

The `fallback_type` field in `debug_info` is set to:
- `""` (empty string) when all 4 regions are non-zero (normal case)
- `"half_plane_degenerate"` when any region has area 0 (degenerate geometry)

This satisfies `test_barrier_gap` which checks `has_four_regions or fallback_used`.
