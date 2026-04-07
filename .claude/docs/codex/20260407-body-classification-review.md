# Codex Review: body classification half-plane bug
Date: 2026-04-07

## Hypothesis Submitted

In `_classify_by_half_planes()`, the body condition requires all four half-planes:
  `body = mask & (s1>=0) & (s2>=0) & (s3>=0) & (s4>=0)`

When line_2 (right foramen posterior wall) is tilted, its normal n2 (computed with
hardcoded reference `centroid+[0,-1]`) has a large x-component. The s2=0 boundary
runs diagonally across the full image. Body pixels on the LEFT side fail s2>=0 even
though they are anatomically body pixels (nowhere near the right foramen).
They fall into the catch-all `posterior` region = yellow-invades-green artifact.

**Proposed Fix A:**
  `body = mask & (s1>=0) & (s3>=0) & ((s2>=0) | (s4>=0))`

**Proposed Fix B:**
  Change n2 reference from `centroid+[0,-1]` to `j_r` (junction point of right foramen corner).

## Codex Response

### Fix B is more principled.

**Why Fix B is correct:**
- s2 and s4 are supposed to be geometric boundary tests.
- If the sign of s2 is defined using a global, nonlocal reference like `centroid + [0,-1]`,
  the meaning of "inside" can change when line_2 tilts.
- Using `j_r` makes the sign of s2 anchored to the anatomically correct local side of
  the right foraminal wall. That preserves the interpretation of `s2 >= 0` as a true
  half-plane constraint.

**Fix A is a heuristic patch, not a principled fix:**
- It changes the model from an intersection of boundaries to a disjunction.
- Geometrically, `(s2>=0) OR (s4>=0)` means a point is accepted if it is valid with
  respect to EITHER foraminal wall, not both.
- That avoids the contralateral over-exclusion, but only by weakening the constraint.

**Overlap/gap issues with Fix A:**
- No "gap" in the sense of missing region caused by the OR itself. OR only enlarges
  the accepted set.
- The real risk is **over-inclusion/leakage**: pixels posterior to one wall can still be
  accepted if they satisfy the other wall.
- Fix A can admit anatomically wrong regions, especially around the posterior midline
  or if the two lines are strongly tilted/crossing.
- It also breaks the nice convex "intersection of half-planes" interpretation.

### Recommended Fix

If the bug is wrong sign orientation for s2, choose **Fix B** (change n2 reference to j_r,
change n4 reference to j_l).

If the deeper issue is that each foraminal wall should only constrain its own side, then
neither A nor B is the ideal formulation. The better model is a **side-gated constraint**:

```python
# Most principled approach: apply s2 only on right side, s4 only on left side
x_mid = (float(j_r[0]) + float(j_l[0])) / 2.0
right_side = xs >= x_mid
left_side = xs < x_mid
body = mask & (s1>=0) & (s3>=0) & np.where(right_side, s2>=0, True) & np.where(left_side, s4>=0, True)
# equivalently:
body = mask & (s1>=0) & (s3>=0) & ((~right_side) | (s2>=0)) & ((~left_side) | (s4>=0))
```

### Summary

- **Fix B** (use j_r for n2, j_l for n4) is the most principled fix
- **Fix A** (OR condition) is acceptable as a pragmatic workaround but has over-inclusion risk
- **Side-gated constraint** is the theoretically cleanest approach

### Coverage Analysis

Fix A (OR):
- Does NOT create gaps (OR enlarges set, never shrinks)
- MAY create overlap leakage near posterior midline when lines are strongly tilted

Fix B (correct reference points):
- Maintains intersection semantics (convex body region)
- n2 normal will point correctly toward body side of right foramen wall
- Requires j_r and j_l to be anatomically correct (they should be from _nearest_endpoint_junction)

### Q&A Summary

1. **Is hypothesis correct?** YES - tilted s2 half-plane root cause is valid
2. **Is Fix A (OR) correct?** Partially - works but has leakage risk near posterior midline
3. **Alternative?** Fix B (j_r/j_l reference) or side-gated constraint
4. **Does Fix A maintain non-overlapping coverage?** Mostly, but can have leakage overlap
5. **Should n2/n4 use j_r/j_l?** YES - this is the more principled fix
