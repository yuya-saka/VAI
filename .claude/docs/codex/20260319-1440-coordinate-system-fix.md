# Codex Analysis: Coordinate System Fix Strategy
Date: 2026-03-19 14:40

## Question
Analyze coordinate system unification for line detection.

Context:
- line_losses.py (training): math coords (center origin, Y-up)
- line_detection_moments.py (evaluation): image coords (Y-down) - WRONG
- line_only/line_detection.py: correct implementation (math coords internally, converts for drawing)

The files line_detection_moments.py and line_only/line_detection.py have nearly identical detect_line_moments() functions.

Question: What's the best way to fix this?
1. Copy line_only/line_detection.py implementation to line_detection_moments.py?
2. Delete line_detection_moments.py and use line_only/line_detection.py instead?
3. Other approach?

Analyze:
- Best implementation strategy
- Risk of breaking existing code
- Testing approach
- Long-term maintainability

## Codex Response

推奨は **3) Other** です。
具体的には、**`line_only` を単一の正とし、`line_detection_moments.py` は互換 shim を経て削除**が最適です（`copy` は非推奨）。

### 根拠

- **旧実装は画像座標（Y-down）**: [line_detection_moments.py:78] `ys = np.arange(H, dtype=np.float64); xs = np.arange(W, dtype=np.float64); X, Y = np.meshgrid(xs, ys)`

- **正実装は数学座標（Y-up）で統一し、出力時のみ画像座標へ変換**:
  - [line_detection.py:84] Grid computation in math coords
  - [line_detection.py:130] Conversion to image coords for drawing

- **loss 側も Y-up 前提**: [line_losses.py:104] `y_grid = -(torch.arange(H, device=device, dtype=torch.float32) - H / 2.0)`

- **さらに旧ファイルは評価指標系も古く、部分コピーだと再び不整合化しやすい**

### 実行方針

1. **`Unet/line_only/line_detection.py` を唯一の実装に固定**
   - This is the correct implementation with proper coordinate handling
   - Uses math coordinates internally (Y-up)
   - Converts to image coordinates only for visualization

2. **`Unet/line_detection_moments.py` を薄い re-export shim（DeprecationWarning付き）に置換**
   - Create deprecation wrapper that imports from line_only
   - Add warning to guide users to new location
   - Maintain backward compatibility temporarily

3. **外部利用がないことを確認後に削除**
   - 現状、リポ内参照は実質なし
   - Only referenced in docs/progress/2026-03-10_weekly.md (documentation)
   - No active code dependencies found

### Why NOT Option 1 (Copy)?

- **重複コードの維持コスト**: Two files would drift apart again
- **再発防止できない**: Same coordinate mismatch could happen again
- **テストの二重管理**: Would need to test both implementations

### Why NOT Option 2 (Direct Delete)?

- **破壊的変更**: Could break unknown dependencies
- **段階的移行なし**: No deprecation period for users

### Why Option 3 (Shim → Delete) is Best?

1. **安全な移行**: Deprecation warning gives users time to update
2. **単一の正**: Only one implementation to maintain
3. **再発防止**: Forces all code to use correct coordinate system
4. **テスト一元化**: Only need to test one implementation

### 実装ステップ

**Phase 1: Create Shim (Safe)**
```python
# Unet/line_detection_moments.py
"""Deprecated: Use Unet.line_only.line_detection instead."""
import warnings
from Unet.line_only.line_detection import (
    detect_line_moments,
    predict_lines_and_eval_test,
    # ... other exports
)

warnings.warn(
    "line_detection_moments is deprecated. Use Unet.line_only.line_detection",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["detect_line_moments", "predict_lines_and_eval_test", ...]
```

**Phase 2: Verify (1-2 weeks)**
- Monitor for deprecation warnings
- Check if any external code breaks
- Update documentation

**Phase 3: Delete (After verification)**
- Remove line_detection_moments.py
- Update docs/progress/2026-03-10_weekly.md reference

### Testing Approach

**Before Shim:**
```bash
# Run existing tests to establish baseline
uv run pytest Unet/line_only/test/ -v

# Verify coordinate system tests pass
uv run pytest Unet/line_only/test/test_coordinate_fix_simple.py -v
uv run pytest Unet/line_only/test/test_angle_consistency.py -v
```

**After Shim:**
```bash
# Test that old imports still work (with warning)
python -c "from Unet.line_detection_moments import detect_line_moments"

# Verify warning is raised
python -W error::DeprecationWarning -c "from Unet.line_detection_moments import detect_line_moments" || echo "OK: Warning raised"
```

**After Delete:**
```bash
# Verify import fails gracefully
python -c "from Unet.line_detection_moments import detect_line_moments" && echo "ERROR: Should fail" || echo "OK: Import removed"

# All tests still pass
uv run pytest Unet/line_only/test/ -v
```

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Unknown external dependencies | Shim with deprecation warning (gradual migration) |
| Breaking internal tests | Verify all tests before/after shim |
| Documentation outdated | Update docs in Phase 2 |
| Re-introduction of bug | Only one source of truth (line_only/) |

### Long-term Maintainability

**Benefits:**
- ✅ **Single source of truth**: line_only/line_detection.py
- ✅ **Coordinate system consistency**: Math coords everywhere internally
- ✅ **Easier testing**: Only one implementation to test
- ✅ **Clear migration path**: Deprecation → Removal

**Trade-offs:**
- ⚠️ Requires updating imports in future code
- ⚠️ Need to monitor deprecation warnings

### Conclusion

**推奨: Option 3 (Shim → Delete)**

要するに、**「1) copy」ではなく「3) shim→削除」**が安全で再発防止になります。

This approach:
- Maintains backward compatibility during transition
- Prevents code duplication and drift
- Ensures single source of truth with correct coordinate system
- Provides clear migration path for any unknown dependencies
