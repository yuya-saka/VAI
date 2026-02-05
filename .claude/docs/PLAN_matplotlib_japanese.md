# Implementation Plan: Matplotlib Japanese Font Support (Python 3.12)

## Purpose
Enable Japanese text rendering in matplotlib graphs (axis labels, titles, legends) for Python 3.12 environment without font garbling issues.

## Requirements Analysis

### Purpose
- Display Japanese characters correctly in matplotlib plots
- Support axis labels, titles, legends, and annotations in Japanese
- Work reliably in Python 3.12 environment

### Scope
**Include:**
- Font configuration for matplotlib
- Japanese font installation/verification
- Test script to verify Japanese rendering
- Documentation for team usage

**Exclude:**
- Other visualization libraries (seaborn, plotly, etc.) - can be added later
- Font customization beyond basic Japanese support

### Constraints
- Python 3.12 environment
- Linux system (based on current environment)
- Must work with existing uv package manager
- Should not break existing plots

## Current State Investigation

### Existing Code
- No matplotlib currently in pyproject.toml
- No existing plotting code found in Unet/
- Clean slate for implementation

### Common Approaches (Knowledge Base)

1. **japanize-matplotlib** (Popular, simple)
   - Single import solution
   - May have Python 3.12 compatibility concerns (TBD from Gemini research)

2. **Manual font configuration** (Flexible, reliable)
   - Use matplotlib.rcParams
   - Specify Japanese fonts explicitly
   - More control, works across versions

3. **System font installation** (Foundation)
   - Install Japanese fonts on system
   - Configure matplotlib to use them

## Implementation Steps

### Step 1: Research Findings & Approach Selection ✅
- [x] Gemini research completed
- [x] Verified japanize-matplotlib compatibility: **❌ NOT compatible with Python 3.12+**
- [x] Identified 2 viable alternatives
- [x] Selected recommended approach

**CRITICAL FINDING: japanize-matplotlib は Python 3.12+ で動作しません**
- 理由: `distutils` 依存（Python 3.12で削除済み）
- 最終更新: 2020年（開発停止）

**代替手段:**

| 手法 | Python 3.12対応 | 使いやすさ | 本番適性 | 推奨用途 |
|------|----------------|----------|---------|---------|
| japanize-matplotlib | ❌ クラッシュ | - | - | **使用禁止** |
| matplotlib-fontja | ✅ 対応 | ★★★ 高 | ★★ 中 | スクリプト/Notebook |
| 手動フォント設定 | ✅ 対応 | ★★ 中 | ★★★ 高 | 本番環境/Docker |

**Selected Approach: matplotlib-fontja (開発段階) + 手動設定（本番オプション）**
- 開発/実験: `matplotlib-fontja` - ゼロコンフィグで即座に動作
- 本番移行時: 手動フォント設定への移行パスを確保

**Verification**: Research completed, approach validated by Gemini

### Step 2: Font Installation
- [ ] Check available Japanese fonts on system (`fc-list :lang=ja`)
- [ ] Install recommended Japanese fonts if missing (e.g., IPAexGothic, Noto Sans CJK JP)
- [ ] Verify font installation

**Verification**: `fc-list :lang=ja` returns Japanese fonts

### Step 3: Package Installation

**Primary Approach: matplotlib-fontja**
```toml
# Add to pyproject.toml
[project]
dependencies = [
    "matplotlib>=3.8",
    "matplotlib-fontja>=0.1",  # Python 3.12 compatible
]
```

- [ ] Add matplotlib and matplotlib-fontja to pyproject.toml dependencies
- [ ] Run `uv add matplotlib matplotlib-fontja`
- [ ] Run `uv sync` to install packages
- [ ] Verify installation

**Verification**:
```bash
uv run python -c "import matplotlib; import matplotlib_fontja; print('OK')"
```

### Step 4: Configuration Implementation

**Approach A: matplotlib-fontja (Recommended for Development)**
```python
import matplotlib_fontja  # Auto-configures Japanese fonts
import matplotlib.pyplot as plt

# That's it! No further configuration needed
```

**Approach B: Manual Font Configuration (Production Option)**
```python
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Option 1: Use bundled font (if available)
font_path = Path("fonts/IPAexGothic.ttf")  # or Noto Sans CJK JP
if font_path.exists():
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(fname=str(font_path))
    mpl.rcParams['font.family'] = font_prop.get_name()

# Option 2: Use system font
mpl.rcParams['font.sans-serif'] = ['IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
```

**Implementation Steps:**
- [ ] Create simple wrapper for matplotlib-fontja approach
- [ ] (Optional) Prepare manual config for future production use
- [ ] Document both approaches in project docs

**Verification**: Configuration code runs without errors

### Step 5: Test Script Creation
- [ ] Create test script `tests/test_japanese_plot.py` or `scripts/test_japanese_plot.py`
- [ ] Test all text elements:
  - Axis labels (xlabel, ylabel)
  - Title
  - Legend
  - Text annotations
- [ ] Save test plot to verify rendering

**Test Script Template:**
```python
#!/usr/bin/env python3
"""Test script for Japanese text rendering in matplotlib."""

import matplotlib_fontja  # Auto-configure Japanese fonts
import matplotlib.pyplot as plt

def test_japanese_rendering():
    """Test all Japanese text elements in matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Test data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Test various text elements
    ax.plot(x, y, label='データ系列1', marker='o')
    ax.set_xlabel('横軸ラベル (X軸)', fontsize=12)
    ax.set_ylabel('縦軸ラベル (Y軸)', fontsize=12)
    ax.set_title('日本語タイトルのテスト - matplotlib-fontja', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.text(3, 5, '注釈テキスト：正常に表示', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = 'test_japanese_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")
    print("   Open the image to verify Japanese text renders correctly")
    plt.close()

if __name__ == "__main__":
    test_japanese_rendering()
```

**Verification**:
- Script runs without errors
- Generated PNG shows Japanese text correctly (no garbled characters/boxes)

### Step 6: Integration & Documentation
- [ ] Add usage instructions to project README or separate doc
- [ ] Update pyproject.toml with final dependencies
- [ ] Create reusable plotting utilities if needed
- [ ] Document font installation steps for new developers

**Verification**:
- Documentation is clear and actionable
- New team member can follow steps successfully

## Scope Details

### New Files (Estimated)
- `.claude/docs/research/matplotlib-japanese-fonts.md` (Gemini research output)
- `tests/test_japanese_plot.py` or `scripts/test_japanese_plot.py` (test script)
- `utils/plotting.py` (optional utility module)
- `.claude/docs/matplotlib_japanese_setup.md` (setup documentation)

### Modified Files
- `pyproject.toml` (add matplotlib + Japanese font support)
- `README.md` (optional: add usage notes)

### Dependencies

**Python packages (pyproject.toml):**
- `matplotlib>=3.8` - Core plotting library
- `matplotlib-fontja>=0.1` - Japanese font support (Python 3.12 compatible)

**System-level (Optional for manual config only):**
- Not required for matplotlib-fontja (bundles fonts)
- If using manual config: IPAexGothic, Noto Sans CJK JP, etc.

## Risks & Considerations

### Potential Issues

1. **Python 3.12 Compatibility** ✅ RESOLVED
   - Risk: japanize-matplotlib does NOT work with Python 3.12+
   - Solution: Use matplotlib-fontja instead (actively maintained, Python 3.12 compatible)
   - Status: Research confirmed - matplotlib-fontja is the correct choice

2. **Font Availability**
   - Risk: System may not have Japanese fonts installed
   - Mitigation: Document font installation steps, check fonts in Step 2
   - Impact: High (no fonts = no Japanese rendering)

3. **Matplotlib Version Conflicts**
   - Risk: Older matplotlib versions may have font handling issues
   - Mitigation: Use matplotlib >= 3.8 (recent stable)

4. **Backend-Specific Issues**
   - Risk: Different matplotlib backends (Agg, TkAgg, etc.) may render fonts differently
   - Mitigation: Test with default backend, document backend-specific configs if needed

5. **Performance**
   - Risk: Japanese font rendering may be slower
   - Mitigation: Minimal impact expected, measure if performance-critical

### Best Practices

- **Always test after setup**: Run test script to verify Japanese rendering
- **Document for team**: Clear setup instructions prevent repeated troubleshooting
- **Version pinning**: Pin matplotlib and font package versions for reproducibility
- **Graceful fallback**: Consider fallback to English labels if fonts fail (optional)

## Open Questions ✅ RESOLVED

1. **japanize-matplotlib status**: Is it actively maintained? Python 3.12 compatible?
   - **Answer**: ❌ NOT maintained (last update 2020), NOT Python 3.12 compatible (distutils dependency)
   - **Action**: Do NOT use

2. **Recommended fonts**: Which Japanese font provides best cross-platform support?
   - **Answer**: matplotlib-fontja bundles fonts, no manual setup needed
   - **Alternative**: IPAexGothic or Noto Sans CJK JP for manual config

3. **Alternative libraries**: Are there newer/better solutions than japanize-matplotlib?
   - **Answer**: ✅ matplotlib-fontja is the modern replacement
   - **Status**: Actively maintained, Python 3.12+ compatible

4. **Integration scope**: Should we create a project-wide plotting utility module?
   - **Decision needed**: Depends on how often plotting is used in the project

5. **Font distribution**: Should fonts be bundled with the project or rely on system fonts?
   - **Recommendation**: System fonts (cleaner, smaller repo)
   - **Alternative**: Bundle if targeting environments without font management

## Next Steps

1. ✅ **Gemini research completed** - matplotlib-fontja confirmed as solution
2. ✅ **Plan updated** with research findings
3. **Ready for implementation** - Begin with Step 2 (font check can be skipped for matplotlib-fontja)
4. **Optional**: Consult with Codex on implementation details if needed

---

**Status**: ✅ COMPLETED & VERIFIED
**Last Updated**: 2026-02-05 (Implementation verified by user)
**Research Agent ID**: a9dc6fc
**Key Decision**: Use `matplotlib-fontja` for Python 3.12 compatibility
**Verification**: User tested successfully with `tests/matplotlib-font.py`
