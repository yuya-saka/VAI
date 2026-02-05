# Matplotlib Japanese Font Support Research (Python 3.12)

**Research Date**: 2026-02-05
**Agent**: Gemini CLI via Claude Code
**Context**: VAI project requires Japanese text rendering in matplotlib graphs

---

## Executive Summary

**CRITICAL FINDING**: `japanize-matplotlib` is **NOT compatible** with Python 3.12+

**Recommended Solution**: Use `matplotlib-fontja` - a modern, actively maintained alternative

---

## Detailed Findings

### 1. japanize-matplotlib Status ❌

| Aspect | Status |
|--------|--------|
| **Python 3.12 Compatibility** | ❌ NOT compatible |
| **Root Cause** | Depends on `distutils` (removed in Python 3.12) |
| **Last Update** | October 2020 |
| **Maintenance** | 💀 Development stopped |
| **Recommendation** | **Do NOT use** |

**Why it fails:**
```python
# japanize-matplotlib imports distutils
from distutils import ...
# ModuleNotFoundError: No module named 'distutils'
```

---

### 2. matplotlib-fontja ✅ (Recommended)

| Aspect | Details |
|--------|---------|
| **Python 3.12 Compatibility** | ✅ Fully compatible |
| **Maintenance** | 🟢 Actively maintained |
| **Ease of Use** | ★★★ Excellent (zero-config) |
| **Dependencies** | Bundles Japanese fonts (no system font required) |
| **Best For** | Scripts, notebooks, prototyping |

**Installation:**
```bash
uv add matplotlib-fontja
```

**Usage:**
```python
import matplotlib_fontja  # Auto-configures Japanese fonts
import matplotlib.pyplot as plt

# Just use matplotlib normally with Japanese text
plt.plot([1, 2, 3], [1, 4, 9], label='データ')
plt.xlabel('横軸')
plt.ylabel('縦軸')
plt.title('日本語タイトル')
plt.legend()
plt.show()
```

**Pros:**
- ✅ Zero configuration required
- ✅ Works out of the box
- ✅ Bundles fonts (no system dependency)
- ✅ Python 3.12+ compatible
- ✅ Drop-in replacement for japanize-matplotlib

**Cons:**
- ⚠️ Adds dependency to project
- ⚠️ Slightly larger package size (includes font files)

---

### 3. Manual Font Configuration ⚙️ (Production Alternative)

| Aspect | Details |
|--------|---------|
| **Python 3.12 Compatibility** | ✅ Fully compatible |
| **Dependencies** | None (Python standard library only) |
| **Ease of Use** | ★★ Moderate (requires setup) |
| **Best For** | Production, Docker, controlled environments |

**Approach A: Use System Fonts**
```python
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [
    'IPAexGothic',        # First choice
    'Noto Sans CJK JP',   # Fallback
    'DejaVu Sans'         # Final fallback
]
mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign
```

**Approach B: Bundle Font Files**
```python
from pathlib import Path
from matplotlib.font_manager import FontProperties

font_path = Path(__file__).parent / "fonts" / "IPAexGothic.ttf"
font_prop = FontProperties(fname=str(font_path))

import matplotlib as mpl
mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['axes.unicode_minus'] = False
```

**Pros:**
- ✅ No external dependencies
- ✅ Full control over font selection
- ✅ Lightweight
- ✅ Consistent across environments (if bundling fonts)

**Cons:**
- ⚠️ Requires manual setup
- ⚠️ Need to manage font files (if bundling)
- ⚠️ System font availability varies across OS

---

## Comparison Table

| Feature | japanize-matplotlib | matplotlib-fontja | Manual Config |
|---------|-------------------|------------------|---------------|
| **Python 3.12** | ❌ Crashes | ✅ Works | ✅ Works |
| **Maintenance** | 💀 Stopped (2020) | 🟢 Active | N/A |
| **Setup Effort** | Low | Low | Medium |
| **Dependencies** | Heavy | Medium | None |
| **Font Control** | None | Limited | Full |
| **Production Ready** | ❌ No | ⚠️ Yes (with caveats) | ✅ Yes |
| **Best Use Case** | ❌ Legacy only | Scripts/Notebooks | Docker/Production |

---

## Recommendations by Use Case

### Quick Scripts / Jupyter Notebooks
**Use: matplotlib-fontja**
```bash
uv add matplotlib-fontja
```
```python
import matplotlib_fontja
import matplotlib.pyplot as plt
# Just works!
```

### Production / Docker Environments
**Use: Manual font configuration**
- Bundle font file in project (`fonts/IPAexGothic.ttf`)
- Configure matplotlib to use bundled font
- Ensures consistency across deployments

### Legacy Projects (Python < 3.12)
**Use: japanize-matplotlib** (only if stuck on old Python)
- Not applicable for this project (Python 3.12)

---

## Implementation Plan Reference

See full implementation plan: `.claude/docs/PLAN_matplotlib_japanese.md`

**Key Steps:**
1. ✅ Research completed (this document)
2. Add `matplotlib-fontja` to dependencies
3. Create test script to verify Japanese rendering
4. Document usage for team

**Selected Approach**: `matplotlib-fontja` for development/research phase

**Future Migration**: Can switch to manual config if production requirements demand it

---

## Important Notes

### License Considerations
- **matplotlib-fontja**: Check license of bundled fonts
- **IPAex Fonts**: IPA Font License (permissive, redistribution allowed)
- **Noto Sans CJK**: SIL Open Font License (permissive)

### Known Issues
- None reported for matplotlib-fontja with Python 3.12
- Manual config may have font cache issues (rare) - solution: `rm -rf ~/.matplotlib/fontlist-*.json`

### Font File Sources
If using manual config:
- **IPAex Fonts**: https://moji.or.jp/ipafont/
- **Noto Sans CJK**: https://fonts.google.com/noto/specimen/Noto+Sans+JP

---

## References

- matplotlib-fontja GitHub: https://github.com/kota7/matplotlib-fontja
- japanize-matplotlib (archived): https://github.com/uehara1414/japanize-matplotlib
- Matplotlib font configuration docs: https://matplotlib.org/stable/users/explain/text/fonts.html

---

**Research completed by**: Gemini CLI
**Validated by**: Claude Code orchestration
**Status**: ✅ Ready for implementation
