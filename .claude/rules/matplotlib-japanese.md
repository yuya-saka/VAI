# Matplotlib Japanese Font Support

Rules for using matplotlib with Japanese text in this project (Python 3.12 environment).

## Critical: Package Selection

| Package | Python 3.12 | Status | Decision |
|---------|-------------|--------|----------|
| `japanize-matplotlib` | ❌ Crash | Abandoned (last update 2020, `distutils` dependency) | **DO NOT USE** |
| `matplotlib-fontja` | ✅ Compatible | Actively maintained | **Use this** |

**`japanize-matplotlib` は Python 3.12 で動作しない。絶対に使用しないこと。**

## Recommended Setup

### Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
    "matplotlib>=3.8",
    "matplotlib-fontja>=0.1",
]
```

```bash
uv add matplotlib matplotlib-fontja
```

### Usage (Development / Scripts / Notebooks)

```python
import matplotlib_fontja  # 先頭でimportするだけで自動設定
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_title("日本語タイトル")
ax.set_xlabel("横軸")
ax.set_ylabel("縦軸")
```

### Manual Font Configuration (Production / Docker)

`matplotlib-fontja` が使えない環境向け:

```python
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
```

## Best Practices

- `import matplotlib_fontja` は必ずファイルの先頭（他の matplotlib import より前）に書く
- バージョンは固定する（再現性のため）: `matplotlib>=3.8`, `matplotlib-fontja>=0.1`
- 新規グラフ作成後は PNG を目視確認して文字化けがないことを確認すること

## Verification

```bash
uv run python -c "import matplotlib_fontja; import matplotlib.pyplot as plt; print('OK')"
```

テストスクリプト: `tests/matplotlib-font.py`（既に動作確認済み）
