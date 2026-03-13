# 実験結果

実験の結果・分析を保存します。

## ディレクトリ構成

各実験ごとにサブディレクトリを作成：

```
results/
├── exp_001/
│   ├── metrics.md      # 定量的結果
│   ├── analysis.md     # 分析・考察
│   ├── plots/          # グラフ・可視化
│   └── samples/        # サンプル出力
├── exp_002/
...
```

## メトリクステンプレート

```markdown
# Experiment <ID> - Results

**Date:** YYYY-MM-DD

## Quantitative Results

| Metric | Value |
|--------|-------|
| Accuracy | |
| Loss | |
| F1-Score | |

## Qualitative Results

[サンプル画像や可視化結果]

## Comparison

[他の実験との比較]

## Observations
```
