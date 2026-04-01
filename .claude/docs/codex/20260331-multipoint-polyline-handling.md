# Codex Analysis: Multi-point Polyline Handling
Date: 2026-03-31

## Question

Medical CT annotation quality problem: cervical vertebra boundary line annotations stored as polylines.

Dataset statistics:
- 3640 total annotations (4 lines × ~910 slices across 40+ patients, 7 vertebrae)
- 49.3% have exactly 2 points (clean endpoints)
- 48.5% have near-duplicate points (segment < 1px), suggesting annotation tool artifact
- All annotations have very high linearity (LI mean=0.9990, min=0.979)
- Low RMSE_perp (mean=0.46px) meaning points lie close to a straight line
- High kappa_bar (mean fold angle 40°) due to near-duplicate points causing degenerate angle computation
- 15 continuity failures (0.5%) = genuinely bad annotations, already removed
- Heatmap generation: distance transform from rendered polyline → Gaussian (sigma=4px)
- GT parameter extraction: PCA on all points → (phi, rho) Hesse normal form

Questions:
1. For the 48.5% with near-duplicate points (LI>0.99, RMSE<0.5px): are they safe to use as-is, or should intermediate near-duplicate points be deduplicated before heatmap generation and PCA?
2. What is the correct deduplication strategy? Options: (a) remove points with segment < threshold, (b) Douglas-Peucker simplification, (c) use only first and last point (endpoints only), (d) keep as-is since LI and RMSE are fine
3. Does rendering near-duplicate points with cv2.polylines cause any heatmap artifacts compared to a clean 2-point line?
4. For genuinely V-shaped 3+ point lines (not near-duplicates), does PCA still correctly capture the intended line direction?
5. Recommendation: preprocess or not? If preprocess, exact algorithm and threshold?

## Codex Response

結論から言うと、**学習品質の観点では「軽い前処理あり」を推奨**です。
ただし影響は用途で分かれます。

### 1. near-duplicateをそのまま使って安全か

- **Heatmap学習（`cv2.polylines`→距離変換→`sigma=4` Gaussian）だけなら概ね安全**です。
- ただし **`phi,rho` 推定や角度系指標（kappa含む）には悪影響**が出ます。今回の `kappa_bar` 高騰は典型的な退化症状です。

### 2. 最適なdedup戦略

- 第一選択: **連続点間距離 `< 1.0 px` を削除**（artifactに直接対応）。
- `Douglas-Peucker` はこの問題には過剰で、V字を壊すリスクがあるので通常は不要。
- `endpoints-only` は「直線性が高いものだけ」に限定して使うのが良いです。
- `keep as-is` は heatmap-only なら可、幾何パラメータ学習/評価には非推奨。

### 3. near-duplicateでcv2.polylinesのheatmapにartifactは出るか

- 完全重複/同一直線上ならほぼ同一。
- 微小な折れがあると1px級の局所ふくらみは出ますが、`sigma=4` なら学習影響は通常ごく小さいです。
- つまり **主問題はヒートマップより幾何ラベル側**です。

### 4. genuine V-shapeにPCAは正しいか

- PCAは「全点の最小二乗1本線（主軸）」としては正しいです。
- でも **V字の片側エッジ方向を表すわけではない**ため、タスク定義次第では不適切です（対称Vなら二等分方向に寄る）。

### 5. 最終推奨（具体閾値）

- **前処理する**。
- 推奨:
  - `tau_dup = 1.0 px`（最終学習座標系で適用）
  - dedup後、`LI >= 0.995` かつ `RMSE_perp <= 1.0 px` のものは2点化して安定化
  - それ以外（真のV字候補）は2点化しない（必要なら角度損失から除外 or 別扱い）

要するに、**heatmap品質はほぼ維持しつつ、幾何学習のノイズ源だけ除去する最小前処理**が最も安全です。
