# Codex Analysis: Region Mask Redesign
Date: 2026-04-05

## Question
4本のポリラインアノテーションから領域マスクを生成する手法の再設計。
現行実装は交差前提で80%がfallback → 破綻。

## Codex Response (Summary)

**推奨: Approach B (Polyline Barrier) 主体 + Approach A (Half-plane) を補助にした Hybrid**

### 核心的な問題
`line_intersection()` で仮想交点を作りrayを延ばす設計が根本的に間違い。

### 推奨アーキテクチャ

1. **body_seed を distance transform で計算**
   - `body_seed = argmax(distance_transform(erode(V, r=3)))` — 重心より安定

2. **交点計算を廃止 → endpoint gap bridge に置き換え**
   - line_1-2, line_3-4 の端点ギャップを短いブリッジで接続

3. **free endpoint の延長方向は local tangent で決定**
   - endpoint 近傍 3-5 点の local tangent を使用
   - ray march で vertebra mask 境界まで延長

4. **half-plane は prior/seed 生成のみに使用**
   - body_seed が含まれる側 = body_side と自動決定
   - 最終マスクの境界には使わない

5. **posterior は残差として計算**

6. **V-shape への対応**
   - バリアとしてそのまま保持（直線化しない）
   - 延長方向の計算のみ local tangent を使用

### 残すべきコード
- `preprocess_polyline()`, `draw_barrier()`, `split_regions()`
- `validate_region_mask()`, `_component_info()`, `_merge_small_components()`

### 廃止すべきコード
- `line_intersection()` 主経路全体
- 多段フォールバック群（junction_snap, local_tangent_refit, junction_clamp, etc.）

### Validation 指標
- `inside_coverage_error`, `outside_leak_error`, `one_hot_error`
- `polyline_boundary_dist_mean/p95/max`
- `bridge_length_norm`, `extension_length_norm`
- `lr_symmetry`, `anatomy_order_ok`
