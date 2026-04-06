# 領域マスク生成 実装計画

<!-- 作成: 2026-04-01 -->
<!-- 更新: 2026-04-01 (Codexレビュー反映) -->
<!-- 目的: 線アノテーション(4ポリライン)から領域マスク(5ch)を生成し、マルチタスク学習に使用 -->

## 背景

現在は境界線検出（4ch ヒートマップ）のみ学習している。
マルチタスク学習（セグメンテーション＋境界検出）に向けて、線アノテーションから領域マスクを生成する必要がある。

参照: `context/Unet_plan.md` §4（9チャンネル出力設計）

---

## コアアルゴリズム（Codex推奨）

**「T-junction強制 → Ray延長 → ConnectedComponents → Seed-basedラベル割当」**

```
4本のポリライン (line_1, line_2, line_3, line_4)
    ↓ TLS/PCAで各ポリラインを直線フィット
右ペア(line_1, line_2) → 交点 J_R（右T-junction）
左ペア(line_3, line_4) → 交点 J_L（左T-junction）
    ↓ 各junctionから外側方向にrayを延長（向き検証あり）
    ↓ vertebra mask の端まで描画 → barrier (cv2.line)
    ↓ vertebra_mask - barrier を cv2.connectedComponents
    ↓ half-planeでseedを生成 → seedが入るcomponentを採番（seed-based）
    ↓ posterior は残差（body ∪ right ∪ left の補集合）
    ↓ 5ch one-hot mask を出力
```

---

## トポロジー定義

| ch | 領域 | 定義 |
|----|------|------|
| 0 | background | `vertebra_mask == 0` |
| 1 | body（椎体中心部） | `V ∩ body_side(L1) ∩ body_side(L3) ∩ nonposterior(L2) ∩ nonposterior(L4)` |
| 2 | right_foramen（右横突孔） | `V ∩ foramen_side(L1) ∩ nonposterior(L2)` |
| 3 | left_foramen（左横突孔） | `V ∩ foramen_side(L3) ∩ nonposterior(L4)` |
| 4 | posterior（後方要素） | `V \ (body ∪ right_foramen ∪ left_foramen)` ← **常に残差** |

**注意**: `line_1`〜`line_4` の解剖学的対応を可視化で確認してから実装すること。

---

## マスク保存形式（確定）

**ディスク保存**: single-channel label PNG（ピクセル値 0〜4）
**学習時**: one-hot 化（5ch）してから使用

> ⚠️ この形式を変更する場合は loader / augmentation / loss 設計すべてに影響する。

---

## 実装ステップ

### Phase 0: マスク保存形式の確定 ← 新規追加

- [x] 保存形式を single-channel label PNG (0-4) に確定
- [ ] `dataset.py`, `trainer.py`, `model.py` への影響範囲を洗い出す（Phase 5 の準備）

### Phase 1: 可視化・確認

- [ ] サンプルスライスにポリラインをoverlay描画し、line_1〜4の解剖学的対応を目視確認
- [ ] 「line_1が右縦境界、line_2が右横境界、line_3が左縦境界、line_4が左横境界」の仮定を検証
- [ ] **まず C3-C7 で確認**（C1/C2 は Phase 6 以降）

### Phase 2: 合成データ単体テスト ← 新規追加

場所: `Unet/preprocessing/tests/test_generate_region_mask.py`

以下のケースで deterministic test を作成してから Phase 3 に進む：

- [ ] 理想的なT字（正常ケース）
- [ ] line_1/line_2 がほぼ平行（junction fallback）
- [ ] junction が vertebra mask 外に飛ぶ
- [ ] barrier に gap が残る
- [ ] 左右の line が入れ替わった annotation
- [ ] C1/C2 相当の非標準形状

### Phase 3: マスク生成コア実装

場所: `Unet/preprocessing/generate_region_mask.py`

```python
# 主要関数
preprocess_polyline(points)       # 近接重複点除去（既存流用可）
fit_tls_line(points) -> FittedLine   # TLS/PCAで直線フィット
line_intersection(a, b)           # 2直線の交点計算
validate_ray_direction(junction, candidate_endpoint, mask_centroid)
                                  # ★ 向き検証：inward事故防止
extend_ray_to_mask(junction, outward_pt, vertebra_mask)  # ray延長
draw_barrier(shape, segments)     # cv2.line + morphological close
split_regions(vertebra_mask, barrier)  # connectedComponents
assign_labels_by_seed(labels, fitted_lines, vertebra_mask)
                                  # ★ half-plane seed → component採番（centroid依存から変更）
generate_region_mask(
    line_1, line_2, line_3, line_4, vertebra_mask
) -> tuple[np.ndarray, dict]      # ★ (5ch mask, debug_info) を返す
```

**`debug_info` に含める情報**:
- junction座標 `J_R`, `J_L`
- fallback の種別と stage
- barrier後の connected component 数
- 各領域の面積

### Phase 4: パイロット前処理（100 slice）← 新規追加

- [x] C3-C7 からランダム100スライスで `generate_region_mask` を実行
- [x] 失敗パターンの taxonomy を作成（`bad_slices.json` に `reason` と `stage` を記録）
- [x] failure rate が許容範囲か確認してから Phase 5 へ
  → 全7椎骨・全909スライスに拡張: 906/909成功 (99.67%)

### Phase 5: バッチ前処理スクリプト

場所: `Unet/preprocessing/preprocess_all.py`

- [ ] 全サンプル・全椎体・全スライスに対して実行
- [ ] 結果を `dataset/sampleXX/Cx/gt_masks/slice_XXX.npy` に保存（5ch uint8）
- [ ] 生成失敗スライスを `bad_slices.json`（`reason` + `stage` 付き）に記録

### Phase 6: QA可視化 + QCスコアリング

- [x] ランダム20スライスをCT + 4色region overlay + 4本separator で描画
  → `Unet/preprocessing/output/region_mask_viz/combined_grid.png`
- [x] QCスコアリング実装: `Unet/preprocessing/qc_score.py`
  → keep 859件 / downweight 43件 / exclude 7件
  → 各椎骨に `qc_scores.json` 保存済み
  → flagged可視化: `Unet/preprocessing/output/qc_viz/qc_flagged.png`
- [x] **方針確定**: exclude はテスト時のみ、downweight は訓練に使用

### Phase 7: データセット・学習系統合

**影響ファイル（Codex指摘）**:
- `Unet/line_only/src/dataset.py:169, 329, 359` — region_mask 読み込み追加
- `Unet/line_only/src/trainer.py:98, 236` — 9ch 出力対応
- `Unet/line_only/src/model.py:35` — 出力チャンネル変更
- visualization / augmentation（左右反転時の class remap 対応必須）

**統合手順**:
1. まず「GT region mask の読込と可視化」だけを通す smoke test（9ch 化は後）
2. smoke test 通過後に model / loss を 9ch に拡張
3. C1/C2 は C3-C7 で成立性確認後に追加

---

## フォールバック戦略（優先順・修正版）

| 優先度 | 戦略 | 説明 |
|--------|------|------|
| 1 | junction snap / bridge | 近傍endpointを短いbridgeで結ぶ |
| 2 | local tangent refit | junction側3-5点で局所接線フィット（強い湾曲に対応） |
| 3 | junction clamp / ray re-direction | junctionをmask内にclamp + 向き再検証 |
| 4 | barrier reinforcement | 1px dilation / morphological closing |
| 5 | seed-based relabel / merge | 小さなゴミ成分を最近接大成分にmerge |
| 6 | watershed（last resort） | **seed を body/right/left/posterior に固定して使用**、marker なし禁止 |

---

## エッジケース対応

| 状況 | 対応 |
|------|------|
| line_1とline_2がほぼ平行 | 最近endpointのmidpointをjunctionに |
| junctionがmask外に飛ぶ | mask内最近点にclamp |
| ポリラインが強く湾曲 | junction側3-5点で局所接線フィット |
| バリア後の成分数 ≠ 4 | bridge強化 → watershed（seed固定） |
| line_i が 2点未満 | そのスライスをinvalidとしてスキップ |
| vertebra_mask 自体に穴/分断あり | invalid扱い、ログに記録 ← **追加** |
| annotation order inconsistency | centroidの左右位置で検出してflag ← **追加** |
| line が mask 外を長く走る | mask境界でクリップ ← **追加** |
| 左右反転 augmentation | class remap (right_foramen ↔ left_foramen) 必須 ← **追加** |
| C1/C2 の非標準 topology | Phase 6 以降で別対応 ← **追加** |

---

## 検証チェックリスト（拡充版）

各スライスで自動チェック（`validate_region_mask` 関数）:

**ハードfail（スキップ）**:
- [ ] one-hot性: `sum(ch axis) == 1` for all pixels
- [ ] vertebra内完全被覆: `vertebra_mask=1` のpixelは必ずいずれかの領域
- [ ] 背景整合: `vertebra_mask=0` のpixelはbackgroundのみ
- [ ] topology整合: right_foramen は L1 の foramen_side かつ L2 の nonposterior_side
- [ ] J_R/J_L が最終的に mask 内にあるか ← **追加**
- [ ] right/left の x-order が保たれているか ← **追加**

**警告のみ（warning）**:
- [ ] 各 region 面積 > 0（ゼロなら warning、hard fail ではない） ← **修正**
- [ ] barrier後の connected component 数 ← **追加**
- [ ] 各 polyline が対応境界に 1-2px 以内で沿うか ← **追加**
- [ ] slice 間の急激な面積ジャンプ ← **追加**

---

## 未確定事項

1. **line_1〜4 の解剖学的対応**: Phase 1 の可視化で確認（実装前必須）
2. ~~マスク保存形式~~ → **確定**: single-channel label PNG (0-4)
3. **事前バッチ生成 vs オンライン生成**: ディスク容量とのトレードオフ（推奨: 事前生成）

---

## 参照

- Codex分析: `.claude/docs/codex/20260401-region-mask-plan.md`
- Codexレビュー: `.claude/docs/codex/20260401-region-mask-review.md`
- 研究計画: `context/Unet_plan.md`
- 現行データセット: `Unet/line_only/src/dataset.py`
