# Codex Review: Region Mask Plan
Date: 2026-04-01

## Plan Reviewed
.claude/docs/region-mask-plan.md

## Codex Response

**Findings**

1. **合成データでの単体テスト段階が欠如**
   現行計画は可視化確認の直後にコア実装と全件バッチへ進んでいるが、幾何バグが入ると全データに静かに混入する。「理想的T字」「ほぼ平行」「junctionがmask外」「barrier gap」「左右入れ替わり」「C1/C2相当の歪形」で deterministic test を先に作るべき。

2. **`assign_labels = centroid + line-side` は脆い**
   成分が細長い、posteriorが大きい、gap補修後に成分形状が崩れるケースで重心が期待領域を外す可能性がある。ラベル割当は「half-plane で作った期待領域から seed を作り、その seed が入る connected component を採番する」方式に変えた方が安定する。

3. **Phase 5 はスコープ不足**
   今の学習系は 4ch heatmap 前提が広く埋め込まれており、dataset/model/trainer/visualization/augmentation 全部を触る。「region_mask を読む」「出力を 9ch にする」だけでは足りない。

4. **マスク保存形式は Phase 3 の前に決めるべき**
   未確定のままバッチ前処理設計に入ると loader・augmentation・loss 設計が二度手間になる。推奨は「disk は single-channel label PNG(0-4)、学習時に one-hot 化」。

5. **フォールバックの `watershed` は marker 定義なしでは危険**
   thin barrier と concave mask の組み合わせでは、解剖学と無関係な分割を作りやすい。watershed を使うなら body/right/left/posterior の seed を固定して最後の最後に限定すること。

6. **「4領域すべて area > 0 を fail」は hard-fail 条件として強すぎる可能性**
   極端な頭尾側スライスや狭小領域で偽陽性 fail を増やす。`warning` と `hard fail` を分けるべき。

7. **`outward ray` の向き決定が未定義**
   endpoint 順序は安定しない前提で設計すべき。単純な端点選択では inward 側へ伸びる事故が出る。junction から最遠点を候補にしつつ、「mask重心から離れる向き」「既存 polyline 末端側」などで検証する必要がある。

8. **C1/C2 は別フェーズに切り出した方が安全**
   まず C3-C7 で成立性を固め、失敗統計を見てから上位頸椎へ拡張するのが妥当。

---

**Answers to Review Questions**

1. **実装ステップの順序について**
   修正推奨。`可視化確認 -> 合成単体テスト -> コア実装 -> 小規模pilot batch -> QA可視化 -> full batch -> dataset/model/loss統合 -> train smoke test` を推奨。

2. **T-junction → Ray → ConnectedComponents → Label assignment アルゴリズムについて**
   アルゴリズム自体は sound。特に「posterior を残差にする」「infinite line ではなく ray を使う」は良い。ただし成功条件は「T-junctionの強制」「seed-based label assignment」「fallback前後の topology check」が入ること。

3. **フォールバック戦略について**
   `junction snap/bridge -> local tangent refit -> junction clamp/ray re-direction -> barrier reinforcement -> seed-based relabel/merge -> watershed(last resort)` の順が安全。

4. **未記載エッジケース**
   - `vertebra_mask 自体の穴/分断`
   - `annotation order inconsistency`
   - `line が mask 外を長く走る`
   - `左右反転 augmentation 後の class remap`
   - `C1/C2 の非標準 topology`

5. **検証チェックリストの不足点**
   追加で以下が必要：
   - J_R/J_L が最終的に mask 内か
   - barrier後 component 数
   - 各 polyline が対応境界に 1-2px 以内で沿うか
   - right/left の x-order が保たれているか
   - slice 間の急激な面積ジャンプ

6. **全体のリスク**
   - annotation semantics の取り違え
   - batch 生成の静かな汚染
   - augmentation 時の左右ラベル崩れ
   - C1/C2 への早すぎる一般化

7. **改善提案（効果大 3点）**
   - label assignment を centroid 依存から seed 依存へ変更
   - 保存形式を単一ラベルPNGに固定
   - 統合前に pilot 100 slice 程度で failure taxonomy を取る

---

**推奨追加項目**

1. `generate_region_mask` の戻り値とは別に `debug_info` を返し、junction座標・fallback種別・component数・各領域面積を保存する。
2. `bad_slices.json` には単なる列挙でなく、`reason` と `stage` を持たせる。
3. 学習統合時は 9ch を一気に入れず、まず「GT region mask の読込と可視化」だけを通す smoke test を 1 段入れる。
