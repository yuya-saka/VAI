# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-06 -->

## 現在の精度（MSE-only, sigma=3.5, 5-fold CV）

| 指標 | 値 |
|------|----|
| 角度誤差 (deg) | 6.263 |
| ρ誤差 (px) | 3.465 |
| Peak Dist (px) | 20.76 |

詳細: `.claude/docs/experiments/2026-03-31-regularization-sigma.md`

---

## コード構成

```
line_only/
├── train.py
├── src/  model.py / dataset.py / data_utils.py / trainer.py
└── utils/  losses.py / metrics.py / detection.py / visualization.py

preprocessing/
├── generate_region_mask.py   ✅ 半平面分割で再実装済み（236行）
├── pilot_region_mask.py
├── preprocess_all.py         ✅ 全件バッチ保存スクリプト（Phase 5完了）
├── visualize_region_masks.py
└── tests/test_generate_region_mask.py
```

実行: `uv run python Unet/line_only/train.py --config Unet/config/config.yaml`

---

## 現在のフェーズ：GT領域マスク保存完了 → 訓練パイプライン統合へ

| Phase | 状態 | 概要 |
|-------|------|------|
| 0〜3 | ✅ | generate_region_mask.py 半平面分割で再実装（870行→236行） |
| 4 | ✅ | パイロット100スライス: 100/100成功・全件909スライス: 906/909成功 |
| 5 | ✅ | `preprocess_all.py` 実行完了 → `gt_masks/slice_{idx:03d}.png` 全件保存（単一ラベルPNG, 値0-4） |
| 6 (QC) | ✅ | QCスコアリング実装済み: keep 859件 / downweight 43件 / exclude 7件 |
| 7 | ⏳ | `dataset.py` 統合: gt_masks読込 + qc_scores.json excludeスキップ |

**保存形式（確定）**:
- `dataset/{sample}/{vertebra}/gt_masks/slice_{idx:03d}.png`
- 単一チャネルラベルPNG（ピクセル値 0=bg, 1=body, 2=right, 3=left, 4=posterior）
- ※値が0〜4なので画像ビューアでは真っ黒に見える（正常）

**次のアクション**:
1. GTマスクのカラーオーバーレイ可視化スクリプト作成（`visualize_gt_masks.py`）
2. Phase 7: `dataset.py` に gt_masks 読込 + exclude スキップを統合

---

## 未解決の設計決定

- `heatmap_threshold` を 0.50 に上げるか（現状 0.20）→ GT確認後に判断
- geometry loss 有効化実験（Config B）→ `codex/20260331-geometry-loss-activation.md`

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
| [2026-04-06](work-logs/2026-04-06.md) | 半平面分割実装・CT-maskアフィンズレ修正・全件バッチ保存(906/909)・QCスコアリング完了 |
| [2026-04-05](work-logs/2026-04-05.md) | 半平面分割プロトタイプ検証(99.9%)・方針決定 |
| [2026-04-02](work-logs/2026-04-02.md) | region-mask 設計ミス判明・再設計候補整理 |
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装 |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep、GT品質確認 |
| [2026-03-30](work-logs/2026-03-30.md) | 椎体条件付け実装、fold1回帰調査 |
| [2026-03-29](work-logs/2026-03-29.md) | 線損失3ステージ実装（テスト21/21通過） |
