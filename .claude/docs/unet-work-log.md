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
├── visualize_region_masks.py
└── tests/test_generate_region_mask.py
```

実行: `uv run python Unet/line_only/train.py --config Unet/config/config.yaml`

---

## 現在のフェーズ：TotalSegmentatorマスク品質対策の検討

| Phase | 状態 | 概要 |
|-------|------|------|
| 0〜3 | ✅ | generate_region_mask.py 半平面分割で再実装（870行→236行） |
| 4 | ✅ | パイロット100スライス: **100/100成功** ・可視化確認OK |
| 5 | ⏳ | TotalSegmentatorマスク歪み対策（次セッション） |
| 6〜 | ⏳ | 訓練パイプラインへの組み込み |

**課題 (2026-04-06)**:
- TotalSegmentatorマスク自体が歪むと領域品質が低下
- Codex推奨優先順: ① QCフィルタリング → ② eroded mask で外周依存を下げる
- 詳細: `codex/20260406-mask-quality-strategy.md`

**次のアクション**:
1. TotalSegmentatorマスクのQCスコアリング実装（`solidity`・隣接スライス連続性・line端点がmask内か）
2. eroded maskベースの領域生成を試験（boundary歪み影響を低減）

---

## 未解決の設計決定

- `heatmap_threshold` を 0.50 に上げるか（現状 0.20）→ GT確認後に判断
- geometry loss 有効化実験（Config B）→ `codex/20260331-geometry-loss-activation.md`

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
| [2026-04-06](work-logs/2026-04-06.md) | 半平面分割実装完了・テスト6/6・パイロット100/100・マスク品質課題整理 |
| [2026-04-05](work-logs/2026-04-05.md) | 半平面分割プロトタイプ検証(99.9%)・方針決定 |
| [2026-04-02](work-logs/2026-04-02.md) | region-mask 設計ミス判明・再設計候補整理 |
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装 |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep、GT品質確認 |
| [2026-03-30](work-logs/2026-03-30.md) | 椎体条件付け実装、fold1回帰調査 |
| [2026-03-29](work-logs/2026-03-29.md) | 線損失3ステージ実装（テスト21/21通過） |
