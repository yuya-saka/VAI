# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-02 -->

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
├── generate_region_mask.py   ← ⚠️ 要再実装（設計ミス）
├── pilot_region_mask.py
├── visualize_region_masks.py
└── tests/test_generate_region_mask.py
```

実行: `uv run python Unet/line_only/train.py --config Unet/config/config.yaml`

---

## 現在のフェーズ：region-mask 再設計待ち

| Phase | 状態 | 概要 |
|-------|------|------|
| 0〜2 | ✅ | 可視化・影響範囲・テスト作成 |
| 3 | ⚠️ 要再実装 | generate_region_mask.py（設計ミス、詳細→2026-04-02.md） |
| 4 | ✅（参考値） | パイロット100slice（結果は不正） |
| 5〜7 | ⏳ | Phase 3 再実装後 |

**次のアクション**: `generate_region_mask.py` の再設計方針を決定して再実装。
候補: 半平面分割 or 折れ線バリア（詳細→`work-logs/2026-04-02.md`）

---

## 未解決の設計決定

- `heatmap_threshold` を 0.50 に上げるか（現状 0.20）→ GT確認後に判断
- geometry loss 有効化実験（Config B）→ `codex/20260331-geometry-loss-activation.md`

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
| [2026-04-02](work-logs/2026-04-02.md) | region-mask 設計ミス判明・再設計候補整理 |
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装 |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep、GT品質確認 |
| [2026-03-30](work-logs/2026-03-30.md) | 椎体条件付け実装、fold1回帰調査 |
| [2026-03-29](work-logs/2026-03-29.md) | 線損失3ステージ実装（テスト21/21通過） |
| [2026-03-18](work-logs/2026-03-18.md) | 重大バグ修正（Y軸座標・NaN処理・warmup式） |
| [2026-03-13](work-logs/2026-03-13.md) | line_only/ 初期実装 |
