---
name: multitask implementation progress
description: Unet/multitask/ の実装進捗状況（Phase 1〜4 完了、Phase 5 未着手）
type: project
---

Unet/multitask/ の実装を Codex に委託して進行中。

**Why:** line_only の line 検出精度向上を目的とした multitask ResUNet の実装。seg（5クラス）と line（4ch ヒートマップ）を同時学習。

**How to apply:** 次セッションは Phase 5（テスト実装 + pytest 実行）から再開。

## 進捗（2026-04-08）

| Phase | 状態 | 内容 |
|-------|------|------|
| Phase 1 | ✅ | config.yaml, __init__.py, detection.py, visualization.py |
| Phase 2 | ✅ | losses.py, metrics.py, model.py（ResUNet 1,581,781 params） |
| Phase 3 | ✅ | dataset.py, data_utils.py（WeightedRandomSampler） |
| Phase 4 | ✅ | trainer.py, train.py（import OK） |
| Phase 5 | ⬜ | test/test_model.py, test/test_losses.py |

## 次のアクション（次セッション）

1. Codex で `test/test_model.py`, `test/test_losses.py` を実装
2. `uv run pytest Unet/multitask/test/ -v` 全 pass 確認
3. 1 fold dry-run（数エポック）で学習動作確認

## 重要な実装メモ

- ResBlock: `groups = min(norm_groups, ch)` でクランプ（in_ch=2 の入力段で必要）
- model.forward(x) は dict を返す: `{'seg_logits': ..., 'line_heatmaps': ...}`
- trainer に VERTEBRA_TO_IDX / v_idx なし
- WeightedRandomSampler: seg GT あり = 重み 3.0、なし = 1.0
- alpha_seg = 0.03（cfg['loss']['alpha_seg']）
