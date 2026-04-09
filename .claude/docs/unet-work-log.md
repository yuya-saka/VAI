# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-09 (session 3) -->

## 精度比較（5-fold CV, 指標統一済み）

| モデル | perp (px) | angle (°) | rho (px) | fg_miou | fg_mdice |
|--------|:-:|:-:|:-:|:-:|:-:|
| line_only sig3.5 | 11.55 | 5.53 | 3.29 | — | — |
| seg_only_v1 | — | — | — | 0.882 | 0.936 |
| multitask_v1 (re-run) | 11.56 | 6.07 | 3.54 | 0.888 | 0.939 |

- multitask: seg は seg_only を上回る。直線は line_only より angle/rho がやや劣る
- C1 perp≈22px（突出して悪い）、C7 perp≈13px。C3〜C5 が最良（8px台）

---

## 現在の問題

**multitask の heatmap が line_only より"太く・鈍く"なる negative transfer**

- 原因: dense な seg 勾配（全ピクセル）が sparse な line 信号を圧倒
- epoch 28 以降: seg_miou は改善し続けるが line metrics は振動・停滞
- seg head は seg_loss なしでは学習不可（line_loss の勾配は seg head に流れない）

## 次のアクション（優先順）

1. **alpha_seg を下げる実験**（現在 0.03）
2. **seg loss 遅延導入**（最初 10〜20 epoch は alpha_seg=0）
3. **line_only checkpoint で encoder 初期化** → seg head だけ立ち上げ → unfreeze
4. **weighted MSE heatmap loss**（peak 帯強調: `w = 1 + 8*gt`）
5. **C1/C7 vertebra-aware weighting**（C1: 1.75〜2.0x, C7: 1.25〜1.5x、line loss のみ）

参考: `.claude/docs/codex/20260409-1830-accuracy-improvement.md`
参考: `.claude/docs/codex/20260409-1900-heatmap-improvement.md`

---

## コード構成

実行: `uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml`
テスト: `uv run pytest Unet/multitask/test/ -v`（13/13 pass）

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
| [2026-04-09](work-logs/2026-04-09.md) | seg_only/ プロジェクト新規作成（11/11テスト pass）・background_weight/gamma_dice 設計方針確認 |
| [2026-04-08](work-logs/2026-04-08.md) | Phase 5完了(13/13 pass)・fold 0実験: angle 5.42deg / mIoU 0.911 |
| [2026-04-07](work-logs/2026-04-07.md) | Phase 7完了: dataset.py に gt_masks 読込統合・bad_slices フォーマット修正 |
| [2026-04-06](work-logs/2026-04-06.md) | 半平面分割実装・CT-maskアフィンズレ修正・全件バッチ保存(906/909)・QCスコアリング完了 |
| [2026-04-05](work-logs/2026-04-05.md) | 半平面分割プロトタイプ検証(99.9%)・方針決定 |
| [2026-04-02](work-logs/2026-04-02.md) | region-mask 設計ミス判明・再設計候補整理 |
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装 |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep、GT品質確認 |
