# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-09 (session 2) -->

## 精度比較

| モデル | Angle (deg) | ρ (px) | Peak (px) | Seg mIoU |
|--------|-------------|--------|-----------|----------|
| line_only（5-fold CV） | 6.263 | 3.465 | 20.76 | — |
| multitask_v1 baseline（fold 0） | **5.424** | **3.368** | 20.58 | 0.911 |

詳細: `Unet/outputs/multitask_v1/baseline/checkpoints/all_folds_summary.json`

---

## コード構成

```
multitask/
├── train.py
├── config/config.yaml
├── src/   model.py / dataset.py / data_utils.py / trainer.py
├── utils/ losses.py / metrics.py / detection.py / visualization.py
└── test/  test_model.py / test_losses.py  ✅ 13/13 pass
```

実行: `uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml`

---

## 現在のフェーズ：multitask 実装完了 → 比較実験フェーズ

| Phase | 状態 | 概要 |
|-------|------|------|
| Phase 1〜4 | ✅ | multitask 全モジュール実装 |
| Phase 5: テスト | ✅ | test_model.py, test_losses.py（13/13 pass） |
| fold 0 実験 | ✅ | multitask_v1/baseline 完了（angle 5.42 deg） |

**損失設計（確定）**:
- `L = L_line(MSE) + 0.03·L_seg(CE)`
- 幾何損失（angle/rho）は導入しない
- Sampler で seg-labeled を 3〜4/8 確保

---

## 次のアクション

1. **seg_only fold 0 実験実行**（次セッション）
   - `uv run python Unet/seg_only/train.py --config Unet/seg_only/config/config.yaml`
   - primary metric: fg_mdice（background除く4クラス平均Dice）
2. multitask_v1 baseline の 5-fold CV 実行（fold 1〜4）
3. seg-only vs multitask の比較（fg_mdice で比較）

---

## seg-only 設計決定（2026-04-09 確定）

Codex 分析（`.claude/docs/codex/20260408-seg-only-design.md`）をもとに確定。

### アーキテクチャ
- `SegOnlyUNet`（Encoder + seg_decoder のみ）を新クラスとして `model.py` に追加
- 既存の `Encoder` / `Decoder` クラスを再利用、line_decoder は持たない
- config フラグ（`model.type: SegOnlyUNet`）で切り替え

### 学習データ
- `has_seg_label==True` のサンプルのみ使用
- seg ラベルは十分な量あるため転移学習不要、スクラッチで学習

### Loss
- **Weighted CE + 0.3〜0.5 × Dice_fg**（background 除く fg クラスの Dice）
- CE weight: `1/sqrt(freq)` で計算、background は 0.25〜0.5 に抑える
- 現在の `compute_multitask_loss` とは別の `compute_seg_only_loss` を新規実装

### 評価指標
- **fg-mDice**（background 除く 4 クラス平均）を primary metric に変更
- fg-mIoU、per-class Dice/IoU（body/right/left/posterior）も出力
- 現在の mIoU/Dice は background 込みなので参考値として残す

---

## 保存形式（確定）

- `dataset/{sample}/{vertebra}/gt_masks/slice_{idx:03d}.png`
- 単一チャネルラベルPNG（値 0=bg, 1=body, 2=right, 3=left, 4=posterior）

---

## 未解決の設計決定

- Dice を summary JSON に追加するか（現状 mIoU のみ保存）
- geometry loss 有効化実験（Config B）→ `codex/20260331-geometry-loss-activation.md`

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
