# Unet/ 作業サマリー

<!-- 最終更新: 2026-03-30 -->
<!-- 大きな変更（新実験結果・設計変更・コード構成変更）があった時だけ更新する -->

## プロジェクト概要

脊椎X線画像から椎体間の直線（φ, ρ）をヒートマップ経由で検出するU-Net。
実装は `Unet/line_only/` に集約。

---

## 現在のコード構成

```
line_only/
├── train.py              (154行)
├── src/
│   ├── model.py          ← 椎体条件付けあり（bottleneck concat + one-hot）
│   ├── dataset.py
│   ├── data_utils.py
│   └── trainer.py
└── utils/
    ├── losses.py         ← L_mse + w(t)·L_line
    ├── metrics.py
    ├── detection.py      ← threshold=0.2、PCA法GT
    └── visualization.py
```

実行: `uv run python Unet/line_only/train.py --config Unet/config/config.yaml`

---

## 現在の精度（MSE-only, sigma=2.0, 5-fold CV）

| fold | angle_error |
|------|------------|
| fold0 | 5.57° |
| fold1 | 7.60° |
| fold2 | 5.08° |
| fold3 | 4.26° |
| fold4 | 7.99° |
| **avg** | **6.10°** |

---

## 確定済み設計決定

| 項目 | 決定内容 | 理由 |
|------|---------|------|
| GT計算 | PCA法（全ポリライン点） | V字ポリラインが57.3%存在、端点法だと角度誤差>10° |
| 予測抽出 | `threshold=0.2` | ノイズ背景によるモーメント不安定性を解消 |
| confidence定義 | `(lam1-lam2)/(lam1+lam2+eps)` | [0,1]正規化、NaN廃止 |
| 角度損失 | `1 - dot²`（sin²(Δφ)と等価） | cuspなし、全域滑らか、π周期 |
| ρ損失 | detach符号整合 + SmoothL1 | 勾配不連続を回避 |
| MSE重み | `L_mse + w(t)·L_line`（MSEは常に1.0） | MSE品質を損なわない |
| confidence gate | `conf_gate_low=0.1`, `conf_gate_high=0.8` | ソフトゲートで安定学習 |
| warmup開始 | `warmup_start_epoch=90`（val_mse収束後） | epoch 96以降で角度抽出が信頼できる |
| 椎体条件付け | bottleneck concat + one-hot | 小規模データ(<100患者)に最適（Codex確認） |
| aug | `A.Affine`（ShiftScaleRotateから移行） | 精度への影響なし（差0.20°） |

---

## 現在の config.yaml 設定

```yaml
sigma: 2.0
use_line_loss: false        # trueにすると L_line 有効
use_vertebra_conditioning: true
num_vertebra: 7
warmup_start_epoch: 90
conf_gate_low: 0.1
conf_gate_high: 0.8
wandb:
  enabled: false            # 実験時に true に変更
```

---

## 次にやること

- [ ] `use_line_loss: true` で fold0 実験（sigma=2.0、`warmup_start_epoch=90`）
- [ ] 椎体条件付けの効果確認（条件なし baseline との比較）
- [ ] **目標: angle_error < 5°**（現状 avg 6.10°）

---

## 過去ログ（詳細は work-logs/ を参照）

| 日付 | 主な内容 |
|------|---------|
| [2026-03-30](work-logs/2026-03-30.md) | 椎体条件付け実装、fold1回帰調査（非決定的挙動と確定） |
| [2026-03-29](work-logs/2026-03-29.md) | 線損失3ステージ実装完了（テスト21/21通過） |
| [2026-03-28](work-logs/2026-03-28.md) | angle_loss設計（Codex相談2回）、実装計画策定 |
| [2026-03-27](work-logs/2026-03-27.md) | sigmaベースライン実験（4条件）、リファクタリング完了 |
| [2026-03-25](work-logs/2026-03-25.md) | PCA法GT・threshold修正、wandb統合 |
| [2026-03-23](work-logs/2026-03-23.md) | threshold処理調査（中央値誤差 28°→10°） |
| [2026-03-18](work-logs/2026-03-18.md) | 重大バグ修正（Y軸座標・NaN処理・warmup式） |
| [2026-03-13](work-logs/2026-03-13.md) | line_only/ 初期実装 |
