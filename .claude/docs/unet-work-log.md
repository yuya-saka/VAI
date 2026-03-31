# Unet/ 作業サマリー

<!-- 最終更新: 2026-04-01 -->
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

## outputs/ 整理（2026-03-31）

フラットだった `outputs/` をテーマ別に再編。

```
outputs/
├── baseline/              ← baseline_sig2.0, baseline_sig3.5
├── augmentation/          ← 拡張弱め実験
├── regularization/        ← 正則化強化実験
├── vertebrae-onehot/      ← 椎体条件付け実験
├── 過去実装のbaseline/    ← ALL系実験（sig2.0/2.5/3.0/3.5）
└── archive/               ← 旧 past_run の中身
```

**config変更**: `experiment.phase` + `experiment.name` を設定するだけでパスが自動導出される。

```yaml
experiment:
  phase: "regularization"
  name: "sig3.5_正則化強化"
# → outputs/regularization/sig3.5_正則化強化/checkpoints/ & vis/ が自動生成
# → wandb project も unet-regularization-sig3.5_正則化強化 に自動設定
```

---

## 現在の精度（MSE-only, 5-fold CV）

**sigma=3.5 が最良（2026-03-31 確定）:**

| 指標 | sig2.0 | sig3.5 |
|------|--------|--------|
| 角度誤差 (deg) | 6.793 | **6.263** |
| ρ誤差 (px) | 3.808 | **3.465** |
| Peak Dist (px) | 21.34 | **20.76** |

詳細: `.claude/docs/experiments/2026-03-31-regularization-sigma.md`

---

## 確定済み設計決定

| 項目 | 決定内容 | 理由 |
|------|---------|------|
| GT計算 | PCA法（全ポリライン点） | V字ポリラインが57.3%存在、端点法だと角度誤差>10° |
| 予測抽出 | `threshold=0.2`（→0.50検討中） | ノイズ背景によるモーメント不安定性を解消。0.50で mean -1.8°、max -14° 改善（GT確認後に更新予定） |
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
sigma: 3.5                  # 2026-03-31 確定（sig3.5 が全指標で優位）
use_line_loss: false        # true にすると L_line 有効
use_vertebra_conditioning: false
warmup_start_epoch: 90
conf_gate_low: 0.1
conf_gate_high: 0.8
wandb:
  enabled: false            # 実験時に true に変更
```

---

## GT アノテーション品質管理（2026-03-31 完了）

- 検証スクリプト: `Unet/debug/gt_validation.py`（全3640アノテーションをスキャン）
- 可視化スクリプト: `Unet/debug/gt_show_flagged.py`
- 除外リスト: `dataset/bad_slices.json`（4スライス）
- 前処理: `preprocess_polyline()` in `dataset.py`（near-dup除去のみ。2点化は [A,B,A'] パターンで phi が 90° ずれるバグがあり削除済み）

詳細: `work-logs/2026-03-31.md` §5

---

## 次にやること

- [ ] heatmap_threshold を 0.50 に更新するか判断
- [ ] geometry loss 有効化実験（Config B、sigma=3.5）
  - `lambda_angle: 0.08`, `lambda_rho: 0.30`
  - `warmup_start_epoch: 50`, `warmup_epochs: 30`
  - `gate_low: 0.05`, `gate_high: 0.65`
  - 根拠: `.claude/docs/codex/20260331-geometry-loss-activation.md`
- [ ] **目標: angle_error < 5.5°**（現状 avg 6.26°）

---

## 過去ログ（詳細は work-logs/ を参照）

| 日付 | 主な内容 |
|------|---------|
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装（誤差分布・Bland-Altman・worst sample可視化） |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep調査(0.20→0.50で23%改善)、GT品質確認待ち |
| [2026-03-30](work-logs/2026-03-30.md) | 椎体条件付け実装、fold1回帰調査（非決定的挙動と確定） |
| [2026-03-29](work-logs/2026-03-29.md) | 線損失3ステージ実装完了（テスト21/21通過） |
| [2026-03-28](work-logs/2026-03-28.md) | angle_loss設計（Codex相談2回）、実装計画策定 |
| [2026-03-27](work-logs/2026-03-27.md) | sigmaベースライン実験（4条件）、リファクタリング完了 |
| [2026-03-25](work-logs/2026-03-25.md) | PCA法GT・threshold修正、wandb統合 |
| [2026-03-23](work-logs/2026-03-23.md) | threshold処理調査（中央値誤差 28°→10°） |
| [2026-03-18](work-logs/2026-03-18.md) | 重大バグ修正（Y軸座標・NaN処理・warmup式） |
| [2026-03-13](work-logs/2026-03-13.md) | line_only/ 初期実装 |
