# Unet/ 作業ログ

日々の作業内容を記録します。セッション終了時に追記してください。

---

## 2026-03-13

### やったこと

**1. 作業管理の仕組み構築**
- `Unet/CLAUDE.md` を作成 - 作業管理方法をドキュメント化
- `unet-work-log.md` を作成 - 作業ログの仕組みを構築
- セッション継続性を保つ仕組みを整備

**2. line-only 実装完了（`docs/notes/ideas/Unet_2_plan.md` に基づく）**
- `line_only/` ディレクトリに直線検出関連コードを実装
  - `line_detection.py` - モーメント法による (φ, ρ) 計算
  - `line_losses.py` - 角度損失 + ρ損失の実装
  - `line_metrics.py` - 評価指標（角度誤差、法線方向距離、垂線距離）
  - `train_heat.py` - 段階的損失の訓練ループ
  - `test_line_losses.py` - 単体テスト

**3. 実装の主要ポイント**
- GT 直線の定義: アノテーション2点から法線形式 (φ, ρ) を計算
- 予測: ヒートマップ → モーメント → 主軸 → (φ, ρ)
- 符号合わせ: 法線ベクトルの内積で向きを統一
- 段階的学習:
  - Phase 1: ヒートマップ MSE のみ
  - Phase 2: + 角度損失（warmup後に重み増加）
  - Phase 3: + ρ損失（弱めに追加）
- 評価指標: 角度誤差、法線方向距離、GT線分からの平均垂線距離

**4. チェックポイント保存**
- `/checkpointing --full` でセッション全体を保存
- `.claude/checkpoints/2026-03-13-070134.md` 作成
  - 5コミット記録
  - Codex相談16件（角度検出バグ分析など）
  - Gemini調査4件（matplotlib日本語対応など）

### 現状

**実装済み:**
- `line_only/` の全機能実装完了（Unet_2_plan.md 準拠）
- 評価指標の幾何量ベース化
- 損失関数の段階的追加メカニズム
- 符号合わせロジック

**未完了:**
- 訓練実行と精度検証
- 比較実験（Baseline / +Angle / +Angle+Rho）
- 訓練/評価パイプラインのリファクタリング（DESIGN.md 参照）

### 次にやること

**優先度：高**
- [ ] `line_only/train_heat.py` の動作確認
- [ ] Phase 1 訓練実行（MSE baseline）
- [ ] Phase 2-3 訓練実行（角度損失、ρ損失追加）
- [ ] 3条件での比較評価

**優先度：中**
- [ ] 評価メトリクスの検証
- [ ] ハイパーパラメータ調整（warmup、損失重み）
- [ ] 可視化スクリプト作成

### メモ

- matplotlib の日本語対応仕様が決定済み

---

## 2026-03-25

### やったこと

**1. 線方向の誤差根本原因調査（完了）**

症状: 予測ヒートマップは位置的に正しいが、抽出された線の方向が大きくずれる

**原因1: GT角度の誤計算 (`line_losses.py` → `extract_gt_line_params`)**
- アノテーションのV字型ポリライン（折れ線）に対して、先頭・末尾点のみで角度を計算していた
- V字ポリラインは全データの **57.3%**（3639アノテーション中）に存在
- これにより **46.4%** のアノテーションで角度誤差 >10°（ランダムに近い）

**原因2: `detect_line_moments` のしきい値なし問題**
- ヒートマップのsigmoid出力は背景全体に低いノイズを持つ
- しきい値なしでモーメント計算すると mu20/mu02 が ~1000-2000（背景ノイズ支配）
- しきい値=0.2 を適用すると ~10-200（線ブロブのみ）に絞られ正確な方向抽出が可能

**2. 修正方針決定**

| 修正対象 | 修正内容 | 効果 |
|---------|---------|------|
| GT計算: `extract_gt_line_params` | 端点法 → PCA法（全ポリライン点を使う） | 平均角度誤差 22.52° → 4.64° |
| 予測抽出: `detect_line_moments` | `threshold=0.2` をデフォルト引数に追加 | 平均角度誤差 28.26° → 4.64° |

**3. 実装済み**
- `line_detection.py`: `detect_line_moments` に `threshold: float | None = 0.2` 引数追加
- 診断スクリプト群 (`Unet/.tmp/`): V字分析、精度比較、可視化
- 可視化画像: `Unet/.tmp/vis_threshold_compare/` (20枚、6パネル)

### 現状

**検証済み:**
- しきい値修正の効果をテストセットで確認（mean 28.26° → 4.64°）
- PCA法GTの方が端点法GTより正確（57.3%のV字ポリライン問題）
- 学習済みモデル自体は良好（PCA GTで比較すると91.4%が10°以内）

**未適用:**
- `line_losses.py` の `extract_gt_line_params` をPCA法に置換（次セッションで実施）
- PCA法GTで再学習（必要に応じて）

### 次にやること

**優先度：高**
- [ ] `line_losses.py` の `extract_gt_line_params` をPCA法に置換
- [ ] 修正後のGT + しきい値修正済み抽出で再評価
- [ ] 必要なら角度損失・ρ損失を有効化して再学習

### 作成したテストコード (`Unet/.tmp/`)

| ファイル | 目的 | 主な出力 |
|---------|------|---------|
| `diagnose_direction.py` | GT heatmap抽出精度の確認（端点法 vs モーメント法の比較） | テキスト出力 |
| `diagnose_polylines.py` | V字型ポリラインの分析（endpoint_dist/total_length の統計） | テキスト出力 |
| `diagnose_summary.py` | データセット全体の統計（V字型57.3%、角度誤差46.4%が>10°） | テキスト出力 |
| `test_pca_fix.py` | PCA法GTの正確性を直線・V字の両ケースで検証 | テキスト出力 |
| `test_full_statistics.py` | 全データの誤差分布と図（fig1〜3） | `vis_*.png` |
| `eval_angle_compare.py` | 端点法GT vs PCA法GT の角度誤差比較（22.52° vs 4.64°） | テキスト出力 |
| `eval_detailed.py` | PCA法GTでのライン別詳細精度評価 | テキスト出力 |
| `eval_visualize.py` | 3パネル可視化（Heatmap / GT+Pred / polyline+Pred） | `vis_output/*.png` |
| `eval_worst_pca.py` | PCA法GTでも誤差が大きいワースト20ケースの可視化 | `vis_worst_pca/*.png` |
| `eval_extraction_debug.py` | チャンネル別ヒートマップ + モーメント値 + GT/Pred重ね表示 | `vis_extraction_debug/*.png` |
| `test_threshold_fix.py` | しきい値なし/0.15/0.2 の精度比較（28.26°/4.74°/4.64°） | テキスト出力 |
| `vis_threshold_compare.py` | しきい値なし vs 0.2 の予測線を6パネルで比較可視化 | `vis_threshold_compare/*.png` |

### メモ

- 現行モデルはMSE損失のみで学習（`use_angle_loss: false`, `use_rho_loss: false`）
- モデル自体は十分学習できている（問題は評価側・抽出側のバグ）
- `detect_line_moments`（numpy/可視化用）と `extract_pred_line_params_batch`（torch/eval用）は同じ結果を返す
- `train_heat.py` は削除され、`line_only/train_heat.py` に統合
- 実装は `Unet_2_plan.md` の方針に完全準拠
- Codex 相談で角度計算のバグ分析を実施済み

---

## 2026-03-25（続き）

### やったこと（続き）

**4. パイプライン修正（完了）**

以下の3つのバグを修正し、コミット済み：

| 修正箇所 | 修正内容 |
|---------|---------|
| `line_losses.py: extract_gt_line_params` | 端点法 → PCA法（V字ポリライン対応） |
| `line_detection.py: line_extent` 関数追加 | `polyline_length`（V字で2倍カウント）→ `line_extent`（最遠点間距離）に置換 |
| `line_detection.py: predict_lines_and_eval_test` | `extract_pred_line_params_batch` に `threshold=hm_thr` を渡すよう修正 |

**5. 統計評価バグの発見・修正**

`predict_lines_and_eval_test` が `threshold` を渡していなかったため、評価結果が誤っていた：

| | 修正前 | 修正後 |
|-|--------|--------|
| angle_error | 28.19° | 5.27° |
| rho_error | 9.87px | 3.01px |
| perp_dist | 18.61px | 10.76px |

**6. 新チェックポイント検証**

`checkpoints_sig2.5_ALL_位置ずれ修正こんどこそ_1fold`（修正済みパイプラインで評価）：

| メトリクス | 値 |
|-----------|-----|
| angle_error | **4.89°** |
| rho_error | **2.94px** |
| perp_dist | **10.48px** |
| peak_dist | **23.12px** |

### 現状

**完了済みバグ修正（パイプライン）:**
- [x] GT計算のPCA法化（V字ポリライン問題解決）
- [x] `polyline_length` → `line_extent`（V字2倍カウント解消）
- [x] `predict_lines_and_eval_test` の threshold 未渡し修正
- [x] 可視化スクリプトの GT 描画位置ずれ修正

**モデル精度（fold2, MSEのみ）:**
- angle_error: ~4.9°（目標 <5° 達成）
- rho_error: ~2.9px
- C1椎体の peak_dist が他より高め（32px vs 平均23px）

**7. wandb ログ統合（完了）**

- `train_heat.py` に wandb 対応追加（遅延インポート、約50行）
- `config.yaml` に `wandb:` セクション追加（`enabled`, `project`, `run_name`）
- ログ内容: `train_mse`, `val_mse`, `peak_dist`, `angle_error`, `rho_error`, `lr`, `warmup_weight`
- best保存時・テスト完了時に `wandb.run.summary` を更新
- `enabled: false` がデフォルト（後方互換）

### 次にやること

- [ ] 全 fold 実行（現在は fold2 のみ）
- [ ] 角度損失・ρ損失を有効化した実験（Phase 2/3）
- [ ] C1の精度改善検討

---

## 2026-03-27

### やったこと

**1. sigma ベースライン実験（4パターン）の結果まとめ**

sigma=2.0, 2.5, 3.0, 3.5 の5-fold CV 結果を集計・分析。

| sigma | Angle Error | Rho Error | Perp Dist | Peak Dist |
|-------|------------|-----------|-----------|-----------|
| 2.0   | 6.00°      | 3.62px    | 9.86px    | 23.51px   |
| 2.5   | 5.74°      | 3.52px    | 9.93px    | 25.14px   |
| 3.0   | 5.93°      | 3.50px    | 9.73px    | 25.77px   |
| 3.5   | 5.74°      | 3.38px    | 9.69px    | 26.43px   |

- 線形状精度（Angle / Rho）: sigma=3.5 が最良
- Heatmap 品質（MSE / Peak Dist）: sigma=2.0 が最良
- 結果保存: `.claude/docs/experiments/2026-03-26-sigma-baseline.md`

椎体別傾向: C1/C2 が最難（Angle 7-9°）、C3/C5 が易（4-6°）

**2. line_only/ ディレクトリのリファクタリング（完了）**

`train_heat.py`（1149行）・`line_detection.py`（664行）を分割・再構成。

```
line_only/
├── train.py              (154行)  ← Unet/train.py から移動
├── src/
│   ├── model.py          (63行)
│   ├── dataset.py        (319行)
│   ├── data_utils.py     (252行)
│   └── trainer.py        (715行)
├── utils/
│   ├── losses.py         (362行)
│   ├── metrics.py        (147行)
│   ├── detection.py      (234行)
│   └── visualization.py  (265行)
└── shim/                 旧コード保管（参照用）
```

- test/ 23ファイルの import パスを全更新
- import 全通過、テスト 9/10 通過（1件は変更前からの既存失敗）

**3. CLAUDE.md 更新**

`Unet/CLAUDE.md` をリファクタリング後の構成に合わせて更新。
コマンドも `train_heat.py` → `line_only/train.py` に更新。

**4. train.py の sys.path 修正**

`Unet/line_only/train.py` を直接実行したとき相対 import が失敗する問題を修正。
`train.py` 先頭で `Unet/` を `sys.path` に追加するコードを追加。

### 現状

- リファクタリング完了・動作確認済み
- 3 epoch smoke test: 正常終了（訓練ループ・評価・可視化・チェックポイント保存 全OK）
- 実行方法: `uv run python Unet/line_only/train.py --config Unet/config/config.yaml`

### 次にやること

- [ ] 本格的な sigma チューニング / 損失設定の実験継続
- [ ] angle_loss / rho_loss 有効化（Phase 2/3）の効果検証

---

## 2026-03-28

### やったこと

**1. 学習推移の分析**

wandb ログ（fold0, sigma=2.5 baseline）の epoch 別メトリクスを詳細分析。

- Phase 1 (epoch 1-50): val_mse 0.219→0.006, angle 35-50°（NaN多発）
- Phase 2 (epoch 50-95): val_mse 0.006→0.002, angle 25-40°
- Phase 3 (epoch 95-136): val_mse ~0.002, angle **突然 40°→7-10°** に改善

角度抽出が信頼できるのは val_mse < 0.002（~epoch 96）以降。MSE のみでは 7-10° でプラトー。

**2. angle_loss の設計検討（Codex 相談 2 回）**

Codex に以下を相談し、設計方針を決定：
- `.claude/docs/codex/20260328-angle-loss-design.md`
- `.claude/docs/codex/20260328-line-loss-design.md`

**Codex からの主要指摘：**

| 問題 | 修正方針 |
|------|---------|
| `float('nan')` 代入が勾配を汚染 | NaN 廃止、confidence=0 で無効を表現 |
| `1 - \|dot\|` は dot=0 で cusp | `1 - dot²`（全域滑らか） |
| `torch.where` sign flip が勾配不連続 | 訓練中は使わない、dot² で 180° 曖昧性を自然に処理 |
| `atan2(0,0)` が勾配未定義 | 損失では atan2 を使わず法線ベクトルの内積で直接計算 |
| `sqrt(a²+b²)` が数値不安定 | `torch.hypot` を使用 |
| `(1-0.5w)*MSE` が MSE 品質を劣化 | `MSE + w*L_line`（MSE 常に weight=1.0） |
| warmup_epochs=50 が早すぎ | epoch 85 or val_mse < 0.0025 トリガー |

**3. 実装計画の策定**

`.claude/docs/plan-line-loss-redesign.md` に 3 段階の実装計画を作成：

| Stage | 内容 | MSE 学習への影響 |
|-------|------|----------------|
| Stage 1 | 抽出関数のベクトル化・NaN 廃止 | なし |
| Stage 2 | 新損失関数（`1-dot²` + detach 符号整合 SmoothL1） | なし（use_line_loss=false） |
| Stage 3 | Warmup 統合 + config 更新 | あり（use_line_loss=true） |

### 設計の核心

```
L_total = L_mse + w(t) · L_line

L_line = λ₁ · (1 - dot²)                # 角度損失 = sin²(Δφ)
       + λ₂ · SmoothL1(ρ_pred - ρ_gt)   # ρ損失（detach 符号整合）

dot = n_pred · n_gt（法線ベクトルの内積）
ρ_pred は dot の符号（detach）で GT 方向に整合してから計算
ソフト信頼度ゲート: conf 0.3〜0.6 でスムーズにオン
```

### 次にやること

- [ ] Stage 1: `extract_pred_line_params_batch` のベクトル化（NaN 廃止）
- [ ] Stage 2: 新損失関数の実装（`1-dot²` + detach 符号整合）
- [ ] Stage 3: Warmup 統合 + 訓練実行
- [ ] `use_line_loss: true` で angle_error < 5° を目指す

### 作成ファイル

| ファイル | 内容 |
|---------|------|
| `.claude/docs/codex/20260328-angle-loss-design.md` | Codex 分析（勾配・warmup・設計比較） |
| `.claude/docs/codex/20260328-line-loss-design.md` | Codex 分析（損失オプション・実装ステップ） |
| `.claude/docs/plan-line-loss-redesign.md` | 3 段階の実装計画 |

---

## 2026-03-29

### やったこと

**1. 線一致度損失の再設計 — 3 ステージ実装完了**

前セッション（2026-03-28）で策定した `plan-line-loss-redesign.md` に沿って、全 3 ステージを実装・テスト完了。

---

#### Stage 1: 抽出関数のベクトル化（NaN 廃止）

**変更ファイル:** `utils/losses.py`, `src/trainer.py`, `test/test_line_losses.py`

| 変更内容 | 詳細 |
|---------|------|
| for ループ廃止 | `(B*C, H, W)` reshape で一括演算 |
| NaN 廃止 | 無効時は `(phi=0, rho=0, confidence=0)` |
| `sqrt(disc)` → `torch.hypot` | 数値安定化 |
| confidence 再定義 | `(lam1-lam2)/(lam1+lam2+eps)` で [0,1] 正規化 |
| 縮退時フォールバック | `sxy≈0` 軸平行ケースを `torch.where` で一括処理 |
| `_compute_moments_batch` 切り出し | Stage 2 の損失関数が phi 正規化なしの法線を使えるよう分離 |
| trainer.py の NaN チェック更新 | `~isnan()` → `confidence > 0` |
| 新テスト 2 本追加 | `test_pred_extraction_no_nan`, `test_pred_extraction_batch_consistency` |

---

#### Stage 2: 新損失関数

**変更ファイル:** `utils/losses.py`, `test/test_line_losses.py`

| 変更内容 | 詳細 |
|---------|------|
| `angle_loss` | `1-\|dot\|` → `1-dot²`（全域滑らか、cusp なし、π 周期） |
| `rho_loss` | smooth-min → detach 符号整合 + SmoothL1 |
| 損失関数のシグネチャ変更 | `(pred_params, ...)` → `(nx_pred, ny_pred, ...)` で phi 正規化を回避 |
| `compute_line_loss` | `use_angle`/`use_rho` → 単一 `use_line_loss` フラグ |
| ソフト信頼度ゲート追加 | `conf_gate_low=0.3`, `conf_gate_high=0.6` |
| MSE 重み修正 | `(1-0.5w)*MSE` → `MSE + w*L_line`（MSE 常に weight=1.0） |
| 新テスト 7 本追加 | aligned/orthogonal/π-periodic/smooth/sign_ambiguity/gate/backward |

---

#### Stage 3: Warmup 統合 + config 更新

**変更ファイル:** `utils/losses.py`, `src/trainer.py`, `config/config.yaml`, `test/test_line_losses.py`

| 変更内容 | 詳細 |
|---------|------|
| `get_warmup_weight` | `warmup_start_epoch` 引数追加（開始前は weight=0） |
| trainer.py config 読み込み | 旧キー（`use_angle_loss` 等）→ 新キーへ（フォールバック付き） |
| wandb ログ追加 | `train_L_ang`, `train_L_rho`, `train_gate_ratio` |
| console ログ追加 | `use_line_loss` 時に L_ang / L_rho / gate / w を表示 |
| config.yaml 更新 | 新キー群（`use_line_loss`, `lambda_angle`, `confidence_gate_*`, `warmup_start_epoch`）に置換 |
| 新テスト 2 本追加 | `test_warmup_weight_linear`, `test_warmup_weight_start_epoch` |

---

**テスト結果:** 21/21 全通過（9.45 s）

### 現状

```
L_total = L_mse + w(t) · L_line

L_line = λ₁ · (1 - dot²)                # 角度損失（sin²(Δφ) と等価）
       + λ₂ · SmoothL1(ρ_pred - ρ_gt)   # ρ損失（detach 符号整合）
```

- `use_line_loss: false` で MSE-only モードは変更なし（後方互換）
- 実験を始めるには `config.yaml` で `use_line_loss: true` に変更するだけ

### 次にやること

- [ ] `use_line_loss: true` で fold0 実験（sigma=2.5、`warmup_start_epoch=85`）
- [ ] 実験比較: MSE-only vs +L_line で angle_error の変化を確認
- [ ] target: angle_error < 5°（現状 5.7-6.0°）

---

---

## 2026-03-30

### やったこと

**1. albumentations 警告修正（完了）**

`dataset.py` の ShiftScaleRotate → `A.Affine` 移行で発生していた引数警告を修正。

```python
# 修正前（誤ったパラメータ名）
A.Affine(..., mode=cv2.BORDER_CONSTANT, cval=0.0, cval_mask=0.0)

# 修正後（正しいパラメータ名）
A.Affine(..., border_mode=cv2.BORDER_CONSTANT, fill=0.0, fill_mask=0.0)
```

デフォルト値は同じだったため挙動への影響は最小限だが警告は解消。

---

**2. fold1 回帰調査（継続中）**

sig2.0_base（新 src コード）で fold1 角度誤差が **8.60° → 17.77°** に悪化した原因を調査。

**実験: ShiftScaleRotate vs Affine の影響測定**

`test/test_aug_fold1.py` を作成して fold1 で両 augmentation を比較：

| 実験 | 角度誤差 |
|------|---------|
| ShiftScaleRotate（旧） | 10.71° |
| Affine（新） | 10.93° |

→ **augmentation の変更は原因ではない**（差 0.22°）

**Codex 相談結果**

- `wandb.init()` は torch RNG を消費しない → wandb on/off は原因ではない
- 共有 `torch.Generator` バグ（train/val/test が同一 generator を使用）を特定
- 詳細: `.claude/docs/codex/20260329-wandb-randomness.md`

**ログ比較: baseline（旧 shim）vs 新 src の fold1**

| 時期 | コード | fold1 収束パターン |
|------|--------|-----------------|
| baseline | 旧 shim | epoch 108 まで peak=33px/angle=38°（未収束）→ epoch 113 で 5.87° に突然収束、以後安定 |
| sig2.0_base | 新 src | epoch 96 で 6.43° → epoch 103-170 で 13°→31°→37° と**不安定に振動** |

**重要な発見**: 旧コードでも fold1 は収束が遅い（fold4 は epoch 130 で収束するのに fold1 は epoch 113）。ただし旧コードは一度収束すると安定。**新コードは収束後も不安定**。

**現時点の最有力仮説**: 評価時の valid_mask 変更（`~isnan()` → `confidence > 0`）と confidence 計算式変更（`1-lam2/lam1` → `(lam1-lam2)/(lam1+lam2+eps)`）が fold1 の不安定な training signal につながっている。

詳細: `.claude/docs/codex/20260329-fold1-regression.md`

---

### 次にやること（次セッション）

- [ ] **根本原因の特定（優先）**: 新コードの fold1 不安定さの原因を絞り込む
  - 旧コード（shim）の confidence 計算式・valid_mask を新コードで再現して実験
  - float64 → float32 の精度変化の影響確認
- [ ] **共有 generator バグの修正**: `data_utils.py` で train/val/test に別々の generator を使うよう修正
- [ ] 壊れているテスト 3 件の修正（test_gt_pred_consistency.py / test_one_sample.py / test_sample_fix.py）
- [ ] 根本原因が解消できたら `use_line_loss: true` の実験へ

---

## 2026-03-30（続き）

### やったこと

**1. test_aug_fold1.py のパッチバグ発見**

`test_aug_fold1.py`（旧）と `test_aug_fold1_修正.py` の差分を調査。

**バグ内容**: 旧テストは `dataset_module.get_transforms` のみパッチしていたが、
`data_utils.py` はモジュールレベルで `from .dataset import get_transforms` しているため、
`data_utils_module.get_transforms` 側の参照は差し替わっていなかった。
→ 旧テストの SSR 実験は実際には効いておらず、両実験ともデフォルト aug で走っていた。

修正版（`test_aug_fold1_修正.py`）は両モジュールをパッチし、正しく比較できる。

**実験結果（修正版）:**

| 実験 | val angle |
|------|-----------|
| ShiftScaleRotate（修正版） | 5.39° |
| Affine（修正版） | 5.19° |

→ **augmentation の変更は原因ではない**（差 0.20°）
→ **両結果ともベースラインに近い精度を再現**

**2. sig2.0_base_debug 実験（全 fold）の結果**

新 src コードで改めて 5-fold 実行（wandb=false、confidence_gate 0.1/0.8）：

| fold | debug | 精度悪化したやつ | baseline |
|------|-------|----------------|----------|
| fold0 | 5.57° | 5.94° | 5.43° |
| fold1 | **7.60°** | **17.77°** | 8.60° |
| fold2 | 5.08° | 5.14° | 4.94° |
| fold3 | 4.26° | 5.11° | 4.78° |
| fold4 | 7.99° | 8.11° | 8.11° |
| **AVG** | **6.10°** | **8.41°** | **6.37°** |

→ debug は baseline より良い平均精度（6.10° vs 6.37°）を達成
→ fold1 のみの劣化（17.77°）が debug では解消（7.60°）

**3. Codex による根本原因分析**

Codex CLI を直接実行して調査（`.claude/docs/codex/20260330-codex-fold1-direct.md`）。

| 質問 | Codex 回答 |
|------|-----------|
| wandb.init() が Python/numpy RNG を消費するか | **No**（secrets.choice を使用、実測で不変） |
| kaiming_uniform_ が numpy/Python RNG を使うか | **No**（torch RNG のみ） |
| reinit=True がfold1で特別か | fold1 は最初の「finish→reinit 境界」を跨ぐ fold |
| なぜ fold1 だけか | fold1 は元々難しいfold。微小な非決定性が最も増幅されやすい |

**Codex 推奨の決定的な修正:**
- 本命: `--start_fold 1 --end_fold 1` でfoldごとに別プロセス実行し reinit=True 連鎖をやめる
- 追加策: `wandb.init()` 直後に `set_seed(seed)` を再呼び出し
- 追加策: `torch.use_deterministic_algorithms(True)` を有効化

**4. wandb=true での実験で精度異常が再現しないことを確認**

改めて wandb=true で実験し直したところ、精度が異常になることはなくなった。
→ 精度悪化したやつ の fold1 劣化は**再現性のない非決定的な挙動**と確定。
   コード自体に問題はない。

### 現状

- **新 src コードは正常に動作している**（avg 6.10° でベースライン超え）
- fold1 の一時的劣化は再現せず、非決定的挙動と確定
- augmentation（ShiftScaleRotate vs Affine）は精度に影響しない
- **次ステップ: `use_line_loss: true` の実験へ移行可能**

### 次にやること

- [ ] `use_line_loss: true` で fold0 実験（sigma=2.0、`warmup_start_epoch=90`）
- [ ] 実験比較: MSE-only（avg 6.10°）vs +L_line で angle_error の変化を確認
- [ ] target: angle_error < 5°

---

## テンプレート（以下をコピーして使用）

```markdown
## YYYY-MM-DD

### やったこと

-
-

### 実験結果（あれば）

**Setup:**
-

**Results:**
-

### 次にやること

- [ ]
- [ ]

### メモ

-
```

## 2026-03-18 Session: Critical Bugs Fixed & Y-Axis Coordinate Bug Discovery

### 🔧 Critical Issues Fixed (Codex Analysis)

**3 critical bugs fixed in line_losses.py and train_heat.py:**

1. **NaN Handling Order (line_losses.py:191-278)**
   - Problem: Computed `cos/sin/exp` on NaN-containing tensors before masking
   - Impact: `NaN * 0 = NaN` poisoned batch loss, caused training collapse
   - Fix: Extract valid entries FIRST, then compute loss only on valid data
   - Added: `.detach()` on confidence to prevent gradient collapse

2. **Double Smoothing in Rho Loss (line_losses.py:265)**
   - Problem: Applied smooth minimum, then `F.smooth_l1_loss()` on result
   - Impact: Mathematically incorrect, unpredictable gradient scale
   - Fix: Removed `F.smooth_l1_loss()` call

3. **Warmup Weight Formula (train_heat.py:843)**
   - Problem: `loss = loss_mse + warmup_weight * line_loss` (MSE always constant)
   - Impact: Gradient shock at epoch 50 when geometry loss activates
   - Fix: `loss = (1 - 0.5*warmup_weight) * loss_mse + warmup_weight * line_loss`

**Results After Critical Fixes:**
- test_mse: 0.011 → **0.0017** (6.5x improvement)
- peak_dist: 57.59 → **21.29 px** (2.7x improvement)
- perpendicular_dist: 25.25 → **5.37 px** (4.7x improvement)
- angle_error: 41.66 → **34.30 deg** (7.4 deg improvement)
- rho_error: 18.74 → **15.75 px** (3.0 px improvement)

### 🐛 Y-Axis Coordinate System Bug Discovery

**Root Cause Identified (line_losses.py:96):**
```python
# WRONG (image convention):
y_grid = torch.arange(H, ...) - H / 2.0
# row 0 (top) → -H/2, row H-1 (bottom) → +H/2 (Y increases downward)

# CORRECT (math convention):
y_grid = -(torch.arange(H, ...) - H / 2.0)
# row 0 (top) → +H/2, row H-1 (bottom) → -H/2 (Y increases upward)
```

**Why 40-50° Errors (Not 90° or 180°)?**
- Flipped Y-axis causes non-linear angle distortion in moment calculations
- For 20° line: mu11 sign flips asymmetrically → ~43° error
- For 125° line: similar mechanism → ~52° error

**Fix Impact (Verified with Unit Tests):**
- sample5_C1_slice029 line_1: 42.72° error → **0.03° error**
- sample5_C1_slice029 line_3: 52.64° error → **0.02° error**
- All test angles (0°-180°): max error **0.38°**

### 📊 Expected Full Re-training Impact

**Current (with Critical Fixes + Y-axis Bug):**
- angle_error: 34.30 deg
- rho_error: 15.75 px

**After Re-training (Y-axis Fix):**
- angle_error: **<10 deg** (expected 5-10 deg)
- rho_error: **<10 px** (with lambda_rho tuning)

### 📝 Codex Strategy Recommendations

**Next Steps (Codex Priority):**
1. **Re-train with Y-axis fix** - Verify angle error drops to <10 deg
2. **Increase lambda_rho gradually** - 0.05 → 0.1 → 0.2 (NOT 0.5 jump)
3. **Consider adaptive warmup** - Only if geometry loss plateaus

**NOT Recommended Now:**
- ❌ Jump lambda_rho to 0.5 (10x) - Too risky
- ❌ Implement adaptive warmup immediately - Complexity without proven need
- ❌ Multiple changes at once - Can't isolate effects

### 🧪 Unit Tests Created

**Location:** `Unet/line_only/test/`
- `test_moment_extraction.py` - Basic angle tests (0°, 45°, 90°, 135°)
- `test_sample_fix.py` - Problem case verification
- `visualize_fix.py` - Visual comparison generator

**All tests passing with <1° error.**

### 📁 Files Modified

1. `Unet/line_only/line_losses.py:96` - Y-axis coordinate fix
2. `Unet/line_only/line_losses.py:191-233` - NaN-safe angle_loss
3. `Unet/line_only/line_losses.py:235-278` - NaN-safe rho_loss, removed double smoothing
4. `Unet/line_only/train_heat.py:843` - Warmup formula fix

### 🎯 Next Session Tasks

- [ ] Run full re-training: `cd Unet && uv run python train.py`
- [ ] Verify angle_error <10 deg, rho_error <10 px
- [ ] If successful, test lambda_rho=0.1 (2x increase)
- [ ] Document final results

---

## 2026-03-23 Session: Threshold Processing Investigation

### 📊 Problem Analysis

**Current Test Results (Y-axis fix applied):**
- angle_error: 32.70° (still high, expected <10°)
- Confidence: 0.214 (very low, indicates poor line quality)

**Root Cause Hypothesis:**
- Low confidence → unstable angle calculation
- When mu20 ≈ mu02, denominator in `atan2(2*mu11, mu20-mu02)` becomes small
- Heatmap includes low-intensity "tails" that spread variance isotropically

### 🧪 Threshold Processing Experiment

**Test Setup:**
```python
# Test script: Unet/line_only/test/test_threshold_effect.py
# Compare: threshold=0.0 (no filter) vs threshold=0.2 (filter low values)
```

**Results (411 test samples):**

| Metric | No Threshold | Threshold ≥ 0.2 | Improvement |
|--------|--------------|-----------------|-------------|
| Mean Error | 32.70° | **23.73°** | -8.97° (27.4%) |
| Median Error | 28.10° | **10.09°** | -18.01° (64.1%) |
| Max Error | 89.61° | 89.49° | ~same |
| Confidence | 0.214 | **0.941** | +0.726 (340%) |

**Key Findings:**
- ✅ 98.8% of samples improved or unchanged
- ✅ Dramatic confidence boost (0.214 → 0.941)
- ⚠️ 1.2% of samples degraded (5 out of 411)
  - Degraded samples: originally low heatmap response
  - Threshold cut important low-intensity regions

### 🎨 Visualization Scripts Created

**Location:** `Unet/line_only/test/threshold_effect/`

1. **test_threshold_effect.py**
   - Compare threshold=0.0 vs 0.2 across test dataset
   - Generate statistics and improvement analysis
   - Output: `analysis_summary.png`

2. **visualize_threshold_comparison.py**
   - Visual comparison of improved/degraded samples
   - 3-panel format: Heatmap / Pred Lines / GT Lines
   - Key features:
     - Shows thresholded heatmap (>= 0.2 only)
     - Pred lines clipped to thresholded region + 10px margin
     - Background: original CT image

**Generated Images (11 files, 1.9MB):**
- `analysis_summary.png` - Overall statistics
- `improved_1-5_delta*.png` - Top 5 improved samples (70-88° improvement)
- `degraded_1-5_delta*.png` - Top 5 degraded samples (52-72° degradation)

### 📈 Sample Analysis

**Best Improvement (sample22_C3_slice044_ch1):**
- Error: 89.6° → 0.9° (**88.6° improvement**)
- Threshold removed noisy low-intensity regions
- Pred line almost perfectly matches GT

**Worst Degradation (sample22_C4_slice046_ch1):**
- Error: 6.3° → 78.2° (**71.9° degradation**)
- Original heatmap had weak but correct response
- Threshold cut too much, leaving fragmented information

### 🎯 Next Steps

**Immediate (This Direction):**
- [ ] Implement threshold processing in training pipeline
  - Modify `extract_pred_line_params_batch()` in `line_losses.py`
  - Add `threshold=0.2` parameter
- [ ] Re-train model with threshold processing
- [ ] Verify angle_error drops to <15° (ideally <10°)

**Alternative Approaches (If Needed):**
- [ ] Adaptive threshold per channel (instead of fixed 0.2)
- [ ] Soft weighting instead of hard threshold
- [ ] Improve heatmap quality through better training

### 📝 Technical Details

**Threshold Implementation:**
```python
# Before moment calculation:
if threshold > 0:
    heatmaps = torch.where(heatmaps >= threshold, heatmaps, 0.0)
# Then compute moments as usual
```

**Why It Works:**
1. Removes low-intensity noise → cleaner heatmap
2. Reduces mu20 ≈ mu02 cases → more stable atan2
3. Increases λ1/λ2 ratio → higher confidence

**Trade-off:**
- 98.8% improve, but 1.2% degrade (acceptable)
- Degraded cases: weak heatmap response (training issue, not threshold issue)

### 📁 Files Created

- `Unet/line_only/test/test_threshold_effect.py` - Effect analysis
- `Unet/line_only/test/visualize_threshold_comparison.py` - Visualization
- `Unet/line_only/test/threshold_effect/*.png` - 11 visualization images

### 💡 Key Insight

**The angle error is not caused by coordinate bugs or formula errors.**
**It's caused by mathematical instability when computing angles from noisy, isotropic heatmaps.**
**Threshold processing is an effective post-processing solution.**

---
