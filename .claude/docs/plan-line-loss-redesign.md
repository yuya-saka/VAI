# 実装計画: 線一致度損失の再設計

**作成日:** 2026-03-28
**ステータス:** 未着手
**関連 Codex 分析:** `.claude/docs/codex/20260328-angle-loss-design.md`, `.claude/docs/codex/20260328-line-loss-design.md`

---

## コンテキスト

TinyUNet で頸椎 CT 画像（224×224）から 4 本の境界線を検出するヒートマップを学習中。
MSE のみの baseline では angle_error が 5.7〜6.5° で頭打ち。
幾何学的制約（角度損失 + ρ損失）を追加してこのプラトーを突破したい。

### 直線表現

Hesse 標準形: `x·cosφ + y·sinφ = ρ`
φ ∈ [0, π), ρ は D = √2·224 で正規化

### 学習推移の観察（baseline, fold0, sigma=2.5）

| フェーズ | Epoch | val_mse | angle_error | 備考 |
|---------|-------|---------|-------------|------|
| Phase 1 | 1〜50 | 0.219→0.006 | 35〜50°（NaN多発） | ヒートマップ未形成 |
| Phase 2 | 50〜95 | 0.006→0.002 | 25〜40° | MSE 漸減 |
| Phase 3 | 95〜136 | 0.002→0.0018 | **突然 40°→7-10°** | MSE が十分下がり角度抽出が安定 |

→ angle_loss は **val_mse < 0.002-0.003 以降** でないと意味がない

---

## 設計方針（Codex 分析に基づく）

### 1. 角度損失: `1 - (n_pred · n_gt)²`

```python
dot = nx_pred * nx_gt + ny_pred * ny_gt   # 法線ベクトルの内積
L_ang = 1 - dot ** 2                      # = sin²(Δφ)
```

- π 周期（180° 曖昧性を自然に処理）
- atan2 不使用（勾配未定義の問題を回避）
- sign flip 不使用（勾配不連続を回避）
- `1 - |dot|` と違い dot=0 で cusp なし（全域滑らか）

### 2. ρ損失: detach 符号整合 + SmoothL1

```python
sgn = torch.sign(dot).detach().clamp(min=1)  # GT 方向に整合（勾配は通さない）
rho_pred = (sgn * nx * cx + sgn * ny * cy) / D
L_rho = F.smooth_l1_loss(rho_pred, rho_gt, reduction='none')
```

- smooth-min 廃止（φ∈[0,π) への一貫した正規化で曖昧性なし）
- detach した符号で法線方向を GT に揃えてから ρ を計算

### 3. 抽出: 完全ベクトル化

```python
# (B, C, H, W) → 一括演算（for ループ・NaN 廃止）
disc = torch.hypot(sxx - syy, 2 * sxy)     # sqrt(a²+b²) より安定
confidence = (lam1 - lam2) / (lam1 + lam2 + eps)  # [0,1]
# 無効時: (phi=0, rho=0, confidence=0) — NaN を使わない
```

### 4. ソフト信頼度ゲート

```python
gate = ((confidence - 0.3) / 0.3).clamp(0, 1).detach()  # conf<0.3:0, conf>0.6:1
gate = gate * gt_valid.float()
L_line = (gate * (λ1 * L_ang + λ2 * L_rho)).sum() / gate.sum().clamp(1)
```

### 5. 損失統合

```python
loss = L_mse + w(t) * L_line    # MSE 重み常に 1.0（旧式の (1-0.5w)*MSE は廃止）
```

### 6. Warmup

```
w(t) = 0                        t < t_start (epoch 85 or val_mse < 0.0025)
w(t) = (t - t_start) / 25       linear ramp over 25 epochs
w(t) = 1                        after ramp
```

---

## スコープ

| ファイル | 変更内容 |
|---------|---------|
| `Unet/line_only/utils/losses.py` | 抽出・損失関数の書き換え |
| `Unet/line_only/src/trainer.py` | 損失統合・warmup ロジック |
| `Unet/config/config.yaml` | 新パラメータ追加 |
| `Unet/line_only/test/test_line_losses.py` | テスト更新・追加 |

**変更なし:** `dataset.py`, `detection.py`, `model.py`

---

## Stage 1: `extract_pred_line_params_batch` のベクトル化

**目的:** for ループ廃止・NaN 廃止・数値安定化。MSE 学習への影響ゼロ。

### Step 1a: `losses.py` — 抽出関数の書き換え

- [ ] for ループ → `(B*C, H, W)` に reshape して一括演算
- [ ] `float('nan')` 代入 → 無効時は `(phi=0, rho=0, confidence=0)` を返す
- [ ] `torch.sqrt(discriminant)` → `torch.hypot(sxx - syy, 2 * sxy)` に変更
- [ ] confidence 計算: `(lam1 - lam2) / (lam1 + lam2 + eps)` に変更
- [ ] φ∈[0,π) 正規化は維持（評価用。訓練の損失パスでは使わない）
- [ ] 関数シグネチャは現行互換を維持

**検証:** 既存チェックポイントで `evaluate()` 実行、angle_error / rho_error が ±0.5° 以内

### Step 1b: `trainer.py` — evaluate() の NaN チェック更新

- [ ] `pred_valid = ~torch.isnan(...)` → `pred_valid = confidence > 0`
- [ ] `predict_lines_and_eval_test()` 内の `np.isnan(pred_phi)` チェックも同様

### Step 1c: テスト更新

- [ ] `test_pred_extraction_gradient_flow` の valid_mask を confidence ベースに
- [ ] `test_pred_extraction_no_nan` 追加
- [ ] `test_pred_extraction_batch_consistency` 追加

**チェックポイント:** MSE-only で 1 fold 50 epoch → baseline と一致

---

## Stage 2: 新しい損失関数

### Step 2a: `losses.py` — `angle_loss` 書き換え

- [ ] 式: `L = 1 - dot²`
- [ ] `* 0.01` スケーリング廃止（L は [0,1] で自然なスケール）
- [ ] ソフト信頼度ゲート（gate_low=0.3, gate_high=0.6）
- [ ] valid 数で正規化

### Step 2b: `losses.py` — `rho_loss` 書き換え

- [ ] smooth-min 廃止 → detach 符号整合 + SmoothL1
- [ ] `* 0.01` スケーリング廃止
- [ ] 同じソフトゲート適用

### Step 2c: `losses.py` — `compute_line_loss` 書き換え

- [ ] `use_angle`/`use_rho` → 単一の `use_line_loss` フラグ
- [ ] ソフトゲートパラメータを引数に追加
- [ ] valid_mask: `gt_valid & (confidence > 0)`（NaN チェック不要）

### Step 2d: テスト追加

- [ ] `test_angle_loss_aligned`: pred=gt → loss=0
- [ ] `test_angle_loss_orthogonal`: 90° ずれ → loss=1
- [ ] `test_angle_loss_pi_periodic`: φ+π → loss=0
- [ ] `test_angle_loss_smooth_at_zero`: Δφ=0 での勾配=0（cusp なし）
- [ ] `test_rho_loss_sign_ambiguity`: (φ+π, -ρ) vs (φ, ρ) → loss≈0
- [ ] `test_soft_gate_bounds`: gate 境界値の確認
- [ ] `test_compute_line_loss_backward`: backward で NaN なし

**チェックポイント:** `use_line_loss: false` で MSE-only と同一結果

---

## Stage 3: Warmup 統合 + config 更新

### Step 3a: `losses.py` — `get_warmup_weight` 修正

- [ ] `warmup_start_epoch` 引数追加（デフォルト 0 で旧互換）
- [ ] `current_epoch < warmup_start_epoch` なら 0.0 を返す

### Step 3b: `trainer.py` — 訓練ループ更新

- [ ] 新 config キー読み込み（旧キーへのフォールバック付き）
- [ ] 損失式: `(1-0.5w)*MSE` → `MSE + w*L_line` に変更
- [ ] val_mse ゲート: `warmup_start_epoch=-1` のとき val_mse が閾値を下回った epoch で開始
- [ ] wandb ログに `L_ang`, `L_rho`, `gate_ratio`, `warmup_weight` を追加

### Step 3c: `config.yaml` 更新

```yaml
loss:
  use_line_loss: false
  lambda_angle: 1.0
  lambda_rho: 1.0
  warmup_start_epoch: 85       # -1 なら val_mse ゲート
  warmup_ramp_epochs: 25
  warmup_mode: "linear"
  warmup_mse_trigger: 0.0025   # warmup_start_epoch=-1 のとき使用
  confidence_gate_low: 0.3
  confidence_gate_high: 0.6
```

### Step 3d: Warmup テスト

- [ ] `test_warmup_delayed_start`: 開始前は 0.0、開始後にランプ
- [ ] `test_warmup_backward_compat`: `start_epoch=0` で旧動作と一致

**チェックポイント:** `use_line_loss: true` で 1 fold 完走、NaN なし

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| Stage 1 で評価値が変わる | 既存チェックポイントで evaluate() 比較 |
| ベクトル化で勾配が壊れる | gradient flow テスト + 5 step 訓練で NaN チェック |
| MSE-only 学習が壊れる | 各 Stage 後に use_line_loss=false で回帰テスト |
| 旧 config との互換 | trainer.py で旧キーへのフォールバック実装 |

---

## 推奨パラメータ（初回実験）

| パラメータ | 値 | 理由 |
|------------|-----|------|
| `warmup_start_epoch` | 85 | angle が信頼できるタイミング |
| `warmup_ramp_epochs` | 25 | 急激な勾配ショック回避 |
| `lambda_angle` | 1.0 | L_ang ∈ [0,1] なのでスケール調整不要 |
| `lambda_rho` | 1.0 | D 正規化済みで同スケール |
| `confidence_gate_low` | 0.3 | 等方性ブロブを除外 |
| `confidence_gate_high` | 0.6 | 信頼できる予測のみ |

---

## 検証手順（最終）

1. `uv run pytest Unet/line_only/test/test_line_losses.py -v` — 全テスト通過
2. `use_line_loss: false` で 1 fold 50 epoch — MSE 曲線が baseline と一致
3. `use_line_loss: true` で 1 fold 完走 — NaN なし、angle_error が改善傾向
