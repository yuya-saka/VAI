# Codex Analysis: Geometry Loss Activation Strategy
Date: 2026-03-31

## Question
How to properly enable and tune the existing geometry loss for angle/rho improvement

System setup:
- LOSS: total = MSE + w(t) * (0.004 * L_angle + 0.04 * L_rho)
- MSE ~ 0.002 at convergence
- L_angle = 1 - (n_pred · n_gt)^2 in [0,1]
- L_rho = SmoothL1(rho_pred_normalized, rho_gt_normalized)
- w(t) = linear warmup from epoch 90 to 120 (total 230 epochs)
- Confidence gate: soft gate based on anisotropy ratio, low=0.1, high=0.8
- Current results (MSE only): angle_error ~6.3 deg, rho_error ~3.5 px

## Codex Response

1. 結論: いまの `lambda_angle=0.004` は小さすぎる可能性が高いです。`lambda_rho=0.04` は **SmoothL1 の beta 次第** です。
- あなたの近似（`L_rho≈0.011`）を使うと、`w=1, gate=1` で
  `angle_term≈4.8e-5`, `rho_term≈4.4e-4`, 合計 `≈4.9e-4`（MSE=0.002の約24%）。
- ただし実際は `gate<1` なので、実効寄与はさらに下がります（例えば平均 gate=0.2 なら約5%）。
- さらに PyTorch デフォルトの SmoothL1（`beta=1`）なら `L_rho` は `~6e-5` になり、rho項はほぼ無視レベルです。ここは実ログ確認が必須です。

2. `0.5–2.0` に上げるべきか
- 一気に両方 `0.5–2.0` はリスク高めです。
- 推奨は段階的増加です。目安としては
  `lambda_angle`: 5x〜30x（`0.02–0.12`）
  `lambda_rho`: 2x〜20x（`0.08–0.8`、beta=1ならさらに上もあり）
- まずは「幾何項の実効寄与（warmup後）を MSE の 20〜60%」に合わせるのが実務的です。

3. `warmup_start=90/230` は遅いか
- 安定性重視としては合理的ですが、改善余地はあります。
- 幾何最適化に使える期間が短くなり、MSE谷から抜けにくくなる可能性があります。
- 今の症状なら `start=40〜70` がバランス良いです。理想は固定epochより「val MSE plateauトリガ」です。

4. gate `[0.1, 0.8]` は適切か
- 典型的には、細長い良好予測で anisotropy ratio は高め（`0.6〜0.95`）、ぼけ/ノイズで低め（`<0.3`）になりやすいです。
- `high=0.8` はやや厳しめで、学習中盤まで有効サンプルを削る可能性があります。
- まず ratio ヒストグラムを epoch 別に確認してください。`P75 < 0.8` が続くなら `high` を下げるべきです。

5. 実験設定（3案）

| Config | lambda_angle | lambda_rho | warmup_start_epoch | warmup_epochs | gate_low | gate_high | ねらい |
|---|---:|---:|---:|---:|---:|---:|---|
| A Conservative | 0.02 | 0.08 | 70 | 40 | 0.08 | 0.75 | 現状から小〜中程度の増強。崩壊リスクを最小化。 |
| B Moderate | 0.08 | 0.30 | 50 | 30 | 0.05 | 0.65 | 幾何を明確に効かせる本命。安定性と改善幅のバランス。 |
| C Aggressive | 0.25 | 1.00 | 30 | 20 | 0.02 | 0.50 | 早期から幾何主導。大幅改善狙いだが不安定化リスク高。 |

補足: `L_rho` が本当に `~0.011` でなく `beta=1` で非常に小さいなら、上表の `lambda_rho` をさらに上げる（特に B/C）判断が必要です。まず 1 run で `mean(L_angle), mean(L_rho), mean(gate), each weighted term` をログしてから最終決定するのが最短です。

## Key Takeaways

### Scale Analysis
- Current lambda_angle=0.004 is likely **too small by 5-30x**
- lambda_rho=0.04 depends on SmoothL1 beta value (PyTorch default beta=1 makes it negligible)
- Effective geometry contribution after gating may be as low as 5% of MSE → insufficient signal

### Critical Action: Log Loss Components First
Before tuning, add logging of:
- mean(L_angle), mean(L_rho)
- mean(gate) per epoch
- Each weighted term contribution

### Recommended Lambda Ranges
- lambda_angle: 0.02–0.12 (5x–30x current)
- lambda_rho: 0.08–0.8 (2x–20x current, adjust based on beta)

### 3 Experiment Configurations

**Config A (Conservative):**
- lambda_angle: 0.02, lambda_rho: 0.08
- warmup_start_epoch: 70, warmup_epochs: 40
- gate_low: 0.08, gate_high: 0.75
- Risk: Low, Expected gain: Small

**Config B (Moderate) — Recommended first experiment:**
- lambda_angle: 0.08, lambda_rho: 0.30
- warmup_start_epoch: 50, warmup_epochs: 30
- gate_low: 0.05, gate_high: 0.65
- Risk: Medium, Expected gain: Moderate

**Config C (Aggressive):**
- lambda_angle: 0.25, lambda_rho: 1.00
- warmup_start_epoch: 30, warmup_epochs: 20
- gate_low: 0.02, gate_high: 0.50
- Risk: High, Expected gain: Large (if stable)

### Key Risks
1. Current lambdas are too small → geometry loss has no effect even when enabled
2. gate_high=0.8 may be too strict → few samples pass during mid-training
3. warmup_start=90 gives only 140 epochs for geometry optimization
4. SmoothL1 beta=1 (PyTorch default) makes rho_term negligible at small errors
