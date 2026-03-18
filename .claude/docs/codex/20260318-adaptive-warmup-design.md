# Codex Analysis: Adaptive Warmup Design
Date: 2026-03-18

## Question

Analyze adaptive warmup strategy for multi-objective loss function.

Current: loss = (1-0.5w)*mse + w*geometry, w linear 0→1 over 50 epochs
Proposed: Adaptive warmup triggered by MSE plateau

Question: Should we use adaptive warmup? If yes, provide concrete formula. If no, explain why fixed is better.

Constraints: 200 epochs, MSE plateaus at 30-50

## Codex Response

結論: **Adaptive warmup を使うべき**です。
理由は、200 epoch で `MSE plateau = 30–50` が分かっているなら、固定 50 epoch は遅れやすく、geometry 学習時間を無駄にしやすいためです。

具体式（そのまま実装可能）:

\[
L_e = (1 - 0.5\,w_e)\,L_{\text{mse},e} + w_e\,L_{\text{geo},e}
\]

### 1. Plateau 検出（`val_mse`）

\[
\text{ema}_e = 0.9\,\text{ema}_{e-1} + 0.1\,\text{val\_mse}_e
\]

\[
r_e = \frac{\text{ema}_{e-1}-\text{ema}_e}{\max(\text{ema}_{e-1}, 10^{-12})}
\]

- `e >= 20` かつ `r_e < 0.002`（0.2%未満改善）を連続 5 epoch で満たしたら `e_start = e`
- それまでに発火しなければ `e_start = 60`（fallback）
- 一度発火したら解除しない（latch）

### 2. Warmup 重み

\[
w_e =
\begin{cases}
0 & (e < e_{start})\\
\frac{1}{2}\left(1-\cos\left(\pi\cdot \mathrm{clip}\left(\frac{e-e_{start}}{20},0,1\right)\right)\right) & (e \ge e_{start})
\end{cases}
\]

この設定だと plateau が 30–50 の範囲で始まりやすく、最悪ケースでも epoch 60 から geometry を入れられます。

設計記録も更新済みです: DESIGN.md

## Key Implementation Details

### Plateau Detection Parameters
- **EMA decay**: 0.9
- **Relative improvement threshold**: 0.2% (0.002)
- **Patience**: 5 consecutive epochs
- **Minimum epoch**: 20
- **Fallback start**: epoch 60
- **Latch behavior**: One-way (no deactivation once triggered)

### Warmup Schedule
- **Duration**: 20 epochs after trigger
- **Function**: Cosine ramp (smooth, avoids gradient shock)
- **Range**: w from 0 to 0.5 (note: max weight is 0.5 in the loss formula)
- **Formula**: `w = 0.5 * (1 - cos(π * clip((e - e_start)/20, 0, 1)))`

### Loss Composition
- **Before warmup**: `L = L_mse` (w=0)
- **During warmup**: `L = (1 - 0.5*w) * L_mse + w * L_geo`
- **After warmup**: `L = 0.75 * L_mse + 0.5 * L_geo` (w=0.5 at max)

## Advantages Over Fixed Warmup

1. **Adaptive timing**: Reacts to actual MSE plateau (typically 30-50) vs fixed epoch 50
2. **Reduced waste**: Doesn't wait unnecessarily if MSE plateaus early
3. **Stability**: Multiple safeguards prevent false triggers
4. **Guaranteed budget**: Fallback ensures geometry training starts by epoch 60

## Trade-offs

- **Complexity**: More hyperparameters and state tracking
- **Debugging**: Harder to reproduce exact behavior across runs
- **Overhead**: Small computational cost for EMA tracking

## Recommendation

**Use adaptive warmup** for this 200-epoch training scenario where MSE plateau timing is variable but predictable within a range. The complexity cost is justified by the efficiency gains and improved training dynamics.
