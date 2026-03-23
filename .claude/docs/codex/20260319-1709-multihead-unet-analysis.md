# Codex Analysis: Multi-Head UNet Validity
Date: 2026-03-19 17:09

## Question
Quick analysis: Multi-Head UNet for 4-channel heatmap crosstalk. Shared encoder + skip connections + 4 separate decoders. Will this reduce 5-10% inter-channel crosstalk? Is 2.5x parameter increase justified? Better alternatives? Give concise answer.

## Context
- 5-10% crosstalk: each channel responds to other lines' positions
- Causes 8-28° angle errors in moment-based extraction
- Proposed: shared encoder + 4 separate decoders, but skip connections still shared

## Codex Response

結論だけ言うと、**いきなり full Multi-Head UNet（4 decoder）は非推奨**です。

### 効果の評価

- **5-10% crosstalk低減**: 多少は効く可能性はありますが、**同等の絶対改善を安定して出す保証は弱い**です（効果は中程度・不確実）。
- **2.5x前後のパラメータ増**: **最初の一手としては割高**。loss設計で詰め切れないときだけ正当化されます。

### 推奨される代替案（優先順）

1. **GT-aware anti-crosstalk loss**（one-vs-rest抑制）
2. **局所マスク付き競合**（`4 classes + background`、GT近傍のみ）
3. **late-branch + head-specific skip adapter**（`1x1` gate/proj）

### 推奨順序

**loss対策 → 軽量branch分離 → 最後にfull multi-head**

### 設計判断

設計判断は [DESIGN.md](/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/DESIGN.md#L88) に記録済み。

## Summary

**Full Multi-Head UNet は段階的アプローチの最後の選択肢とすべき**

1. まずloss設計で対策（anti-crosstalk loss）
2. 効果不足なら軽量なbranch分離
3. それでもダメなら full multi-head を検討

理由：
- 効果が不確実（5-10%低減の保証なし）
- コスト高（2.5倍のパラメータ）
- より低コストな代替案が存在
