# Codex Analysis: Phase 1 Implementation Order
Date: 2026-03-18 03:27

## Question

Review the implementation order of these 3 fixes:

1. Warmup formula fix: Change loss = loss_mse + warmup_weight * line_loss
   to: loss = (1 - 0.5*warmup_weight) * loss_mse + warmup_weight * line_loss

2. Remove double smoothing: Delete F.smooth_l1_loss() call after smooth minimum

3. Fix NaN handling: Extract valid entries BEFORE computation in angle_loss/rho_loss

Questions:
- Is this the optimal order?
- Should NaN handling (#3) come first since it causes training failure?
- Are there dependencies between these fixes?
- Should we implement all at once or one-by-one?

## Codex Response

推奨順序は `#3 -> #2 -> #1` です。

- `#3`（NaN handling）を最優先: これは安定性の前提条件です。現状は `cos/sin/exp` を NaN 含みテンソルに先に適用しており、`NaN * 0` で損失が壊れます。
- `#2`（double smoothing除去）を次: `rho_loss` の曲率・スケールを先に確定させるべきです。
- `#1`（warmup式変更）を最後: 全体の loss バランス調整なので、`#2` 後の line loss スケールを前提に適用するのが最も解釈しやすいです。

質問への回答:
- 最適順序か: 元の `#1 -> #2` より、`#2 -> #1` の方が効果分離しやすく、原因追跡が容易です。
- `#3` を先にすべきか: はい、必須です（training failure 直結）。
- 依存関係:
  - 強依存: `#3` は他2つの実験前提。
  - 弱依存: `#1` は `#2` で変わる line loss スケールに影響を受けます。
- 一括か段階か: 原則は段階実装（3コミット）。時間優先なら最低でも `#3` を単独コミット、その後 `#2+#1` をまとめる2段階。

この判断は設計記録にも追記済みです: [DESIGN.md](/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/DESIGN.md:61)

## Summary

**Recommended Order: #3 → #2 → #1**

**Rationale:**
1. **#3 (NaN handling) must come first** - Critical stability fix that prevents training failure
2. **#2 (double smoothing) second** - Establishes correct rho_loss scale before warmup adjustment
3. **#1 (warmup formula) last** - Overall balance adjustment depends on line_loss scale from #2

**Implementation Strategy:**
- Best: 3 separate commits (one per fix, for clear effect isolation)
- Acceptable: 2 commits (#3 alone, then #2+#1 together)
- Avoid: All 3 at once (makes debugging harder)
