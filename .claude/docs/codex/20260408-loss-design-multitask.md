# Codex Analysis: Multitask Loss Design
Date: 2026-04-08

## Question

We are designing a multitask learning loss for medical image analysis (cervical vertebra CT segmentation + boundary line prediction).

Architecture: ResUNet [24,48,96,192], shared encoder, dual decoder (seg head: 5-class CE, line head: MSE heatmap)
Data: ~500 CT slices, batch_size=8, ~50% have segmentation GT, 100% have line GT

Loss design questions:

1. Does the ordering in the loss formula matter philosophically?
   - Option A: L = L_line + alpha * L_seg  (current spec, line is 'base')
   - Option B: L = L_seg + beta * L_line   (seg is 'base', line is 'auxiliary')
   Both are mathematically equivalent. But which convention is better for:
   (a) Monitoring training dynamics
   (b) Hyperparameter tuning intuition
   (c) Communicating task priority to readers

2. Loss magnitude calibration:
   - MSE (line heatmap, sigmoid output): typically 1e-3 to 1e-2 range
   - CrossEntropy (5-class segmentation): typically 1e-1 to 1e0 range
   - With alpha=0.03: alpha*CE ≈ 0.003 to 0.03
   - Is this ratio appropriate? What is the principled way to set the weight?
   - Should we normalize each loss by its typical magnitude? Or use gradient-based balancing?

3. Partial supervision risk:
   - Line loss: computed for all 8 samples per batch
   - Seg loss: computed for only ~4 samples per batch (those with GT masks)
   - The seg loss contribution per batch = alpha * CE_mean(4 samples)
   - Risk: is alpha=0.03 too small to learn useful segmentation from only 4 samples?
   - Alternative: increase alpha when seg GT is available (conditional alpha)

4. Monitoring strategy:
   - What metrics should we track to detect negative transfer?
   - When should we intervene with alpha adjustment?

5. Practical recommendation:
   - Given a small dataset (~500 samples, 5-fold CV), simple 2-class split (seg GT vs no GT)
   - Should we start with fixed alpha and only move to dynamic weighting if problems arise?

## Codex Response

結論から言うと、あなたの設定では **`L_total = L_line + α L_seg` をそのまま採用し、まずは固定 `α` で始める**のが一番妥当です。今回の条件では line が `100%` 教師あり、seg は `~50%` 教師ありなので、seg は「主目的を補助する表現正則化」として扱うのが自然です。

### 1. 式の順序

数式としては等価ですが、表記は意味を持ちます。今回なら **Option A** がよいです。

- **監視**: `line` をベース項にすると、「主損失」と「補助損失」が見やすい。ログは `raw_line`, `raw_seg`, `weighted_seg=α·raw_seg`, `total` を分けて出せば十分。
- **チューニング直感**: `α` は「seg をどれだけ line に食い込ませるか」と読めるので解釈しやすい。
- **論文上の伝達**: 順序は優先度の宣言。今回のデータ構成なら「line primary, seg auxiliary」の方が読者に誤解が少ない。

### 2. 重みの決め方

`MSE ~ 1e-3 to 1e-2`、`CE ~ 1e-1 to 1e0` なら、`α=0.03` は十分に妥当な初期値です。`α·CE ≈ 0.003 to 0.03` なので、少なくともオーダーは揃っています。

ただし、**本当に揃えたいのは loss 値ではなく shared encoder への勾配寄与**です。原則は次の通り：

- 最初の基準: `weighted seg loss` が `line loss` の `0.3x〜1.0x` 程度
- より principled な基準: shared encoder の最後のブロックで `||∇L_line||` と `||∇(αL_seg)||` を比較
- 目標: 勾配ノルム比が極端にならないこと
  - 目安として `seg/line` が `0.2未満` なら seg が弱すぎる
  - `>2` なら seg が強すぎることを疑う

**各 loss を典型値で正規化する方法は、初手では勧めない。**
理由は、typical magnitude 自体が学習中に動くので、重みの意味が不安定になるため。動的化したいなら、`moving-average normalization` より **uncertainty weighting**、さらに厳密にやるなら **GradNorm** の方が筋が良い。

### 3. 部分教師のリスク

重要なのは、seg を **ラベルありサンプルだけで mean** しているなら、seg loss 自体は「4/8 だから半分に薄まる」わけではないことです。薄まるのは **更新頻度と有効サンプル数** です。

懸念の整理：

- **本質的な問題**: seg の 1 step あたりの強さより、seg supervision の総量が少ない
- **`α=0.03` が危険か**: 危険というより「少し弱めの安全側スタート」
- **本当に効かなくなる条件**:
  - seg GT ありサンプルが batch 内で安定して確保されない
  - `α·L_seg` が常時 `L_line` の 10% 未満
  - val mIoU が早期から全く立ち上がらない

**conditional alpha は勧めない。**
seg GT がある batch だけ重みを上げると、batch ごとの有効 objective が揺れて最適化がノイジーになります。補正したいなら：

- まず sampler を調整して、毎 batch `3〜4 / 8` の seg-labeled を確保
- その上で fixed `α=0.03` から始める
- seg が弱ければ `0.05` まで上げる

推奨 α 探索: `0.03`, `0.05` の 2 点。`0.1` はこの設定では強すぎる可能性が高い。

### 4. 監視項目

negative transfer を見るなら、loss より **単独学習との比較** が重要。

**最低限の監視:**
- Line validation: `angle_error_deg`, `rho_error_px`, `perpendicular_dist_px`
- Seg validation: `mIoU`, `per-class IoU`, 必要なら Dice
- Optimization logs:
  - `raw_line_loss`
  - `raw_seg_loss`
  - `α·raw_seg_loss`
  - `total_loss`
  - seg-labeled sample count per batch

**追加推奨（可能なら）:**
- shared encoder 上の task-wise gradient norm
- task gradient cosine similarity

**介入の目安:**

- seg の介入:
  - val mIoU が 5〜10 epoch ほぼ横ばい
  - かつ line 指標だけ改善している
  - かつ `α·L_seg / L_line < 0.1〜0.2` が継続
- line 側の介入:
  - `α` を上げた後に angle/rho が単独 line 学習より悪化
- negative transfer 判定:
  - multitask が single-task line より line 指標で悪い
  - かつ single-task seg より seg 指標でも悪い

このとき初めて `α` 調整、必要なら動的重みを検討。

### 5. 実務上の推奨

この規模なら、最初から動的 weighting に行く必要はない。むしろ baseline 解釈を壊す。

**推奨順:**
1. `L_total = L_line + αL_seg`
2. `α = 0.03` で開始
3. seg loss は labeled samples only mean
4. sampler で毎 batch `3〜4/8` の seg-labeled を確保
5. 5-fold CV では `α ∈ {0.03, 0.05}` だけ軽く比較
6. 明確な stagnation / negative transfer が出たときだけ dynamic weighting へ移行

dynamic に行くなら、順番は **Uncertainty Weighting → GradNorm** を勧める。前者の方が軽く、今の 2-task・小規模データには合う。

**一言まとめ:** 今回は「line 主、seg 補助」を明示した fixed-α baseline が正解。`α=0.03` は合理的な出発点で、まずは sampler と監視設計を固め、問題が観測されたときだけ `0.05` や dynamic weighting に進むのがよい。
