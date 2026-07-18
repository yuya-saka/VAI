# Stage3 実装設計書：階層的弱教師 骨折局在モデル

**確定日**: 2026-07-15
**位置づけ**: `memo/計画書/弱ラベル学習設計書.md`（上流設計）の忠実実装。次セッションで本書を元に `train_models/stage3/` を実装する。
**参照**: Codex設計レビュー `.claude/docs/codex/20260715-stage3-design-review.md`（P0修正・退化・比較交絡の指摘を本書に織り込み済み）

---

## 0. 一行サマリ

椎体レベルの二値ラベル **のみ** で学習し、`p[z,r] → q[r] → ŷ_V` の階層だけを通して椎体骨折を予測する。direct head を持たない。副産物としてスライス×領域の contextual evidence を出力し、**将来の領域アノテーションの候補順位づけ**に使う。

**重要**: `p/q/a` は「骨折確率」でも「局在の正解」でもなく **contextual evidence score**（文脈依存の証拠スコア）である。呼称・報告・下流利用でこの区別を厳守する。

---

## 1. 目的と研究仮説

### 1.1 目的
1つの椎体CTから、椎体・4領域・スライスの各レベルで骨折の証拠を同時出力するモデルを、椎体レベル弱ラベルのみで学習する。

### 1.2 研究仮説（上流設計書§13）
椎体ラベルを全スライスに複製して学習するより、slice×region の潜在構造を明示し階層集約するほうが、意味のある局在を得つつ椎体分類の頑健性も保てる。

### 1.3 stage1/stage2 との関係
| | stage1 | stage2 | **stage3（本書）** |
|---|---|---|---|
| 椎体予測経路 | direct head（flat MIL相当） | direct head 主＋領域補助 | **階層のみ（head無し）** |
| q[r]（4領域スコア） | 無 | 形成しない（面ごとNoisy-OR→平均） | **明示形成** |
| スライス集約 | BiLSTM+平均 | 面平均 | **p由来tied attention** |
| 領域→椎体 | — | 面ごとNoisy-OR→平均 | **正規化Smooth-Max** |
| 学習教師 | 椎体ラベル（per-slice複製） | 椎体ラベル（primary複製＋領域bag） | **椎体ラベルのみ（bag loss）** |

stage3 は「純粋な弱教師階層」であり、stage1（flat）/stage2（aux）と並ぶ第3の比較点。

---

## 2. 全体アーキテクチャ（forward）

入力: `images (B, 15, 6, 224, 224)`（5ch CT window + 1ch 椎体マスク）、`region_masks (B, 15, H, W)` 整数ID or `(B,15,4,H,W)` soft。

```
1. 共有encoder（EfficientNetV2-S, ImageNet init, end-to-end, 凍結なし）
   + FPN + マスク正規化プーリング          ← stage2 を流用
   → f[z,r] ∈ R^256   （slice z × region r=4 の特徴）

2. 領域共有BiLSTM（スライス方向, 重み共有）→ h[z,r]
   → Linear_p → p[z,r]（contextual evidence logit）
   ※ 独立attentionヘッドは持たない

3. スライス集約（p由来 tied attention）:
   a[z,r] = softmax_z( p[z,r] / T_a )         （有効plane限定, 無効は-inf）
   q[r]   = Σ_z a[z,r] · p[z,r]                （region-pooled evidence score）

4. 領域集約（正規化Smooth-Max, LSE）:
   ŷ_V_logit = ( logsumexp_{r∈valid}(τ·q[r]) − log|valid| ) / τ
   ŷ_V       = sigmoid(ŷ_V_logit)
   ※ direct head 無し。椎体予測はこの経路のみ。

5. スライスレベル出力:
   s_z_logit = ( logsumexp_{r∈valid_z}(τ·p[z,r]) − log|valid_z| ) / τ
```

**tied attention の性質**: `a=softmax(p/T_a)` は evidence 自身を重みにする self-weighted pooling で、実質 soft-max pooling（temperature T_a）。よって上流設計§6「attentionでスライス集約」の趣旨を保ちつつ、独立ヘッドが持つ「高evidenceを隠す」退化の自由度を持たない。

---

## 3. ディレクトリ・モジュール構成

```
train_models/stage3/
├── train.py                 # entry point（stage2ミラー: mp.spawn 自動マルチGPU）
├── config/
│   └── config.yaml          # 実験設定
├── src/
│   ├── model.py             # Stage3Model, HierarchicalRegionHead
│   ├── dataset.py           # stage2 流用（ct.npy + vertebra_mask.npy + region_4class.npy）
│   ├── data_utils.py        # stage2 流用（split/loaders/DistributedSampler）
│   ├── staging.py           # stage2 流用（NFS→ローカルstaging）
│   ├── evaluation.py        # 椎体metrics + 退化診断 + 局在診断
│   ├── experiment.py        # stage2 流用（wandb/paths/logging）
│   └── trainer.py           # stage2 を改修（AMP/DDP/early stop/guard, head無し対応）
└── utils/
    ├── losses.py            # stage3_loss, 正規化smooth-max, tied attention pooling
    └── metrics.py           # stage1/stage2 流用
```

### 3.1 再利用元コード（file:line、実装時に参照）
- **encoder + FPN + マスクプーリング**: `train_models/stage2/src/model.py`
  - `Stage2Model.__init__` の encoder/FPN構築（`:110-157`）
  - `_pool_regions`（`:227-254`, マスク正規化average pooling, `region_plane_valid` 生成）
  - `forward` の encoder→FPN→pooling 部分（`:170-216`）
- **数値安定Noisy-OR primitive**（参照実装・無効logitを読まない`torch.where`パターン）: `train_models/stage2/utils/losses.py:28-46`
- **weighted BCE / positive_weight**: `train_models/stage1/utils/losses.py`（`weighted_bce`, positive_weight=2.0）
- **flat baseline**: `train_models/stage1/src/model.py`（`TimmModel`）
- **data pipeline / trainer骨格 / staging / metrics**: stage2 の同名モジュール

### 3.2 新規実装
- `Stage3Model`（head無し forward, 下記集約を接続）
- `HierarchicalRegionHead`（BiLSTM → p[z,r]。stage2 `SharedRegionHead:32-65` を p単出力に簡約）
- `tied_attention_pool` / `normalized_smoothmax` / `slice_evidence`（全て all-invalid 対応）
- `stage3_loss`（bag weighted BCE + vertebra-balanced negative-instance）
- 退化・局在診断（evaluation.py）
- 比較統制の control 実装（4-slot global, spatial-scramble）

---

## 4. コンポーネント詳細仕様

### 4.1 HierarchicalRegionHead
- 入力: `f[z,r]`（B,15,4,256）を `(B*4, 15, 256)` に整形
- BiLSTM（hidden=256, layers=2, bidirectional, batch_first, FP32実行）
- Linear_p: `hidden*2 → 1` → `p[z,r]`（B,15,4）
- **bias有無**: Linear_p は bias 可。attention用の独立層は無いので zero-init 議論は不要。

### 4.2 tied_attention_pool（スライス集約）
```
入力: p (B,S,R) logits, region_plane_valid (B,S,R) bool, T_a
masked = where(region_plane_valid, p / T_a, -inf)
a = softmax over S of masked                      # (B,S,R)
q = sum over S of ( a * where(region_plane_valid, p, 0) )   # (B,R)
region_valid = region_plane_valid.any(dim=S)      # (B,R)
# 全スライス無効の(領域)は softmax が NaN → region_valid=False の列は
# a を 0 埋め・q を 0 埋めし、後段LSEで除外（値は使われない）
return q, a, region_valid
```
FP32・`torch.where` で `0*inf` 回避。masked softmax は FP32。

### 4.3 normalized_smoothmax（領域集約）
```
入力: q (B,R) logits, region_valid (B,R) bool, tau
masked = where(region_valid, tau*q, -inf)
lse = logsumexp over R of masked                  # (B,)
log_count = log(region_valid.sum(R).clamp_min(1)) # (B,)
y_logit = (lse - log_count) / tau                 # (B,)
# 全領域無効の椎体（data error）は loss 対象外にする（後述）
```
**正規化必須**（P0）: 未正規化だと全 `q_r=c` で `y=c+log|valid|/τ` となり領域数依存の陽性バイアス＋global control比較が交絡する。この形なら全 `q_r=c` で `y=c`。

### 4.4 slice_evidence（s_z）
`normalized_smoothmax` を「領域方向」に各スライス独立で適用。valid_z = そのスライスの有効領域集合。all-invalid スライスは s_z=無効扱い。

### 4.5 出力（すべて contextual evidence）
`ŷ_V=sigmoid(y_logit)` / `q[r]` / `p[z,r]` / `a[z,r]` / `s_z` / `region_valid` / `plane_valid`。

---

## 5. 損失関数

```
L = L_bag + λ_neg · L_neg

L_bag = weighted_BCE_with_logits(y_logit, y_V; positive_weight=2.0)   # stage1と一致
        （全領域無効の椎体は除外）

L_neg（陰性椎体のみ, vertebra-balanced）:
  対象 = 陰性椎体( y_V=0 ) かつ mixup時は「両ソースが陰性」の標本のみ
  per_v[b] = Σ_{z,r} valid·BCE(p[z,r], 0) / Σ_{z,r} valid   # 椎体内平均
  L_neg    = mean_b per_v[b] over 対象椎体                   # 椎体間平均（対象0件なら0）
```

**規約（Codexレビュー反映）**
- **positive bag に `BCE(p,0)` を適用しない**（恣意的sparsity prior、単一slice spikeを助長）。正例に正則化が要るなら augmentation consistency / z平滑性 / attention entropy 等 label-free を使う。
- negative-instance は **椎体内平均→椎体間平均の2段**（フラット平均は有効数の多い椎体を過重）。
- mixup で正負が混ざった標本を陰性instanceとして扱わない。
- `L_bag` の positive_weight は stage1（=2.0）と一致させないと比較不能。
- empty reduction は 0 loss として明示（NaN防止）。

**λ_neg 注意**: 「小さいから安全」と固定しない。attention が false-positive を隠し始めると 0.1 では弱い可能性。gradient-norm 比を見て調整。

---

## 6. 数値安定性・堅牢性

- **all-invalid 処理**（P0）:
  - `region_valid[r] = any_z(region_plane_valid[z,r])`、all-invalid 領域は LSE から除外。
  - all-invalid 椎体（valid領域0）は data error として検出し loss 対象外＋ログ。
  - masked softmax / LSE は FP32。無効値との積は `torch.where` かvalid indexing。
- **精度**: CNN/FPN は BF16、BiLSTM・heads・attention・LSE・loss は FP32（stage2 `SharedRegionHead` も FP32実行 `model.py:60`）。
- **guard**: global grad clip 1.0、non-finite output/grad の fail-fast（stage1/stage2 のガードを流用）。
- **初期化**: evidence 最終 weight は小さめ init、bias は class-prior logit に初期化。

---

## 7. ハイパーパラメータ

| 項目 | 既定 | アブレーション範囲 |
|---|---|---|
| τ（region smooth-max温度） | 1.0 | {0.5, 1, 2, 4}（初回は学習可にしない） |
| T_a（slice attention温度） | 1.0 | 後で探索 |
| λ_neg | 0.1（候補、grad-norm比で調整） | {0, 0.1, 0.3, 1.0} |
| 時系列モデル | BiLSTM | Transformer |
| slice集約 | tied attention | LSE固定 / 独立attention |
| region集約 | 正規化Smooth-Max | Noisy-OR / Max |
| positive_weight | 2.0（stage1一致） | — |

---

## 8. 学習設定（stage1/2 と厳密一致）

比較を交絡させないため、以下を **3ステージで一致**させる:
- ImageNet init・end-to-end・凍結なし（stage3 だけ warm-start/freeze/長epoch/別LR は禁止）
- backbone LR / scheduler / effective batch / epoch budget / augmentation / mixup / selection metric
- 同一seed群、同一 study split、同一 OOF 行（同一 sample manifest で対象椎体が一致することを確認）

必要なら「parity protocol（完全一致）」と「各モデル個別tune protocol」を分けて報告。

---

## 9. 比較統制（control）

椎体AUROCの差を「階層」に帰属するため、以下の control を用意（Codexレビュー）:
1. **capacity-matched global control**: FPN + 同一 attention/bag loss + no-anatomy pooling。**4-slot 数・valid数・aggregator を一致**させる（stage2 の 1-region global では容量不一致）。
2. **spatial-scramble control**: 面積・valid pattern を保ったままマスクの**空間対応を破壊**する。
   - ⚠ **領域ID置換は無効**: `SharedRegionHead` は重み共有・aggregator は region permutation 不変なので、4ch名の入れ替えは数学的に同一出力。空間マスク自体を崩すこと。
3. 完全に同じ OOF 行で比較。

---

## 10. 評価・診断

### 10.1 椎体レベル（guardrail）
stage1 の metrics（AUROC/AUPRC/per-vertebra）を流用。**非劣性**を確認: paired bootstrap で ΔAUROC の95%下限 > −0.01（対 stage1 0.92）。AUPRC・固定specificityでのsensitivityも非悪化。

### 10.2 退化・局在診断（アノテーション不要）
- 実効領域数 `1/Σ_r g_r²`（g_r=softmax(τq)_r, 領域LSE重み）
- 実効スライス数 `1/Σ_z a_{z,r}²`（領域ごと）
- 領域 argmax 分布 / attention entropy
- **seed間の帰属安定性**（winner領域・attentionの順位相関）
- 左右反転 equivariance
- global control / spatial-scramble control との差（anatomyが効いているか）
- p/q/y の logit 分布

**judge**: 椎体AUROCが stage1 に非劣 **かつ** 単一領域collapse・seed不安定・attention一様（局在してないのにAUROCだけ出る）でない。

### 10.3 位置づけの厳守
`p/q/a/s_z` は「将来の領域アノテ候補を順位づける仮説」。「slice fracture probability」「region fracture probability」と呼ばない・報告しない。最終的な解剖学的主張には将来の領域アノテによる外部検証が必須。seed安定性は必要条件にすぎない（安定したshortcutはあり得る）。

---

## 11. 最初の実験（アノテーション不要・go/no-go）

- 1 fold・3 seeds・既定構成（tied attention + 正規化smooth-max + head無し + λ_neg=0.1）、椎体ラベルのみ。
- 同fold で **global(4-slot) control** と **spatial-scramble control** も実行。
- **go 条件**: 椎体AUROCが stage1 に非劣 かつ 退化診断で masked が control 群（global/scramble）を局在指標で安定して上回る。
- **no-go 時**: full 5-fold や Transformer・複雑attentionに進む前に、bbox 20〜40例からの soft positive anchor（部分教師）導入を検討（＝`4領域骨折検出_部分教師付きMIL学習計画.md` への接続）。

---

## 12. 実装タスク分解（次セッション）

1. **scaffold**: `train_models/stage3/` を stage2 ミラーで作成。dataset/data_utils/staging/experiment/metrics を流用・調整。
2. **Stage3Model**: encoder+FPN+`_pool_regions` を再利用（import/subclass or 抽出）。primary head/lstm を削除。`HierarchicalRegionHead`（BiLSTM→p）を実装。
3. **集約関数**: `tied_attention_pool` / `normalized_smoothmax` / `slice_evidence`（all-invalid 対応）。
4. **stage3_loss**: bag weighted BCE(pw=2.0) + vertebra-balanced negative-instance（mixup対応・empty=0）。
5. **trainer**: FP32/BF16分離・grad clip 1.0・non-finite guard・init。head無しに対応（selection metric は椎体AUROC）。
6. **evaluation**: 実効領域/スライス数・argmax分布・entropy・seed安定性ハーネス・椎体metrics。
7. **config.yaml**: §7/§8 の既定。encoder_init/freeze/region_mode(masked|global|scramble)/slice_agg/region_agg/tau/T_a/lambda_neg/temporal を切替可能に。
8. **control**: 4-slot global、spatial-scramble を実装。
9. **単体テスト**（下記）。
10. **最初の実験**（§11）。

---

## 13. 単体テスト（testing rules準拠）

- `normalized_smoothmax`: 全 `q_r=c` → `y_logit≈c`（正規化の検証）。
- all-invalid 領域: NaN が出ず、LSE から除外される。
- all-invalid 椎体: loss 対象から除外される。
- `tied_attention_pool`: a が有効スライス上で和1、無効スライスで0。
- `stage3_loss`: バッチに陰性椎体が無ければ `L_neg=0`。
- mixup: 正負混合標本に negative-instance が適用されない。
- parity 的検証: region_agg/attention を通した経路が既知の小入力で手計算と一致。

---

## 14. 決定履歴（確定事項）

- スライス集約 = **p由来 tied attention**（`a=softmax_z(p/T_a)`、独立eヘッド廃止）— ユーザー確定
- 領域集約 = **正規化 Smooth-Max**（LSE, `−log|valid|`）— ユーザー確定 + Codex P0
- **direct head 無し** — ユーザー確定（上流§9）
- encoder = **ImageNet init・end-to-end・凍結なし**（stage1 warm-start却下）— ユーザー確定（fair comparison）
- 損失 positive_weight=2.0 / negative-instance 2段正規化 / positive bagに BCE(p,0)禁止 — Codexレビュー反映
- 出力は contextual evidence（確率ではない）— Codexレビュー反映

---

## 15. 参照
- 上流設計: `memo/計画書/弱ラベル学習設計書.md`
- 部分教師拡張（将来）: `memo/計画書/4領域骨折検出_部分教師付きMIL学習計画.md`
- Codexレビュー全文: `.claude/docs/codex/20260715-stage3-design-review.md`
- Codex研究優先度: `.claude/docs/codex/20260715-weak-label-research-priorities.md`
- 再利用元: `train_models/stage2/`（encoder/FPN/pooling/Noisy-OR/pipeline）, `train_models/stage1/`（TimmModel/weighted_bce/metrics）
