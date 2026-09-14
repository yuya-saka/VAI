# weak/test_v2_beta=0: 5fold アブレーションとwhole検出アンサンブル

再現方法: `.claude/docs/experiments/2026-09-12-weak-test-v2-beta0/analyze.py`が、
学習済みcheckpointの`outer_predictions.csv`からこのディレクトリの`metrics.json`を
生成する。本文書のどの数値も再学習なしで再現できる。v1/v2の文書と同様、
`fold_metrics.json.best_metrics`はinner検証のスナップショットなので使わず、
すべてouter predictionsから再計算している。

## 対象run

- `fracture_detection/weak/outputs/09_12/test_v2_beta=0/`（`beta=0.0`、LSE集約
  tau=0.5、batch N/A/U=8/4/4、patience=15）— 5 outer fold全完走。
- `fracture_detection/weak/outputs/09_12/test_v2/`（`beta=1.0`、patience=10以外は
  同一設定）— outer0のみ。
- `fracture_detection/baseline0/outputs/09_04/baseline0_aug追加/` — 専用のwhole分類器
  （同じencoder、通常のwhole label BCE、region branchやFPNなし）、5 outer fold全て。

## 1. outer0限定のmatched比較: beta=1 vs beta=0（beta以外は同一設定）

| | beta=1 | beta=0 | baseline0 |
|---|---:|---:|---:|
| whole AP | 0.7405 | 0.7748 | 0.7726 |
| whole AUROC | 0.9160 | 0.9284 | 0.9186 |
| 陰性p_whole平均 | 0.0670 | 0.0221 | 0.0678 |
| 陰性p_whole p90 | 0.1717 | 0.0357 | 0.1738 |
| region macro AP (56bag) | 0.7717 | 0.7677 | - |

outer0だけを見ると、beta=0はAP・AUROCともbeta=1を上回り、baseline0にも並ぶか上回る。
局在(region macro AP)はbetaの値にほぼ影響されない。設計上、betaが触るのはGT annotation
のない群だけで、局在はGT annotation群だけから学習されるため、これは想定通り。

## 2. 5fold全体像: outer0の結果は一般化しない

| fold | 陽性数 | beta=0 AP | baseline0 AP | 差 | beta=0 AUROC | baseline0 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 262 | 0.7748 | 0.7726 | +0.0022 | 0.9284 | 0.9186 |
| 1 | 268 | 0.7024 | 0.7238 | -0.0214 | 0.9033 | 0.9106 |
| 2 | 269 | 0.7444 | 0.7591 | -0.0147 | 0.9179 | 0.9167 |
| 3 | 263 | 0.6317 | 0.6993 | -0.0676 | 0.8821 | 0.8834 |
| 4 | 270 | 0.6993 | 0.7596 | -0.0602 | 0.9035 | 0.9169 |
| **pooled** | 1332 | **0.7021** | **0.7405** | **-0.038** | **0.9061** | **0.9085** |

beta=0が勝っているのはouter0だけで、残り4foldは全てbaseline0が上回り、うち2foldは差が
大きい。outer0だけの好結果は一般化しない(`[[feedback_validate_across_all_folds]]`参照)。

## 3. なぜ陰性はクリーンなのに4/5foldでbaseline0に負けるのか

fold別診断(`analyze.py`の`whole_metrics`)を見ると、原因は陰性への侵入ではない——
beta=0の陰性平均/p90はどのfoldでもbaseline0より低い(クリーン)。原因は真陽性bagの
確信度そのものの崩壊にある。

| fold | beta=0陽性中央値 | baseline0陽性中央値 | beta=0のtop100内陽性数 | baseline0のtop100内陽性数 |
|---|---:|---:|---:|---:|
| 0 | 0.4845 | 0.9277 | 99 | 99 |
| 1 | 0.3143 | 0.7480 | 91 | 94 |
| 2 | 0.5529 | 0.8945 | 91 | 95 |
| 3 | 0.4008 | 0.7617 | 88 | 96 |
| 4 | 0.4729 | 0.8965 | 92 | 94 |

**根本原因**（`weak/modeling/losses.py::compute_weak_losses`より）:
`total = (negative_sum + annotated_sum + beta * weak_sum) / total_bags`。
betaが掛かるのは`weak_sum`（bboxのない弱陽性=陽性の約80%）だけで、`negative_sum`と
`annotated_sum`にはbetaが一切関与しない。これは実装のクセではなく、whole labelだけで
region単位に何が確定するかという情報量の違いから必然的に出てくる非対称性である。

- whole**陰性**ラベルは4領域すべてを確定させる(全て0)ので、N群の教師には集約関数が
  不要であり、betaのゲートを一切経由しない。
- bboxのないwhole**陽性**ラベルは「少なくとも1領域は陽性」という不完全な情報しか
  与えず、これを勾配に変換するには集約関数(そしてbeta)が必須になる。

beta=0にすると陰性への供給は完全に残る(だから陰性スコアが非常にクリーン)一方、
陽性の約80%への唯一の勾配経路が消え、確信度が崩壊する。その結果、少数の残存する
高スコア陰性がランキング上位で真陽性を追い越してしまい、約10%のprevalenceでは
APへの影響が特に大きい。

## 4. whole分類専用headの設計相談（`.claude/docs/codex/20260913-1440-whole-head-vs-region-branch-design.md`参照）

ユーザーから「whole分類専用headを並列で追加すれば両方うまく学習できないか」という
提案があった。これは`REGION_MIL_DESIGN.md`の「whole/region並列分岐を作らない」という
明示的な制約に抵触する。この制約の目的は、whole判定の成否をregion-branchアプローチ
自体の診断材料として使える状態に保つことにある（`p_whole`は4つのregion logit経由
以外に学習可能な経路を持たない）。

Codexの分析（全文は上記リンク先）：既存のArm Bにwhole headを追加してはならない。
試すなら科学的主張の異なる新規Arm Dとして分離し(`p_global`と`p_region`は常に別々に
報告する)、まず非侵襲的なfrozen probe（beta=0 checkpointを凍結し、stop-gradientの
whole headを後付け）から始め、readoutのボトルネックか表現自体の不足かを切り分ける
べき。単一経路を維持したまま試せる優先度の高い代替案（アーキテクチャ変更なし）は、
U群のpositive BCEをpairwise ranking lossに置き換える案、および集約をLSEからmax/
hard-LSEに変えてU群の勾配を1領域に集中させる案。いずれも未実施。

## 5. baseline0とのアンサンブル（実装済み、再学習なし）

beta=0の出力とbaseline0の`vertebra_score`をパラメータの少ない方法で事後結合する
2案を実測した。いずれも既存の`outer_predictions.csv`だけから計算できる。

### 5a. ロジット空間での加重平均

`combined = sigmoid(alpha * logit(p_whole) + (1-alpha) * logit(vertebra_score))`。
alphaは各foldについて、他の4fold(pooled)上でのgrid search(step 0.05)で選び、
held-out foldへ適用した(`analyze.py`の`select_alpha_by_ap`/`ensemble_analysis`)——
評価対象のfold自身でチューニングしていない。

| | AP | AUROC |
|---|---:|---:|
| beta=0単体(pooled) | 0.7021 | 0.9061 |
| baseline0単体(pooled) | 0.7405 | 0.9085 |
| 固定50/50ブレンド、fold3(最悪ケース) | 0.7019 | 0.8981 |
| **CV選択alpha、pooled** | **0.7527** | **0.9190** |
| 全data事後最適alpha(0.35)、参考値 | 0.7528 | 0.9186 |

CVで選ばれたalphaはfold0-2で0.35、fold3-4で0.40と安定しており、事後最適値とほぼ
一致するので過学習ではない。単純な50/50ブレンドだけでも5foldのうち4foldで
baseline0・beta=0の両方を上回った。

### 5b. パラメータフリーのmax結合

`combined = max(vertebra_score, q_1, q_2, q_3, q_4)` —— チューニング一切不要。

| | pooled AP | pooled AUROC |
|---|---:|---:|
| max(q_1..q_4)単体(baseline0なし) | 0.6993 | 0.9050 |
| baseline0単体 | 0.7405 | 0.9085 |
| beta=0自身のp_whole(LSE) | 0.7021 | 0.9061 |
| **max(baseline0, q_1..q_4)** | **0.7474** | **0.9195** |

outer0/1/2/4でbaseline0に勝り、outer3はほぼ同着(0.6992 vs 0.6993)、パラメータは
ゼロ。`max(q_1..q_4)`単体はbeta=0自身のp_wholeより悪いので、効果はbaseline0を
土台にregionスコアを安全弁として足す組み合わせ自体にあり、region側のmax単独には
価値がない。

### 5c. なぜアンサンブルが効くのか: 誤り方が相補的(同一ではない)

beta=0の`p_whole`とbaseline0の`vertebra_score`の相関(ロジット空間、pooled):
陰性0.66、陽性0.86——相関はあるが完全には一致しない。これが平均化を有効にする条件。

| 陰性(n=12,100)、閾値0.5 | 件数 |
|---|---:|
| 両方≥0.5(直せない) | 85 |
| baseline0だけ≥0.5(beta=0が正しく抑制) | **277** |
| beta=0だけ≥0.5(baseline0が正しく抑制) | 31 |

| 陽性(n=1,332)、閾値0.3 | 件数 |
|---|---:|
| 両方確信 | 739 |
| beta=0が弱く、**baseline0が救済** | **238** |
| baseline0が弱く、beta=0が救済 | 18 |
| 両方弱い(直せない) | 337 |

これは第3節の機序そのものである: beta=0の常時密な陰性supervisionがbaseline0の
誤検知277件を抑制し、baseline0の常時密な陽性supervision(80%starvationがない)が
beta=0の確信度不足238件を救済する。2つのモデルは損失構造が正反対の方向で非対称
なため、正反対の・相補的な方向で失敗している。

## 注意点

- `test_v2`(beta=1)はouter0のみ実行のため、第1節のbeta=1 vs beta=0比較は他4fold
  では未実施。
- `patience_gt_passes`がbeta=1(10)とbeta=0(15)で異なる。`best_gt_pass`
  (outer0で21 vs 25)から見て、この交絡の影響は軽微とみられる。
- アンサンブルの結果は実用上の問い(beta=0のregion出力を局在用に保ったまま、
  達成可能な最良のwhole検出は何か)に答えるものであり、元の研究上の問い
  (region evidenceだけでwhole状態を説明できるか、Arm A/B/Cが答えるS1)には
  答えていない。後者を解決したとは読まないこと。
- alpha選定の母数は5foldと小さい。fold間でalphaが0.35〜0.40と安定している点は
  安心材料だが、外部検証データの代わりにはならない。
