# weak/test_v2_beta=0: N/A/U比率変更(12/4/0)の5foldアブレーション

再現方法: `.claude/docs/experiments/2026-09-13-weak-beta0-N12A4U0/analyze.py`が、
学習済みcheckpointの`outer_predictions.csv`からこのディレクトリの`metrics.json`を
生成する。旧run(8/4/4)・baseline0との比較関数は
`.claude/docs/experiments/2026-09-12-weak-test-v2-beta0/analyze.py`の実装を再利用しており、
`fold_metrics.json.best_metrics`(inner検証のスナップショット)は使わず、すべて
outer predictionsから再計算している。

## 対象run

- `fracture_detection/weak/outputs/09_13/test_v2_beta=0_N12A4U0/`
  (`beta=0.0`、batch N/A/U=**12/4/0**、他は09/12のbeta=0 runと同一設定) — 5 outer fold全完走。
- `fracture_detection/weak/outputs/09_12/test_v2_beta=0/`
  (batch N/A/U=**8/4/4**) — 比較対象、5 outer fold全完走。
- `fracture_detection/baseline0/outputs/09_04/baseline0_aug追加/` — 専用whole分類器、5 outer fold全完走。

## 動機

`beta=0`ではU群(GT annotationなし陽性)の勾配は厳密にゼロで、optimizerはAdamWのため
N/A/U比率を変えても更新量にはほぼ影響しない。一方でA群(annotated positive)の
提示頻度・1 GT-passあたりのstep数・LR scheduleは維持したまま、陰性への露出だけを
1.5倍(N=8→12、batchは16のまま、U=0で穴埋め)に増やすと、
[[project_weak_beta0_ablation_result]]で見つかった「陽性確信度の崩壊」が緩和されるか
どうかを検証するのが目的。**結論: 変わらなかった。比率はレバーではなかった。**

## 1. pooled比較(5fold, N=13,432)

| | N12A4U0 | N8A4U4(旧) | baseline0 |
|---|---:|---:|---:|
| whole AP | 0.7048 | 0.7021 | 0.7405 |
| whole AUROC | 0.8979 | 0.9061 | 0.9085 |
| region macro AP (268 bag) | 0.7545 | 0.7649 | - |
| 陰性平均score | 0.0246 | 0.0335 | 0.0733 |
| 陰性p90 | 0.0459 | 0.0680 | 0.1885 |
| 陰性score≥0.5件数 | 78 | 116 | 362 |
| unweighted BCE | 0.2019 | 0.1881 | 0.1819 |
| ECE | 0.0354 | 0.0243 | 0.0316 |
| 平均score(全体) | 0.0637 | 0.0749 | 0.1307 |

study単位paired bootstrap(1000 replicate、`analyze.py::paired_bootstrap`):

| 差分 | mean | 95%CI | 有意か |
|---|---:|---:|---|
| whole AP: N12A4U0 − 旧 | +0.0025 | [-0.0081, +0.0141] | **有意差なし**(frac>0: 0.673) |
| whole AP: N12A4U0 − baseline0 | -0.0359 | [-0.0497, -0.0222] | 明確に負け(frac>0: 0.0) |
| whole AUROC: N12A4U0 − 旧 | **-0.0084** | [-0.0134, -0.0034] | **有意に悪化**(frac>0: 0.0) |
| region macro AP: N12A4U0 − 旧 | -0.0106 | [-0.0274, +0.0080] | 有意差なし(frac>0: 0.129) |

APは横ばい(信頼区間がゼロを跨ぐ)だが、AUROCは統計的に有意に悪化している。
陰性への露出を増やしたことで陰性スコア自体はさらにクリーンになった
(平均0.0335→0.0246、p90 0.068→0.046、score≥0.5が116→78件)にもかかわらず、
較正(BCE 0.188→0.202、ECE 0.024→0.035)はむしろ悪化した。
陰性判別の閾値付近の精度は上がったが、確率としての較正は崩れている。

## 2. fold別

| fold | 陽性数 | N12A4U0 AP | 旧 AP | 差 | best_gt_pass(new/old) |
|---|---:|---:|---:|---:|---|
| 0 | 262 | 0.7645 | 0.7748 | -0.0103 | 43 / 25 |
| 1 | 268 | 0.6986 | 0.7024 | -0.0038 | 51 / 35 |
| 2 | 269 | 0.7433 | 0.7444 | -0.0011 | 27 / 28 |
| 3 | 263 | 0.6739 | 0.6317 | **+0.0422** | 20 / 35 |
| 4 | 270 | 0.6958 | 0.6993 | -0.0035 | 34 / 36 |

outer3だけがはっきり改善し、他4foldは同等か微減。best_gt_passは前回2foldで見えた
「学習が遅れる」傾向(旧の方がbest passが早い)が5foldでは一貫せず、outer0/1では
むしろN12A4U0の方が遅く、outer2/4はほぼ同じ、outer3は逆に早い。60 pass上限に
当たったのはouter1(N12A4U0)だけ。

## 3. baseline0とのアンサンブルも変化なし

`analyze.py::ensembles`で09/12と同じ手法(logit CV加重平均、max結合)を再実行。

| | pooled AP | pooled AUROC |
|---|---:|---:|
| logit CV加重平均(N12A4U0) | 0.7529 | 0.9166 |
| logit CV加重平均(旧8/4/4) | 0.7527 | 0.9190 |
| max(baseline0, q1..q4)(N12A4U0) | 0.7468 | 0.9156 |
| max(baseline0, q1..q4)(旧8/4/4) | 0.7474 | 0.9195 |

いずれもほぼ同一。N/A/U比率を変えてもbaseline0とのアンサンブル性能に実質的な差はない。

## 結論

- N=12/A=4/U=0への変更は[[project_weak_beta0_ablation_result]]で確認された
  「beta=0はbaseline0の単体whole分類器に届かない」という結果を覆さなかった。
  whole APは実質横ばい(+0.003、CI片側)で、AUROCはむしろ有意に悪化した。
- 陰性露出を増やすと陰性スコアの分布そのものは改善する(より低く、より集中する)が、
  較正やAUROCには寄与しない。これは根本原因(`weak/modeling/losses.py`で
  `beta`が`weak_sum`(陽性の約80%)にしかかからず`negative_sum`は常時密という非対称性)
  が陽性側の確信度崩壊に起因するためで、陰性側の供給量を増やしても陽性側の
  starvationは解消されない、という[[project_weak_beta0_ablation_result]]の機序説明と整合する。
- region macro APも有意差なし。region-branchの局在性能はN/A/U比率に依存しない
  (A群のみから学習される設計通り)。
- 実用上の結論として、baseline0とのアンサンブル(logit CV加重平均 or max結合)を
  使うなら、beta=0側のN/A/U比率をどちらにしても結果はほぼ変わらない。
  したがって比率調整はこれ以上追求する価値がない。

## 注意点

- N=0のU群を使わない設定なので、`weak_bags_per_batch=0`のconfig/sampler対応が前提
  (schema側でbeta>0との組み合わせは拒否するガードを実装済み、テスト109件pass)。
- patience_gt_passesは両runとも15で揃っており、09/12のbeta=1 vs beta=0比較にあった
  交絡(patience 10 vs 15)はここでは発生していない。
- alpha選定・max結合の母数は引き続き5foldと小さい。
