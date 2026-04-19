# 実験レポート: `multitask_v4(aug修正)` vs `seg_only_v2` 総比較

- 作成日: 2026-04-19
- 比較対象:
  - `Unet/outputs/seg_only_v2/baseline`
  - `Unet/outputs/seg_only_v2/lambda_bd_0.2`
  - `Unet/outputs/multitask_v4(aug修正)/sig3.5-alpha0.02`
  - `Unet/outputs/multitask_v4(aug修正)/sig3.5-alpha0.07`
  - `Unet/outputs/multitask_v4(aug修正)/sig3.5-alpha0.07-dual`
  - `Unet/outputs/multitask_v4(aug修正)/sig3.5-alpha0.15`
- 評価単位: 5-fold CV の `all_folds_summary.json` を使用

## 要点

- セグメンテーション単体で最良なのは `multitask_v4/a0.07` で、平均 mIoU は `0.9084`。
- multitask の中でライン精度が最良なのは `multitask_v4/a0.02` で、angle error `5.53°`、rho error `3.35 px`。
- `alpha=0.07` はセグメンテーション最良、`alpha=0.02` はライン最良、`alpha=0.07-dual` はその中間の妥協点という構図。
- `seg_only_v2/lambda_bd_0.2` は baseline を明確には上回らず、少なくともこの設定では境界重み付けの利益は確認できない。
- `alpha=0.15` は seg/line の両方で悪化傾向があり、seg loss の比重が強すぎる可能性が高い。

## 1. セグメンテーション総比較

| Run | mIoU | fg_mIoU | fg_mDice | vs baseline mIoU | vs baseline fg_mDice | fold勝ち数(mIoU) |
|---|---:|---:|---:|---:|---:|---:|
| seg_only_v2/baseline | 0.9052 | 0.8815 | 0.9351 | 0.0000 | 0.0000 | - |
| seg_only_v2/lambda_bd_0.2 | 0.9048 | 0.8810 | 0.9347 | -0.0004 | -0.0004 | 1 |
| multitask_v4/a0.02 | 0.9048 | 0.8810 | 0.9349 | -0.0003 | -0.0002 | 1 |
| multitask_v4/a0.07 | 0.9084 | 0.8855 | 0.9376 | 0.0032 | 0.0025 | 2 |
| multitask_v4/a0.07-dual | 0.9058 | 0.8823 | 0.9352 | 0.0007 | 0.0001 | 3 |
| multitask_v4/a0.15 | 0.9034 | 0.8793 | 0.9338 | -0.0018 | -0.0013 | 2 |

### paired t-test vs `seg_only_v2/baseline`

| Run | ΔmIoU | p(mIoU) | Δfg_mDice | p(fg_mDice) |
|---|---:|---:|---:|---:|
| seg_only_v2/lambda_bd_0.2 | -0.0004 | 0.926 | -0.0004 | 0.916 |
| multitask_v4/a0.02 | -0.0003 | 0.950 | -0.0002 | 0.957 |
| multitask_v4/a0.07 | 0.0032 | 0.480 | 0.0025 | 0.462 |
| multitask_v4/a0.07-dual | 0.0007 | 0.860 | 0.0001 | 0.966 |
| multitask_v4/a0.15 | -0.0018 | 0.660 | -0.0013 | 0.654 |

- いずれも `n=5` のため有意差は出ていない。
- ただし effect size の方向を見ると、`alpha=0.07` だけが seg 指標で一貫して正方向。

### クラス別 Dice / IoU

| Run | body Dice | right Dice | left Dice | posterior Dice |
|---|---:|---:|---:|---:|
| seg_only_v2/baseline | 0.9436 | 0.9114 | 0.9120 | 0.9679 |
| seg_only_v2/lambda_bd_0.2 | 0.9434 | 0.9097 | 0.9128 | 0.9666 |
| multitask_v4/a0.02 | 0.9449 | 0.9120 | 0.9089 | 0.9673 |
| multitask_v4/a0.07 | 0.9445 | 0.9172 | 0.9127 | 0.9693 |
| multitask_v4/a0.07-dual | 0.9450 | 0.9133 | 0.9073 | 0.9667 |
| multitask_v4/a0.15 | 0.9437 | 0.9097 | 0.9090 | 0.9672 |

- `alpha=0.07` は `right` と `posterior` で最良。
- `alpha=0.02` と `alpha=0.07-dual` は `body` が微増だが、`left` が baseline を下回る。
- `lambda_bd_0.2` は `left` だけ微改善、`right/posterior` は悪化。境界強調がクラス間で一様には効いていない。

## 2. multitask のライン検出比較

| Run | seg mIoU | angle error (deg) | rho error (px) | perp dist (px) | peak dist |
|---|---:|---:|---:|---:|---:|
| multitask_v4/a0.02 | 0.9048 | 5.53 | 3.35 | 11.56 | 21.45 |
| multitask_v4/a0.07 | 0.9084 | 5.93 | 3.55 | 11.55 | 21.96 |
| multitask_v4/a0.07-dual | 0.9058 | 5.71 | 3.43 | 11.45 | 22.70 |
| multitask_v4/a0.15 | 0.9034 | 6.21 | 3.73 | 11.28 | 23.23 |

- `alpha=0.02` が angle/rho の両方で最良。aug 修正後も、seg loss を軽く保った方が line task には有利。
- `alpha=0.07` は seg mIoU を最も押し上げるが、line は `alpha=0.02` より悪化。典型的な task trade-off。
- `alpha=0.07-dual` は dual 構成で perp dist が最良だが、angle/rho は `alpha=0.02` に届かない。共有 trunk の干渉は少し減ったが、C2/C6 の角度にはまだ不利。
- `alpha=0.15` は peak dist が最も悪く、heatmap 品質の劣化が line 指標悪化に繋がっている。

### 椎骨別 angle error

| Vertebra | a0.02 | a0.07 | a0.07-dual | a0.15 | best |
|---|---:|---:|---:|---:|---|
| C1 | 7.22 | 7.75 | 7.30 | 8.62 | a0.02 |
| C2 | 7.16 | 7.55 | 8.19 | 9.37 | a0.02 |
| C3 | 4.44 | 3.95 | 3.90 | 4.16 | a0.07-dual |
| C4 | 4.48 | 4.96 | 4.31 | 4.18 | a0.15 |
| C5 | 3.77 | 4.16 | 4.06 | 4.17 | a0.02 |
| C6 | 3.95 | 4.65 | 4.84 | 5.48 | a0.02 |
| C7 | 7.51 | 8.12 | 7.03 | 7.28 | a0.07-dual |

- `C1`, `C2`, `C6` は全体的に難しく、どの設定でも誤差が大きい。
- `C3-C5` は 4° 前後まで収まっており、ライン自体の学習は安定している。難所は上位椎骨と下位端。
- `alpha=0.07-dual` は `C3`, `C4`, `C7` で強いが、`C2` と `C6` がボトルネック。

## 3. 解釈

### 3.1 aug 修正後の最適点

- `seg` を主目的に置くなら `multitask_v4/a0.07` が最有力。baseline 比で mIoU `+0.0032`、fg_mDice `+0.0025`。
- `line` を主目的に置くなら `multitask_v4/a0.02` が最有力。angle `5.53°`、rho `3.35 px` は今回比較中ベスト。
- 両方のバランスを見るなら `multitask_v4/a0.07-dual` も候補だが、seg 改善量は `a0.07` に届かず、line も `a0.02` 未満。明確な主役にはなっていない。

### 3.2 `seg_only_v2` 側の示唆

- `lambda_bd_0.2` は baseline を超えなかった。少なくともこの重みでは boundary loss が強すぎるか、あるいは境界帯の定義が main objective と競合している。
- したがって次の seg-only 改善を考えるなら、`lambda_bd` の再 sweep か、boundary term の局所化を先に検討した方がよい。

### 3.3 multitask 側の示唆

- aug 修正後も `alpha_seg` の最適域は広くなく、`0.07` を超えると seg/line 両方に逆効果が出始める。
- `alpha=0.07` で seg が伸びることから、augmentation 修正は negative transfer を減らし、旧版より multitask の seg 便益を出しやすくした可能性が高い。
- 一方で line 最適はまだ `alpha=0.02` に留まるので、shared representation だけでは line degradation を完全には防げていない。dual の追加改善も限定的。

## 4. 結論

1. `multitask_v4(aug修正)` は設定次第で `seg_only_v2` と同等以上の seg 精度を維持しつつ line を追加できる。
2. ただしその成立条件は `alpha_seg` に敏感で、今回の最良 seg 設定は `alpha=0.07`、最良 line 設定は `alpha=0.02`。
3. 単一の万能設定はまだなく、用途別に checkpoint を分ける判断が合理的。
4. 次の本命は `alpha=0.07` 周辺の微調整と、`a0.02` の line 優位を保ったまま seg を底上げする設計。
