# v3完走・評価指標の汚染発見・CAM再監査・conditional指標の実装

作成日: 2026-09-04
状態: **v3 outer fold 0 完走。評価系の欠陥を特定し計測を追加。fold 1-4 は未実行**

参照する正本:

- `.claude/docs/work-logs/2026-09/2026-09-03-unified-region-target-and-v3-calibration.md`
- `.claude/docs/work-logs/2026-08/2026-08-28-region-loss-diagnosis.md`
- `fracture_detection/region_branch/config/region_branch_all.yaml`

---

## 1. v3 学習の完走

`09_03_region_branch_all/test_v3/outer0/`、GPU 1、2026-09-03 14:28 開始・18:54 終了。

- epoch 71 で早期終了（`bad_epochs=20`、patience=20）
- `best_region_epoch=51`（`val_region_centered_loss=0.132416`）
- `best_whole_epoch=5`（`val_whole=0.314235`）
- `collapse_alarm` は全 71 epoch で未発火
- 1 epoch 約 220 秒（epoch 1 のみ torch.compile 初回で 1218 秒）

## 2. 発見: 学習中に見ていた region 指標がすべて汚染されていた

「精度が伸びない」という観測の実体は、**性能ではなく計測の問題**だった。学習ログに出る
region 系の指標は、選択基準も含めて局在性能を測っていない。

### 2.1 `val_region_*_ap` / `_auroc` / `val_region_macro_ap`

`trainer.py` の評価関数がスコアを marginal で作っている。

```python
region_probability = conditional_region_probability * whole_probability[:, None]
```

さらに母集団が `region_hard_valid`（`dataset.py:_resolve_region_targets`）で、椎体陰性 bag の
4 領域すべてが有効セルになる。inner val (fold 1, 2,687 bag) の内訳:

| 内訳 | セル数 |
|---|---:|
| 椎体陰性 bag の論理 0 | 9,676 |
| 人手 GT | 183 |
| 合計 | 9,859 |

**母集団の 98% が自明な陰性**で、実質 whole の陰性除去性能を測っている。加えて whole head は
epoch 5 以降過学習しており（`val_whole` rho=+0.314 p=0.021 で悪化、`train_whole_loss`
rho=-0.916、`val_whole_auroc` は rho=+0.148 p=0.29 で横ばい、`is_best_whole` は epoch 1/2/5 のみ）、
その drift が region 指標へ写り込む。

wandb 全 epoch での実測:

| region 指標 | vs `val_whole_ap` | vs `val_whole_auroc` |
|---|---:|---:|
| region_4_auroc | +0.796 (4e-13) | +0.373 |
| region_2_auroc | +0.665 (3e-08) | +0.173 |
| region_1_auroc | -0.613 (7e-07) | -0.074 |

whole_auroc とはほぼ無相関で whole_ap とだけ強相関 = 「過学習で上位が尖った whole head」を
見ている。

### 2.2 `val_region_centered_loss`（early stopping と checkpoint 選択の基準）

whole を掛けないので whole 汚染は無いが、母集団が `region_target_valid` ∧ 椎体陽性で、内訳が

| 内訳 | セル数 | 割合 |
|---|---:|---:|
| CAM 疑似 | 889 | 82.9% |
| 人手 GT | 183 | 17.1% |

**83% が「教師にどれだけ似たか」**。`student_teacher_spearman` 平均が rho=+0.887 (p=2e-18) で
上昇、`val_region_centered_loss` が rho=-0.753 で低下しており、両者は同じものを符号違いで
測っている。`early_stopping_metric: val_region_centered_loss` なので、checkpoint 選択は実質
「CAM 教師に最も似た epoch を選ぶ」になっている。

### 2.3 人手 GT の陽性が少なすぎる

inner val で AP/AUROC を決めている陽性は **72 個**（R1 15 / R2 11 / R3 14 / R4 32）。
epoch 単位のトレンドはノイズに埋もれる。

### 2.4 v2 と v3 は同名フィールドでも比較不能

コード改修でスコア定義とマスクが両方変わっている。

| | v2（旧コード） | v3（新コード） |
|---|---|---|
| `region_*_score` | `bag_logits.sigmoid()` = **conditional** | conditional × p(whole) = **marginal** |
| AP 母集団 | `region_target_valid`（人手 GT のみ）983 セル | `region_hard_valid`（＋陰性）49,383 セル |
| 陽性率 | 0.243〜0.629 | 0.0048〜0.0128 |

同名指標を並べた比較はすべて無効。正しくは v2 は `region_*_score`、v3 は
`region_*_conditional_score` を人手 GT に突合する。

### 2.5 汚染されていない指標

- `val_whole_auroc`
- `student_teacher_spearman_region_*`（ただし測っているのは精度ではなく教師との一致度）
- `spearman_region_i_vs_j` / `first_pc_explained_variance`（collapse 監視は正常。領域間
  Spearman 最大 0.284、閾値 0.95）
- `outer_predictions.csv` の `region_*_conditional_score`（これだけが局在性能を測れる）

## 3. 正しい母集団での実測結果（outer fold 0）

教師 / v2 / v3 を **完全に同一の 262 bag・177 studies** に揃え、人手 GT 有効セル・椎体陽性で
比較。教師スコアは `pseudo_region_targets.csv` の `student_outer_fold=4, teacher_outer_fold=0`
の行（fold 0 を学習に使っていない教師による採点）。CI は study_id の患者クラスタ
bootstrap 2,000 回。

### AUROC

| 領域 | n | 陽性 | 教師 | v2 | v3 | v3−v2 [95%CI] | v3−教師 [95%CI] |
|---|---:|---:|---:|---:|---:|---|---|
| R1 椎体 | 51 | 15 | 0.8796 | 0.9630 | 0.9426 | -0.020 [-0.110, +0.054] | +0.063 [-0.040, +0.179] |
| R2 右横突孔 | 49 | 12 | 0.8423 | 0.6959 | 0.8176 | +0.122 [-0.101, +0.325] | -0.025 [-0.124, +0.051] |
| R3 左横突孔 | 52 | 14 | 0.9211 | 0.6259 | 0.9643 | **+0.338 [+0.125, +0.525]** | +0.043 [-0.014, +0.138] |
| R4 後方要素 | 52 | 31 | 0.8295 | 0.8111 | 0.8848 | +0.074 [-0.052, +0.190] | +0.055 [-0.039, +0.165] |
| 平均 | | | 0.8681 | 0.7740 | 0.9023 | +0.128 | +0.034 |

### AP

| 領域 | 教師 | v2 | v3 | v3−v2 [95%CI] | v3−教師 [95%CI] |
|---|---:|---:|---:|---|---|
| R1 椎体 | 0.6643 | 0.9210 | 0.9103 | -0.011 [-0.145, +0.126] | **+0.246 [+0.015, +0.465]** |
| R2 右横突孔 | 0.7156 | 0.5084 | 0.6470 | +0.139 [-0.150, +0.410] | -0.069 [-0.217, +0.071] |
| R3 左横突孔 | 0.8935 | 0.4436 | 0.9119 | **+0.468 [+0.202, +0.634]** | +0.018 [-0.072, +0.144] |
| R4 後方要素 | 0.8943 | 0.8907 | 0.9314 | +0.041 [-0.030, +0.109] | +0.037 [-0.021, +0.130] |
| 平均 | 0.7919 | 0.6909 | 0.8501 | +0.159 | +0.058 |

### 有意なのは 2 つだけ

1. **R3 の v3 > v2**（AUROC +0.338、AP +0.468、いずれも CI が 0 を跨がない）
2. **R1 の v3 > 教師**（AP +0.246。ただし AUROC では非有意）

**それ以外は全て CI が 0 を跨ぐ。** 特に「v3 が教師を平均 +0.034 上回った」は**有意ではない**
（R2 はむしろ負側）。平均 0.774 → 0.902 の改善は R3 が単独で牽引しており、他 3 領域は
「悪化していない」までしか言えない。

**現時点で成功と主張できるのは R3 のみ。**

### 椎体単位（whole）は動いていない

| モデル | AUROC | AP |
|---|---:|---:|
| baseline0（初期化元） | 0.8988 | 0.7276 |
| v2 | 0.9035 | 0.7383 |
| v3 | 0.9002 | 0.7328 |

3 者とも誤差の範囲。region branch は whole 性能を壊しても上げてもいない（設計どおり）。

## 4. CAM ゲート再監査（cam_share 基準）

### 4.1 8/23 監査は別の量を測っていた

`.tmp/cam_gate_audit.py` を確認したところ、8/23 監査のスコアは
**`{region}_density_enrichment`**。一方いま損失に入っているのは `cam_share`
（→ 校正して `pseudo_target`）で、`calibration.py:enrichment_to_share` が bag ごとに
4 領域の合計で割るため、両者は bag をまたぐ順位が一致しない。

**9/1 の疑似ラベル再設計後、実際に使っている `cam_share` でのゲート監査は一度も
走っていなかった。**

### 4.2 再監査結果（GPU 不要）

`pseudo_region_targets.csv` の inner-val ブロック（`teacher_outer_fold ==
(student_outer_fold + 1) % 5`）が、5 fold 分で 1,332 bag を「その fold を学習に
使っていない教師」で**ちょうど 1 回ずつ**覆っている。これで完全 OOF の教師スコアが
既存ファイルだけで得られる。

| 領域 | 陽性 | 陰性 | AUROC | 患者クラスタ 95%CI | AP |
|---|---:|---:|---:|---|---:|
| R1 椎体 | 78 | 167 | 0.7666 | [0.699, 0.830] | 0.6042 |
| R2 右横突孔 | 59 | 184 | 0.8326 | [0.778, 0.883] | 0.6450 |
| R3 左横突孔 | 72 | 172 | 0.8156 | [0.738, 0.880] | 0.7281 |
| R4 後方要素 | 158 | 93 | 0.8246 | [0.770, 0.873] | 0.8895 |

| ゲート | 判定 |
|---|---|
| R2/R3 AUROC >= 0.70 | **PASS** |
| R2/R3 CI 下限 > 0.50 | **PASS** |
| laterality 勝率 > 0.55 | **PASS**（0.8415、CI [0.753, 0.921]、82 bag / 64 studies） |

8/23（enrichment 基準）との差は平均 +0.033 だが、**スコア定義の変更と 4-view TTA 導入が
同時に起きている**ため、どちらの寄与かは分離できない（enrichment 値は最終 CSV に
保存されていない）。

### 4.3 注意: fold 0 は教師にとって当たり fold

| | R1 | R2 | R3 | R4 | 平均 |
|---|---:|---:|---:|---:|---:|
| 教師（fold 0 のみ、262 bag） | 0.8796 | 0.8423 | 0.9211 | 0.8295 | 0.8681 |
| 教師（5 fold OOF、1,332 bag） | 0.7666 | 0.8326 | 0.8156 | 0.8246 | 0.8099 |

差 +0.058。3 章の教師比較は、教師が強く出た fold での値である。

## 5. 校正の実装調査（結論: 実装は正しい）

`calibration.py` の凍結変換:

```
s_vr = e_vr / Σ_j e_vj          # view 内で 4 領域をシェア化
s_r  = mean_v(s_vr)             # シェア化してから view 平均
x_r  = logit(clip(s_r, 0.01, 0.99))
q*_r = sigmoid(a_k · x_r + b_k) # fold k ごとの共有 (a_k, b_k)
q_r  = q*_r  if Σ_j q*_j >= 1 else q*_r / Σ_j q*_j
```

順位を壊しうる 2 箇所はどちらも実質不活性:

| 箇所 | 実測 |
|---|---|
| `shares_to_logit_features` の floor clip | **0.08%**（16/21,312 セル）。R2 0.2% / R3 0.1% |
| `project_sum_at_least_one` の行ごと除算 | **`projected_fraction: 0.0` = 一度も発火せず**。行合計の最小 1.0416、平均 1.38 |

`fit_shared_logit_share_calibration` が `slope > 0` を強制しているため、単一 fold 内では
sigmoid(a·logit(s)+b) は s に対して厳密に単調。fold 内に閉じて測ると **全 fold・全領域で
Spearman = 1.000000、AP 差 = 0.000000** で完全一致した。

fold ごとに (a_k, b_k) が異なる（fold 0: slope 1.4759 / fold 3: slope 2.4383）ため、
**5 fold をプールして 1 本のランキングにすると順位が入れ替わる**。学習は fold ごとに
閉じているので実運用上は無害。

## 6. 実装した修正（計測の追加のみ）

`fracture_detection/region_branch/training/trainer.py`、3 箇所。**損失・checkpoint 選択・
early stopping には一切触れていない**ので学習挙動は不変。

1. `_conditional_region_metrics` を新規追加。人手 GT × 椎体陽性 cell だけを対象に
   `{column}_conditional_score`（whole 確率を含まない）で AP/AUROC を集計。出力キーは
   `region_{i}_cond_ap` / `_cond_auroc` / `_cond_n_positive` / `region_cond_macro_ap`。
   退化時は例外を投げず NaN（人手 GT の陽性は 1 fold あたり数十個なので退化は正常に起こる）。
   `_cond_n_positive` を必ず出すのは、「その数字が陽性何個に乗っているか」をログだけで
   判断できるようにするため。
2. 評価関数と `_hard_region_metrics` の両方から呼び出し。`trainer.py:320` の
   `**{f"val_{key}": ...}` により history.csv / training.log / wandb へ自動伝播し、
   `fold_metrics.json` にも入る。
3. `_append_log` に `val_region_cond_macro_ap` を追加。

検証: v3 の `outer_predictions.csv` に当てて手作業の値と完全一致
（cond_ap 0.9103 / 0.6470 / 0.9119 / 0.9314、macro 0.8501）。
`ruff check` / `ruff format --check` 通過、**pytest 245 passed**。

`early_stopping_metric` を条件付き指標に変えるかは別判断。陽性 12〜31 個では分散が
大きすぎるため現時点では非推奨。

## 7. 私 (Claude) の判断ミスの記録

`2026-08-28-region-loss-diagnosis.md` §4 の前例に倣う。今回も同じ型の誤りを繰り返した。

1. **v2 と v3 の `val_region_macro_ap` を同名だからと比較**し「0.35 → 0.454 に改善」と報告
   → 母集団もスコア定義も違い無効。撤回
2. **「教師収束が有害」と結論**（R1/R3 の AUROC 低下を根拠に）→ その AUROC 自体が whole 汚染
   を受けており、完走後の正しい測定では R3 は 0.626 → 0.964 だった。撤回
3. **教師の監査値 0.777 と生徒を比較**して「生徒が教師とほぼ同じ = 蒸留天井」と結論
   → 監査値は別サンプル・別スコア定義。同一セルで測ると教師 0.868。撤回
4. **「校正が順位を保存していない」と報告** → 5 fold をプールした自分の集計が原因。実装は
   正しい。撤回
5. **CI を付けずに「生徒が教師を超えた」と報告** → bootstrap を付けたら 4 領域中 3 領域で
   CI が 0 を跨ぎ、有意なのは R3 の v3>v2 と R1 の v3>教師(AP) のみだった

**教訓（前回と同じ）: 指標を比較する前に、母集団・スコア定義・サンプル範囲が一致して
いるかをコードで確認する。母数が小さい場合は点推定を報告せず CI を付ける。**

## 8. 次にやること

1. **fold 1〜4 の学習**（各約 2 時間、計 8 時間）。983 セル・陽性 372 個の 5 fold OOF に
   すれば、いま検出できない ±0.05〜0.10 の差が検出可能域に入る。現状 R3 以外は何も
   主張できない
   - 事前に `experiment.phase/name` の変更（既存出力の上書きガードに当たる）と
     fold 1〜4 の校正（`cli/calibrate.py`）が必要
   - 実行はユーザーが手動でトリガーする
2. whole head が epoch 5 以降過学習している件の扱い（凍結するか否か）は未判断。最終推論は
   `best_whole`(epoch 5) を使うので成果物は守られているが、学習中の region 診断指標は
   drift の影響を受ける
3. 本ログ 4.1 の「8/23 監査は density_enrichment 基準だった」件を、設計文書側にも反映するか
   未判断
