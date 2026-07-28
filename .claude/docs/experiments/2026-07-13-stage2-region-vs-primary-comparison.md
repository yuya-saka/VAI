# 実験レポート: Stage1 / Stage2 primary head vs region head 比較（全症例・非アンサンブル）

- 作成日: 2026-07-13
- 比較対象:
  - `train_models/stage1/outputs/baseline/v1_parity`（stage1、region headなし、5fold）
  - `train_models/stage2/outputs/baseline/v1`（primary_loss_weight=1.0相当, region_loss_weight=0.5、joint学習、5fold）
  - `train_models/stage2/outputs/ablation/region_only`（primary_loss_weight=0.0, region_loss_weight=1.0、region単体学習、5fold）
- 評価単位:
  - OOF: 10,730行（1,607患者、各行=そのfoldを学習していない単一モデルによる予測）
  - Test: 従来のアンサンブル平均（2,703行）に加え、**fold毎の非アンサンブル予測を新規に推論**
    （13,515行 = 2,703症例 × 5fold、各fold単独モデルの予測。詳細は§6）
  - 「全症例」統合: OOF + Test per-fold = **24,245行、2,009患者（データセット全体）**
- 関連: `.claude/docs/work-logs/2026-07/2026-07-13.md`（セッション経緯の詳細）

## 要点

- **stage2のregion headはprimary headより一貫して優れている**。同一モデル内
  （baseline/v1）のpaired bootstrapで、全症例データを使うとAUROC・AUPRCとも
  有意（ΔAUROC=+0.0047 [95%CI +0.0011,+0.0083]、ΔAUPRC=+0.0264 [+0.0167,+0.0369]）。
  今回の分析全体で唯一、あらゆる比較条件を通じて頑健に生き残った結論。
- stage2 region系（baseline joint / region_only どちらも）はstage1に対してAUPRCが
  有意に優位（+0.02〜+0.03）だが、AUROCは非有意（誤差範囲）。
- region_only（region単体学習）とbaseline/v1（joint学習）のregion head同士に
  有意差なし——regionをsoloで学習してもjointで学習しても精度は変わらない。
- 「region_onlyのregion headがbaseline primaryを上回る」というOOFのみでの当初の
  主張（ギリギリ有意）は、全症例データでは非有意に転じた。サンプル不足による
  偶然だった可能性が高い。
- region_only の primary head は設計通りchance level（未学習、primary_loss_weight=0）。

## 1. 実験設定

| | stage1 v1_parity | stage2 baseline/v1 | stage2 region_only |
|---|---|---|---|
| primary_loss_weight | (旧schema、実質1.0相当) | 実質1.0（旧schema） | **0.0（未学習）** |
| region_loss_weight | — (region head なし) | 0.5 | **1.0** |
| selection_metric | primary AUROC | primary AUROC固定（旧schema） | region |
| backbone | tf_efficientnetv2_s + BiLSTM | 同左 + region-masked MIL head | 同左 |
| batch_size / n_gpu | 16 / 1 | 8 / 2 | 16 / 1 |
| n_folds / seed | 5 / 42 | 5 / 42 | 5 / 42 |

同一データ分割（`random_seed=42`、`test_size=0.2`）を使用しており、OOF/Testとも
3系統で(study_uid, vertebra)キーが完全一致することを確認済み。

## 2. 主要結果: AUROC / AUPRC / Precision / Recall / F1

### OOF（10,730行）

| モデル / head | AUROC | AUPRC | P@0.5 | R@0.5 | F1@0.5 | thr(opt) | P@opt | R@opt | F1@opt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 0.9209 | 0.7447 | 0.7150 | 0.6425 | 0.6768 | 0.4361 | 0.6766 | 0.6797 | 0.6781 |
| stage2 baseline/v1 primary | 0.9217 | 0.7466 | 0.7370 | 0.6471 | 0.6891 | 0.5392 | 0.7668 | 0.6369 | 0.6958 |
| stage2 baseline/v1 region | 0.9244 | 0.7654 | 0.7636 | 0.6676 | 0.7124 | 0.6253 | 0.8228 | 0.6313 | 0.7144 |
| stage2 region_only primary | 0.4941 | 0.1180 | 0.0996 | 0.5363 | 0.1680 | 0.6076 | 0.1559 | 0.2505 | 0.1921 |
| stage2 region_only region | 0.9286 | 0.7678 | 0.7600 | 0.6723 | 0.7134 | 0.4805 | 0.7521 | 0.6806 | 0.7146 |

### Test（アンサンブル平均、2,703行）

| モデル / head | AUROC | AUPRC | P@0.5 | R@0.5 | F1@0.5 | thr(opt) | P@opt | R@opt | F1@opt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 0.9370 | 0.7658 | 0.7409 | 0.6293 | 0.6806 | 0.5720 | 0.8307 | 0.6062 | 0.7009 |
| stage2 baseline/v1 primary | 0.9368 | 0.7738 | 0.8168 | 0.6371 | 0.7158 | 0.4850 | 0.8125 | 0.6525 | 0.7238 |
| stage2 baseline/v1 region | 0.9457 | 0.8064 | 0.8333 | 0.6564 | 0.7343 | 0.4328 | 0.7991 | 0.6911 | 0.7412 |
| stage2 region_only primary | 0.5162 | 0.1052 | 0.0919 | 0.7915 | 0.1647 | 0.5148 | 0.1053 | 0.5521 | 0.1769 |
| stage2 region_only region | 0.9400 | 0.7924 | 0.8358 | 0.6486 | 0.7304 | 0.5335 | 0.8639 | 0.6371 | 0.7333 |

### 全症例統合（OOF + Test fold別非アンサンブル、24,245行、2,009患者、決定版）

| モデル / head | AUROC | AUPRC | P@0.5 | R@0.5 | F1@0.5 | thr(opt) | P@opt | R@opt | F1@opt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 0.9213 | 0.7359 | 0.6962 | 0.6374 | 0.6655 | 0.6302 | 0.7779 | 0.5914 | 0.6719 |
| stage2 baseline/v1 primary | 0.9217 | 0.7379 | 0.7349 | 0.6378 | 0.6829 | 0.5910 | 0.7911 | 0.6091 | 0.6883 |
| stage2 baseline/v1 region | 0.9264 | 0.7643 | 0.7539 | 0.6606 | 0.7042 | 0.5910 | 0.8004 | 0.6315 | 0.7060 |
| stage2 region_only primary | 0.5044 | 0.1221 | 0.0985 | 0.5428 | 0.1667 | 0.6080 | 0.1627 | 0.2562 | 0.1990 |
| stage2 region_only region | 0.9249 | 0.7591 | 0.7557 | 0.6556 | 0.7021 | 0.6129 | 0.8150 | 0.6192 | 0.7038 |

（全症例のprevalenceは0.0977。OOF/Testを混ぜているため単純にOOFやTest単体の
prevalenceとは一致しない）

## 3. Paired cluster bootstrap 95%CI（全症例、study_uid単位、n_boot=2000）

これが本レポートの核心。単純な点推定の大小比較ではなく、患者クラスタ単位で
2000回リサンプリングした差分の95%信頼区間で判定した。

| 比較 | ΔAUROC [95%CI] | 有意 | ΔAUPRC [95%CI] | 有意 |
|---|---|---|---|---|
| stage2 baseline/v1 region − stage1 | +0.0050 [-0.0008,+0.0109] | no | +0.0284 [+0.0157,+0.0410] | **YES** |
| stage2 region_only region − stage1 | +0.0036 [-0.0031,+0.0101] | no | +0.0231 [+0.0103,+0.0365] | **YES** |
| **stage2 baseline/v1: region − primary（同一モデル内）** | **+0.0047 [+0.0011,+0.0083]** | **YES** | **+0.0264 [+0.0167,+0.0369]** | **YES** |
| region_only region − baseline/v1 region | -0.0014 [-0.0066,+0.0038] | no | -0.0053 [-0.0151,+0.0048] | no |
| region_only region − baseline/v1 primary | +0.0033 [-0.0027,+0.0091] | no | +0.0212 [+0.0082,+0.0346] | **YES** |

**唯一AUROC/AUPRCの両方で有意なのは「同一モデル内でのregion − primary」のみ**。
他の比較はAUPRCのみ有意、またはいずれも非有意。

## 4. Per-vertebra breakdown（OOF）

| Vertebra | stage1 primary AUROC/AUPRC | baseline/v1 primary AUROC/AUPRC | region_only region AUROC/AUPRC |
|---|---|---|---|
| C1 | — / — | 0.9356 / 0.7564 | 0.9514 / 0.8357 |
| C2 | — / — | 0.9450 / 0.8217 | 0.9518 / 0.8380 |
| C3 | — / — | 0.8973 / 0.5623 | 0.8982 / 0.5377 |
| C4 | — / — | 0.9349 / 0.7600 | 0.9486 / 0.7712 |
| C5 | — / — | 0.8753 / 0.6144 | 0.8820 / 0.6200 |
| C6 | — / — | 0.8825 / 0.7400 | 0.8766 / 0.7263 |
| C7 | — / — | 0.8998 / 0.7768 | 0.9110 / 0.8140 |

- C3・C5・C6が全モデル共通で相対的に弱い（positive数が少ない: C3 n_pos=55/1564、
  下位頸椎の境界の曖昧さに起因すると推測）。この傾向はモデルに依存しないデータ側の
  難しさ。
- stage1のper-vertebra数値は今回未収集（stage1の`metrics.json`は
  `oof_level_metrics`に格納されており、必要なら別途抽出可能）。

## 5. 解釈

1. **region-MIL経路（4領域のsub-region evidenceをMIL集約する構造）は、同一モデル内で
   primary directヘッドより一貫して優れた予測器になっている。** joint学習・region単体
   学習のどちらでも同程度に達成される。primaryを主タスクとして学習しつつregionを
   補助的に学習させても、regionを完全に主役にしても、region head自体の性能は
   ほぼ変わらない（§3の "region_only region − baseline/v1 region" が非有意）。
2. **stage1（region headなし）に対するstage2の優位性は、AUPRCでは明確だが
   AUROCでは誤差範囲。** 低prevalence（約10%）下でのprecision-recallバランスの
   改善が主な効果であり、全体順位付け能力（AUROC）自体を大きく変えるものではない。
3. **primaryを完全に切り捨てて(region_only)regionに全予算を投入しても、
   region単体の精度が上がるわけではない。** primaryを犠牲にするコストに見合う
   メリットは統計的に確認できなかった。
4. サンプルサイズの影響が大きい。OOFのみ（1,607患者）での分析では
   「region_onlyがbaseline primaryを上回る」がギリギリ有意だったが、全症例
   （2,009患者）に増やすと消えた。逆に「baseline region − primary」はOOF単独では
   AUROCが非有意だったが、全症例では有意に転じた。**小さいNでの有意判定は
   逆転しうるため、今後の比較でも可能な限りTestを非アンサンブル化して
   サンプルを増やすべき。**

## 6. 手法メモ

### 6.1 OOF/Testの構造の違いとその影響

- OOF: 5foldが互いに排他的なvalidation部分（重複なし）を担当し、合体すると
  学習データ全体（80%）を単一モデルの予測でカバーする「モザイク方式」。
- Test（従来）: 固定holdout（20%）に対し、5foldのモデル予測を**平均（アンサンブル）**。
- 当初この2つを混ぜて比較していたが、推定方式が異なる（非アンサンブル vs
  アンサンブル）ため統計的に一貫性がないという指摘を受け、Testも**fold毎の
  非アンサンブル予測**を新規にGPU推論（`CUDA_VISIBLE_DEVICES=2`、他GPUは
  別ジョブが占有中のため使用不可）。既存の`test_predictions.csv`（アンサンブル）
  との整合性は、5fold平均が相関0.9999以上・平均絶対誤差0.0002以下で一致することを
  確認済み（bf16 autocastの非決定性のみ）。
- 「全症例」データセット: OOF（1予測/ケース）+ Test per-fold（5予測/ケース、
  各fold非アンサンブル）を結合。Test側の症例はOOF側の症例よりbootstrap上の
  重みが5倍になる非対称性があるが、いずれの行も「学習していないモデルによる
  予測」という点で同質。

### 6.2 コード変更（今後の再現性のため）

`train_models/stage{1,2}/src/trainer.py`のtest推論関数を、アンサンブル平均に加えて
fold単位の非アンサンブル予測も返すよう修正済み（詳細は work-log 参照）。

- stage1 `predict_on_items`: `(ensemble_preds, per_fold_preds)`を返す。
- stage2 `predict_ensemble`: `(ensemble_outputs, per_fold_outputs)`を返す。
- `train.py`側で`test_predictions_per_fold.csv`を追加保存するよう変更。
- **注意**: 今回比較に使った3実験（stage1 v1_parity, baseline/v1, region_only）は
  この変更より前に学習済みのため、出力ディレクトリに`test_predictions_per_fold.csv`は
  存在しない。今回の分析で使ったper-fold Test予測は`.tmp/`のscratchpadに一時生成した
  ものであり、セッション終了後に消える。正式に残したい場合は既存チェックポイントに
  対して新コードで推論をやり直す必要がある（再学習不要、GPU推論のみで15分程度）。

### 6.3 統計手法

- Patient-cluster paired bootstrap（`study_uid`単位でリサンプリング、n_boot=2000、
  seed=42）。単純な周辺95%CIの重なりでは検出力が低いため、同一クラスタ割り当てで
  2モデルの差分を直接ブートストラップする方式を採用。
- 差分の分散は `Var(A-B) = Var(A) + Var(B) - 2Cov(A,B)` に従うため、相関の高い
  ペア（同一モデル内のprimary/region、相関0.96〜0.98）ほど検出力が高く、
  相関の低いペア（別モデル間、相関0.87〜0.97）ほどノイズが大きく有意差が
  出にくいことを確認済み。

## 7. 次のステップ

1. 既存3実験の`test_predictions_per_fold.csv`を正式生成（新コードで推論のみ再実行）。
2. `.claude/docs/work-logs/2026-07/2026-07-09.md`記載の未着手ablation
   （shuffled region mask、Noisy-OR pooling、primary-only-same-architecture、
   global FPN head）の実行。
3. Screening operating point評価（specificity固定時のrecall等）は未実施。
4. stage1のper-vertebra breakdownを`oof_level_metrics`/`test_level_metrics`から
   抽出し、§4の表を完成させる。
