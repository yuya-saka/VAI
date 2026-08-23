# Baseline 0 CAM-onlyとStage3の領域局在比較 worklog

**日付**: 2026-08-21  
**状態**: 注釈coverage訂正、Baseline 0 CAM-only再集計、Stage3領域評価、同一症例比較まで完了。

## 目的

1. Baseline 0が高い骨折スコアを出すとき、椎体内のどの解剖領域へ注目しているか確認する。
2. 4領域アノテーションを使って、CAM局在のAUROC/PRAUCを探索的に評価する。
3. `train_models/stage3`が出力する4領域contextual evidenceとCAM-onlyを比較する。

Baseline 0は領域教師も4領域mask入力も使わない。CAM-onlyスコアは正式OOF checkpointの
`encoder.bn2`に対するGrad-CAMを4領域mask内で積算し、領域面積で補正したCAM密度である。
Stage3スコアは4領域mask pooling後のcontextual region evidence logitであり、領域教師で
直接学習された確率ではない。

---

## 1. 4領域ラベルcoverageの訂正

### 発覚した問題

`fracture_region_labels_dicom.csv`の1行は、1椎体全体ではなく連続する骨折bboxの1 `run`を
判定した結果である。複数runを持つ椎体で一部runが未注釈の場合、記録済みrunの論理和が
`0`でも「その領域に骨折なし」とは言えない。

アノテーションツールの全targetとラベルCSVを照合した結果:

- 少なくとも1 runが注釈済み: **268 bag / 160 study**
- 268 bag内のannotatable run: **321本**
- 注釈済みrun: **285本**
- 未注釈run: **36本、33 bag**
- 全run注釈完了: **235 bag**
- 注釈済みbag内の`bbox_missing` target: **0**

### 確定した有効性規則

- 記録済みラベルが`1`: 未完了bagでも**既知陽性**として使用可能
- 記録済みrunの論理和が`0`かつ全run完了: **既知陰性**
- 記録済みrunの論理和が`0`かつ未注釈runあり: **unknown**

したがって領域rの評価対象は、`region_r == 1 OR annotation_complete`とする。
以前の「268 bagすべてで0を陰性」としたCAM AUROC/PRAUCは無効であり、以下の値で置き換える。

なお、ここでの陰性は**骨折陽性椎体の中で、その領域に骨折がない例**である。
椎体骨折陰性症例は領域局在評価に含めていない。

---

## 2. Baseline 0 CAM-onlyの訂正後結果

対象は正式run
`fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/`の
4領域注釈付き268 bag。スコアは領域面積補正CAM密度。

| 領域 | 陽性 | 陰性 | unknown | AUROC | PRAUC | level内rank AUROC | level内rank PRAUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 椎体 | 78 | 167 | 23 | **0.798** | **0.645** | 0.746 | 0.598 |
| R2 右横突孔 | 59 | 184 | 25 | **0.786** | **0.518** | 0.767 | 0.487 |
| R3 左横突孔 | 72 | 172 | 24 | **0.788** | **0.642** | 0.745 | 0.572 |
| R4 後方要素 | 158 | 93 | 17 | **0.736** | **0.814** | 0.673 | 0.773 |

CAM-onlyでも全領域でAUROC 0.74〜0.80であり、骨折部位の局在をある程度反映している。
ただし7×7最終特徴からの事後説明であり、画素単位の骨折検出器または因果的根拠ではない。

---

## 3. Stage3の領域evidence評価

対象はlegacy Stage3正式成果物
`train_models/stage3/outputs/baseline_3/v1/`。`oof_evidence.npz`の`region`を
領域evidence logitとして使用した。Stage3の固定20% testをOOFへ混ぜず、partition別に評価した。

### OOF: 216注釈bag / 129 study

| 領域 | 陽性 | 陰性 | unknown | AUROC | PRAUC |
|---|---:|---:|---:|---:|---:|
| R1 | 70 | 133 | 13 | 0.758 | 0.747 |
| R2 | 44 | 156 | 16 | 0.755 | 0.590 |
| R3 | 63 | 136 | 17 | 0.662 | 0.506 |
| R4 | 126 | 80 | 10 | 0.772 | 0.833 |

### 固定test: 52注釈bag / 31 study

5 foldそれぞれのregion evidence logitをbagごとに算術平均したensemble。

| 領域 | 陽性 | 陰性 | unknown | AUROC | PRAUC |
|---|---:|---:|---:|---:|---:|
| R1 | 8 | 34 | 10 | 0.783 | 0.708 |
| R2 | 15 | 28 | 9 | 0.576 | 0.629 |
| R3 | 9 | 36 | 7 | 0.583 | 0.481 |
| R4 | 32 | 13 | 7 | 0.769 | 0.867 |

固定testではR2/R3 AUROCが0.58前後まで低下した。特にR1陽性8件、R3陽性9件と少ないため、
testの領域別推定値は不確実性が大きい。

---

## 4. Stage3対CAM-onlyの同一症例比較

Stage3の各partitionと同じ`study_id × level × region`へCAM-onlyを限定した。
ラベル有効性も両者で完全に同じである。数値は`AUROC / PRAUC`。

### OOF 216 bag

| 領域 | Stage3 | CAM-only | 記述的な傾向 |
|---|---:|---:|---|
| R1 | 0.758 / **0.747** | **0.785** / 0.651 | AUROCはCAM、PRAUCはStage3 |
| R2 | 0.755 / **0.590** | **0.783** / 0.495 | AUROCはCAM、PRAUCはStage3 |
| R3 | 0.662 / 0.506 | **0.787 / 0.669** | CAMが両方で優位 |
| R4 | **0.772 / 0.833** | 0.722 / 0.790 | Stage3が両方で高い |
| Macro | 0.736 / **0.669** | **0.769** / 0.651 | AUROCはCAM、PRAUCはほぼ同等 |

### 固定test 52 bag

| 領域 | Stage3 | CAM-only | 記述的な傾向 |
|---|---:|---:|---|
| R1 | 0.783 / **0.708** | **0.842** / 0.660 | AUROCはCAM、PRAUCはStage3 |
| R2 | 0.576 / 0.629 | **0.826 / 0.742** | CAMが両方で高い |
| R3 | 0.583 / 0.481 | **0.799 / 0.616** | CAMが両方で高い |
| R4 | 0.769 / 0.868 | **0.796 / 0.911** | CAMが両方で高い |
| Macro | 0.678 / 0.671 | **0.816 / 0.733** | CAMが両方で高い |

### Paired bootstrap

差を`Stage3 − CAM-only`とし、study単位のpaired percentile bootstrap 2,000回で95% CIを計算した。

- **OOF R3 AUROC差**: `-0.125`、95% CI `[-0.227, -0.016]`
- **OOF R3 PRAUC差**: `-0.163`、95% CI `[-0.289, -0.012]`
- その他の領域・partitionの差: 95% CIがすべて0を跨いだ

### 結論

- CAM-onlyは4領域全体でStage3より安定している。
- **R3はCAM-onlyがStage3より明確に良い**。
- Stage3はOOF R1/R2のPRAUCとR4で高いが、paired CIでは優位性を確認できない。
- 固定testではCAM-onlyが全領域のAUROC、R2/R3/R4のPRAUCで高い。
- よって、**Stage3がCAM-onlyより領域局在を改善した証拠はない**。むしろCAM-only、とくに
  R3の方が良好という結果である。

ただしこれは同一症例上の**スコア比較**であり、因果的なarm比較ではない。Stage3とBaseline 0は
学習データ分割、checkpoint選択、入力、領域mask使用法、スコア定義が異なる。

---

## 5. 保存した成果物

### Baseline 0

- `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/gradcam_annotated/attention_metrics.csv`
- `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/gradcam_annotated/annotation_coverage.csv`
- `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/gradcam_annotated/annotated_localization_metrics.csv`
- `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/gradcam_attention/`

### Stage3

- `train_models/stage3/outputs/baseline_3/v1/region_evidence_localization_metrics.csv`
- `train_models/stage3/outputs/baseline_3/v1/region_evidence_localization_samples.csv`
- `train_models/stage3/outputs/baseline_3/v1/region_evidence_localization_metadata.json`
- `train_models/stage3/outputs/baseline_3/v1/region_evidence_vs_baseline0_cam_metrics.csv`
- `train_models/stage3/outputs/baseline_3/v1/region_evidence_vs_baseline0_cam_samples.csv`
- `train_models/stage3/outputs/baseline_3/v1/region_evidence_vs_baseline0_cam_metadata.json`

---

## 6. 実装・検証

- `fracture_detection/baseline0/cli/attention.py`
  - アノテーションツールのtarget inventoryからrun完了状況を自動導出
  - `陽性 OR annotation_complete`のper-region有効性maskを追加
  - `n_unknown`と`annotation_coverage.csv`を出力
- `fracture_detection/folds/load_labels.py`
  - run論理和は「全run」ではなく「記録済みrun」の論理和であることを明記
- `fracture_detection/baseline0/tests/test_attention.py`
  - 未完了bagの`0`を除外し、`1`を保持する回帰テストを追加
- Baseline 0テスト: **51 passed**
- Ruff format/check: pass
- mypy: pass
- `git diff --check`: pass

---

## 7. 後続作業への影響

Baseline 0学習は4領域教師を使わないため、今回のlabel missingnessによる学習影響はない。
一方、region-supervised armの現行datasetは`has_region_target=True`なら4領域すべてをvalidにするため、
未完了33 bagの`0`を陰性教師にしてしまう。**正式な領域教師あり学習では、同じper-region validity
maskを導入するか、残り36 runの注釈を完了する必要がある。**

---

## 8. Baseline 0事前学習からの領域fine-tuning案（未実施）

CAM-onlyが領域ラベルなしでも局在信号を持つため、Baseline 0を初期値として4領域教師へ
fine-tuningすることは技術的に可能である。ただし268 bagを使ったfull-model fine-tuningを
そのまま行うのは過学習とwhole-task破壊の危険が大きい。

最小かつ安全なpilotは以下。

1. 6chの`Control-B`型モデルを使い、各outer foldで対応するBaseline 0 checkpointから
   `encoder`、`lstm`、`head -> whole_head`を転送する。4出力`region_head`だけ新規初期化する。
2. まずencoder・BiLSTM・whole headを凍結し、region headだけを学習する。
3. region lossは`region_r == 1 OR annotation_complete`の有効性maskを使う。未完了bagの0を
   陰性教師にしない。
4. outer foldの注釈は学習・checkpoint選択に使わない。Baseline 0も同じouter foldを見ていない
   fold対応checkpointを使い、3 training foldsでregion head学習、inner foldで選択、outer foldで
   1回だけ評価する。
5. head-onlyで不足する場合だけ、最終CNN blockとBiLSTMを低LRで解凍する。この段階では
   268注釈bagだけに限定せず、full natural streamのwhole lossを併用して既存の骨折識別能力を保つ。

直接Grad-CAMをlossにすると高階微分が必要になり不安定・高コストなので、最初のpilotでは行わない。
CAMは事前表現に局在信号がある根拠とbaseline評価に使い、学習教師は4領域BCEとする。

このpilotは現行の凍結済み正式6-arm試験を置き換えず、独立した探索的ablationとして扱う。

---

## 9. `mtl_type2` outer0の精度低下診断

`baseline1_type2`はouter0だけが完了しており、`control_type2`とouter1-4は未実施である。
したがって以下は一foldの探索的診断であり、方式全体の確定的な結論ではない。

### 同一outer0での性能

| model | whole AUROC | whole PRAUC |
|---|---:|---:|
| Baseline 0 | 0.8988 | 0.7276 |
| Baseline 1-B | 0.8828 | 0.6869 |
| `baseline1_type2` | 0.8543 | 0.6090 |

注釈未完了部位をunknownとして除外した領域別成績は以下。

| region | Type2 AUROC / PRAUC | Baseline 1-B | CAM-only |
|---|---:|---:|---:|
| R1 | 0.822 / 0.777 | 0.895 / 0.835 | 0.889 / 0.745 |
| R2 | 0.618 / 0.293 | 0.775 / 0.623 | 0.788 / 0.590 |
| R3 | 0.317 / 0.230 | 0.592 / 0.355 | 0.925 / 0.867 |
| R4 | 0.636 / 0.767 | 0.885 / 0.898 | 0.731 / 0.754 |

Type2ではR1-R3 scoreの相関が0.973-0.983で、56例中55例でR4が4領域中の最大scoreに
なった。R3は陽性平均0.222、陰性平均0.272と向きも逆転しており、4領域の解剖学的識別が
collapseしている。

### 主因

1. Type2は共有CNNの後にwhole/region BiLSTMを分離したが、`region_lstm`はBaseline 0から
   転送されずrandom initializationである。`pretrained: true`はtimm/ImageNet初期化だけで、
   Baseline 0 checkpointはloadしていない。
2. region専用parameterは約486万で、共有LSTM型Baseline 1-Bのregion専用head約13.3万の
   約36.6倍である。それをouter0では159 annotated bag、40 optimizer step/epochだけで学習する。
   whole側は約505 step/epochだが、分離された`region_lstm`にはwhole lossの教師が届かない。
3. 4 logitsはいずれも同じglobal pooled CNN sequenceから作られ、領域別spatial poolingがない。
   少数教師では解剖学的位置よりbag-level shortcutを共有しやすく、実際の高相関とR4 biasが
   このcollapseを支持する。
4. `region_lambda`を校正値約0.476から1.0へ変え、branch分離・batch size・更新頻度も同時に
   変更したため効果が交絡している。さらにgradient計測stepとregion更新stepが一度も一致せず、
   region gradient列は全epochでNaNなので、lambda=1の勾配balanceは監査できていない。
5. 現行datasetは`has_region_target=True`なら4 targetすべてをvalidにする。outer0 trainの
   未完了16 bagとinner validationの未完了10 bagでは、未注釈の0が陰性として学習・選択に混入した。
6. region validation PRAUCはepoch41で最大だったが、outer predictionはwhole AUROC-bestの
   epoch61を使用した。epoch41から61でtrain region macro PRAUCは0.506から0.655へ上昇する一方、
   validationは0.527から0.500へ低下しており、region branchのmemorizationも認める。

### 結論と次の扱い

最大の原因は、**少数の領域教師しかないのに大きなregion BiLSTMを分離・random初期化し、
whole-taskから得られていた転移学習を切ったこと**である。ラベルmissingness、未校正lambda、
checkpoint選択は悪化要因だが、観測された4領域collapseの主説明はbranch設計と更新量の不釣り合いである。

`mtl_type2` v1を現状のままouter1-4へ展開しない。次の試験は、per-region validity maskを直し、
gradient監査をregion step上で行ったうえで、fold対応Baseline 0からregion sequence modelも転送する
head-first fine-tuning、または共有BiLSTMを維持する構成を、他条件を一度に変えず独立ablationとして行う。
なお実行中にtrainer/diagnostics sourceが更新され、保存historyのregion loss集計は現在のsourceと
一致しないため、このrunは正式な再現可能結果ではなく探索結果としてのみ扱う。
