# 4領域骨折検出モデル 全体設計

作成日: 2026-08-25  
状態: **設計確定・実装未着手**

## 1. この文書の位置づけ

本書は、Baseline 0を基礎とする4領域骨折検出モデルについて、目的、architecture、
教師信号、損失、sampling、評価、監視、実装順序を一つにまとめた日本語の全体設計書である。
実装時は本書を主な入口とし、数式の導出、代替案の比較、参考文献の詳細は関連資料を参照する。

- 損失設計の確定記録:
  `.claude/docs/work-logs/2026-08/2026-08-25-region-loss-final-decisions.md`
- 文献調査と比較:
  `.claude/docs/research/20260825-region-loss-balancing.md`
- 設計過程と実測結果:
  `.claude/docs/work-logs/2026-08/2026-08-25-region-branch-design.md`

本書は設計の正本であり、実装開始そのものの承認を意味しない。

---

## 2. 目的

椎体単位の骨折有無を予測するBaseline 0の性能と学習条件を保ちながら、同じCTから
4領域それぞれの骨折scoreを出力する。

主な検証対象は次の2点である。

1. 4領域を一つのmodelで共有学習する構成が、単一領域model 4本より有効か。
2. 少数の人手4領域ラベルに、whole-negative由来の論理0とCAM pseudo rankingを加えることで、
   領域識別を安定して学習できるか。

### 対象外

- region scoreを校正済みの無条件骨折確率として解釈すること
- attention、Transformer、region専用CNNなどを初期modelへ追加すること
- human-only同一architecture armを追加すること
- 268件の人手ラベルを見ながらloss係数を繰り返し調整すること

---

## 3. 基準系とデータ

### 3.1 Baseline 0

Baseline 0をwhole-vertebra分類の基準系、CAM pseudo教師のteacher、実装上の参照系として
維持する。

```text
15 planes × (5-channel 2.5D CT + whole-vertebra mask)
    -> EfficientNetV2-S
    -> whole BiLSTM
    -> plane-level whole logits
    -> mean sigmoid
    -> vertebra-level fracture score
```

### 3.2 データ規模

quality filter後の全データは13,432 bagsである。

| 教師source | Bag数 | 4領域への教師 |
|---|---:|---|
| Human-annotated | 268 | 有効セルの人手0/1ラベル |
| Whole-negative | 12,100 | 論理的に確定する`[0, 0, 0, 0]` |
| Pseudo-positive | 1,064 | fold-matched CAM pairwise ranking |

人手ラベルは1,072セル中983セルが有効である。無効セルはlossから除外し、有効な人手セルには
pseudo教師を適用しない。常にhard GTを優先する。

### 3.3 分割

- patient-grouped nested five-foldを維持する。
- pseudo scoreは各outer foldに対応するfold-matched teacherから取得する。
- held-out patientの情報をtraining、sampling、係数校正へ混入させない。

---

## 4. Model architecture

### 4.1 全体構成

whole用とregion用にBiLSTMを1本ずつ置く。共有するのはEfficientNetV2-SのCNN trunkだけで、
Baseline 0のwhole pathは変更しない。

```text
15 planes × 6 channels
    -> Shared EfficientNetV2-S trunk
         ├─ Baseline 0 whole path
         │    -> conv_head / bn2
         │    -> whole BiLSTM
         │    -> whole head
         │    -> whole-vertebra score
         │
         └─ region path
              -> stage 1-4 feature maps
              -> FPN fusion at stride 4
              -> four region masks
              -> mask-normalized pooling
              -> shared region BiLSTM
              -> four region heads
              -> z_R1, z_R2, z_R3, z_R4
```

### 4.2 Whole path

- Baseline 0と同じ入力、BiLSTM、head、bag probability aggregationを維持する。
- whole pathの比較条件を変えないため、region BiLSTMとは統合しない。
- whole lossは全bagを自然分布で供給する既存streamから計算する。

### 4.3 Region path

- EfficientNetV2-Sのstage 1-4をFPNでstride 4へ融合する。
- 4領域maskごとにmask-normalized poolingを行う。
- 4領域は同じregion BiLSTMを共有する。
- 最終headは領域別に4つ置く。
- hard GTとpseudo rankingは同じ4領域logitを学習する。

### 4.4 勾配の流れ

| Parameter群 | `L_whole` | Region loss |
|---|:---:|:---:|
| Shared CNN trunk | ✓ | ✓ |
| Whole conv head / BiLSTM / head | ✓ | — |
| FPN / region BiLSTM / region heads | — | ✓ |

したがって`lambda`は主にshared trunkでwhole表現とregion表現の相対的な影響を制御する。

### 4.5 初期化とfine-tuning

student outer fold `k`はfold-matched Baseline 0の`outer{k}/best_model.pt`から開始する。
`encoder`、whole BiLSTM、whole headを移し、FPN、region BiLSTM、region headsだけを
seed固定でランダム初期化する。checkpointのouter/inner/train fold対応とroleを読込時に検証する。

全parameterを学習対象とし、freeze/warmupは設けない。Baseline 0由来parameterの初期LRは
`2.3e-5`、新規region pathは`2.3e-4`とし、それぞれcosineで10分の1まで減衰させる。
同じ初期化規約を統合modelと単一領域4 modelへ適用する。

### 4.5 各領域専用single model 4本

統合modelとの比較対象として、R1、R2、R3、R4だけをそれぞれ学習する独立modelを4本作る。
これは統合modelの4 headを別々に評価するだけの構成ではなく、**CNN trunkを含むmodel全体を
領域ごとに独立して学習する構成**である。

single model `M_r`の構成は次とする。

```text
M_r: 15 planes × 6 channels
    -> Independent EfficientNetV2-S trunk
         ├─ Baseline 0 whole path
         │    -> whole BiLSTM
         │    -> whole head
         │    -> whole-vertebra score
         │
         └─ single-region path
              -> stage 1-4 FPN at stride 4
              -> target region mask M_r only
              -> mask-normalized pooling
              -> region BiLSTM
              -> one region head H_r
              -> z_r
```

| 項目 | 統合4領域model | Single model `M_r` |
|---|---|---|
| 学習本数 / fold | 1本 | R1-R4の4本 |
| CNN trunk | 1本を4領域で共有 | 領域ごとに独立 |
| Whole path | 1本 | 各single modelに1本ずつ |
| FPN | 1本を4領域で共有 | 各single modelに1本 |
| Region BiLSTM | 1本を4領域で共有 | target領域専用に1本 |
| Region head | 4 heads | target領域の1 head |
| Region mask | 4枚 | target領域の1枚だけ |
| Region出力 | `z_R1`-`z_R4` | `z_r`だけ |

single modelにもBaseline 0と同じwhole pathを残し、

\[
L^{(r)}=L_{\mathrm{whole}}^{(r)}+\lambda_k
\left(L_{\mathrm{exact}}^{(r)}+\alpha_kL_{\mathrm{rank}}^{(r)}\right)
\]

で学習する。active region集合は`R={r}`であり、対象外3領域のmask、logit、label、lossは
forwardにもloss計算にも入れない。

4本のsingle model間ではparameterを共有しない。各modelのwhole scoreはwhole-path parityを
確認するdiagnosticとして保存するが、4つのwhole scoreをensembleしてprimary endpointにはしない。
single model 4本の`z_R1`-`z_R4`を対応領域ごとに並べたものを、統合modelの4出力との比較対象とする。

---

## 5. 教師信号

### 5.1 Exact教師

`L_exact`には次の2種類を入れる。

1. 人手で付与された4領域0/1ラベル
2. whole-negativeから論理的に導かれる4領域`[0, 0, 0, 0]`

両者は信頼度の異なる教師ではなく、どちらもexact GTとして扱う。ただし件数差が大きいため、
loss reductionではsourceを分けて均衡化する。

### 5.2 Pseudo教師

- fracture-positive bagsに対するfold-matched CAM scoreを使う。
- CAMの絶対値を直接target probabilityにはせず、bag間のpairwise rankingとして使う。
- undefined CAM scoreだけを既存規則に従って除外する。
- CAM magnitudeによる追加confidence weightは使わない。
- validな人手ラベルが存在するセルではpseudo教師を除外する。

---

## 6. 損失関数

### 6.1 全体損失

全体損失は次で固定する。

\[
L=L_{\mathrm{whole}}+\lambda
\left(L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}\right).
\]

- `L_whole`: whole-vertebra分類loss
- `L_exact`: 人手4領域ラベルとwhole-negative論理0のexact loss
- `L_rank`: fracture-positive bags間のCAM pairwise ranking loss
- `alpha`: exact教師に対するpseudo rankingの相対係数
- `lambda`: whole taskに対するregion task全体の相対係数

各lossは領域内で平均した後、active region間で平均する。統合4領域modelだけlossが4倍に
ならないよう、領域方向の単純な総和は使わない。

### 6.2 Whole loss

Baseline 0の契約を維持する。

- BCE with `pos_weight=2.0`
- positive要素のlossを2倍する。
- `weighted_loss.sum() / weight.sum()`で正規化する。
- bag分布、batch size、whole exposureをBaseline 0と揃える。

### 6.3 Exact loss

active region集合を`R`、人手ラベルvalidityを`v_ir`とする。

\[
L_H=\frac{1}{|R|}\sum_{r\in R}
\frac{\sum_{i\in H}v_{ir}\,\mathrm{BCE}(z_{ir},y_{ir})}
     {\sum_{i\in H}v_{ir}},
\]

\[
L_N=\frac{1}{|R|}\sum_{r\in R}
\frac{1}{|N|}\sum_{i\in N}\mathrm{BCE}(z_{ir},0),
\]

\[
L_{\mathrm{exact}}=0.5L_H+0.5L_N.
\]

12,100件のwhole-negativeが268件の人手教師を件数で希釈しないよう、sourceごとに平均して
から1:1で結合する。region側ではplain BCEを使い、`pos_weight`を追加しない。

### 6.4 Ranking loss

既存の`region_balanced_pairwise_ranking_loss`を利用する。exact cell数とpseudo pair数を
共通分母へ入れず、それぞれを独立に平均してから`alpha`で結合する。

`L_exact`はlogitの符号、bias、絶対scaleを固定し、`L_rank`はpositive bag間の相対順序を
与える。この2つを同じheadへ入れることで相互補完する。

---

## 7. 不均衡対策とsampling

### 7.1 Auxiliary region batch

region batch size 16では毎updateに次の3 sourceを含める。

| Source | Bags / batch | 対応loss |
|---|---:|---|
| Human-annotated | 4 | `L_H` |
| Whole-negative | 4 | `L_N` |
| Pseudo-positive endpoints | 8 | `L_rank` |

- 各poolはpersistentなshuffle-without-replacement queueで循環する。
- negative-only update、pseudo-only updateは禁止する。
- 人手bagは一様抽出する。
- rare region positiveを基準にしたbag oversamplingは行わない。
- source-balanced samplingとregion `pos_weight`を併用しない。

この構成における`L_exact`内の概算実効陽性率は、R1 15.9%、R2 12.1%、R3 14.8%、
R4 31.5%である。samplingだけでBaseline 0のweighted whole lossと同程度のpositive寄与を
確保できるため、region側へ`pos_weight=2.0`を追加しない。

### 7.2 採用しない不均衡対策

- 全whole-negativeを自然頻度のまま`L_exact`へ流す。
- corpus全体のpositive / negative比からregion `pos_weight`を作る。
- source-balanced samplingと`pos_weight`を二重に適用する。
- region別positive oversamplingでmulti-labelの共起分布を変える。
- ASLまたはfocal lossをprimary lossにする。

---

## 8. `alpha`と`lambda`の決定

### 8.1 共通方針

- 268件の人手ラベルの性能を使ったgrid searchは行わない。
- 各outer foldのtraining foldsだけを使う。
- model更新前の64 deterministic calibration batchesで一度だけ決める。
- 統合4領域modelをreferenceとして校正する。
- 同じouter foldの統合modelと単一領域4 modelへ同じ係数を適用する。
- model別、region別、性能確認後の再校正は禁止する。

### 8.2 `alpha_k`

region BiLSTM parameters上で次を測る。

\[
g_H=\|\nabla L_{\mathrm{exact}}\|_2,\qquad
g_P=\|\nabla L_{\mathrm{rank}}\|_2
\]

\[
\alpha_k=
\operatorname{clip}_{[0.01,1]}
\left[
0.25\exp\left\{
\operatorname{median}_b
\log\frac{g_{H,b}+\epsilon}{g_{P,b}+\epsilon}
\right\}
\right].
\]

初期pseudo勾配をexact勾配の約1/4にする。

### 8.3 `lambda_k`

`alpha_k`決定後、wholeとregionの両方が通る最後のshared CNN block上で次を測る。

\[
g_W=\|\nabla L_{\mathrm{whole}}\|_2,
\qquad
g_R=\|\nabla(L_{\mathrm{exact}}+\alpha_kL_{\mathrm{rank}})\|_2
\]

\[
\lambda_k=
\operatorname{clip}_{[0.01,10]}
\left[
0.25\exp\left\{
\operatorname{median}_b
\log\frac{g_{W,b}+\epsilon}{g_{R,b}+\epsilon}
\right\}
\right].
\]

初期shared-trunk region勾配をwhole勾配の約1/4にする。非有限gradientが生じた場合は
implementation failureとしてtrainingを開始しない。

one-time calibrationを実装できない場合だけ、暫定値`alpha=0.25`、`lambda=0.25`を使う。
その場合もlabel-blind preflightでraw gradient比が0.5-2倍に収まることを確認する。

dynamic GradNormとlearned uncertainty weightingは採用しない。

---

## 9. 学習・validation・比較

### 9.1 学習

- whole streamはBaseline 0と同じnatural distributionを維持する。
- region streamは4/4/8のsource-balanced batchを使う。
- hard / pseudo streamは固定cadenceで更新する。
- optimizer、scheduler、augmentationなど、変更理由のないBaseline 0条件は維持する。
- optimizerはBaseline 0と同じAdamWを使うが、fine-tuningのため学習済み部分と新規region部分を
  10倍差のlearning rate groupへ分ける。

whole batchとregion batchを一つのoptimizer update内でどうforward / backwardするかは、
実装開始時にmemory使用量とBatchNorm stateを含めて固定し、unit test可能なtraining-step契約として
記録する。損失式、source比、更新cadence自体は変更しない。

### 9.2 Validation

fold-matched pseudo scoreはstudentのheld-out bagsには存在しないため、validationでは
`L_rank`を計算しない。

\[
L_{\mathrm{val}}=L_{\mathrm{whole}}+\lambda L_{\mathrm{exact}}.
\]

trainとvalidationでlossの集約単位を揃える。selection metricの最終的な実装契約は、
既存nested CVのcheckpoint規則と合わせて実装前に固定する。

### 9.3 比較実験

- 統合4領域model 1本
- 対応する1領域だけを独立学習するsingle model 4本
- 5 outer folds

合計は5 models × 5 foldsの25 runsである。比較では次を固定する。

- single modelにもwhole pathを持たせ、統合modelと同じ全体損失式を使う。
- `alpha_k`と`lambda_k`は統合modelで校正したfold別値を4本すべてへ流用する。
- whole natural batch、human bag、whole-negative bagは、同一fold・対応stepで同じbag IDを使う。
- pseudo教師は、統合head `r`とsingle model `M_r`で同じtarget-region pairを使う。
- target regionのvalidity mask、augmentation、update数、checkpoint規則を揃える。
- region別OOF比較は同じheld-out patients上で、統合head `r`対single model `M_r`として行う。
- 統合modelとsingle modelのparameter数、学習時間、推論時間も併記する。

比較目的はregion representation共有の効果を測ることであり、single modelごとの係数再校正、
学習回数追加、region別hyperparameter tuningは行わない。human-only armも追加しない。

---

## 10. 評価とcollapse監視

### 10.1 主評価

- locked OOF logitsを使用する。
- region別AP / AUROCを報告する。
- balanced samplingとrankingによりinterceptが変わるため、region sigmoidは確率ではなく
  scoreとして扱う。

### 10.2 毎epochの記録

- region logit間Spearman 6組
- 各region logitとwhole logitのSpearman
- 標準化4-logit行列の第1主成分説明率
- region別student-teacher rank correlation
- raw / weighted `g_rank / g_exact`
- raw / weighted region / whole shared-trunk gradient比
- source別visit数、unique coverage、queue周回数
- region別valid cell数と実効陽性率

### 10.3 Collapse alarm

次の両方が3回連続した場合をcollapseとする。

- inter-region Spearman中央値 `>= 0.95`
- region-whole Spearman中央値 `>= 0.95`

alarm後に係数を変更して同じ人手ラベルへ再適合しない。事前定義したfailureとして扱う。

---

## 11. 再現性とデータ保護

- split、sampler、calibration batch、model初期化のseedをartifactへ保存する。
- patient IDのfold重複を開始前assertで検出する。
- pseudo scoreのteacher foldとstudent foldの対応を検証する。
- human validity maskとpseudo除外maskの優先関係をtestする。
- source別のvisit数とunique coverageを保存する。
- `alpha_k`、`lambda_k`、gradient summary、clip前後の値をfold別artifactとして保存する。
- 校正artifactはconfigの`calibration.version`ごとに分離し、校正関連config fingerprintを保存する。
- 出力先、GPU、active region以外の学習条件がfingerprintと一致しない校正artifactは拒否する。
- calibrationでoptimizer updateや意図しないmodel state更新を発生させない。

---

## 12. 実装構成

新規実装は`fracture_detection/region_branch/`へ置き、Baseline 0の責務分割に合わせる。

想定する責務は次のとおり。

- model: two-BiLSTM、FPN、mask-normalized pooling、four-region heads
- model variant: 統合4領域とtarget-region指定single modelを同じcomponentから構築
- data: human / whole-negative / pseudo sourceの構築とfold検証
- sampling: persistent three-source queuesと固定cadence
- losses: source-balanced `L_exact`と既存`L_rank`の統合
- calibration: fold別`alpha_k` / `lambda_k`算出とartifact保存
- training: whole / region streamsの統合、logging、checkpoint
- evaluation: locked OOF region metricsとcollapse diagnostics

既存資産として次を優先的に再利用する。

- `baseline0/pseudo_labeling/scoring.py`のpair構築とranking loss
- `baseline0/modeling/losses.py`のwhole lossとbag aggregation
- `baseline0/data/splits.py`のfold分割
- 既存stage 3のFPN / mask pooling参照実装

---

## 13. 実装前後の検証項目

### 13.1 Unit test

- whole-negativeを複製してもsource平均後の`L_exact`が変わらない。
- invalid human cellがlossへ入らない。
- valid human cellでpseudo教師が除外される。
- 4/4/8 batchが毎region updateで維持される。
- queueがshuffle-without-replacementで全sampleを循環する。
- 統合modelと単一領域modelでactive-region平均のscaleが一致する。
- single modelがtarget以外の3領域maskとlabelを参照しない。
- 統合head `r`とsingle model `M_r`で対応領域の教師manifestが一致する。
- `L_whole`はwhole pathとshared trunkだけへ勾配を流す。
- region lossはregion pathとshared trunkだけへ勾配を流す。
- calibrationがmodel parameterと永続的なmodel stateを変更しない。
- collapse alarmが閾値と連続回数どおり発火する。

### 13.2 短縮preflight

- Baseline 0のwhole exposureとloss契約が維持される。
- 3 sourceのvisit数とcoverageが期待値どおりである。
- raw / weighted gradient比が有限で、想定scaleにある。
- region logitsが開始直後から完全に同一化していない。
- checkpoint、calibration、diagnostic artifactを再読込できる。

### 13.3 本学習開始条件

- unit test、ruff、format check、type checkが通る。
- fold leakageとteacher-student fold対応のassertが通る。
- one-time calibration artifactが全foldで生成される。
- 短縮preflightで非有限loss / gradientとdata-source欠落がない。

---

## 14. 実装順序

1. `fracture_detection/region_branch/`の責務構成を作る。
2. two-BiLSTM model、FPN、region poolingを実装する。
3. source-balanced `L_exact`を実装する。
4. 既存pairwise ranking lossを接続する。
5. three-source samplerとqueueを実装する。
6. whole / regionのtraining-step契約を固定して実装する。
7. `alpha_k` / `lambda_k` calibrationとartifactを実装する。
8. collapse監視とsource exposure loggingを実装する。
9. unit testと短縮preflightを実施する。
10. 統合4領域model 5 runsとsingle model 20 runsの本学習へ進む。

---

## 15. 主要な参考文献

- Durand et al. (CVPR 2019),
  [Learning a Deep ConvNet for Multi-Label Classification with Partial Labels](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf)
- Wu et al. (ECCV 2020),
  [Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf)
- Ridnik et al. (ICCV 2021),
  [Asymmetric Loss for Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)
- Radhakrishnan et al. (WACV 2024),
  [Design Choices for Enhancing Noisy Student Self-Training](https://openaccess.thecvf.com/content/WACV2024/papers/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.pdf)
- Sohn et al. (NeurIPS 2020),
  [FixMatch](https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f73-Paper.pdf)
- Xie et al. (CVPR 2020),
  [Noisy Student](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper)
- Chen et al. (ICML 2018),
  [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
- Kendall et al. (CVPR 2018),
  [Multi-Task Learning Using Uncertainty to Weigh Losses](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)

12件の文献と本設計への対応関係は、損失設計の確定記録に記載する。
