# Region branch 損失設計・最終決定

作成日: 2026-08-25

## 1. 状態

4領域骨折検出モデルのarchitecture、教師信号、損失構造、不均衡対策、
`lambda` / `alpha`の決定方法を確定した。

**設計は確定、実装は未着手。**

詳細な文献調査と代替手法の比較は
`.claude/docs/research/20260825-region-loss-balancing.md`を参照する。

---

## 2. Architecture

- Shared EfficientNetV2-S trunk
- stage 1-4をFPNでstride 4へ融合
- 4領域maskによるmask-normalized pooling
- whole用BiLSTMとregion用BiLSTMの**2本構成**
- whole pathはBaseline 0と同じ構造を維持
- region pathはshared region BiLSTMの後に4領域headを置く
- CNN trunkだけをwhole / region間で共有する
- hard GTとpseudo教師は**同じ4領域logit**を学習する

```text
Shared EfficientNetV2-S trunk
    ├─ Baseline 0 whole path
    │    └─ whole BiLSTM -> whole head
    └─ FPN -> region pooling
         └─ region BiLSTM -> four-region heads
```

### Single-region comparator

比較対象はR1-R4ごとの独立single model 4本とする。各single modelはheadだけを分けるのではなく、
CNN trunk、Baseline 0 whole path、FPN、region BiLSTM、対象領域headを独立に持つ。
対象領域maskだけを使い、active region集合を`R={r}`として統合modelと同じ損失式で学習する。

- `alpha_k` / `lambda_k`は統合4領域modelで校正した値を同一foldの4本へ流用する。
- 対応領域のhuman / whole-negative / pseudo教師露出を統合headと揃える。
- single model間でparameterを共有しない。
- 4本のregion出力を対応領域ごとに並べ、統合modelの4出力と比較する。
- 各single modelのwhole出力はdiagnosticであり、4本ensembleをprimary endpointにしない。

---

## 3. 教師信号

| Source | Bag数 | 教師 |
|---|---:|---|
| Human-annotated | 268 | 有効セルの人手4領域ラベル |
| Whole-negative | 12,100 | 論理的に確定する`[0, 0, 0, 0]` |
| Pseudo-positive | 1,064 | fold-matched CAM pairwise ranking |

人手ラベルの有効セルは983 / 1,072。無効セルはlossから除外する。
validな人手セルにはpseudo教師を入れず、hard GTを常に優先する。

---

## 4. 全体損失

損失は次で固定する。

\[
L=L_{\mathrm{whole}}+\lambda
\left(L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}\right).
\]

- `L_whole`: 全bagのwhole-vertebra分類
- `L_exact`: 人手4領域ラベルとwhole-negative由来の論理0
- `L_rank`: fracture-positive bag間のCAM pairwise ranking蒸留

統合4領域モデルと単一領域4モデルで同じ式を使う。
各lossは領域内平均後、active region間で平均する。4領域モデルだけlossが4倍に
ならないよう、領域方向の総和は使わない。

---

## 5. Exact-label不均衡対策

人手GTとwhole-negativeは同じexact教師として`L_exact`へ入れるが、12,100件の
whole-negativeが268件の人手教師を件数で希釈しないよう、内部でsource-balancedにする。

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

`L_H`と`L_N`はどちらもexact GTであり、分ける理由は教師の信頼度差ではなく
sampling母数の差だけである。

### Auxiliary region batch

batch size 16では次のsplitを使う。

| Source | Bags / batch | Loss |
|---|---:|---|
| Human-annotated | 4 | `L_H` |
| Whole-negative | 4 | `L_N` |
| Pseudo-positive endpoints | 8 | `L_rank` |

各poolはpersistentなshuffle-without-replacement queueで循環する。
全region updateに3 sourceを必ず含め、negative-only / pseudo-only updateは禁止する。

人手bagは一様抽出し、R2などのrare positiveを基準にしたbag oversamplingは行わない。
multi-labelでは1領域をoversampleすると、同じbagに共存する他領域も同時に増え、
multi-region症例の共起分布が変わるためである。

この1:1 human / whole-negative構成では、`L_exact`内の実効陽性率は概ね
R1 15.9% / R2 12.1% / R3 14.8% / R4 31.5%になる。

---

## 6. `pos_weight`の適用範囲

### `L_whole`

Baseline 0の契約をそのまま維持する。

- `pos_weight=2.0`
- 陽性要素のlossを2倍
- `weighted_loss.sum() / weight.sum()`で正規化
- natural streamのbag分布、batch size、whole exposureをBaseline 0と揃える

### Region losses

`L_exact`と`L_rank`には`pos_weight`を使わない。

- `L_exact`: source-balanced sampling後のplain BCE
- `L_rank`: 既存のregion-balanced pairwise BCE

Baseline 0のwhole陽性率9.9%に`pos_weight=2.0`を適用した実効陽性寄与は約18%。
region側はsamplingだけで約12-32%なので既に同程度である。
regionへさらに`pos_weight=2.0`を重ねると実効寄与が約22-48%となり、特にR4を
過剰に重くするため採用しない。

全hard GT比からregion `pos_weight`を計算する案も不採用。概算値が
R1 157 / R2 208 / R3 170 / R4 77となり、source-balanced samplingとの
二重補正になる。

---

## 7. Hard GTとpseudo教師の重み付け

同じ4領域logitに対し、exactとrankingを別々に平均してから結合する。

\[
L_{\mathrm{region}}=L_{\mathrm{exact}}+\alpha L_{\mathrm{rank}}.
\]

- exact cell数とpseudo pair数を同じ分母へ入れない
- sample数比例で`alpha`を決めない
- CAM magnitudeによる追加confidence weightは使わない
- undefined CAM scoreだけを既存規則どおり除外する
- hard / pseudo streamは同じ固定cadenceで更新する

`L_exact`は各logitの符号・bias・絶対scaleをanchorする。
`L_rank`はpositive bag間の順序を学ぶが、regionごとの共通logit shiftには不変なので、
exact教師と同じheadへ入れることで絶対位置が定まる。

---

## 8. `alpha`の決定

268人手ラベルの性能を使ったgrid searchは行わない。
各outer foldのtraining foldsだけから、更新前の64 deterministic calibration batchで
一度だけ勾配normを測る。

region BiLSTM parameters上で、

\[
g_H=\|\nabla L_{\mathrm{exact}}\|_2,\qquad
g_P=\|\nabla L_{\mathrm{rank}}\|_2
\]

を測り、

\[
\alpha_k=
\operatorname{clip}_{[0.01,1]}
\left[
0.25\exp\left\{
\operatorname{median}_b
\log\frac{g_{H,b}+\epsilon}{g_{P,b}+\epsilon}
\right\}
\right]
\]

とする。初期pseudo勾配をexact勾配の約1/4にする。

統合4領域モデルをreferenceとして`alpha_k`を一度だけ決め、同じouter foldの
統合モデルと単一領域4モデルすべてへ同じ値を適用する。model別・region別の
再校正は禁止する。

---

## 9. `lambda`の決定

`alpha_k`決定後、whole / regionの両方が通る最後のshared CNN block上で、

\[
g_W=\|\nabla L_{\mathrm{whole}}\|_2,
\qquad
g_R=\|\nabla(L_{\mathrm{exact}}+\alpha_kL_{\mathrm{rank}})\|_2
\]

を測り、

\[
\lambda_k=
\operatorname{clip}_{[0.01,10]}
\left[
0.25\exp\left\{
\operatorname{median}_b
\log\frac{g_{W,b}+\epsilon}{g_{R,b}+\epsilon}
\right\}
\right]
\]

とする。初期shared-trunk region勾配をwhole勾配の約1/4にする。

同じ`lambda_k`を、そのouter foldの統合モデルと単一領域4モデルへ適用する。
非有限gradientはimplementation failureとしてrunを開始しない。

one-time calibrationを実装できない場合のみ、暫定値として
`lambda=0.25`, `alpha=0.25`を使う。ただしlabel-blind preflightでraw gradient比が
0.5-2倍の範囲にあることを確認する。

dynamic GradNorm、learned uncertainty weighting、性能を見た再調整は採用しない。

---

## 10. 監視項目

毎epoch、固定diagnostic subsetで次を記録する。

- region logit間Spearman 6組
- 各region logitとwhole logitのSpearman
- 標準化4-logit行列の第1主成分説明率
- region別student-teacher rank correlation
- raw / weighted `g_rank / g_exact`
- raw / weighted region / whole shared-trunk gradient比
- source別visit数、unique coverage、queue周回数
- region別valid cell数と実効陽性率

collapse alarmは次で固定する。

- inter-region Spearman中央値 `>= 0.95`
- region-whole Spearman中央値 `>= 0.95`
- 上記が3回連続

alarm後に係数を調整して再実行しない。事前定義したcollapseとして失敗扱いにする。

balanced samplingとranking supervisionによりlogitのinterceptはpopulation prevalenceを
直接表さない。region sigmoidは校正済み無条件骨折確率ではなくscoreとして扱い、
locked OOFのregion別AP / AUROCで評価する。

---

## 11. 採用しない手法

- 全12,100 whole-negativeを自然頻度のまま`L_exact`へ流す
- corpus比から作るregion `pos_weight`
- source-balanced samplingと`pos_weight`の併用
- rare region陽性に基づくbag oversampling
- ASL / focal lossをprimary lossにする
- exact cell数とpseudo pair数による件数比例weight
- CAM magnitudeによるper-pair confidence weight
- model別・region別の`lambda` / `alpha`調整
- dynamic GradNorm / learned uncertainty weighting
- 268 human labelsまたはouter fold性能を見た再調整

---

## 12. 実装時の次工程

1. `fracture_detection/region_branch/`を作成する
2. two-BiLSTM modelとFPN region poolingを実装する
3. source-balanced `L_exact`を実装する
4. 既存`region_balanced_pairwise_ranking_loss`を接続する
5. three-source sampler / queue / cadenceを実装する
6. `alpha_k` / `lambda_k` calibration artifactを実装する
7. collapse監視とsource exposure loggingを実装する
8. unit test、ruff、type check、短縮preflightを実施する

本work-logは設計記録であり、実装開始の承認を意味しない。

---

## 13. 参考文献

### Partial multi-label・不均衡対策

1. Durand, T. et al. (2019),
   [Learning a Deep ConvNet for Multi-Label Classification with Partial Labels](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf),
   CVPR 2019.
   既知セルだけをpartial BCEへ入れ、未知セルを除外する設計の根拠。
2. Ben-Baruch, E. et al. (2022),
   [Multi-Label Classification with Partial Annotations using Class-Aware Selective Loss](https://openaccess.thecvf.com/content/CVPR2022/html/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.html),
   CVPR 2022.
   annotation済み教師を推定教師より重視し、positive / negative不均衡を非対称に扱う事例。
3. Wu, T. et al. (2020),
   [Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf),
   ECCV 2020.
   multi-labelにおけるnegative dominanceと、label単位oversamplingが共起labelの露出も
   変える問題の根拠。
4. Ridnik, T. et al. (2021),
   [Asymmetric Loss for Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf),
   ICCV 2021.
   abundant easy negativeを抑える代替手法。今回はsource-balanced samplingとの二重補正を
   避けるためprimary lossには採用しない。

### Clean GT・pseudo labelの同一head学習

5. Radhakrishnan, A. et al. (2024),
   [Design Choices for Enhancing Noisy Student Self-Training](https://openaccess.thecvf.com/content/WACV2024/papers/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.pdf),
   WACV 2024.
   clean / pseudoをsplit batchで供給し、別々に平均したlossを結合してpseudo poolの件数支配を
   防ぐ設計の根拠。
6. Sohn, K. et al. (2020),
   [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f73-Paper.pdf),
   NeurIPS 2020.
   labeled / pseudo-labeled sampleを同じstudent outputで学習し、pseudo寄与を独立lossで
   制御する代表例。
7. Xie, Q. et al. (2020),
   [Self-Training with Noisy Student Improves ImageNet Classification](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper),
   CVPR 2020.
   clean labelとpseudo labelを同一studentへ入力し、samplingやstudent noiseでpseudo教師の
   影響を制御する代表例。
8. Hinton, G., Vinyals, O., and Dean, J. (2015),
   [Distilling the Knowledge in a Neural Network](https://research.google.com/pubs/archive/44873.pdf).
   hard targetとsoft targetを別々の重み付き目的として結合する基本的な根拠。
9. Tarvainen, A. and Valpola, H. (2017),
   [Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html),
   NeurIPS 2017.
   pseudo / consistency supervisionを独立した重み付きregularizerとして扱う事例。

### `lambda`・`alpha`とmulti-task weighting

10. Chen, Z. et al. (2018),
    [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html),
    ICML 2018.
    shared parameters上のgradient normでtask間の寄与を観測する根拠。本設計ではdynamicな
    重み更新ではなく、学習前のone-time calibrationだけに限定して使う。
11. Kendall, A., Gal, Y., and Cipolla, R. (2018),
    [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html),
    CVPR 2018.
    auxiliary taskの相対重みがshared representation学習へ影響する代表例。ただしexact教師を
    pseudo教師より優先する制約を表現しないため、learned uncertainty weightingは採用しない。
12. Telesco, A. et al. (2025),
    [Semi-Supervised Multi-Task Learning for Interpretable Quality Assessment of Fundus Images](https://arxiv.org/abs/2511.13353).
    医用画像でmanual primary taskとpseudo-labeled auxiliary taskを同一studentへ組み込む
    architecture上の参考例。CAM ranking教師そのものの妥当性を保証する文献ではない。

各文献から本タスクへの適用を導いた詳細な比較と留保事項は、
`.claude/docs/research/20260825-region-loss-balancing.md`の`Evidence Base`を参照する。
