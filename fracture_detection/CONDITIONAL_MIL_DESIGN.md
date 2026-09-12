# 旧案: 陽性椎体に限定した局所4領域モデルの設計検討

> 2026-09-10のユーザー指定により本案は置き換え済み。
> 現行設計は `REGION_MIL_DESIGN.md`。CNNをfine-tuneし、陰性も含めて4領域を学習し、
> wholeは4領域出力だけから計算する。以下の陽性限定・並列whole・凍結pilotは過去の検討記録。
> 注釈契約も訂正済み: GTあり椎体は全4領域が既知で、0は骨折なしを意味する。
> 以下の部分GT・未知セルの解釈と旧validity由来の集計は、現行設計の根拠として使わない。

作成日: 2026-09-09
設計レビュー更新: 2026-09-10

状態: 文献調査・設計案。モデルコード、設定、学習処理はまだ作成しない。
実装候補の配置先は `fracture_detection/conditional_mil/` とする。

調査方法: 指定PDFの本文、公開一次論文、現行コード、manifest集計を突き合わせた。
補助調査でGemini CLIも試したが認証待ちとなり、研究回答は得られなかった。
以下の根拠は直接確認できた一次資料と、自前の数式・データ確認による。

## 1. 現時点の推奨と未解決点

検証候補は「領域maskでpoolした局所特徴 → 領域ごとのbagスコア」に、
陽性椎体の既知領域GTと、未注釈陽性のat-least-one制約を同時に与える構成とする。
whole分類は全椎体で学習し、region headの損失・専用経路は陽性椎体だけで学習する。

ただし、陽性だけのOR損失は、どこが骨折しているかを単独では識別できない。
領域を区別する教師の中心は、領域GTに含まれる陽性と陰性の組合せである。
局所特徴はその教師を未注釈症例に一般化するための構造上の制約であり、
局在精度を保証するものではない。

最初の比較は、同じ局所モデルの「GTのみ」と「GT＋noisy-OR」とする。
2026-09-10の推奨では、両損失からregion FPN・BiLSTM・最終classifierへ通常の勾配を流す。
ORは未注釈症例にも存在制約を満たすよう求める補助損失であり、係数を0から緩やかに上げる。
正則化として使うことと、最終classifierへの勾配を止めることは別の設計判断である。

前案で第一候補にしたhead detachは追加比較へ下げる。直接の実証がなく、headが固定でも
特徴側の変化で全領域高値になり得るため、「GTが意味を固定する」との説明は強すぎた。
弱教師を追加して良くなるかを先に検証し、その後で勾配経路を制限する効果を調べる。
共有head、mask pooling、陰性region GTのいずれもcollapseの防止を保証しない。

第一pilotの主目的は陽性椎体内の領域判別とし、fold-matched Baseline 0のCNNとwhole経路を固定する。
これはregion正則化を切り分ける比較条件であり、最終モデルのCNN凍結を必須とはしない。
whole分類は既存Baseline 0の学習段階で全椎体の教師を利用済みである。
2026-09-10追記: CNN固定は比較実験を限定するための提案で、ユーザー承認済みの条件ではない。
凍結時は弱ORがCNNを正則化・適応させることはできず、検証範囲はFPN以降に限られる。
CNNまで含めたGT-onlyとGT＋ORの比較も、両armの更新条件をそろえれば成立する。
したがって凍結はORの効果を検証する必要条件ではなく、共同学習とどちらを主比較にするかは検討中。

### 要件と提案の区別

| 区分 | 内容 |
|---|---|
| ユーザー要件 | region headの学習は陽性椎体だけ |
| ユーザー要件 | region GTなし陽性は「少なくとも1領域に骨折あり」で学習 |
| ユーザー要件 | region GTあり陽性は領域GTで直接学習 |
| ユーザー要件 | 各領域スコアを、その領域maskで制限した特徴から計算 |
| 今回の作業範囲 | 調査と設計書まで。実装・GPU学習は対象外 |
| 本書の提案 | GT＋rampしたORからregion専用経路全体を更新。head detachは追加比較 |
| 本書の提案 | region最終classifierは4領域で共有する案を第一候補とし、部分GTはセル単位で管理 |
| 検証が必要 | ORの有効性、特徴の局所性、損失係数、較正。追加でhead共有・detach・trunk解凍 |

以下の数値設定はレビュー用の提案値であり、ユーザーが承認済みの設定や文献の最適値ではない。
既存CAM実験へ遡及適用せず、実装開始前にこの新モデルの設定として固定する。

## 2. 実データから確認した教師の構造

参照: `fracture_detection/baseline0/resources/input_manifest.csv`。
以下は2026-09-09に現行manifestを再集計した値であり、過去のメモからの転記ではない。

| 教師の状態 | 全5foldのbag数 | regionで使える情報 |
|---|---:|---|
| 椎体陰性 | 12,100 | 今回はregion学習に使わず、whole分類に使う |
| 椎体陽性・完全region GT | 235 | 4領域すべての0/1 |
| 椎体陽性・部分region GT | 33 | 確認済みのセルのみ |
| 椎体陽性・region GTなし | 1,064 | 少なくとも1領域が陽性 |
| 合計 | 13,432 | 陽性は1,332袋 |

268 annotated bagsは160 studiesに属する。4領域×268の1,072セルのうち、
有効GTは983セル、未知は89セルである。有効GTの内訳は陽性367、陰性616。

| 領域 | 解剖学的な意味 | 確認陽性セル | 確認陰性セル |
|---|---|---:|---:|
| R1 | 椎体本体 | 78 | 167 |
| R2 | 右椎間孔領域 | 59 | 184 |
| R3 | 左椎間孔領域 | 72 | 172 |
| R4 | 後方要素 | 158 | 93 |

完全GT235件の陽性領域数は、1領域171件、2領域42件、3領域19件、4領域3件。
複数領域骨折があるため、4領域softmaxや「必ず1領域だけを正解にする」教師は適合しない。

部分GT33件はすべて、少なくとも1個の確認陽性を持つ。
現行validity生成規則では、未完了アノテーションの0は未知であり、陰性ではない。
確認陰性616セルは完全GT側から得られる。`has_region_target`だけでは完全性を判定しない。
この契約は `baseline0/data/region_validity.py` の
`attach_region_target_validity` と一致する。

| fold | 椎体陽性 | annotated | 完全GT |
|---|---:|---:|---:|
| 0 | 262 | 56 | 49 |
| 1 | 268 | 53 | 43 |
| 2 | 269 | 53 | 48 |
| 3 | 263 | 53 | 50 |
| 4 | 270 | 53 | 45 |

outer 0ではtrainがfold 2,3,4、inner validationがfold 1、outer testがfold 0。
trainの陽性802件の内訳は完全GT143、部分GT16、未注釈643である。

### 注釈の偏り

annotated陽性とunannotated陽性の椎体レベル構成は同じではない。
例えばC7の割合は13.4%対30.9%、C4は16.0%対5.5%だった。
これは骨折領域分布の差を直接証明しないが、annotated集合を無条件に代表標本とする
根拠にはならない。全268件から得た領域頻度・平均骨折領域数を未注釈集合に強制する
distribution matchingは初期設計に加えない。
将来検討する場合も、train fold内の完全GTだけで推定し、選択バイアスを別途評価する。

### 現行ログから言えること・言えないこと

既存outer 0 pilotでは、epoch 4から24にかけてhuman train lossが0.349から0.101へ下がる一方、
human validation lossは0.542から1.030へ上がり、region AUROCは0.8305から0.8155だった。
少数GTへの過信・較正悪化という懸念とは整合する。ただしこのrunはGT-onlyではなく
`L_exact + CAM ranking`であり、teacher順位とのSpearmanも0.615--0.754から0.826--0.890へ
上がっている。したがって「GTだけが過学習した」と原因を確定できず、GT-onlyとGT＋ORの
matched比較が必要である。hard BCEは順位が同程度でもlogitを極端化してvalidation lossを
上げ得るため、BCEだけで正則化成功・失敗を判定しない。

## 3. 参考論文から使えること・そのまま使えないこと

### Fang et al., 2021: 指定PDF

[Weighing Features of Lung and Heart Regions for Thoracic Disease Classification](https://arxiv.org/abs/2105.12430)
が指定PDFに対応する。ローカルPDFの§3.3、式(3)、§4、Discussionを確認した。

画像分類CNNの特徴mapに、別途求めた肺・心臓のmaskを乗算し、poolingして疾患分類する。
本件に使えるのは「解剖学的maskを特徴の読み出し範囲にする」という発想である。
原論文は肺・心臓を合成したmaskを使い、4領域それぞれの骨折を予測するものではない。
また、病変bboxは主に局在の評価・可視化に使用しており、今回の部分region GT＋OR損失の
直接的な根拠にはしない。原論文自身も、臓器mask内の正常部位まで除去できるわけでは
ないことを説明している。

原論文のmask乗算後の平均poolingに対し、本件では領域面積で正規化する。
これは領域サイズの違いを抑えるための本件の設計判断であり、論文の忠実な再現とは区別する。

### Li et al., CVPR 2018: GTあり／なしで教師を切り替える近い例

[Thoracic Disease Identification and Localization with Limited Supervision](https://arxiv.org/abs/1711.06373)
の§3.2では、位置注釈あり画像にpatchの陽性・陰性教師を与え、位置注釈なしの陽性画像には
noisy-ORによる「少なくとも1patch」を使用する。教師の粒度を切り替える発想が本件に近い。

ただし論文のinstanceは画像内のpatch、本件のinstanceは15面をまとめた解剖学的領域。
さらに論文は疾患陰性画像も学習に利用する。陽性だけのregion headが同様に学習できる
という実証にはならない。多数patchの積に対する独自のスコア変換もあり、
4領域の較正済み確率を得る手法としてそのまま移植しない。

### Durand et al., CVPR 2019: 部分GTの扱い

[Learning a Deep ConvNet for Multi-label Classification with Partial Labels](https://arxiv.org/abs/1902.09720)
は部分ラベルを扱う損失と既知ラベル割合に応じた正規化を検討する。
本件では未知セルをBCEから除く考え方を参照する。
既知セル数による再重み付けまで自動的に採用するのではなく、§6の固定したreductionで
部分陽性セルを過大に繰り返さないようにする。

### Tourniaire et al., 2021 / MS-CLAM系: 補助的な先行例

[Attention-based Multiple Instance Learning with Mixed Supervision on the Camelyon16 Dataset](https://proceedings.mlr.press/v156/tourniaire21a.html)
は少数の局所注釈とbagラベルの併用による局在改善を報告している。
ただしWSIのtile注釈であり、正常slideも利用する。注釈付きの段階と全体学習の段階を持つ
2021年版の手順を、今回の陽性限定・同時学習の根拠として同一視しない。
「詳細教師が弱教師だけの局在を改善し得る」という補助的根拠にとどめる。

### Wang et al., Interspeech 2018: OR集約の失敗例

[Comparing the Max and Noisy-Or Pooling Functions in Multiple Instance Learning for Weakly Supervised Sequence Learning Tasks](https://www.isca-archive.org/interspeech_2018/wang18_interspeech.html)
は音声・音イベントの系列課題で、noisy-ORの局在失敗とmaxとの差を報告する。
骨折4領域でmaxが優れる証拠ではないが、「noisy-ORなら局在が得られる」とは言えない
根拠になる。maxにも1領域だけに勾配が集中する問題があるため、置換だけで解決したとしない。

### Xu et al., ICML 2018: 論理制約を損失にする位置付け

[A Semantic Loss Function for Deep Learning with Symbolic Knowledge](https://proceedings.mlr.press/v80/xu18h.html)
のDefinition 1は、出力の独立Bernoulli積の下で制約を満たす状態の確率を合計し、
その負の対数を損失にする。制約を4変数のORにすると、本書のnoisy-OR項になる。
これは「既知の論理を未注釈出力へ与える正則化」という位置付けを支える。
条件付き周辺確率の較正や骨折局在への有効性を保証する論文ではない。

### 少量局所GTへ粗ラベルを足す効果は一貫しない

[Thoracic Disease Identification and Localization with Limited Supervision](https://arxiv.org/abs/1711.06373)
はbboxあり画像と画像ラベルだけの画像を混ぜ、未注釈画像を増やすことで局在が改善する例を示した。
一方、[Localization with Limited Annotation for Chest X-rays](https://proceedings.mlr.press/v116/rozenberg20a.html)
では、bbox学習集合へ画像ラベルだけの症例を加えた改善は8疾患中4疾患であり、一様ではない。
両研究とも正常画像を含み、instance数や教師構造も本件と違うため、弱ラベル追加を成功保証にはしない。

[CheXseg](https://proceedings.mlr.press/v143/gadgil21a.html)は、少数の専門家pixel GTと
DNN saliency由来の粗いmaskを混ぜ、専門家GTだけのsegmentationよりmIoUを相対9.7%改善した。
これは弱教師が強教師モデルの汎化を改善し得る直接的な医用画像例だが、saliency maskは
本件の1-bit ORより情報量が多く、同じ効果量を期待する根拠にはしない。

[Evaluating Weakly Supervised Object Localization Methods Right](https://openaccess.thecvf.com/content_CVPR_2020/html/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.html)
は画像ラベルだけの局在がill-posedであり、少数の完全教師baselineを厳密に比較すべきことを示す。
したがって本件でも、GT-onlyを省略して「データが増えたから正則化された」と結論しない。

### 弱損失は最初から強く掛けない

[Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)
は、ラベルなし整合性損失の重みを0からrampすることが退化解を避けるうえで重要と報告する。
ORと整合性は同じ損失ではないが、ランダムなregion headが作る初期出力を大量の弱症例で
自己強化しない、という設計原理は本件にも適用する。係数はloss値だけでなく、GT項と弱項が
region表現へ与える勾配ノルムを別々に記録して決める。

### 階層分類論文との違い

[Semi-Supervised Learning with Taxonomic Labels](https://arxiv.org/abs/2111.11595)
の粗ラベルは、相互排他的な葉クラスの確率を足して求める。
本件の4領域は同時に陽性になり得るため、同じ確率の足し上げやsoftmaxを適用できない。
同論文の別headとの比較から、本件のwhole headの廃止を結論することもできない。

過去の `.claude/docs/research/20260729-mixed-supervision-coarse-fine-literature.md`
にある「268件すべて完全GT」「陰性bagをregionへ必ず投入」「事前分布matchingが必須」
という記述は、現行データ・今回の要件・上記の適用範囲に合わせて引き継がない。

## 4. 何を予測するか

椎体骨折ラベルを `Y`、領域rの骨折ラベルを `R_r` とする。
解剖学的4領域が対象を覆うという前提で、`Y = OR(R_1,...,R_4)`。

whole headの出力を `p_whole`、局所region headのsigmoid出力を `q_r` とする。
`q_r`の用途は「骨折陽性の椎体について、どの領域が骨折しているか」の判別である。
noisy-ORで学習する案では、まず条件付き局在スコアと呼び、
`P(R_r=1 | X,Y=1)`として較正されているとは仮定しない。

推論時にはGTのYは利用できない。全bagでwholeとregionを計算し、
領域のend-to-end検出スコア候補を `p_whole * q_r` とする。
両者が同じXに基づく適切な全体・条件付き確率なら連鎖律に対応するが、学習済みスコアの積が較正済み確率になる
保証はない。条件付き局在とend-to-end検出は別々に評価する。

### 条件付き確率とnoisy-ORの区別

一般の従属した領域ラベルでは、

\[
P(\cup_r\{R_r=1\}\mid X,Y=1)
\ne 1-\prod_r\{1-P(R_r=1\mid X,Y=1)\}.
\]

左辺は条件付けによって1である。右辺は独立Bernoulliを仮定した式であり、
真の条件付き周辺確率を入れても1になるとは限らない。
例えば「必ず1領域だけ骨折し、画像からは4候補が等確率」の場合、真の周辺確率は
`q=(0.25,0.25,0.25,0.25)`だがnoisy-ORは0.683594である。
正しい不確実性を残す予測にもOR損失0.380391が掛かってしまう。

したがって本案のORは、独立Bernoulliを作業上のモデルにした弱教師の近似損失であり、
厳密な条件付き周辺確率の尤度とは区別する。この点は実装上の数値安定化では解決しない。

## 5. 局所特徴の構成案

```text
15面 × (2.5D CT 5ch + 椎体mask 1ch)
    ↓ shared EfficientNetV2-S trunk
    ├─ whole path → whole BiLSTM/head → p_whole
    │                全椎体にwhole教師
    └─ 学習時は陽性bagのfeature mapを選択
         ↓ stage feature maps → stride-4 FPN
         ├─ R1 mask pool → R1の面系列 → sequence pool ┐
         ├─ R2 mask pool → R2の面系列 → sequence pool ├→ shared Linear(1) → q1..q4
         ├─ R3 mask pool → R3の面系列 → sequence pool ┤
         └─ R4 mask pool → R4の面系列 → sequence pool ┘
```

regionの面系列を処理するBiLSTMは4領域で重みを共有するが、入力系列は領域ごとに独立。
whole BiLSTMとも分離する。異なる領域のfeatureを連結して4出力を作る経路は設けない。
最終headも「局所特徴に骨折があるか」という同じ述語として4領域で共有する小さい
`Linear(1)`を第一候補とする。これにより領域別biasだけでR4へORを押し付ける自由度と
パラメータ数を減らす。領域ごとに骨折所見が異なり共有headがunderfitする可能性はあるため、
4個の独立Linearは容量ablationとし、GT-onlyとGT＋ORでhead構成を変えない。

第一pilotの容量案は、既存と同じstride-4 FPNの256ch、共有BiLSTMのhidden 128・1層・双方向、
有効面の特徴平均、Dropout 0.30、共有Linear(256,1)とする。既存の2層hidden 256と
4個のBatchNorm付きMLPよりregion経路を小さくする提案であり、この容量自体の優位性は未検証。
FPNのGroupNormは維持し、最終headにBatchNorm・領域ID embedding・global feature連結を置かない。
同じ入力を同じ重みで処理しても、maskから取り出す特徴が違うため4個のqは異なり得る。

面s・領域rの特徴は、FPN出力 `F_s` と領域mask `M_sr` から、

\[
f_{sr}=\frac{\sum_{u,v}\widetilde M_{sr}(u,v)F_s(u,v)}
                 {\sum_{u,v}\widetilde M_{sr}(u,v)+\epsilon}
\]

とする。maskはone-hot化してからarea poolingで特徴解像度に合わせる。
ラベルID自体をbilinear補間しない。小領域の消失と面積バイアスを抑える狙いであり、
既存 `region_branch/modeling/pooling.py` の構成を参照できる。

### 面方向の集約を領域方向のORと分ける

region GTはbagのラベルであり、15面すべての骨折ラベルではない。
第一候補は有効面を元の順序で集めてregion BiLSTMに通し、その出力特徴をmasked meanし、
最後に1bag・1regionのlogitを作る方式とする。可変長系列はpaddingを再帰計算に混ぜず扱う。
GTのBCEもregion間のORも、そのbag logitに1回だけ掛ける。
有効面を詰める場合も元の面indexを保存し、途中欠損の数と位置を監査する。
初期案は有効面の順序だけを利用するため、詰めた系列では物理的間隔を表現しない限界がある。

既存の「面sigmoidを平均してbag確率にする」方式とは異なる提案である。
面sigmoid平均では、bagスコアを1に近づけるには有効面のスコアも広く高くする必要があり、
狭い骨折所見と相性が悪い可能性がある。一方、特徴のmeanにも希釈はあり得る。
その場合の比較候補は領域内だけのattention poolingとする。
attention重みは面の選択重みであり、骨折確率とは呼ばない。

この変更の効果をORに帰属させないよう、GTのみとGT＋ORで同じ面集約を使う。

### 「局所」の保証範囲

mask poolingが保証するのは、特徴mapのどの位置を読み出すかである。
CNNの受容野、FPNの融合、正規化などによって、その特徴には領域外情報も含まれ得る。
2.5D入力の隣接面についても、中心面のmaskだけで厳密な3D領域分離にはならない。

今回の第一候補はこの「maskで読み出し位置を制限する」局所性とする。
領域外画素への依存が強い場合は、各領域ROIをencoderより前でcrop/maskし、
共有encoderで個別に特徴抽出する案を次に比較する。これは入力段階の制限が強い反面、
計算量、骨皮質境界の文脈、領域サイズ差、隣接面maskの扱いが追加課題になる。

局所性の確認では、feature map上の領域外位置を変えてpool結果が不変かを見る検証と、
入力CTの領域外を変更してqがどれほど変わるかを見る検証を区別する。

## 6. GTと弱教師をどう同時に学習するか

bag iについて、確認済みセルの集合を `V_i`、未知セルの集合を `U_i`、
確認済みの0/1を `t_ir` とする。幾何学的なmaskの有効性とGTの既知／未知は別に管理する。

### 第一候補: 観測された情報に応じた損失

\[
\ell_{GT,i}=\sum_{r\in V_i}\operatorname{BCEWithLogits}(z_{ir},t_{ir})
\]

\[
\ell_{weak,i}=
\begin{cases}
-\log\left[1-\prod_{r\in U_i}(1-q_{ir})\right],
 & \text{確認陽性なし、未知候補あり、候補を観測可能}\\
0, & \text{それ以外}
\end{cases}
\]

\[
L_{GT}=\frac{\sum_i\ell_{GT,i}}{\sum_i|V_i|},\qquad
L_{OR}=\frac{\sum_i\ell_{weak,i}}{N_{weak}},
\]

\[
L_{region}=L_{GT}+\beta(t)L_{OR},\qquad
L=L_{whole}+\lambda L_{region}.
\]

GTと弱教師を別々に正規化して `beta(t)` を正則化強度として明示する。
基準候補は `beta_max=0.3`、innerで調べる範囲は `{0.1,0.3,1.0}` の3点に限定する。
これらは文献の推奨値ではなく、セル平均BCEとbag平均ORの尺度を固定した上での探索案である。
global optimizer stepをt、最初の5epoch相当のstep数をT_rampとして、
`beta(t)=beta_max*min(t/T_ramp,1)` とする。GT-onlyでは全期間beta=0。
最初のstepからGTと弱症例を同じbatchへ入れ、独立したGT teacherの事前学習は行わない。

GT・ORは同じqへ作用し、どちらからもregion FPN・BiLSTM・classifierを更新する。
ORを補助損失として追加すること自体が正則化の仮説であり、head detachはその必要条件ではない。
GTとORのregion専用部への勾配ノルム・内積を固定train診断集合で別々に記録する。
GT勾配が小さくなる終盤の比率発散を考慮し、絶対ノルムも併記する。
以前の「弱勾配をGTの25--50%にする」案は、正しい位置への更新や過学習抑制を保証せず、
GT勾配自体が過学習へ向かう可能性もあるため、初期係数の選択規則には採用しない。

追加比較のdetachでは `z_ir=stopgrad(w)^T h_ir+stopgrad(b)` を弱項だけに使う。
ただしwが0でなければ、`h_ir -> h_ir+c*w` によりlogitは `c*||w||^2` だけ上がる。
十分な容量を持つ特徴抽出器ではこのような共通変化が可能で、head固定だけでは全高値化を防げない。
GTだけがheadを更新しても特徴はORで変わるため、出力の意味や較正が固定されるとは言えない。
この勾配制限を直接支持する先行例は確認できておらず、主比較の後に検証する。

`lambda`はwholeとregionの相対重みである。第一pilotでfold-matched Baseline 0 trunkを
固定する場合はwhole側を再学習せず、`lambda`は不要になる。joint fine-tuningへ進む場合だけ
innerで少数候補を事前に決め、outer結果で調整しない。region既知GTにはplain BCEを使い、
wholeの既存重み付けをそのままregionへ転用しない。

| bagの状態 | 直接GT | 弱教師 |
|---|---|---|
| 陰性椎体 | regionにはなし | なし |
| 完全GT陽性 | 4セルの0/1 | なし |
| GTなし陽性 | なし | 4領域のOR |
| 部分GT・確認陽性あり | 既知セルだけ | なし。陽性の存在はすでに確認済み |
| 部分GT・確認陰性だけ | 既知陰性セル | 未知領域だけのOR |
| 完全GTが全部0なのに椎体陽性 | 学習へ入れる前に矛盾として調査 | ORで無理に解消しない |

現行データのpartial 33件は「確認陽性あり」なので、未知89セルはBCEもORも受けない。
例えば `[1, ?, ?, ?]` に対して未知3領域へORを掛けると、「ほかにも骨折がある」という
存在しない教師を追加してしまう。完全未注釈1,064件には4領域のORを掛ける。

独立Bernoulliを作業モデルにすると、既知陽性がある場合の観測事象の尤度は既知セルの積。
既知陽性がない場合だけ、その積に未知領域のOR確率が掛かる。
上記のsum BCE＋ORはこの分解から得られるが、§4の条件付き確率の問題は残る。
さらに本案の別平均とbetaによる重み付け後は、その観測尤度をそのまま最尤推定する目的ではない。

### 損失の数と勾配をそろえて考える

outer 0 trainの既知GTは596セル、未注釈陽性は643袋。
全q=0.5の仮想初期状態では、既知BCEの総和は413.116、弱ORの総和は41.498。
袋数が多いだけで弱教師が損失・勾配を必ず支配するとは言えない。
逆に、この損失値の比から共有encoderへの勾配比を決めることもできない。
将来の診断ではGT項とOR項の値・region専用表現への勾配ノルムを別に記録する。
別平均は弱項の母数が増えても自動的に強くならない反面、弱症例1袋とGT 1セルの尺度は違う。
betaはinnerのGT指標で選び、勾配比は弱項の影響を説明する診断として扱う。

### 空maskと観測範囲

空maskの面は系列から除き、全有効面がないbag×regionは予測無効として扱う。
GT陽性なのにその領域が観測できない場合はQC対象とし、0を教師にしない。
未知4領域のうち1領域が見えないとき、見える3領域だけにORを強制すると誤教師になり得る。
未知候補をすべて観測できることを確認できないbagでは、その弱項を除外して件数を記録する。

4領域maskが非空であることと、骨折所見が選択された15面に含まれることも同一ではない。
2026-09-10にユーザーが15面への所見被覆と領域maskによる被覆に問題がないことを確認した。
本設計では確認済みの前提として扱い、追加の目視確認を実装開始の条件にしない。
これはユーザーの確認に基づく前提であり、今回新たに画像監査を実施したという意味ではない。
maskは解剖学的領域を表し、骨折のsegmentation GTではない。

### 将来実装する場合の数値契約

ORはfloat32以上で `a=sum(logsigmoid(-z))` を求め、
`-log(1-exp(a))` を安定なlog1mexpとして計算する。
sigmoidをBF16のまま掛けて1から引く実装は避ける。
未知targetのNaNはloss計算より前に除外し、計算後に0を掛けて処理しない。

## 7. ORがどのように失敗するか

以下は論文の実験結果ではなく、4領域のnoisy-ORを直接計算した例である。

| q | OR値 | 弱損失 |
|---|---:|---:|
| (0.25, 0.25, 0.25, 0.25) | 0.683594 | 0.380391 |
| (0.5, 0.5, 0.5, 0.5) | 0.937500 | 0.064539 |
| (0.8, 0.8, 0.8, 0.8) | 0.998400 | 0.001601 |
| (0.01, 0.01, 0.01, 0.99) | 0.990297 | 0.009750 |

全領域を高くしても、1領域に押し付けても、弱教師にはよく適合できる。
`A=prod(1-q_r)` とするとlogitへの勾配は

\[
\frac{\partial\ell_{weak}}{\partial z_r}=-\frac{Aq_r}{1-A}.
\]

すべてを上げる向きであり、高いqほど大きい勾配を受ける。
全q=0.5では各領域の勾配は−0.033333、全q=0.8では−0.001282に小さくなる。
局所特徴を使っても、headのbiasだけで定数出力に近づく経路は残る。

GT内の陰性が上昇を抑えるが、未注釈画像での一般化は未検証である。
whole lossが正常／骨折の特徴を共有trunkに学習させても、領域ラベルが自動的に得られる
わけではない。このため、weak OR lossの低下を局在成功の判定にしない。

### 条件付き確率に整合する比較候補

真の条件付き周辺確率なら、陽性bagで
`sum_r q_r = E[陽性領域数 | X,Y=1] >= 1` が必要である。
したがって比較候補は次の弱い制約とする。

\[
\ell_{bound}=[\max(0,1-\sum_{r\in U_i}q_r)]^2.
\]

未知候補だけに適用できるのは、確認陽性がなく、既知領域が陰性と確定している場合。
`q=(0.25,...,0.25)`を不当に罰しない一方、全領域を高くする解も許す。
これはOR尤度そのものではなく、条件付き周辺確率の必要条件に基づく正則化である。
較正を重視する場合の理論上の利点と、弱い制約ゆえの学習信号不足を比較する。

この必要条件の導出は、4個の周辺確率をすべて同じ情報Xに条件付ける前提を持つ。
厳密に局所情報だけを見るheadの出力を `P(R_r=1 | f_r(X),Y=1)` と解釈すると、
領域ごとに条件付けが異なり、各bagでの `sum(q)>=1` は保証されない。
その場合、boundも追加の仮定に基づく正則化であり、各局所確率の較正を保証しない。
真の局所条件付き確率であれば、同じ陽性母集団全体で平均した和には下限1が成立するが、
これは症例ごとの局在を決める情報にはならない。
また、全q=0.5ではboundの損失・勾配はすでに0であり、ORと同じ強さの学習信号ではない。

15個の非空ラベル組合せ上に正規化した厳密な条件付きモデルも考えられるが、
未注釈陽性の許される状態集合が全15状態なので、その粗ラベルの尤度は常に1となる。
結果として弱損失は0であり、今回求める未注釈症例からのregion学習信号にはならない。
構造制約だけでは未知領域の正解を生成できないことを示す比較として位置付ける。

## 8. 学習プロトコルの提案

GTと未注釈陽性は同じ学習期間に混ぜるが、完全GTだけでregion teacherを収束させ、
その予測を正解として広げる段階は置かない。初期案ではCAM、hard pseudo-label、
固定cardinality、分布matchingを追加しない。

ORの正則化効果を最も解釈しやすくする第一pilotは次とする。

1. 現行のpatient-grouped nested 5foldを維持する。3 train / 1 inner / 1 outer。
2. fold-matched Baseline 0のCNN trunkを初期特徴抽出器として使い、まずweightとBatchNorm統計を固定する。
3. region更新は陽性bagだけで作り、train陽性を各epochに1回だけ使う。GTだけをoversamplingしない。
4. 各batchへannotatedとunannotatedを両方割り当て、GTがclassifierの意味をanchorした同じupdateでORを掛ける。
5. `beta`を0からrampし、GT・ORの両方から全region parameterを更新する。
6. GT-onlyとGT＋ORで初期化、batch割当、augmentation、epoch上限、checkpoint規則を固定する。

outer 0 trainは陽性802件中159件がannotatedである。positive-only batch 16なら、比率を変えず
各bagを1回ずつ割り当てても、51 batchへ159件を3--4件ずつ配置できる。
小さな末尾batchを作らず15--16件へ再配分し、GTを反復せず各updateにGTと弱症例を共存させる。
QCで弱教師が無効になる症例がある場合は、その除外後にbatch数・教師件数を再計算する。
これは全8,074件のnatural batchで約72.7%のstepにGTがない問題を避ける。

trunkを固定する第一pilotではencoder全体をeval状態に保ち、weight・BatchNorm統計に加え
Dropout/DropPathも固定特徴抽出器として無効化する。既存のBNだけevalへ戻す凍結helperを、
この契約を満たすものとして無条件に流用しない。positive-only forwardによるBatchNorm汚染が起きず、whole性能の
変化もOR効果へ混ざらない。局所表現の容量不足が確認された場合に限り、次の段階でtrunk最終blockを
解凍する。その場合はBatchNorm統計を固定するか、現行v7のsingle natural-batch forwardを使い、
陽性だけの第2 encoder forwardで統計を更新しない。

### ミニバッチとreduction

outer 0のnatural train 8,074件中、annotatedは159件。
独立抽出近似ではbatch 16の約72.7%にannotated bagが含まれない。
これは学習を否定する値ではないが、毎stepのGT lossを独立に正規化したり、
GTなしstepを「GT平均0」としてfold平均へ混ぜたりしない。

regionのfold objectiveは `L_GT + beta(t)L_OR` とし、QC後のtrain集合で既知GTセル数G、
弱bag数W、1epochのoptimizer step数Kを固定する。各batch Bのbackward用損失は

\[
L_B=\frac{K}{G}\sum_{i\in B}\ell_{GT,i}
 +\beta(t)\frac{K}{W}\sum_{i\in B}\ell_{weak,i}.
\]

parameterとbetaを固定して全batchのL_Bを平均すれば上記fold objectiveと一致する。
optimizerでparameterを更新しながら1epochを回すため、1回のfull-batch更新との等価性は主張しない。
batchの実件数や途中までの累積件数を分母にして勾配尺度を変えない。
epochログではGT損失総和/Gと弱損失総和/Wを計算し、ramp中の重み付き和とは別に報告する。
G=0では本実験は成立せず停止、W=0ならORを無効化してGT-onlyとして扱う。
annotated／unannotatedの並び順だけを層化し、出現回数と母集団比率は変えない。

第一pilotの最適化案はAdamW、region LR 2.3e-4、weight decay 1e-4、75epoch上限、
cosineで最小学習率2.3e-5、seed `{20260807,20260808,20260809}` とする。
LR・上限・weight decayは既存region設定を参照した初期値であり、凍結trunk条件での最適値ではない。
早期停止によるGT露出差を避けて各armを同じ上限まで回し、inner macro AP最大のcheckpointを選ぶ。
同点では最も早いepochを選び、outerは係数・epoch・seed選択に使わない。
GT-onlyにも同じbatchの全陽性をforwardし、OR項だけを0にする。
これにより初期化、GT露出、optimizer step、乱数消費、正規化層が見る症例を可能な範囲でそろえる。

追加候補として、同じ弱陽性の2つの解剖学的に妥当なaugmentation間で4-vector `q`を揃える
EMA consistencyを検討できる。ただし定数出力でも満たせるためORの代替にはならず、最初のA/Bへ
同時投入しない。OR単独の寄与を確認後、係数を0からrampする独立armとして追加する。

### データ拡張

CT・whole mask・region maskを同期変換し、移動したmaskの領域IDは保持する。
初期比較では全視野を保つflip/transposeなどを中心にする。
未知の骨折箇所を消し得るcrop/Cutout、異なるbagを混ぜるMixUpは、
region教師の論理を維持する方法が定まるまで初期案に含めない。
whole側の比較も同じaugmentationにそろえ、拡張変更の効果をORの効果としない。

## 9. 評価とモデル選択

本案はregion GTを学習に使うため、268 bags全部を「外部の完全未使用holdout」とする
過去のpseudo-only案とは別のプロトコルになる。
各GT bagの評価予測は、そのstudyをtrainにもinnerにも含めないouter modelから取得する。
単に5modelを平均して全268件に当てると、GTを見たmodelが混ざるため不適切である。

| 評価対象 | 推奨指標 | 主な解釈 |
|---|---|---|
| outerの完全GT陽性235件をOOF集約 | R1–R4 AP/AUROC、macro AP、BCE/Brier | 主たる条件付き局在性能 |
| partial GT33件の観測セル | 有効GTに限った補助指標と件数 | 全4領域の精度とは区別 |
| 全椎体のwhole OOF | AUROC/AP、whole loss | whole分類への影響 |
| 完全GT陽性＋whole陰性 | `p_whole*q_r` のAP/AUROC | end-to-end領域検出。未知陽性は除外 |
| 未注釈陽性 | q分布、sum(q)、全高値率、R4最大率、領域間相関 | collapse診断。正解率ではない |

partial GTの未確認セルは評価時にも陰性扱いしない。
end-to-end評価でもmask無効セルを無理に0スコアとして埋めず、評価対象数・被覆率を報告する。
全高値率などの診断閾値は事前に固定する。診断だけでモデルを有効と判定しない。
AP/AUROCの信頼区間と差の比較はstudy単位のpaired bootstrapを候補とする。
椎体レベル別の症例数・注釈被覆と性能も確認し、C7などの構成比の違いを隠さない。
単一クラスしかない小集団のAUROCは未定義として扱う。

region checkpointはinnerの完全GT上のmacro APを主選択候補とし、BCE/Brierを較正診断にする。
理由はORが高くても局在の正しさを判断できず、hard BCEは順位を保ったまま過信だけで悪化し得る
一方、目的がregionの識別だからである。これは混合training lossと意図的に異なる選択基準であり、
AP/AUROC、BCE/Brier、GT lossとOR lossの曲線を併記する。inner完全GTは43–50件なので
不安定さは残る。§8の固定上限・macro AP最大・最早同点規則を全armに適用し、outerで選び直さない。

第一pilotではwhole pathを固定するため、armごとのwhole checkpoint選択は行わない。
joint fine-tuningへ進む場合だけwholeをinner whole lossで独立選択する。
同じreductionによるvalidation total lossも記録するが、total低下だけをregion改善と呼ばない。
比較arm間で評価用の選択規則を統一し、都合のよいcheckpointだけを採用しない。

## 10. 最小の比較実験と判断順序

| arm | 局所特徴 | region GT | 未注釈陽性のregion教師 | 目的 |
|---|---|---|---|---|
| A: local GT | 同一のmask-local構造 | 既知セルBCE | なし | 弱教師なしの基準 |
| B: local GT＋OR | Aと同じ | Aと同じ | rampしたnoisy-OR。全region parameterを更新 | 弱教師追加の主比較 |
| C: local GT＋detach OR | Aと同じ | Aと同じ | Bと同じ係数・schedule、弱項だけheadをdetach | 勾配制限の追加比較 |
| D: local separate heads＋OR | Bの最終Linearだけ4個に分離 | Bと同じ | Bと同じ | shared headの制約の影響 |
| E: global separate heads＋OR | Dの4領域poolを共通全体poolへ変更 | Dと同じ | Dと同じ | D/Eで局所読み出しの影響 |

第一pilotでは全armが、全椎体で学習済みの同じfold-matched Baseline 0 trunk/whole pathを
固定して使う。したがってA/B差は未注釈陽性からregion表現へ与える弱教師だけである。
後にjoint fine-tuningする場合もwhole側のbag、loss、更新回数をarm間でそろえる。
Eは提案モデルではなく、局所性の有用性を測る比較用モデルである。

最初はA/Bを同じ初期化、seed、batch順序、拡張、epoch上限で比較する。
beta=0.3を基準候補とし、係数を選ぶ場合は各outerのinnerだけで3候補を比較する。
各候補について3seedの選択epochでのinner macro APを平均し、平均APが最大の候補のbetaをそのouterの
全seedへ共通採用する。同点は小さいbetaを優先する。outer 0の結果を見て別outerの設定を変えない。
固定beta=0.3の比較とinnerでbetaを選んだ比較を区別し、都合のよい方だけを報告しない。
C以降は主比較と独立した探索とし、結果を見て新設計を選んだouterを未使用testとは呼ばない。

以前の「shared headのまま全領域を同じglobal poolingへ変更する」比較は取り下げる。
入力系列・有効面も同じなら、共有BiLSTMと共有Linearは4個とも同じqを出し、比較として退化する。
局所性を調べる場合はD/Eのように最終4headを両armでそろえ、poolだけを変更する。
さらにbagごとの有効面集合がD/Eで違わないようそろえ、対象件数も報告する。
`sum(q)>=1`やEMA consistencyは初期比較から外し、ORの挙動を確認した後の候補として保留する。

さらにGT量の効果を調べる場合は、train内の完全GTをstudy単位で間引き、
隠したregionラベルを「未知陽性」として扱う。inner/outerのGTは固定する。
25%/50%/100%などの注釈予算は事前に固定し、全armで同じ症例集合を使う。
これにより少量GTとの組合せという主張を、単なる混合lossの比較より直接検証できる。
同じtrain画像をGT-onlyにも提示するため、画像数やencoder事前学習の差を弱教師の効果としない。
隠したGTを診断に使う場合も、訓練・係数選択・checkpoint選択には戻さない。

採用判断では、BがAを上回るGT上の局在性能、whole性能の変化、seed間の変動を確認する。
OR lossの低下、R4だけでの改善、未注釈集合の見栄えだけでは採用しない。
改善がなければ係数やpseudo処理を次々と足す前に、局所性、入力での所見被覆、
ラベルの選択バイアス、region head容量を調べる。
正確な非劣性幅・最小改善幅は研究の評価基準として実験前に決める必要があり、
根拠のない数値を本書では性能保証として置かない。

正則化の仮説は「未使用GTへの汎化改善」で判定する。Bのtrain GT lossがAより高くても、
inner/outer APが良くなれば目的に合う。train-validation差が小さくなるだけでは、両方が悪化する
underfitでも成立するため十分ではない。AP向上のみなら識別の改善、BCE/Brierも良くなれば
過信の軽減、と分けて記述する。qの平均・標準偏差・sum(q)・全q>0.8率・各領域argmax率を
各epochで記録し、OR低下と同時の定数化を確認する。未知群ではこれらを精度指標と呼ばない。
study単位のpaired bootstrapを用い、OOF集約とfold別・seed別の差を併記する。

## 11. 設計後の実装順序と検証事項

以下は将来の作業計画であり、今回の実装開始を意味しない。

1. 教師契約を固定する。既知／未知／mask有効性、ORとbound、reduction、選択指標を決める。
2. 新規packageに局所modelとlossを実装する。既存pooling・whole経路などを再利用する。
3. 部分GTと損失を合成データで検証する。陰性bagへのregion勾配0、partial未知セルへの
   誤教師0、known-positive時の追加ORなし、既知陰性をOR候補から除く動作を確認する。
4. 単一encoder forward、空系列、推論時GT不要、極端logit、mixed precisionを検証する。
5. 学習・評価を接続し、小規模動作確認後に事前定義したA/B比較を行う。

変更範囲は新規package・そのtests・設計参照に限定する案とする。
学習入力manifestや既存checkpointを上書きせず、実験出力には別の識別子を付ける。
問題があれば新規armの利用を止め、既存Baseline 0／region_branchの実行経路へ戻せる構成にする。

2026-09-10のレビュー用第一候補は「凍結Baseline 0から局所特徴を読み、共有BiLSTMと
共有Linearを既知セルBCE＋rampしたORで通常更新する」方式である。
第一出力は陽性椎体内の局在スコア、主評価はmacro AP。較正済み確率とは呼ばない。
面集約・容量・係数候補・ミニバッチreduction・checkpoint選択を本書で具体化した。
15面被覆とmaskによる所見被覆はユーザー確認済み。残る設計判断はtrain分割とcheckpointの対応、
CNNを凍結する比較とCNNも更新する比較のどちらを主実験とするかである。
