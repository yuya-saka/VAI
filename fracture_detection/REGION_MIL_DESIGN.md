# 4領域の骨折出力からwholeを判定するMILモデル

> **2026-09-12 v2 amendment:** 実装済みv1のouter0結果を受け、ユーザー指定により
> region-to-whole集約を正規化logit-LSEへ変更する。既定τ=0.5、batch構成は
> N/A/U=8/4/4。N/Aは従来どおり4領域BCE、UはLSE whole logitへの陽性BCEとする。
> `loss.whole_aggregation` と `loss.lse_temperature` で設定可能にし、noisy-ORも比較用に
> 残す。完了済みv1から独立したv2実験ではearly-stopping patienceも設定可能とし、
> 現行値は15 GT-passとする。この追補は以下のnoisy-ORおよび4/4/8記述に優先する。

更新日: 2026-09-10。状態: ユーザーが基本設計を採用。実装済み（Arm B のみ）、学習は未実施。

本書が新モデルの現行設計である。ユーザーの2026-09-10の指定により、
`CONDITIONAL_MIL_DESIGN.md`の陽性限定・並列whole経路・凍結CNN案を置き換える。
実装は `fracture_detection/weak/` に配置した（本書は当初 `region_mil/` としていたが、
ユーザー指定により `weak/` が正式な配置先である。設計内容自体の変更はない）。
局所4領域からnoisy-ORでwholeを導く構造、全4領域GT/弱陽性/陰性の混合教師、
CNN fine-tuning、陽性中心samplingを採用する。今回の依頼は設計の再整理であり、実装開始の指示ではない。
4/4/8・beta=1などは採用設計を検証する初期設定であり、実測済みの最適条件ではない。
実装は Arm B（提案モデル）のみで、Arm A/C の切替機構は作っていない。
augmentationはbaseline0の凍結設定をそのまま使うが、MixUpは構造的に適用不可のため
常に無効（理由は `fracture_detection/weak/README.md` 参照）。詳細は同READMEを参照。

## 1. 確定した要件と推奨構成

- CNNをfine-tuneして4領域の骨折を学習する。
- whole判定は4領域出力から計算する。別のwhole classifierやglobal-feature bypassを作らない。
- 陰性椎体もregion経路へ入力し、4領域すべてが陰性という情報を学習する。
- 陽性・region GTありは全4領域の0/1を直接学習し、GTなし陽性は少なくとも1領域の存在を学習する。
- GTあり椎体には未注釈領域がなく、0は骨折なしを意味する。ユーザーが2026-09-10に明示した契約を優先する。
- 各領域スコアは、その領域maskでpoolした局所特徴から計算する。
- 選択15面への骨折所見被覆と領域maskによる被覆は、ユーザー確認済みの前提とする。
- 陰性・GTあり陽性・GTなし陽性を層化samplingし、batch内の陰性を少なめにする。
- fine-tuningの目的に合わせて損失を設計する。Baseline 0のpos_weight・sampling分布への一致は求めない。

推奨構成は、4個のsigmoid出力を固定のnoisy-ORで集約し、教師の観測範囲に応じた損失で
CNN・region経路を同時にfine-tuneする方式である。GTあり陽性は全4セルBCE、GTなし陽性は
OR、陰性は4領域の陰性BCEの和で学習する。3群samplingをそのまま学習目的に反映し、
元の陰性多数の分布へ戻すimportance補正と追加pos_weightは使わない。
以下は今回の条件で優先して検証する設計判断であり、性能最適性を実測済みとはしない。

## 2. 一方向のモデル構造

```text
15面 × (2.5D CT 5ch + 椎体mask 1ch)
    ↓ fine-tuneするEfficientNetV2-S
    ↓ stage features → stride-4 FPN
    ├─ R1 mask pooling → R1面系列 → 共有BiLSTM → 有効面平均 → h1
    ├─ R2 mask pooling → R2面系列 → 共有BiLSTM → 有効面平均 → h2
    ├─ R3 mask pooling → R3面系列 → 共有BiLSTM → 有効面平均 → h3
    └─ R4 mask pooling → R4面系列 → 共有BiLSTM → 有効面平均 → h4
                    ↓ 各h_rへ共有Linear(1)を個別適用
                    z1, z2, z3, z4
                    ↓ sigmoid
                    q1, q2, q3, q4
                    ↓ 固定noisy-OR
                    p_whole
```

ここで「共有」は重みを共有する意味であり、4領域の入力特徴を混ぜる意味ではない。
whole専用BiLSTM/headは置かず、全損失がこの1本の経路を通る。
region内の面集約を終えてから4領域を集約する。15面×4領域の60個をORする構造ではない。
複数領域が同時に骨折し得るため、4出力にはsoftmaxを使わない。

初期容量案はFPN 256ch、共有BiLSTM hidden 128・1層・双方向、Dropout 0.30、
共有Linear(256,1)。最終headにはBatchNorm・領域ID embeddingを置かない。
領域ごとに所見の見え方が違うため、共有headが不十分なら4個のLinearを追加比較する。
これはwhole/regionの並列分岐を意味しない。

面s・領域rのpoolingは、one-hot maskをarea poolingで特徴解像度に合わせ、

\[
f_{sr}=\frac{\sum_{u,v}\widetilde M_{sr}(u,v)F_s(u,v)}
 {\sum_{u,v}\widetilde M_{sr}(u,v)+\epsilon}
\]

とする。maskラベルIDをbilinear補間しない。有効面だけを元の順序で系列化し、
元の面indexは保存する。詰めた系列で物理的間隔を表現しない限界は記録する。
mask poolingは読み出し位置を限定するが、CNNの受容野・FPN・GroupNormを通じた
領域外情報まで遮断するものではない。入力での所見被覆とは別の論点である。

## 3. 出力の意味と集約

\[
q_r=\sigma(z_r),\qquad
p_{whole}=1-\prod_{r=1}^4(1-q_r).
\]

qは陰性椎体を含む集団での領域骨折スコアであり、旧案の陽性条件付きスコアではない。
領域検出にはqをそのまま使う。`p_whole*q_r`を再度掛けず、wholeとregionで別epochの
parameterを組み合わせない。推論時の入力にGTのwhole陽性ラベルは不要。

noisy-ORは、独立Bernoulliを作業モデルにした少なくとも1領域の陽性確率である。
実際の領域骨折は相関するため、真の周辺確率からこの式でwhole確率が厳密に得られる
とは主張しない。ラベルの独立性が実証されたわけではなく、qとpの較正を別々に評価する。

この集約には学習可能なweight/biasを置かない。全q=0ならp=0、いずれかのq=1ならp=1、
各qに対して単調増加という構造を保つ。

注意: softな集約なので、全q=0.2でもp=0.5904となる。
`p>0.5`と「いずれかのq>0.5」は同じ判定ではない。初期評価は連続スコアのAP/AUROCを使い、
二値化する場合はinnerでwholeとregionの閾値規則を事前に決める。
両者の二値判定の完全一致が必要なら、`p=max(q)`またはregion閾値判定の論理ORを
別の出力契約として検討する。noisy-ORで学習した後に無説明でmaxへ切り替えない。
maxはwhole勾配が最大領域に集中するため、初期の第一候補はnoisy-ORとする。

## 4. 教師の契約

2026-09-09に確認したmanifest集計を参照する。新しい学習を始める際は入力versionを固定する。

| 椎体の状態 | 件数 | 利用する教師 | 訓練損失 |
|---|---:|---|---|
| whole陰性 | 12,100 | 4領域がすべて0 | 4領域の陰性BCEの和。whole陰性BCEと同値 |
| whole陽性・region GTなし | 1,064 | 少なくとも1領域が1 | whole陽性OR損失 |
| whole陽性・region GTあり | 268 | 全4領域の0/1 | 4セルBCEの和。whole項は重ねない |

ユーザー確認により、GTあり椎体には領域単位の未注釈は存在せず、全4セルを教師とする。
従来の「235完全・33部分」「983既知・89未知」という集計は旧validity処理によるもので、
今回の注釈の意味を表す区分として撤回する。上記bag数を維持するなら268×4=1,072セルが教師になる。
既存 `baseline0/data/region_validity.py` はannotation runの完了状況により0を無効化するため、
新モデルへそのまま流用しない。実装時はGTありflagで全4セルを有効にし、入力versionと件数を再確認する。
本設計更新では既存コード・manifest・過去の評価値を変更または再計算していない。
whole陰性の4個の0は、個別の領域注釈ではなくwholeラベルから論理的に得られる教師である。

全領域が観測可能であることを前提とする。実行時の空maskを黙ってq=0として埋めない。
全体として領域が観測できない症例は無効として件数を報告し、whole/regionのloss対象を明示する。
whole陰性と確認陽性region GTが併存する場合や、whole陽性の完全GTが全0なら入力矛盾として扱う。

## 5. whole lossがregionに与える教師

1袋のwhole損失を `ell_whole=BCE(Y,p_whole)` とする。

陽性の場合は、従来検討していた存在制約そのものになる。

\[
\ell_{whole}^{+}=-\log\left(1-\prod_r(1-q_r)\right).
\]

陰性の場合は、4領域の陰性BCEの和と厳密に一致する。

\[
\ell_{whole}^{-}=-\log(1-p_{whole})
 =-\sum_r\log(1-q_r)
 =\sum_r\operatorname{BCEWithLogits}(z_r,0).
\]

ここでの等式はnoisy-ORの定義から成立し、実データの独立性を証明するものではない。
陰性lossの各logitへの微分はq_rとなり、陽性GTがなくても全領域に直接下降方向の勾配が流れる。
したがって `whole BCE + 4領域の陰性BCE` を同時に加えると、同じ陰性情報を二重に数える。
陰性BCEを4領域平均に置き換えるとwhole陰性lossの1/4になり、同じ目的ではなくなる。

陰性症例は全症例への定数高値を抑えるが、「陰性では全部低い、陽性では全部高い」という
解までは排除しない。陽性症例内の正常領域を教えるすべての0のGTが引き続き重要である。
CNNまでfine-tuneするため、これらすべての教師が局所特徴の学習にも作用する。

## 6. 採用を推奨するmixed-supervision目的

ユーザーがBaseline 0への一致要件を外したため、前案のimportance sampling補正、
pos_weight=2、全症例whole BCE＋追加GT BCEという目的を置き換える。
今回の主目的は陽性椎体内の領域判別であり、少数の陰性を大きく再重み付けして
元の陰性多数の目的へ戻す必要はない。

### batch構成とsampling

第一候補は、batch 16で陰性Nを4、GTあり陽性Aを4、GTなし陽性Uを8とする。
同じupdateに正常例・局所の正解・多様な弱陽性を含め、75%の入力を陽性椎体に使う。
4:4:8は今回の初期提案値であり、文献から得た最適比率ではない。

| 群 | 件数 | lossに与える教師 |
|---|---:|---|
| N: whole陰性 | 4 | 全4領域の0 |
| A: GTありwhole陽性 | 4 | 確認された領域の0/1 |
| U: GTなしwhole陽性 | 8 | 4領域の少なくとも1つが1 |

A群は全4領域GTのある陽性椎体であり、群内はbag一様samplingとする。
N群の陰性椎体と、A群に含まれる正常領域のGTは区別する。
A群の陰性GTもすべて使う。初期案には患者/椎体レベル/領域頻度による追加samplingを重ねない。

### 1袋に、実際に観測された教師の損失を掛ける

GTありbag iの全4領域のGTをt_irとする。
以下のsum BCEは、4領域全体で1つのjointラベルを観測したとする作業モデルに対応する。

\[
\ell_i=
\begin{cases}
\sum_{r=1}^4\operatorname{BCEWithLogits}(z_{ir},0), & i\in N\\
\sum_{r=1}^4\operatorname{BCEWithLogits}(z_{ir},t_{ir}), & i\in A\\
-\log\left(1-\prod_{r=1}^4(1-q_{ir})\right), & i\in U.
\end{cases}
\]

全損失は袋数bで平均する。

\[
L_B=\frac{1}{b}\left(
 \sum_{i\in B\cap N}\ell_i+
 \sum_{i\in B\cap A}\ell_i+
 \beta\sum_{i\in B\cap U}\ell_i
\right).
\]

初期値beta=1。pos_weight・source別の逆頻度weight・importance補正・追加GT係数lambdaは置かない。
陰性bagにもA群にも全4セルBCEの和を使う。GTありbag内の0を無効化しない。
領域BCEだけ4で割るとweak ORとの相対強度が変わるので、このreductionを固定する。
完全4セルと粗い存在ラベルでは情報量が違うため、1袋当たりのloss/勾配が等しくなるとは考えない。
変更が必要な場合はbetaだけを {0.5,1,2} の範囲でinner評価する。sampling比率とbetaを同時に探索しない。

この設計には学習前の校正phaseを設けない。CAM確率校正、疑似ラベル生成、
loss間のgradient normによるlambda校正、calibration artifact/versionは不要である。
beta=1は校正で推定する値ではなく、最初の比較実験で固定するhyperparameterとする。
必要になった場合もinner data上の離散候補比較として扱い、専用calibration CLIは作らない。

4:4:8の標準batchでは、目的は
0.25 E_N[ell] + 0.25 E_A[ell] + 0.50 beta E_U[ell]
である。意図的に変更した訓練分布上の目的であり、元データ分布での最尤推定とは異なる。
各項の値の大小だけで教師の強さを判定せず、source別lossとCNNへの勾配ノルムを診断する。
初期q=0.5では弱ORは約0.0645、全0の4セルBCEは約2.7726であり、
「弱袋が半数なので弱lossが半分を占める」とは言えない。

### GTあり陽性にwhole陽性損失を重ねない理由

2026-09-10追記: ユーザーの「GT陽性の領域だけを使って椎体分類する」という問いに対し、
領域単独の陽性判定と、GT陽性領域をまとめたrestricted ORを区別する。
GT陽性集合Pが1領域なら、restricted ORはそのq_rそのもので、陽性BCEは直接領域BCEと一致する。
Pが複数領域なら、restricted ORへの陽性BCEは「Pのどれか1つが陽性」で足りる。
ユーザーは後者、各陽性領域の単独BCEを足す方法を選択した。
陰性領域には別途0のBCEを与える。GTは訓練の損失選択にのみ使い、
推論時は全4領域をORする。restricted ORだけへの置き換えは採用しない。
確認済み0の領域も、その領域mask内の特徴からq_rを出してBCE(q_r,0)で学習する。
椎体全体が陽性でも、局所の0をwholeラベル1で上書きしない。0の領域を入力や損失から除外もしない。
GTあり椎体の0はすべて骨折なしであり、旧validityを理由に直接BCEから除外しない。
なお、この「領域だけ」は局所特徴の読み出しを指し、領域外の画素情報を完全に遮断する保証ではない。

完全GTが [0,1,0,0] なら、領域BCEがq2を上げ、q1/q3/q4を下げる。
q2が上がれば、同じ4出力から計算するp_wholeも上がる。
この袋へ追加ORを掛けなくても、全体の陽性情報はGTの中に含まれている。

独立Bernoulliを作業モデルにすると、全4領域GTの観測確率は
4セルのBernoulli確率の積であり、その負の対数が上記の4セルBCEの和である。
追加ORは新しい情報を与えず、全qを上げる方向の勾配を足すので初期案から除く。
これはGTから得られる領域の0/1を優先する設計判断である。

領域内の部分注釈を仮定した損失分岐は設けない。GTありとGTなしを椎体単位で区別する。
GTが全0なのにwhole陽性なら、入力矛盾として除外/調査する。

陰性Nではwhole陰性BCEと全4セル陰性BCEが同じ情報なので、どちらか一方だけを計算する。
訓練中にwholeスコアはすべてのbagで計算できるが、訓練損失として重ねる必要はない。
GT・陰性・弱ORのすべてが同じregion経路とCNNへ勾配を流す。head detachは使わない。

### GTを繰り返し過ぎないstream管理

A群を1周する期間をGT-passと呼び、同じGT袋を1pass内で反復しない。
N/U群はshuffleしたqueueを次passへ引き継ぐ。同じ少数陰性ばかりを使わない。
outer 0 trainではA=159、U=643、N=7,272。
4:4:8を39step、最後を4:3:8の1stepにすればGT159、弱320、陰性160を提示する。
最終stepでは実袋数15でlossを平均する。そのstepだけ群比率が少し変わることを明記し、
その影響を隠すためのGT重複やimportance補正は行わない。

各passのGT出現数、N/Uの一意症例数、queueの周回数を記録する。
1 GT-passはBaseline 0の自然分布1epochとは違う。
GTの反復が増えれば依然過学習し得るため、samplingだけで正則化が保証されたとはしない。

## 7. 初期化とfine-tuning

1. patient/study-grouped nested 5foldの3 train / 1 inner / 1 outerを維持する。
2. 対応するtrain 3foldで学習したBaseline 0 checkpointからCNN trunkのみを転送する。
3. 旧whole head・whole BiLSTMは新モデルへ接続しない。FPN・region BiLSTM・headは新規初期化。
4. CNNを含めて最初からfine-tuneする。初期LR候補はCNN 2.3e-5、新規部 2.3e-4。
5. §6の3群samplerで4/4/8を基準にbatchを構成し、CNNを1回だけforwardする。
6. 全bagに同じregion経路を適用し、N/A/Uごとに観測に対応した損失を計算し、1つのbatch lossとしてbackwardする。

新規region BiLSTMはFPN入力次元が旧whole BiLSTMと異なるため、そのweightを直接転送しない。
陽性75%のsamplingと少量GTの反復でCNNのBatchNorm統計まで変える要因を抑えるため、
初期案ではBN running mean/varianceだけBaseline 0の値で保持する。
CNN convolutionとBNの学習可能なweight/biasは最初からfine-tuneする。CNN全体の凍結ではない。
このBN方針はBaseline 0との損失一致のためではなく、pretrained featureを安定して適応させる提案である。
BN固定でも特徴分布の変化へ追従できない可能性はあり、有効性を保証しない。
model.train()後もBNだけevalに戻す契約を検証し、陽性だけの第2forwardは作らない。
初期案は全sourceの教師を開始時から使い、GT-only事前学習・弱項ramp・hard pseudo-labelを追加しない。

optimizerの初期案はAdamW、weight decay 1e-4、batch 16、cosine scheduler。
初期の探索予算案は上限60 GT-pass、最短10 GT-pass、inner region macro APが10pass改善しなければ停止。
同点では早いcheckpointを保持する。CNN/新規部の最小LRは各初期LRの1/10とし、
cosine schedulerの終点は60 GT-pass相当のstepに固定する。
この停止規則と数値は本件の初期提案であり、最適性の根拠は未取得。
GT-pass数をBaseline 0の自然epoch数と同じ学習量として比較しない。
seedは `{20260807,20260808,20260809}` を比較armで共通にする。数値は未実測の提案値。
CTと全maskを同期してflip/transposeする。初期案は所見を消し得るcrop/Cutout/MixUpを使わない。

## 8. 評価と比較実験

単一checkpointがqとそこから計算したp_wholeの両方を出す。同じ症例の両出力は常に同じ
parameterから計算し、別モデルのwhole確率でqをgateしない。
checkpoint主選択候補はinner GTあり陽性の全4領域でのregion macro AP。whole AP/AUROCとlossも
同じepochで併記し、whole性能とのtrade-offを明示する。whole優先で選ぶ実験は別に事前定義する。
比較armには同じstep/GT露出上限、同じ停止規則、同じsource順序を用意する。
早期停止による実露出差を報告し、共通stepまでの学習曲線も比較する。
outerで停止時点や係数を調整しない。validation/testはsamplingせず全件を評価する。
whole BCEは自然分布でのplain bag BCE、region BCEは既知セル平均として集計する。
訓練のN/A/U lossもsource別に記録するが、違う母数の訓練lossと検証BCEを同じ尺度と呼ばない。

- 条件付き局在評価: GTあり陽性268袋の全4領域のOOF qを評価。旧complete flagで235袋に絞らない。陰性を大量に足して局在失敗を隠さない。
- 全椎体の骨折判定: 全bagのOOF p_wholeでAP/AUROCを評価。
- 領域検出: GTあり陽性とwhole陰性の全4領域qでAP/AUROCを評価。GTなし陽性を陰性扱いしない。
- BCE/Brierでqとpの過信も確認する。旧部分GT区分による補助評価は設けない。
- qの全高値率・領域間相関・陽性bag内argmax分布・陰性bag内の偽陽性を監視する。

samplingで訓練分布を変更するため、qやpを較正済み確率と呼ばず、閾値は自然分布のinnerで決める。
A群のannotation selectionも関係するので、単純な陽性率のlogit補正で較正できるとは仮定しない。
このinner閾値決定は、学習前の校正工程とは別である。AP/AUROCの主評価には閾値自体が不要で、
precision/recall/F1などを報告するときだけinnerで固定する。Brier/ECEは出力の性質を監視する診断であり、
初期実験でtemperature scalingやisotonic regressionを学習pipelineへ追加することを意味しない。
局在APが上がってもwholeの偽陽性が増える場合はtrade-offとして報告する。
最初からhard-negative miningを入れず、偽陽性の増加が確認された場合に陰性枠を増やす比較を検討する。

複数seedをensembleする場合も、平均qからnoisy-ORを再計算して最終wholeを出す。
`OR(mean(q))`と`mean(OR(q))`は一致しないため、後者を混ぜない。

主比較は以下とする。全armでCNNもfine-tuneし、same seed/入力/GT露出/step数をそろえる。

| arm | whole陰性 | GTあり陽性 | GTなし陽性 | 目的 |
|---|---|---|---|---|
| A: 弱陽性教師なし | 全4領域の陰性BCE和 | 既知セルBCE和 | forwardのみ、lossなし | 弱陽性ORの追加効果を切り分ける対照 |
| B: 提案モデル | Aと同じ | Aと同じ | whole陽性OR | 主比較 |
| C: region GTなし | 全4領域の陰性BCE和 | whole陽性ORのみ | whole陽性OR | 局所GTの必要性の追加比較 |

Aは陰性教師を含むため「人手GTだけ」とは呼ばない。A/Bで同じ3群samplerとbatch袋数の分母を
維持し、AではU群のOR項だけを0にする。Aの分母からU群を外してGT/陰性を強めない。
Aでも弱陽性をforwardし、入力・乱数消費をそろえる。BN統計の扱いも全armで一致させる。
初期化CNNは全train wholeラベルを既に学習しているため、A/B差は弱ラベルを初めて見た効果ではなく、
region出力へ弱陽性を直接与えてfine-tuneする追加効果である。
最初のA/Bはbeta=1で比較し、必要な場合だけ各outerのinnerでbeta候補を選ぶ。
study単位paired bootstrapとfold/seed別結果を報告する。主目的は未使用GTへの汎化改善であり、
train-validation差の縮小やOR低下だけでは採用しない。

## 9. 根拠と実装前の契約

[Li et al., CVPR 2018](https://arxiv.org/abs/1711.06373)は同じ局所予測モデルに対し、
注釈ありではpatch教師、注釈なし陽性ではat-least-one、陰性では全patch陰性を用いる。
今回の陰性込みの構造は、旧陽性限定案よりこの教師構造に近い。
ただしinstanceはpatchと固定解剖領域で異なり、論文も弱データ追加の局在改善が一様でないと報告する。
今回のGTあり/なしで損失を切り替える構造は同論文に近いが、4領域mask・sampling比・容量の
数値は本件の設計判断であり、論文の忠実な再現ではない。

[CheXseg, MIDL 2021](https://proceedings.mlr.press/v143/gadgil21a.html)はGTと弱maskの混合比を
実験し、純GT/純弱教師より混合が良い条件を報告している。sampling比を学習目的の一部として
検証する参考になるが、saliency maskと1-bit ORは情報量が違い、同論文の比率を本件へ移植しない。
[Rozenberg et al.](https://proceedings.mlr.press/v116/rozenberg20a.html)では弱データ追加により
IoU基準で8疾患中4疾患は改善、2疾患はほぼ不変、2疾患は悪化した。混合教師は成功保証ではない。

[Fang et al., 2021](https://arxiv.org/abs/2105.12430)からは、解剖maskで特徴の読み出しを限定する
考え方を参照する。4領域からwholeを導く構造や椎体単位のGTあり/なし混合は本件の設計判断である。

[Wang et al., Interspeech 2018](https://www.isca-archive.org/interspeech_2018/wang18_interspeech.html)
は系列局在でnoisy-ORの失敗を報告する。骨折4領域でmaxが優れる証拠ではないが、
noisy-ORのwhole精度だけを領域局在の成功としない理由になる。

将来の実装で検証する契約:

- whole出力に4領域以外の学習可能な経路がなく、推論にGTが不要である。
- 陰性whole BCEと4領域陰性BCEの和、およびその勾配が一致する。
- GTあり椎体では0を含む全4セルが訓練・評価対象となり、旧annotation-complete flagで除外されない。
- GTなし陽性の保存値を領域GTとして扱わず、ORだけで監督する。
- N/A/Uそれぞれのloss勾配がCNNに届き、1batchのCNN forwardが1回である。
- BN統計保持時はrunning mean/varianceが不変で、CNN/BN affineへの勾配は有効である。
- 4/4/8と最終4/3/8の件数、袋数によるreduction、GT-pass内のGT重複なしを確認する。
- GTあり陽性に追加whole ORを掛けず、全4セルBCEを使う。
- A/BでU項以外のloss尺度・入力順序が変わらない。検証は自然分布で行う。
- 空mask、入力ラベル矛盾、全正負logit、mixed precisionを扱う。

数値実装では `a=sum(logsigmoid(-z))` をfloat32以上で求め、陰性lossは直接 `-a`、
陽性lossは安定な `-log(1-exp(a))` として扱う。BF16確率の積と差をそのまま使わない。
極端logitでのunderflowも含め、値と勾配の有限性を別途検証する。
既存モデル・checkpoint・manifestは変更せず、新規実験として識別する。
