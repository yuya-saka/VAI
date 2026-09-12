# Region-MIL 実装引き継ぎ

日付: 2026-09-10

状態: ユーザーが基本設計を採用済み。実装・学習は未着手。

本ログを次セッションの開始点とする。実装中に矛盾が見つからない限り、
アーキテクチャの検討からやり直さない。詳細設計の正本は
`fracture_detection/REGION_MIL_DESIGN.md`、プロジェクト全体の設計判断は
`.claude/docs/DESIGN.md`にある。`fracture_detection/CONDITIONAL_MIL_DESIGN.md`は
置き換え済みの旧案である。

## 採用済みの設計

- 各領域の局所出力からwhole判定までを1本の直列モデルとしてfine-tuneする。
- whole専用classifier、whole専用BiLSTM、global featureへの迂回経路は作らない。
- 各領域スコアは、その解剖領域maskでpoolingした特徴から計算する。
- 入力は15面の2.5D CT 5chと椎体mask 1ch。
- fold対応のBaseline 0からCNN trunkだけを転送する。FPN、region BiLSTM、
  region classifierは新規初期化し、旧whole head・BiLSTMは転送しない。
- 4領域は同時陽性になり得るため、各出力はsigmoidとする。
- whole確率は固定noisy-OR
  `p_whole = 1 - product_r(1 - q_r)`
  で求める。
- GTの有無は椎体単位。GTあり陽性椎体では全4領域が注釈済みで、
  0は必ず「骨折なし」を意味する。
- 新モデルでは、0を無効化する目的で
  `baseline0/data/region_validity.py`を流用しない。
- 選択15面と領域maskが骨折所見を覆うことは、ユーザー確認済みの前提とする。

## 学習目的

各batchに次の3群を入れる。

| 群 | batch 16での初期件数 | 1椎体あたりの損失 |
|---|---:|---|
| N: whole陰性 | 4 | 4領域すべてtarget 0のBCE-with-logitsの和 |
| A: region GTあり陽性 | 4 | 4領域の0/1 GTに対するBCE-with-logitsの和 |
| U: region GTなし陽性 | 8 | 陽性noisy-OR損失 `-log(p_whole)` |

各椎体内で損失を足し、batch内の実際の椎体数で平均する。
弱教師係数は`beta=1`から開始する。Baseline 0の`pos_weight`、
source importance補正、GT項だけの別正規化、A群への重複whole陽性損失は加えない。
N群ではwhole陰性BCEと4領域陰性BCEの和が同値なので、一度だけ計算する。

4/4/8と`beta=1`は開始値であり、実証済みの最適値ではない。
弱教師の強さを調整するときはsampling比率を固定し、
まず`beta ∈ {0.5, 1, 2}`を比較する。

## 校正工程

学習開始前の校正工程は設けない。今回の設計では、次はすべて不要である。

- CAM確率の校正と疑似ラベル生成
- loss間のgradient normを使った`lambda`校正
- calibration用CLI、artifact、version管理

`beta=1`は校正で推定せず、最初のArm A/B比較では固定する。
調整が必要な場合もinner data上で離散候補を比較し、専用の校正phaseは作らない。

一方、precision・recall・F1など二値指標を報告するときの閾値は、outer testを見ずに
inner dataで決める。これは学習前の校正とは別であり、AP/AUROCの主評価には不要である。
Brier/ECEは確率の診断として記録するだけで、初期実装にtemperature scalingや
isotonic regressionを追加しない。

## データstream

- A群を1周する期間を1 GT-passとする。
- 同じGT-pass内でA群の同じ椎体を反復しない。
- N群・U群はshuffle済みqueueをpass間で引き継ぎ、同じ一部症例に偏らせない。
- outer 0 trainの設計上の件数はA=159、U=643、N=7,272。
- 4/4/8なら39 full stepと、最後の4/3/8の1 stepになる。
- 最終stepは実際の15椎体で平均し、GT重複やimportance補正を行わない。
- source別loss、一意bag提示数、queue周回数、CNN gradient normを記録する。

## 初期モデル・最適化設定

- 新規package候補は`fracture_detection/region_mil/`。Baseline 0の挙動は変えない。
- 初期構成はstride-4 FPN 256ch、mask正規化pooling、
  hidden size 128の共有1層双方向region BiLSTM、有効面平均、Dropout 0.30、
  各領域へ個別適用する共有`Linear(256, 1)`。
- CNNも最初からfine-tuneする。初期学習率は転送CNNが`2.3e-5`、
  新規moduleが`2.3e-4`。
- optimizer初期案はAdamW、weight decay `1e-4`、cosine scheduler。
- Baseline 0のBatchNorm running mean/varianceは初期案では保持する。
  CNN convolutionとBN affine parameterは学習可能にする。
- CNN forwardは1 batchにつき1回とし、BN bufferが更新されないことを検証する。
- 3種類の教師を開始時から使う。最初はGT-only warmup、弱損失ramp、
  疑似ラベル、MixUp、Cutout、cropを追加しない。
- 空間augmentationはCTと全maskへ同期適用する。

## 次セッションの実装順

1. dirty worktreeを保護し、Baseline 0の再利用可能なinterfaceを確認する。
2. data contractとtestを先に実装する。N/A/U分類、A群の全4セル有効、
   0の保持、決定的な4/4/8 GT-pass queueを対象とする。
3. model部品とunit testを実装する。mask正規化pooling、region面系列、
   共有classifier、4 logits、安定なnoisy-ORを確認する。
4. lossと数式testを実装する。3群の式、bag数による正規化、
   陰性損失の非重複、極端logitでの有限性、CNNまでのgradientを確認する。
5. fold対応初期化、学習率別optimizer group、BN保持、trainer、
   checkpoint、resume、source・提示数logを実装する。
6. validationとOOF評価、config、train/evaluate CLI、説明書を実装する。
   calibration CLIは作らない。
   targeted testの後に広い回帰testを実行する。

## 学習前の必須検証

- 全A群bagに4個の有効なbinary targetがあり、
  version固定した全体inventoryで268 bag / 1,072 region cellを再現する。
- GTあり椎体の0が陰性教師として残り、U群には直接region targetを与えない。
- `p_whole`に学習可能な迂回経路がなく、各`q_r`に対して単調増加する。
- whole陰性BCEと4領域陰性BCEの和が、値・gradientとも一致する。
- N/A/UすべてのlossがCNNへ届き、encoder forwardが1回だけである。
- BN running bufferが保持され、CNNとBN affine parameterにはgradientが届く。
- 空mask、ラベル矛盾、極端logit、mixed precision、最終short batch、
  resumeの決定性をtestする。

## 最初の比較実験

初期値、入力、source順、loss分母、step数、seedを揃える。

- Arm A: N群とA群の4セル損失を使用。U群はforwardするがlossを0にする。
- Arm B: Arm Aに`beta=1`のU群陽性noisy-ORを加える。
- Arm C（任意）: N群は全0、すべての陽性bagはORだけで学習する。

checkpointはinnerのGTあり陽性における4領域macro APを主指標として選択する。
同じcheckpointからwhole AP/AUROC、偽陽性、領域間相関、全領域高値率、
較正指標も報告する。validation/testは自然分布で評価する。
Arm Bがheld-out局在を改善し、whole性能の悪化が許容範囲なら弱教師を採用する。

## 学習量と再現性の初期案

- 上限60 GT-pass、最短10 GT-pass、inner region macro APのpatience 10 GT-pass。
- 同点なら早いcheckpointを保持する。
- seedは`20260807`、`20260808`、`20260809`。
- 数値変更はinner dataの根拠だけで行い、outer testを見て変更しない。

## ロールバックと現在のworktree

実装は`fracture_detection/region_mil/`へ隔離する。
Baseline 0、既存checkpoint、manifest、過去のregion実験は変更しない。
phaseが失敗した場合は、そのphaseの新package/config変更だけを戻し、
原因調査用に本設計とログを残す。testを通すためにdata artifactを変更しない。

引き継ぎ時点ではモデル実装・学習は行っていない。
worktreeには次の設計文書変更がある。

- 変更済み: `.claude/docs/DESIGN.md`
- 未追跡: `fracture_detection/REGION_MIL_DESIGN.md`
- 未追跡: `fracture_detection/CONDITIONAL_MIL_DESIGN.md`
