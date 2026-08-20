# 2026-08-19: MTLアーム実装計画（Baseline 1–B / Proposed 3構成）

> 前セッション（v8実測結果と却下事項の確定）: `2026-08-19-baseline0-v8-results-and-frozen-decisions.md`
> 設計本体: `memo/計画書/提案手法.md` / 進捗台帳: `fracture_detection/PROGRESS.md`
>
> **本ファイルの位置づけ**: 凍結済み設計に書かれていない実装レベルの穴を埋め、
> Baseline 1–B と Proposed 3構成の実装手順を確定するための計画書。
>
> ⚠️ **承認状態に注意**。§0の表で「確定」と「提案（未承認）」を明示している。
> 未承認項目はユーザー承認を得るまで実装に着手しない。2026-08-19のClaude/Codex共同レビューで
> 修正したBN・RNG・backward・branch境界・凍結順序・校正artifact・資源見積りはユーザー指示により確定。

---

## 0. 決定事項の一覧と承認状態

| # | 項目 | 内容 | 状態 |
|---|---|---|---|
| A | コード構成 | `fracture_detection/core/` へ共有学習コアを切り出す | **確定**（2026-08-19ユーザー決定） |
| B | 進め方 | 先に全構成を実装・smoke testし、code/config/λ/βを凍結してからfull trainingを開始する | **確定**（2026-08-19共同レビュー後のユーザー指示） |
| C | Controlアーム | 学習runは延期するが、Baseline 1と同じ最終code hashで後から実行できるconfig経路まで今回実装する | **確定**（同上） |
| D | 設計の詰め方 | Claude/Codex共同レビューを行い、実装上の7修正を計画へ反映する | **確定**（同上） |
| 1 | region logit粒度 | 面単位 `[B,15,4]`、bag確率は面sigmoidの平均 | 提案（未承認） |
| 2 | 方式A（max）の適用レベル | 面ごとに `max_r z_r,t` → 他アームと同一のbroadcast BCE | 提案（未承認） |
| 3 | two-stream forward | natural と annotated を連結せず別forward | 提案（未承認） |
| 4 | annotated側のBN | model/dropoutはtrainのまま、`_BatchNorm` moduleだけ一時的にevalへ切り替える | **確定**（共同レビュー修正） |
| 5 | backward | naturalをbackwardしてgraph解放後、annotatedをbackwardし、optimizer stepは1回だけ | **確定**（共同レビュー修正） |
| 6 | 勾配ノルム比ログ | 1 epochあたり4 step（0/126/252/378）の中央値 | 提案（未承認） |
| 7 | Proposedの分割点 | `blocks[4]` 出力（14×14×160、stride 16） | 提案（未承認） |
| 8 | Proposedのbranch | `blocks[5]`+`conv_head`+`bn2` の独立パラメータ複製 ×4 | **確定**（共同レビュー修正） |
| 9 | `L_att` の回帰対象 | spatial attention map `s` (14×14×1)。融合後の `m` ではない | 提案（未承認） |
| 10 | `s` の活性化 | sigmoidを掛けて[0,1]へ収める（原論文からの逸脱として記録） | 提案（未承認） |
| 11 | `L_att` の構成 | 4領域のみ。global（whole mask）branch用の項は入れない | 提案（未承認） |
| 12 | Proposed–Bのwhole head | attentionを掛けない5本目のbranchに置く | 提案（未承認） |
| 13 | maskの縮小方法 | area平均（`adaptive_avg_pool2d`）で14×14へ | 提案（未承認） |
| 14 | 領域が存在しない面 | 全ゼロを教師にする（skipしない） | 提案（未承認） |
| 15 | 原論文のSA module | 導入しない（Block I/II直後、branch内Block IV直後のいずれも） | 提案（未承認） |
| 16 | 校正用の共有ブロック | `blocks[4]`（全アーキテクチャに共通して存在する最後の共有ブロック） | 提案（未承認） |
| 17 | β=0の実装 | `L_att` は常に計算し、βを0倍する（RNG消費とコード経路を一致させる） | 提案（未承認） |
| 18 | RNG分離 | mixup・augmentation・annotated forwardへ独立RNG streamを割り当て、resume checkpointへ状態を保存する | **確定**（共同レビュー修正） |
| 19 | code hash凍結 | 全構成の実装・検証完了後に1つの最終hashを凍結し、そのhash以外でfull trainingしない | **確定**（共同レビュー修正） |
| 20 | 校正artifact | λ/βのraw calibrationを別artifactへ保存し、両方確定後に`loss_weights.json`を一度だけ生成する | **確定**（共同レビュー修正） |
| 21 | 資源gate | 実測parameter数を使い、full training前に各構成のpeak VRAM・step時間をsmoke実測する | **確定**（共同レビュー修正） |
| 22 | 複数GPU実行 | Baseline 1以降は1 foldを1 GPUの独立processとして並列実行できるoptionを持つ | **確定**（2026-08-19ユーザー指示） |

---

## 1. 実測した前提数値

### 凍結manifestからの分割（`common/outputs/input_manifest.csv`）

全体: 13,432 bag / 2,009 study / 陽性1,332 / annotated 268。
領域陽性の内訳: R1 78 / R2 59 / R3 72 / R4 158。

| outer | train bag | steps/epoch | annotated train | annotated passes/epoch | val annotated | outer annotated |
|---|---|---|---|---|---|---|
| 0 | 8,074 | 505 | 159 | 3.18 | 53 | 56 |
| 1 | 8,055 | 504 | 162 | 3.11 | 53 | 53 |
| 2 | 8,056 | 504 | 162 | 3.11 | 53 | 53 |
| 3 | 8,048 | 503 | 162 | 3.10 | 53 | 53 |
| 4 | 8,063 | 504 | 159 | 3.17 | 56 | 53 |

- annotated bagは **1 epochあたり約3.1周**する。v5相当の55 epochなら1 bagあたり**約170回**提示される。
  region headの過学習リスクが高く、`region_passes` のログで監視すべき対象
- inner側のannotatedは53〜56 bagでR2陽性は11〜12件。**epoch毎の領域APはログ専用**とし、
  checkpoint選択は椎体AUROCのままとする（既存決定と整合）

### backbone（`tf_efficientnetv2_s`、224×224入力）

| 段 | ブロック数 | パラメータ | 出力 | 1セルの実寸（0.4 mm/px） |
|---|---|---|---|---|
| stem（10ch） | — | 0.002M | 112×112×24 | — |
| blocks[0] | 2 | 0.010M | 112×112×24 | 0.8 mm |
| blocks[1] | 4 | 0.304M | 56×56×48 | 1.6 mm |
| blocks[2] | 4 | 0.589M | 28×28×64 | 3.2 mm |
| blocks[3] | 6 | 0.918M | 14×14×128 | 6.4 mm |
| blocks[4] | 9 | 3.464M | 14×14×160 | 6.4 mm |
| blocks[5] | 15 | **14.562M（全体の72%）** | 7×7×256 | 12.8 mm |
| conv_head+bn2 | — | 0.330M | 7×7×1280 | 12.8 mm |

10ch backbone合計 20.179M。**横突孔は実寸5〜6 mm程度なので、7×7（12.8 mm/セル）にattention mapを置くと
孔がセル1個未満になり `L_att` の教師が潰れる。** これがProposedの分割点を決める支配的な制約。

branchを実module境界どおり `blocks[5] + conv_head + bn2 + BiLSTM + head` として数え直すと、
共有部は5,286,928 parameter、1 branchは19,750,953 parameter。MA moduleをまだ含めない下限でも
**Proposed–maxは84,290,740、Proposed–Bは104,041,693 parameter**になる。

### 環境

- albumentations 2.0.8 の `CoarseDropout` は `fill_mask` 未指定でmaskを書き換えない。
  領域maskにも穴は開かず、baseline0と同じ挙動を維持できる

---

## 2. 決定案1・2: region logitは面単位、方式Aのmaxも面単位

### 内容

- region head は BiLSTM の面ごと特徴から **`[B,15,4]`** を出す
- `L_region` は bagラベルを15面へbroadcastした**素のBCE**（pos_weightなし）を面・領域で平均
- bag領域確率 = 面sigmoidの平均（whole と同じ readout）
- 方式A（max集約）は**面ごとに** `z_whole,t = max_r z_r,t` を取り、
  他アームと**同一の**broadcast重み付きBCE（pos_weight 2.0、重み合計正規化）へ渡す
- 報告するbag whole確率 = `mean_t sigmoid(max_r z_r,t)`

### 根拠

「`L_whole` の関数形と実効重みが全アームで一致すること」が凍結仕様の hard constraint
（`提案手法.md` §2「`L_whole` は常に `B_W` で mean」「whole taskの経験リスク・bag分布・
1 stepあたりのwhole係数・1 epochあたりのwhole exposureが全アームで完全一致」）。

bag単位で方式Aを実装すると Proposed–max だけ whole loss の項数が 240 → 16 になり、
重み合計正規化の分母も変わる。すると Proposed–max は「集約関数が違うアーム」ではなく
「損失の定義が違うアーム」になり、はしご全体が崩れる。面単位なら `[B,15]` のlogitに落ちるので
Baseline 0 と1文字も違わないBCEを通せる。

### 設計書との差分（`提案手法.md` §4 の修正が必要）

§4 は `p_whole = max(p_L,p_R,p_B,p_P)` と bag 単位で書いている。面単位実装は数学的には別物で、
sigmoidが単調なため

```
mean_t sigmoid(max_r z_rt)  ≥  max_r mean_t sigmoid(z_rt)
```

と常に上界側になる（各面のargmaxが全面一致するときだけ等号）。
どちらも `y_whole = OR_r y_r` と整合する単調なOR型集約なので、
**面レベルの定義として §4 に明記して採用する**。

- **学習と評価で定義を変えない**（損失は面max・報告はbag max、という折衷はしない）。
  訓練目的と報告指標が乖離するため

### H2への影響

β>0 と β=0 は同じ集約を共有するので、この選択がH2を偏らせることはない。影響するのは感度だけ。

### 登録すべき副作用

方式Aでは whole loss が **13,432 bag すべてで region logit へ逆伝播する**。
陽性bagでは「そのとき最大の領域」だけが押し上げられる自己確認的な挙動になり、
事前確率の高い R4（268中158）へ収束しやすい。したがって
**Proposed–max の領域APは Proposed–B / Baseline 1–B と同じ土俵では読めない。**
床ゲートの対象が Proposed–B のみ、H2が椎体AUROC、という既存の凍結仕様とは整合するが、
限界として明記する。

### 実装への影響

`common/losses.py::region_bce` は `[B,4]` 前提なので `[B,N,4]` を受ける形へ拡張する
（validマスクの論理は据え置き）。

---

## 3. 決定案3・4・5・18: two-stream、BN、backward、RNG

### 内容

- natural (16 bag = 240面) と annotated (1 bag = 15面) を**連結せず別々にforward**する
- annotated forward の間だけ全 `_BatchNorm` moduleを **eval** にし、natural streamが作った
  running mean/varianceで正規化する。model全体はtrainのままなのでDropoutは有効
- annotated forward も**同じ autocast/bfloat16 コンテキスト内**で行う
- natural lossをbackwardしてgraphを解放した後、annotated forwardと`λ·L_region`のbackwardを行う
- gradient clippingとoptimizer stepは、両backwardが完了した後に**1回だけ**行う
- 実行順序は全アームで **natural forward → natural backward → annotated forward → annotated backward → optimizer step** に固定する

### 根拠

連結案は却下。BN統計が240面→255面になり、「whole taskの分布・勾配が全アーム完全一致」という
要件を直接壊す。さらに A_t が `L_whole` の正規化統計に混入し、
「A_t は `L_region` にのみ寄与」という厳守事項にも反する。

annotated側のBNの扱いは4択:

| 案 | 推論時running statsへの影響 | 勾配経路 | 判定 |
|---|---|---|---|
| train mode・統計更新あり | **BN更新の半数が1 bag・268の偏った部分集団由来になる** | 正常 | 却下 |
| model全体をeval | なし | head dropoutまで切れる | 却下 |
| train mode + momentum 0 | running bufferは不変 | **annotated batch自身の統計で正規化**し、推論時と不一致 | 却下 |
| **BN moduleだけeval** | **なし** | running stats使用、BN affine/input gradientとDropoutを維持 | **採用** |

`momentum=0` はbuffer更新を止めるだけでforward時のbatch統計利用は止めない。annotatedは1 bagの
偏った部分集団なので、これを学習時だけ使うと推論時のrunning statsとの不一致が生じる。
context managerは各 `_BatchNorm.training` flagを保存して`False`へ切り替え、終了時に個別復元する。
これにより`running_mean` / `running_var` / `num_batches_tracked`はすべて不変で、affine parameterと
入力へのgradientは残り、Dropoutも切れない。

2回backwardはlossの定義を変えない。parameter update前なので共有parameter値は同じで、PyTorchは
`.grad`へnatural gradientとannotated gradientを加算する。一方、natural graphをannotated forward前に
解放できるため、両streamのactivationを同時保持する1回backward案よりpeak VRAMを下げられる。

### その他の制御事項

- sampler、augmentation、mixup、annotated model stochasticityを**別RNG stream**に分ける
  - natural/annotated samplerは既存どおりepochと専用seedから決定
  - augmentationはglobal RNGを使わず、`(outer, epoch, stream, sample ordinal)`から導くper-sample seedを
    `ReplayCompose.set_random_seed()`へ渡す。natural側のseedは全アーム共通
  - mixupの発火判定・λ・permutationは専用CPU `torch.Generator`から生成し、permutationだけdeviceへ転送する
  - annotated forwardは保存済みの専用CPU/CUDA RNG stateへ一時切替え、終了後に専用stateだけ進めて
    natural側のglobal RNG stateを復元する
- checkpointへPython / NumPy / torch CPU / 全CUDAのglobal RNG state、mixup generator state、
  annotated専用CPU/CUDA stateを保存・復元する。resumeはepoch境界で同じ乱数軌跡へ戻す
- `cudnn.benchmark=True` に対しshapeが1種類増えるだけ
  （epoch末尾の端数batchで既に2種類ある）。実害なし
- ⚠️ **MTLアームがBaseline 0とビット一致することはあり得ない**（パラメータ構成もdropoutの
  RNG消費も違い、cuDNNも決定論modeではない）。一致させるのは **natural sampler順序・natural
  augmentation/mixup draw・step数・損失の関数形/係数**であり、weightのビット一致ではない
- コスト: forward/backwardのFLOPsが +6.25%、カーネル起動オーバーヘッド込みで
  **epoch時間 +8〜10%（250s → 270〜275s）** の見込み

---

## 4. 決定案6: 勾配ノルム比のログは1 epochあたり4点の中央値

毎stepで3回 `autograd.grad` を回すのは論外。epoch先頭1回だけだと「shuffle後の最初のbatch」という
1標本になり、比が数倍ぶれる。

**epochあたり4 step（0 / 126 / 252 / 378）で `torch.autograd.grad(..., retain_graph=True)` を
共有ブロックのparameterに限って実行し、中央値を `history.csv` へ記録する。**

- natural graph上で`L_whole`と`L_att`のnormを取得してからnatural backwardし、annotated graph上で
  `L_region`のnormを取得してからannotated backwardする。通常stepでは`autograd.grad`を呼ばない
- `autograd.grad`自体は`.grad`を汚さず、計測後の2回backwardだけがoptimizer用gradientを加算する
- 505 stepに対し4回なので追加コストは小さいが、割合は各構成のsmoke profileで実測する
- この数値の登録目的は「λ/βが意図した比を学習中も保っているかの監査」なので、
  75 epoch × 4点のトレースで足りる

記録する量（`提案手法.md` §2 の要求どおり）: `‖∇L_whole‖` / `‖λ∇L_region‖` /
`‖∇L_att‖` / `‖β∇L_att‖` とその比。β=0ではweighted normを0、βを分母に含む比を`NA`とし、
raw `‖∇L_att‖`はβ>0アームとの経路監査のため残す。
併せて `region_optimizer_steps` / `region_passes = T / N_annotated_train` /
annotated bagごとのvisit回数の min・median・max / epochあたり unique annotated bag 数。

---

## 5. Proposed: PMGAN原論文の精読結果と本研究への写像

原論文: Zhang et al., "Part-Aware Mask-Guided Attention for Thorax Disease Classification",
*Entropy* 2021, 23, 653。
PDF: `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`（p.1-13を精読）

### 原論文の構造（ResNet50 / 512×512入力）

| 項目 | 原論文 | 出典 |
|---|---|---|
| baseline | ResNet50。Conv1→256²、Block I→128²、Block II→64²、Block III→32²、Block IV→16² | Table 1 |
| SA（soft attention） | Block I 直後と Block II 直後 | Fig. 2 |
| **MA（mask-guided attention）** | **Block III 直後（32×32 = stride 16）に4本独立** | Fig. 2, §3.3 |
| **branch** | **Block IV（最終残差ブロック）の独立パラメータ複製 ×4**、各々 SA→GAP→FC | Fig. 2, §3.3 |
| 再重み付け | `f̂ = (1+m) ⊗ f` | 式(3) |
| spatial attention | channel方向GAPで h×w×1 へ圧縮 → encoder-decoder（conv stride2 / 3×3、Block III–IVでは**1層**） | 式(4), §3.2.1 |
| channel attention | GAPで1×1×c → 2 conv、reduction **r=16** | 式(5)(6) |
| 融合 | `m = Conv_{1×1}(s × t)` → **sigmoid** で[0,1]へ | 式(7) |
| **`L_att` の回帰対象** | **spatial attention map `s^b_3` (h×w×1)**。融合後の `m` (h×w×c) **ではない** | 式(8) |
| `L_att` の正規化 | 画素で総和 → 画像数 N で割る → sqrt。標準的なRMSEではない | 式(8) |
| 全体の `L_att` | `L_att = L_att^0 + β Σ_{b=1}^{3} L_att^b`（globalとlocalの相対重み） | 式(9) |
| 分類損失 | `L_ce = L_ce^1 + α L_ce^2`、α=0.5。`L_ce^2` は local 3 branch の **max** | 式(10)(11) |
| 全体 | `L = L_ce + L_att` | 式(12) |
| 推論 | maskは学習時のみ使用。推論に追加計算なし | §3.3 |

原論文は global（all-organ mask）branch と local 3 branch の**両方**を持ち、localをmaxで束ねている。
本研究はこれを方式B（独立whole head）と方式A（max集約）の**2アームに分解**した形になる。

### 本研究への写像（提案）

```text
入力 10ch（CT5 + whole mask + R1..R4）
  ↓
共有CNN: stem + blocks[0..4]        → 14×14×160（stride 16）
  ↓
  ├─ MA_1 → s_1(14×14×1), m_1(14×14×160) → f̂_1 = (1+m_1)⊗f → blocks[5]複製#1 + conv_head + bn2 → GAP → BiLSTM#1 → [B,15] logit (R1)
  ├─ MA_2 → ... → BiLSTM#2 → [B,15] logit (R2)
  ├─ MA_3 → ... → BiLSTM#3 → [B,15] logit (R3)
  ├─ MA_4 → ... → BiLSTM#4 → [B,15] logit (R4)
  └─ （Proposed–Bのみ）attention無しの5本目: blocks[5]複製#5 + conv_head + bn2 → GAP → BiLSTM#5 → whole logit [B,15]

L_att = (1/4) Σ_r sqrt( (1/(B·P)) Σ_{b,p} Σ_{j,k} ( M^r_{b,p,j,k} − s^r_{b,p,j,k} )² )
Proposed–max: z_whole,t = max_r z_r,t
Proposed–B:   z_whole,t = 5本目のbranchの出力
```

| # | 決定案 | 理由 |
|---|---|---|
| 7 | 分割点は `blocks[4]` 出力（14×14×160、stride 16） | 原論文のMAはBlock III直後＝stride 16。かつ7×7では横突孔がセル1個未満に潰れる |
| 8 | branchは `blocks[5]`+`conv_head`+`bn2` の**独立パラメータ複製 ×4** | timmの実forward境界と原論文の "four network branches with independent parameters"（§3.3）に一致。`bn2`はactivation込みの`BatchNormAct2d` |
| 9 | `L_att` は **`s`（14×14×1）** へ回帰。`m` ではない | 式(8)。前回の説明（`m`へ回帰）は誤りだったため訂正 |
| 10 | `s` に **sigmoid** を掛ける | 原論文は式(7)の `m` にしかsigmoidを明記していないが、二値maskへ回帰させる以上 `s` も[0,1]に収める方が安定。**原論文からの逸脱として記録** |
| 11 | `L_att` は **4領域のみ**。global（whole mask）用の `L_att^0` は入れない | 入れるとProposed–B（5項）とProposed–max（4項）で `L_att` の定義が変わり、Proposed–Bで校正したβをProposed–maxへ流用できなくなる |
| 12 | Proposed–Bのwhole headは **attentionを掛けない5本目のbranch** | 方式Bの定義（独立whole head）に沿う。かつ#11と整合する |
| 13 | maskの14×14への縮小は **area平均**（`adaptive_avg_pool2d`） | 横突孔は約5〜6 mm、1セルは6.4 mm。nearestだと消える。area平均なら占有率0.2〜0.6のソフト教師になる |
| 14 | その面に領域が存在しない場合は **全ゼロを教師**にする（skipしない） | 「ここには無い」も正しい教師。attentionの誤発火を抑える。式(8)の全画素和の形とも整合 |
| 15 | 原論文の **SA moduleは導入しない**（Block I/II直後、branch内Block IV直後のいずれも） | 入れるとBaseline 1との差が「mask注入」以外にも広がる。H2はどちらでも成立するが、登録済みの主張範囲を超える |
| 16 | λ/β校正の「最後のshared CNN block」は **`blocks[4]`** | 全アーキテクチャに共通して存在する最後の共有ブロック。Baseline 1でも定義できる |
| 17 | β=0でも `L_att` を計算して **βを0倍**する | β>0とβ=0でコード経路・RNG消費・パラメータ数・初期化が完全に一致する |

### `L_att` の正規化について

原論文の式(8)は「画素で総和 → 画像数で割る → sqrt」であり、画素数で割る標準的なRMSEではない。
本研究では面（plane）が画像に相当するので、**画素で総和 → bag×面の数で割る → sqrt** とする。
画素数で割る形との差は定数倍 `sqrt(196)=14` だが、**βは勾配ノルム比で校正するため定数は吸収される**。
ただし定義は凍結して記録する。

### β=0 ablation の妥当性

MA moduleは存在し続け、教師なしの `m` で `f̂ = (1+m)⊗f` を適用する。
パラメータ数・dropout位置・初期化・RNG消費が同一で、落ちるのは損失項だけ。
これが H2 が測ろうとしている「attention回帰教師の新規性」の正しい対照である。

### 未確定（実装時に実測して決める）

- GPU memoryが49 GBに収まるか。2回backwardでnatural graphを先に解放した条件で、各構成について
  1 optimizer stepのpeak allocated/reserved VRAMを実測する
- 収まらない場合はfull trainingへ進まず停止する。候補は (1) Proposed–Bの5本目branchを共有trunk直結の
  軽量headへ、(2) branchを重み共有（4回適用）へ、(3) `blocks[5]` の複製範囲を後半へ限定、の順だが、
  いずれも**研究構成を変えるため自動適用しない**。ユーザー承認後に計画・parameter数・β校正を更新し、
  新しい最終code/config hashを凍結し直す

---

## 6. Control アームの扱い（延期）

**2026-08-19ユーザー決定により、Control（no-region-mask MTL）の学習runは今回の対象外。**
ただし後から共有codeを変更せず同じhashで実行できるよう、Baseline 1内のconfig経路・unit test・
1 step smokeまでは今回実装する。Baseline 1–B と Proposed 3構成の結果後に、Controlのrunを判断する。

申し送り2点:

1. **H1（primary）が `AUROC(Baseline 1–B) > AUROC(Control–B)` なので、Controlがないと
   確証的検定は H1 が空になり、固定順序 H1→H2 の規則により H2 も探索的な位置づけに落ちる。**
   実験後にControlを追加すれば元の検定計画へ戻せるが、その場合は Baseline 1–B と
   **同じ凍結λ・同じseed・同じcode hash** で走らせる必要がある。したがってControl実行のために
   shared coreへ変更が必要になった時点で同一hash条件は失敗とし、Baseline 1を含む再実行なしには採用しない
2. Control と Baseline 1 の差は**入力チャンネルだけ**なので、
   **コード上はconfigのフラグ1つ（`in_chans: 6` と領域mask不使用）で済む。**
   実装はBaseline 1に内包し、最終hash凍結前にControl経路のunit/1 step smokeまで完了させる。
   学習runだけを走らせない

---

## 7. Phase構成

| Phase | 内容 | 完了条件 |
|---|---|---|
| 1 | `fracture_detection/core/` へ共有学習コアを切り出す。`staging` / `optimization` / `experiment` と `trainer` のarm非依存部（epoch loop、checkpoint、resume、RNG state、early stopping、閾値決定、outer 1回推論ガード）を移し、arm固有部を adapter へ（`build_datasets` / `build_model` / `compute_train_losses` / `predict` / `shared_parameters`）。baseline0はadapter実装だけに縮小 | 既存tests通過 + 同一seed/configの1 epoch parity + CPU上でuninterrupted/resumeのmodel・optimizer・RNG state一致 |
| 2 | 全アーム共通データ層。10ch（CT5 + whole + R1..R4）をcanonical sampleとし、adapterが6ch/10chを選択。hflipは`common.dataset.flip_horizontal`へ一本化し、albumentationsへimage 75ch / mask 30ch（whole 15 + region label map 15、`INTER_NEAREST`）を1回渡す。augmentationはper-sample seed、uint8転送後GPU正規化 | R2↔R3のmask値/label swap、dtype、同一natural keyに対する全アームのCT/whole augmentation一致、resume後のaugmentation seed一致 |
| 3 | **Baseline 1–B + Control経路**。`[B,15,4]` region head、two-stream、BN-only-eval context、2 backward/1 step、独立RNG、exposure/勾配ログ。予測CSVへ`region_{r}_score` / `region_{r}_target` / `has_region_target`を追加。Controlはconfigで6chを選ぶがfull runしない。Baseline 1以降のCLIへsingle-GPUとfold-process並列launcherを追加 | unit tests + 両経路の実データ1 step。annotated前後でBN buffer不変、Dropout有効、annotated実行有無で次natural mixup draw不変、2 backwardのgradient和が参照合算lossと許容誤差内で一致。2 GPU smokeでfold別output/RNG/resumeが衝突しない |
| 4 | Proposed 3構成をすべて実装。MA at 14×14、branch=`blocks[5]+conv_head+bn2`独立複製、`L_att`は`s`へ回帰。各構成のparameter数、1 step peak VRAM、step時間を同一GPUで測定 | unit tests + 各構成の実データ1 step forward/backward。49 GBに収まらなければ停止し、構造変更は別途承認 |
| 5 | λ校正（reference Baseline 1–B）とβ校正（reference Proposed–B）をouter毎64 batchで実行。raw結果を`lambda_calibration.json` / `beta_calibration.json`へ別々にatomic保存し、両方揃った時だけ`loss_weights.json`を一度生成してimmutable化 | 5 foldのλ/βが有限。校正前後でparameter/optimizer/BN/RNG state完全一致。3 artifactのSHA256とreference config hashを記録し、既存file上書きを拒否 |
| 6 | **本番凍結後にのみ学習**。全構成、λ/β、検定順序、source/config/dependency hashとfold-to-GPU実行計画を`frozen_experiment_manifest.json`へ固定し、run開始時とresume時に照合。その後Baseline 0再run、Baseline 1–B、Proposed 3構成を学習し、各checkpointからouterを規定回数だけ推論 | 凍結記録とhash guard完成。すべての正式runが同一frozen manifestを参照し、hash不一致runを拒否。並列時も各foldがsingle-GPU実行と同じconfig/seed/output schemaを保持 |
| 7 | 解析（pooled OOF、領域別AP、床ゲートHolm、H1/H2のpatient-cluster bootstrap、感度解析2件） | 全outer予測のfold/manifest/hash整合性が確認できること |

Phase 2でBaseline 0もcanonical data pathへ載せ替えるのは、全アームでaugmentation実装を1本にするため。
v4/v5とのビット比較は失われるが、**本番runは凍結後なので問題ない**
（v5の数値はv8設定のpilot証拠として有効なまま）。

**Phase 1〜5の途中ではfull trainingを開始しない。** smoke/calibration結果だけで修正し、outer予測は一切作らない。
Phase 6のmanifest凍結後はshared core・arm実装・config・λ/β artifactを変更しない。変更が必要なら既存runを
正式結果から外し、新しいmanifest hashで対象となる全比較アームを最初から再実行する。

---

## 8. 計算資源の見積り

前提: 1 epoch 503〜505 step、Baseline 0 v5実測は約250 s/epoch。Proposedの旧見積りは
late CNN branchだけを倍率計算し、複数BiLSTM、MA module、annotated stream、両graph同時保持を
十分に含めていなかったため、正式な日程見積りには使わない。

### parameter下限（MA moduleを除く実module集計）

| 構成 | parameter | FP32 weight+grad+Adam m/vの理論下限 |
|---|---:|---:|
| Proposed–max | 84,290,740 + MA | 1.35 GB + MA |
| Proposed–B | 104,041,693 + MA | 1.66 GB + MA |

16 byte/parameter（weight 4 + grad 4 + Adam states 8）の単純計算であり、activation、autocast内部buffer、
cuDNN workspace、allocator fragmentationを含まない。49 GB適合性はparameter数では判断せず実測する。

### full training前の必須profile

同一GPU・同一software環境・natural batch 16・annotated batch 1で、Baseline 0 / Baseline 1–B /
Proposed–B / Proposed–maxを各20 optimizer step実行し、warmup後10 stepのmedianを採る。

- `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`
- optimizer step時間、natural forward/backward時間、annotated forward/backward時間
- MA・各branch・BiLSTMのparameter数
- β=0とβ>0の経路・VRAM・時間が一致すること

### Baseline 1以降の複数GPU option

正式に許可する並列化は**fold-process parallelism**。1 outer foldのmodel/optimizer/DataLoaderは1 GPU内に
完全に閉じ、複数foldを別process・別GPUで同時実行する。これならglobal batch 16、BN、loss、sampler、
optimizer step数を変えず、結果へ影響しない実行時間短縮として扱える。λ/β校正もouter fold単位で同じ
launcherを利用できる。

```yaml
parallel:
  mode: fold          # single | fold
  gpu_ids: [0, 1]
  max_concurrent_folds: 2
```

- launcherは未実行outer foldを昇順に割り当て、`fold_to_gpu`をeffective configと
  `frozen_experiment_manifest.json`へ保存する
- 同じouter foldは全アームで同じGPU model/compute capabilityへ割り当てる。異種GPU混在は拒否する
- 各processは従来どおりfold固有output directory、checkpoint、W&B run、RNG stateを持つ
- dataset stagingはmanifest hash付きREADY cacheとlockを共有し、processごとの全量copyを作らない
- 1 foldの失敗は他foldを停止・削除せず、同じGPU割当とcheckpointから個別resumeする
- full training前に2 GPU × 2 foldの20-step smokeを行い、output衝突、GPU oversubscription、host RAM、
  tmpfs/NFS throughput、single実行とのsample order/mixup draw一致を確認する

**DDP / DataParallel / FSDPで1 foldのbatch/modelを複数GPUへ分割する方式は今回の正式optionに含めない。**
global batchまたはper-rank BN統計を変え、DDPはmodelを複製するため1 GPUあたりVRAM不足も解消しない。
将来必要なら、SyncBatchNorm、global batch 16維持、single-GPUとの数値parityを含む別計画として承認する。
fold並列も1 runのpeak VRAMは減らさないため、49 GB resource gateの代替にはしない。

このprofile完了後に、実測秒/epoch × 実際の停止epoch見込みでGPU日数を更新する。それまでは旧
**174 h / 7.3日 / 3.6日を撤回し、予約や完了予定の根拠にしない。**

- batch size変更は却下済み（2026-08-19）。並列化はfold assignmentだけに使う

---

## 9. 実装前に必ず記録すること

1. 方式Aを**面レベル**で定義し直したこと（`提案手法.md` §4 の修正）
2. `L_att` は **`s`（14×14×1）** への回帰であり、`m` ではないこと
3. `s` に sigmoid を掛けたこと（原論文からの逸脱）
4. `L_att` を4領域のみで構成し、global用の項を入れないこと
5. 原論文のSA moduleを導入しないこと
6. `L_att` の正規化を「画素総和 → bag×面数で割る → sqrt」に固定したこと
7. 領域が存在しない面へ全ゼロ教師を与えること
8. maskの縮小をarea平均で行うこと
9. λ/β校正の共有ブロックを `blocks[4]` に固定したこと
10. MTLアームはBaseline 0とweightがビット一致しないこと。一致させるのは natural sampler順序・
    augmentation/mixup draw・step数・損失の関数形/係数
11. Controlを後回しにした結果、H1が空になり H2 が探索的になること
12. Phase 2 で Baseline 0 のaugmentation経路を変更したため、v4/v5とビット比較できないこと
13. annotated forwardではmodel全体でなくBN moduleだけをevalにすること
14. natural backward → annotated backward → 1 optimizer stepの順序と、2 loss graphを同時保持しないこと
15. mixup / augmentation / annotated forwardのRNG分離とcheckpoint復元契約
16. Proposed branchが`blocks[5] + conv_head + bn2`であり、`bn2`もbranch独立であること
17. λ/β raw calibration artifactを分け、`loss_weights.json`は両方確定後に一度だけ生成すること
18. MA除外時点でProposed–max 84,290,740 / Proposed–B 104,041,693 parameterであること
19. 全構成実装・校正・資源profile・frozen manifest作成前にfull trainingしないこと
20. Baseline 1以降の複数GPU optionは1 fold＝1 GPUの独立processであり、fold-to-GPU割当を凍結すること。
    1 fold内のDDP等ではなく、1 runのpeak VRAMも減らさないこと

---

## 10. やってはいけないこと

既存の禁止事項（`2026-08-19-baseline0-v8-results-and-frozen-decisions.md` §9、
`2026-08-18-implementation-handoff.md` §5）に加えて:

- `folds/outputs/folds.csv` の再生成
- outer fold を checkpoint選択・構成選択・ハイパラ調整に使うこと
- λ を arm別にチューニングすること
- 領域AP評価に whole-negative bag を足すこと
- macro-AP へ潰すこと / SideAcc を復活させること
- 複数構成の結果を見てから最終構成を選ぶこと
- 領域ラベルを持つアームで `A.HorizontalFlip` を直接使うこと（R2/R3が静かに壊れる）
- `min_epoch` を上げること
- vertical flip / transpose を導入すること（恒久禁止）
- mixup を `p=0.2` から変えること（恒久却下）
- Codex CLI に `--full-auto` を付けること
- 指示なしの commit / push
- **追加**: 学習と評価で方式Aの定義を変えること（損失は面max・報告はbag max、という折衷）
- **追加**: annotated stream にメインプロセスのRNGを引かせること
- **追加**: annotated forward で BN の running statistics を更新すること
- **追加**: natural batch と annotated batch を連結して1回でforwardすること
- **追加**: annotated forwardでmodel全体をevalにしてDropoutを切ること
- **追加**: annotated BNを`momentum=0`のtrain modeで動かし、1 bagのbatch統計を使うこと
- **追加**: naturalとannotatedのgraphを同時保持する合算1 backwardへ戻すこと
- **追加**: λだけを書いた`loss_weights.json`を凍結後にβ追記で更新すること
- **追加**: resource smoke失敗時に未承認の軽量head・branch共有・分岐変更を自動適用すること
- **追加**: `frozen_experiment_manifest.json`作成前にfull trainingを開始すること
- **追加**: frozen manifest作成後にshared core/config/calibration artifactを変更してrunを継続すること
- **追加**: 正式runで1 foldをDDP / DataParallel / FSDPにより複数GPUへ分割すること
- **追加**: 1 GPUへ複数fold processを同時割当し、GPU oversubscriptionを起こすこと
- **追加**: 並列process間でfold output directory、checkpoint、W&B run、RNG stateを共有すること
- **追加**: 異種GPUを混在させること、または凍結後にアーム間のfold-to-GPU割当を変えること

---

## 11. 今セッションの経緯（記録）

- Codexへ設計4点（region logit粒度 / Proposed構造 / two-stream forward / 勾配ログ頻度）を
  一括相談したが、**55分無応答のため打ち切り**。原因は4問一括＋`model_reasoning_effort=high`＋
  10ファイル読み込みで重すぎたこと。2問へ絞って再投入したがこれもユーザー判断で取り下げ、
  **Claudeが検討する方針へ変更**（2026-08-19ユーザー決定）
- 今後Codexへ投げる場合は、**質問を2問以内に絞り、読ませるファイルを3本以内に限定し、
  `timeout` と `-c model_reasoning_effort="medium"` を付ける**
- 本計画をCodexがレビューし、BN/RNG/凍結順序/branch境界/backward/artifact/parameter見積りの7点を指摘。
  Claudeと論点を分割して再検討し、最初に見解が割れたbackwardとcode hashも前提を明示した再質問で
  合意した。2026-08-19ユーザー指示により、本ファイルへ共同レビュー結果を反映した
- 2026-08-19ユーザー指示により、Baseline 1以降へ複数GPU optionを追加した。比較条件を変えないため、
  1 fold＝1 GPUの独立processとして複数foldを並列実行し、1 fold内のDDP等は正式runから除外する

## 12. 参照

- 設計本体: `memo/計画書/提案手法.md`
- 進捗台帳: `fracture_detection/PROGRESS.md`
- 前セッション: `.claude/docs/work-logs/2026-08/2026-08-19-baseline0-v8-results-and-frozen-decisions.md`
- 実装ハンドオフ: `.claude/docs/work-logs/2026-08/2026-08-18-implementation-handoff.md`
- PMGAN原論文: `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`
  （Zhang et al., *Entropy* 2021, 23, 653）
- Codex回答（未決4点・検定計画・λ校正・構成削減）: `.claude/docs/codex/20260818-remaining-four-decisions.md`
