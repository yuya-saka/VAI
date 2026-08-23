# 疑似ラベルMTL計画のCodexレビューとCAM生成段階の監査

**日付**: 2026-08-23
**状態**: 設計レビュー完了、GPU不要なkill criteria判定は全てPASS。方針3点はユーザー決定済み。
監査CLIを実装・検証済みで、正式監査runの起動待ち。

対象計画: `memo/進捗/研究計画書_2026-08-21.md`（Grad-CAM由来の4領域疑似ラベル + 椎体ラベルのMTL）

---

## 1. Codexレビュー（全文: `.claude/docs/codex/20260823-pseudo-label-mtl-design.md`）

計画書 4.5「これから決めること」の3項目 + 追加で洗い出したリーク・評価の論点を相談した。
結論は「探索的研究として実施価値はあるが、現状の `CAM値 → BCE疑似確率` には反対」。

主要な推奨:

| 論点 | Codex推奨 |
|---|---|
| Teacher割当 | fold-matched `Teacher_k` が自分の訓練3 foldをラベル化（outer/inner未使用＝リークなし） |
| 領域損失の母集団 | **骨折陽性1,332 bagのみ**。領域headは `P(R_r=1 | Y_whole=1, x)` の条件付き局在 |
| CAM variant | 現行 `encoder.bn2` 7×7 plain Grad-CAMを主解析に固定（268で選び直さない） |
| 15面の集約 | 全面のCAM massと面積を先に合計してから密度。max-over-planeは不採用 |
| 面積補正 | 現行の density enrichment（region密度 / 椎体全体密度）を維持 |
| 4領域内正規化 | softmax・rank正規化とも不採用（70例がmulti-region、相互排他は誤り） |
| 疑似ラベルの形 | 二値化も確率もせず、**同一領域のbag間ペアワイズ順位蒸留**（CAMが実証したのは順位情報だけ） |
| 損失 | `L = L_whole + λ_k(L_H + α_k L_P)`。件数比例の重み付けは禁止。人手・疑似を別々に平均 |
| branch | 4領域maskでmask-normalized pooling → **BiLSTM重みは4領域で共有しBaseline 0から初期化**。領域固有パラメータは小さなprojectionとscalar headのみ |
| 成功判定 | student > teacher CAM は必要条件にすぎない。**human-only arm に対する増分**で判定 |

### 計画書との重要な相違

- 計画書は「領域の教師が268から**全データ規模**へ広がる」としているが、Codexは陰性12,100 bagへの
  論理的0投入に反対。理由は「論理ゼロは正しいが**どの領域かの情報を全く持たない**」ため、
  4出力すべてが「骨折があるか」を学ぶのが最短経路になり、`mtl_type2`で観測した ρ≥0.97 の
  collapseを**別原因で再現しうる**こと。Codex案では疑似ラベルの拡大は 268 → 1,332（5倍）に留まる
- 参考論文（Telesco et al. 2025）のteacherは少数の詳細ラベルで直接学習されている。今回は
  「領域ラベルを一切見ていない分類器のCAM」を教師にするため方法論的な飛躍があり、
  同論文は着想源ではあっても本手法を直接支持する先行証拠ではない、と指摘

### 統計的検出力（Codex概算）

teacher CAMへの患者bootstrapで単一AUROCのSEは約0.030–0.035。paired AUROCのMDEは
student–teacher相関0.9で0.04–0.05、0.8で0.05–0.06、4領域Holm補正まで見込むと
**実質0.06–0.09**。したがって +0.02 程度の改善は「意味がない」ではなく
「**この標本では判定不能**」。

---

## 2. 生成段階のkill criteria判定（GPU不要な項目）

既存の `gradcam_annotated/attention_metrics.csv`（268 bag / 160 study）だけで判定できる
ゲートを実測した。患者クラスタbootstrap 10,000回、seed 20260823。
この2つの表（領域別AUROC + CI、左右弁別win rate）は第6節の監査CLIへ取り込み済みで、
`cam_audit_localization.csv` と `cam_audit_verdict.json` の `laterality` として再現できる。

### zero / undefined CAM

| 項目 | 実測 | ゲート | 判定 |
|---|---:|---|---|
| `cam_zero` bag | 0 / 268 (0.00%) | <1% | **PASS** |
| undefined density cell | 0.00% | <1% | **PASS** |

### 領域別CAM AUROC と患者クラスタ95% CI

| 領域 | 陽性 | 陰性 | AUROC | 95% CI | AP |
|---|---:|---:|---:|---|---:|
| R1 椎体 | 78 | 167 | 0.7980 | [0.7352, 0.8552] | 0.6451 |
| R2 右横突孔 | 59 | 184 | 0.7856 | [0.7256, 0.8431] | 0.5177 |
| R3 左横突孔 | 72 | 172 | 0.7882 | [0.7135, 0.8529] | 0.6419 |
| R4 後方要素 | 158 | 93 | 0.7356 | [0.6662, 0.8011] | 0.8143 |

- ゲート「R2/R3 AUROC ≥ 0.70」: **PASS**
- ゲート「R2/R3 CI下限 > 0.50」: **PASS**（0.726 / 0.714）
- ゲート「R2/R3 within-level AUROC ≥ 0.65」: **PASS**（既存実測 0.767 / 0.745）

### 左右弁別（R2 xor R3）

有効性maskを両側に課したうえで R2≠R3 となる 82 bag / 64 study。

| 項目 | 実測 | ゲート | 判定 |
|---|---:|---|---|
| correct-side win rate | **0.8537** 95% CI [0.7692, 0.9286] | >0.55 | **PASS** |

領域教師を1件も使っていないモデルが、左右どちらの横突孔かを85%当てている。
これは椎骨動脈損傷リスクという本研究の臨床動機に直結する指標であり、
疑似ラベルの種としての妥当性を最も強く支持する結果である。

### collapse耐性（`mtl_type2`失敗との対比）

4領域CAMスコアのSpearman相関:

| | R1 | R2 | R3 | R4 |
|---|---:|---:|---:|---:|
| R1 | 1.000 | 0.362 | 0.177 | **-0.806** |
| R2 | | 1.000 | -0.220 | -0.345 |
| R3 | | | 1.000 | -0.353 |
| R4 | | | | 1.000 |

argmax領域の分布: R1 50 (18.7%) / R2 52 (19.4%) / R3 71 (26.5%) / R4 95 (35.4%)。

`mtl_type2` の崩壊状態（相関 0.973–0.983、56例中55例=98%がR4）とは**質的に別物**である。
唯一大きい R1–R4 の -0.806 は負の相関で、椎体と後方要素が面積・位置で相補的であることの
反映であり、collapseの兆候ではない。

### 未判定（GPUが必要）

| ゲート | 内容 | Kill criterion |
|---|---|---|
| teacher memorization | teacher訓練bag vs teacher未学習innerでCAM成績を比較 | 2領域以上でAUROC差 >0.05 |
| mask境界感度 | maskを1 feature cell相当 erosion/dilation/shift、TTA再計算 | 領域順位Spearman <0.80 または argmax変化 >10% |
| provenance | 疑似ラベル各行にteacher ID・訓練fold・checkpoint hashを記録 | outer/innerが1件でも混入 |

---

## 3. 併せて確認した既存実装の状態

- **`region_target_valid` の不備は未修正**。`common/canonical_dataset.py:98` と
  `common/dataset.py:158` は `has_region_target` が真なら4領域すべてをvalidにする。
  2026-08-21のworklogで指摘済みの、未注釈runの`0`を陰性教師にしてしまう問題。
  **`mtl` の `baseline1_b` outer0〜3 はこの状態で学習済み**
- 正式6アーム実験は停止中。`baseline1_b` は outer0〜3 完了・**outer4未実施**、
  `control_b` と Proposed 3構成は未着手。GPU 0/1/2 はすべて空き
- Grad-CAM基盤は再利用可能。`baseline0/cli/attention.py` の `run_analysis` は
  fold別checkpointでbagをbatch処理する構造で、`--selection` に全bag/陽性bagの
  選択肢を足すだけで生成段階へ拡張できる

---

## 4. 決定待ちだった3点（→ 第5節で解決）

1. **領域損失の母集団**: 骨折陽性1,332のみ（Codex推奨・条件付き局在）か、
   計画書どおり陰性12,100へ論理的0を入れて全13,432にするか。
   後者は2026-08-17の「論理的0教師は使わない」というPI決定の解禁にあたる
2. **疑似ラベルの形**: bag間ペアワイズ順位蒸留（Codex推奨）か、二値化か、ソフト確率か
3. **停止中の正式6アーム実験の扱い**: `baseline1_b` outer4以降を先に完走させるか、
   疑似ラベル路線を優先するか

---

## 5. ユーザー決定（2026-08-23）

| 論点 | 決定 |
|---|---|
| 領域損失の母集団 | **両方を独立アームで比較**（骨折陽性1,332のみ / 論理的0を入れた全13,432）。どちらが正しいかをデータに決めさせる |
| 疑似ラベルの形 | **bag間ペアワイズ順位蒸留**（Codex推奨）。二値化・ソフト確率とも不採用 |
| 正式6アーム実験 | **保留**。`baseline1_b` outer4以降とcontrol_b・Proposed 3構成は起動しない |
| 残りの監査 | **実装より先に実施** |
| teacher割当 (a)/(b) | **監査結果を見てから決める**。判定基準（2領域以上でAUROC差0.05超）だけ先に固定 |

論理的0アームを残す決定により、2026-08-17の「論理的0教師は使わない」は
「主解析では使わないが、対照アームとして比較する」へ変わった。

---

## 6. 実装（監査ツール）

`Baseline 0`のGrad-CAM基盤を再利用して監査CLIを追加した。**学習は行わず、疑似ラベルも
書き出さない。** 生成へ進んでよいかだけを判定する。

### 追加・変更したファイル

| ファイル | 内容 |
|---|---|
| `baseline0/analysis/cam_audit.py` | mask摂動グリッド、面積補正CAM密度、teacher役割判定、水平反転ヘルパ |
| `baseline0/analysis/cam_audit_report.py` | 役割別AUROC・paired bootstrap CI・SMD、摂動安定性、TTA安定性、ゲート判定 |
| `baseline0/cli/cam_audit.py` | 5 checkpoint × 268 bagの採点と成果物出力 |
| `baseline0/tests/test_cam_audit.py` | 18テスト |
| `baseline0/analysis/gradcam.py` | GPU上のeval modeでcuDNN RNN backwardが失敗する問題を修正 |
| `baseline0/cli/attention.py` | `attach_annotation_validity`を公開名へ（監査から再利用） |

### 設計上の判断

- **役割の割当**: 凍結protocolが`outer=k / inner=(k+1)%5 / train=残り3`なので、
  268 bagを5 checkpoint全部で採点すると、同じbagが3回`train`（in-sample）、1回`inner`、
  1回`outer`として現れる。`inner`と`outer`はbagごとに一意になる
- **`train`役割の代表値は最小teacher index**。3つのteacherのスコアを平均すると
  ノイズが減ってAUROCが上がり、単一teacherの`outer`スコアとの比較が壊れるため採らない
- **mask摂動のゲートは4 px（1.6 mm）のみ**。8 px以上は感度曲線として記録するだけ
- **膨張・平行移動後のmaskは椎体全体maskで切り取る**。椎体の内訳を誤っても
  組織が椎体の外へは出ないため、また椎体全体密度が比の分母だから
- **未定義スコアは除外し、低スコアとして扱わない**。摂動で小領域が消えた場合や
  CAM質量が0の場合は分母が定義できない
- **水平反転TTAは判定に使わない**（下記の実測により）

### 実測で判明した既存モデルの性質

smoke実行中に、**Baseline 0のbag確率が水平反転で大きく振れる**ことが分かった。
同一bag・同一checkpointで `0.986 → 0.029`、`0.771 → 0.011`、`0.996 → 0.376` など、
16 bagでの平均絶対差は0.257。Baseline 0は`horizontal_flip_probability=0.5`で
学習済みであるにもかかわらず、個々の予測は反転不変ではない。

これは疑似ラベル計画とは独立した観察だが、次の意味を持つ。

- 反転TTAは「CAM集計の安定性」ではなく「モデル自体の反転安定性」を測っている。
  したがってゲートには使わず記述的指標として残す
- 逆に、CAMを反転TTAで平均することは分散低減として有効な可能性がある。
  疑似ラベル生成時の選択肢として登録する

### 検証

- 監査の`identity`/`tta=none`/`role=outer`のbag確率が、正式成果物の
  `recomputed_fp32_score`と6桁一致することを実データ6 bagで確認（パイプラインの正当性）
- 18 unit tests追加。`fracture_detection`全体で**175 tests passed**
- Ruff check/format、mypy（新規3ファイルは既存stub不足を除きクリーン）
- 実データsmoke: 16 bag × 5 teacher × 15 mask変種 + 反転TTAが完走

### 実行

```bash
uv run python -m fracture_detection.baseline0.cli.cam_audit --device cuda:0 --batch-size 8
```

A6000 1枚で約20〜25分。出力は
`fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/cam_generation_audit/`。

---

## 7. 正式監査run（本番268 bag、2026-08-23実施）

```bash
uv run python -m fracture_detection.baseline0.cli.cam_audit --device cuda:0 --batch-size 8
```

268 bag × 5 checkpoint × 15 mask変種 + 反転TTAが完走（GPU 0、約26分）。
出力: `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/cam_generation_audit/`
（`outputs/`配下は既存`.gitignore`により追跡対象外）。

### 結果: `proceed_to_pseudo_label_generation: false`

#### memorizationゲート: 完全PASS

| 領域 | AUROC train (in-sample) | AUROC outer (held-out) | 差 | 95% CI |
|---|---:|---:|---:|---|
| R1 椎体 | 0.8124 | 0.7980 | +0.0144 | [-0.053, 0.085] |
| R2 右横突孔 | 0.8172 | 0.7856 | +0.0317 | [-0.026, 0.091] |
| R3 左横突孔 | 0.8177 | 0.7882 | +0.0295 | [-0.024, 0.088] |
| R4 後方要素 | 0.7764 | 0.7356 | +0.0408 | [-0.018, 0.102] |

0領域がAUROCゲート（差>0.05）に抵触、0領域がSMDゲート（>0.25）に抵触。
**teacherがtrain bagを記憶して局在精度を水増ししている証拠はない。**

#### mask境界感度ゲート: `erode_4`のargmax副条件のみFAIL

ゲート対象6変種（erode_4 / dilate_4 / shift_x±4 / shift_y±4）のうちSpearman順位相関は
**全変種・全領域で0.80を上回った**（最低はerode_4のR2で0.829）。argmax変化率（bag内で
4領域中どれが最大スコアかの入れ替わり）は `erode_4` のみ全領域で0.200（>0.10ゲート抵触）、
他5変種は0.052〜0.082でPASS。

erosion半径を振ると argmax変化率は 2px 0.083（PASS） → 4px 0.200（FAIL） → 8px 0.300 と
単調悪化。8pxではR2/R3の**領域未定義率が73〜75%**（横突孔が侵食で消滅）に達する一方、
dilation・shiftは8pxでも安定（argmax変化0.11〜0.13、Spearman大半0.92以上）。
**侵食方向だけが不安定で、膨張・平行移動は安定**という非対称なパターン。

### Codexへの解釈相談（全文: `.claude/docs/codex/20260823-cam-audit-gate-interpretation.md`）

結論: **「進めてよい」。ただし事前登録上は「ゲート合格」とは書けず、「ゲート逸脱を承認して
継続」と明記すること。**

1. **argmax失敗は採用済みの疑似ラベル方式に直接関係しない**。bag間ペアワイズ順位蒸留は
   領域r内のbag間順位だけを使い、4領域間のargmaxには依存しない。整合する指標は
   領域内Spearman順位相関で、こちらはゲート対象6変種全てでPASS（0.829〜0.994）
2. **限定事項として明記すべき**。「境界侵食に対する領域スコアの感度、特にR2/R3の消失と
   領域間winnerの不安定性」と記載し、**以後、領域間argmax・softmax・領域横断のスコア比較には
   使わない**
3. **teacher割当はfold-matched `Teacher_k`を推奨**。memorization水増しが検出されず
   最大AUROC差も0.041未満だったため。bagごとに異なるteacherを混ぜる案は
   teacher間のスコア尺度差がbag間ランキングに混入し、学生のouter foldを学習した
   teacherを経由しうるため不採用。**比較ペアは必ず同一teacher・同一領域内で作る**
4. 追加の停止要因なし。実装前に同点・近接スコアと未定義領域の扱いを固定すること。
   Kendall順位相関／ペア反転率をerode 4pxで記述的に出すとpairwise lossとの対応が
   より直接的（ただし新しい事後ゲートにはしない）

### 進行承認（2026-08-23）

ユーザーは、上記を「ゲート合格」とは扱わず、`erode_4`の領域間argmax副条件の逸脱を
明記した上で、fold-matched `Teacher_k`による疑似ラベル生成へ進むことを承認した。

書面に必須の制限2点:
- 4px侵食で事前登録argmaxゲートが失敗（20.0%）し、R2/R3では領域消失も発生したため、
  **領域間winnerや領域横断比較には使用しない**
- 監査は268 annotated bags上の内部検証であり、未注釈の疑似ラベル対象集団への一般化と
  水平反転に対するteacherの頑健性は保証されない

### ユーザー指摘（2026-08-23）: argmaxはそもそも本研究の対象指標ではない

上記1点目の「限定事項」は不正確だった。argmaxは**採用済みの疑似ラベル方式が使わないだけ**
ではなく、**2026-08-17の設計時点から本研究のどの段階でも使う予定のない量**である。

- 268例中70例が複数領域陽性であり、4領域は互いに排他ではない
- Codex Q3dで確認済み: 4領域内のsoftmax・rank正規化とも不採用（「70例がmulti-regionであり、
  softmaxは誤った相互排他制約」）
- 各領域は独立に「骨折があるか／ないか」を判定する設計であり、これはこれまで測ってきた
  領域別AUROC（R1 0.798 / R2 0.786 / R3 0.788 / R4 0.736、いずれも他領域と独立に算出）と
  同じ枠組み

したがって「erode_4のargmax失敗」は、そもそも測る必要のない指標がゲートに紛れ込んでいた
ことによるものであり、単に「今回選んだ方式には無関係」以上に、**本研究のどの構成でも
無関係な失敗**と訂正する。ゲート設計（`perturbation_table`のargmax_change_rate列）自体は
今後も記録は残すが、判定根拠としては使わない。**進行判断はより強い根拠に基づく。**

### 未判定分の解消

第2節「未判定（GPUが必要）」の3項目を解消した。teacher memorizationとmask境界感度を
監査runで確認し、provenanceは疑似ラベル各行へteacher ID・訓練fold・student outer fold・
checkpoint hashを記録して整合性監査を通過した。


---

## 8. 疑似ラベル生成の第一段実装（2026-08-23）

ユーザー承認（argmax失敗は本研究のどの構成でも無関係という訂正込み）を受けて、
疑似ラベル生成と順位ペア構成を実装した。**モデル構成・実学習は未着手**。

### 追加したファイル

| ファイル | 内容 |
|---|---|
| `baseline0/analysis/pseudo_label.py` | 温度・soft順位target・循環ペア構築・領域均等pairwise loss |
| `baseline0/cli/generate_pseudo_labels.py` | fold-matched in-sample `Teacher_k`で学習3 fold全bagを採点するCLI |
| `baseline0/tests/test_pseudo_label.py` | 29テスト |

### 設計上の判断

- **教師割当はfold-matched in-sample `Teacher_k`に確定**（Codex推奨、本番監査の
  memorization結果を根拠）。outer fold kの学生の学習集合＝Teacher_kの学習集合＝
  完全一致なので、比較ペアは常に同一teacher・同一スケール内
- **温度`T_r`はサンプリングしたペアのlogスコア差のIQR**。全ペア（学習foldは最大8,074 bag→
  約3,260万ペア）を数え上げず、固定seedで最大10万ペアを抽出して近似する。決定論的なので
  スコア生成物と温度をセットで凍結する必要がない（再計算すれば必ず同じ値になる）
- **`pairwise_ranking_loss`は教師スコアへ勾配を流さない**（`.detach()`）。教師は固定済み
  Baseline 0であり、学習対象は学生のregion logitだけ
- **温度はfold-matched teacherの骨折陽性train bagだけ**から計算する。CAM順位ペアも
  骨折陽性内に限定し、負例同士・陽性負例間のCAMペアは作らない。論理的0込みアームは
  exact-negative BCEを別sourceとして追加する

### 未実装として明示的に残した点

- 論理的0込みアームに追加するexact-negative BCEのsource-balanced weighting
- 4領域maskでのmask-normalized pooling、Baseline 0からのBiLSTM転移込みモデル構成
- λ/α勾配校正、MTL trainerへの統合、provenance（teacher ID・fold・checkpoint hash）を
  実際の学習ループでどう記録するか
- 実際の学習run（ユーザーの手動起動が必要）

### 検証

- 29 unit tests。`fracture_detection`全体で**207 tests passed**
- Ruff check/format、mypy（新規2ファイルは既存stub不足以外クリーン）
- 実データsmoke: 5 outer fold × 40 bag、200行・81 bag分のスコアと温度表を生成、
  `teacher_checkpoint_sha256`によるprovenanceを確認
- 本番生成: 40,296行・13,432 bag（各bag 3 teacher）、温度20件。骨折陽性の未定義スコアは
  全20 teacher-region集計で2件。fold割当、重複なし、各bag 3行、provenance、成果物SHA256を検証
- 実データouter0の802陽性bagで3,207ペア（R1/R2/R3/R4=802/802/801/802）を生成し、
  自己ペアなし、有限loss、有限gradientを確認

---

## 9. 部分注釈validity修正（2026-08-23）

学生モデル実装前の必須修正として、`has_region_target=True`なら4領域すべてをvalidにしていた
`common/canonical_dataset.py` / `common/dataset.py`の不備を修正した。

- 注釈run coverageを`common/region_validity.py`へ共通化し、manifest生成時に領域別validityを固定
- 235完全注釈bagは4領域valid、33部分注釈bagは記録済み陽性だけvalid
- 領域別valid cellは245/243/244/251、部分注釈bagの未確認zero cell 89件をunknown化
- 水平反転時にR2/R3のvalidityもtarget・maskと同期交換
- prediction CSVへ領域別`*_target_valid`を保存
- input manifest SHA256は
  `b5d46161c374b38456393c0dfd65893d535f12eb17595c42f0c78c2b4a36b955`へ更新
- 全210 tests、Ruff、実partial-zero bag smoke、stub不足を除いたmypyが通過

既存`baseline1_b` outer0〜3の誤陰性教師は遡及修復できない。新しい疑似ラベルMTLは旧runを
resumeせず、修正版manifestを含む新しいfrozen experiment manifestを作る。

---

## 10. セッション進捗保存（2026-08-23、ユーザー確認済み）

ユーザー確認「おけ」を受け、ここまでを再開可能なcheckpointとして保存する。

### 完了済み

1. CAM生成段階監査を完了。memorizationゲートはPASS、`erode_4`の領域間argmax副条件だけ
   形式上FAIL。argmaxは独立multi-label設計で使用しないため、ゲート逸脱を明記して継続承認済み
2. fold-matched in-sample `Teacher_k`で全13,432 bagを採点。40,296 score行、温度20件を生成
3. 骨折陽性・同一teacher・同一領域内の固定seed循環ペアを実装。同点・近接差はsoft target、
   未定義2件は除外、`human+pseudo`では人手validセルをpseudo lossから除外
4. 部分注釈validityを修正。235完全注釈bag / 33部分注釈bag、未確認zero 89セルをunknown化
5. 修正版input manifest、prediction validity列、水平反転時のR2/R3 validity同期交換を実装

### 固定成果物

- 疑似スコア: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_label_scores.csv`
  - 40,296行 / 13,432 unique bag
  - SHA256: `2a78aededc11b3231aaf906cbf907e2104486c6db18205d6a0d79f212bfea22f`
- 温度表: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_label_temperatures.csv`
  - SHA256: `ed91915d0b30e0a83a0c2f72898ec10d6b9da402c4362249998c262ba426559b`
- 修正版manifest: `fracture_detection/common/outputs/input_manifest.csv`
  - SHA256: `b5d46161c374b38456393c0dfd65893d535f12eb17595c42f0c78c2b4a36b955`

### 検証状態

- `fracture_detection`全210 tests passed
- Ruff check/format passed
- `git diff --check` passed
- unavailable type stubsと既存unused-ignoreだけを抑制したmypyで変更6 source files passed
- 実partial-zero bag `1.2.826.0.1.3680043.13203 / C6`で
  target `[1,1,1,0]`、valid `[true,true,true,false]`を確認

### 再開時の次工程

1. 学生モデルを実装: 4領域mask-normalized pooling、4領域共有BiLSTM、Baseline 0から転移、
   小さなregion固有projection/embedding/scalar head
2. 論理的0込みアームのexact-negative BCEをsource-balanced化し、重み規則を固定
3. human-only / pseudo-only / human+pseudo / shuffled-pseudo controlを同一architectureで実装
4. λ/α gradient calibrationと短縮inner pilotを実装し、outer推論前にkill criteriaを確認

### 再開時の注意

- `baseline1_b` outer0〜3は修正前validityで学習済み。結果を新MTL比較へ流用しない
- 旧manifest hashを持つrunを新manifestでresumeしない
- 新MTLは修正版manifestと疑似成果物hashを含む別frozen experiment manifestを作る
- 現時点で学生モデル・trainer統合・新MTL学習runは未着手
