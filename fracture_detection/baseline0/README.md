# Baseline 0：椎体単位骨折分類

4領域教師を使わず、CT 5 channelと椎体全体mask 1 channelから椎体骨折を分類する比較の起点です。
正式な再学習は旧`baseline0`専用loopではなく、6構成共通の`fracture_detection/core/`を使います。

## モデル

- 入力: canonical 10 channelから先頭6 channelを選択した`[B,15,6,224,224]`
- backbone: ImageNet事前学習済み`tf_efficientnetv2_s`
- sequence: hidden 256、2層BiLSTM
- head: `Linear -> BatchNorm -> Dropout(0.3) -> LeakyReLU -> Linear`
- 出力: 15面logit。bag確率は面sigmoidの平均

canonical datasetは全アーム共通でCT、whole mask、4領域maskを同期変換します。Baseline 0のmodel adapterは
4領域channelを入力へ渡さず、annotated streamも実行しません。これによりnatural sampler、augmentation、
mixup、optimizer step数を他アームと共有します。

## 学習契約

- 品質除外済み`13,432 bag / 2,009 study / 陽性1,332`
- 凍結済み5-fold、`outer=k / inner=(k+1)%5 / train=残り3 fold`
- natural batch 16、BF16、AdamW、weight decay `1e-4`
- LR `2.3e-4 -> 2.3e-5`、75 epoch単一cosine周期
- `pos_weight=2.0`、mixup `p=0.2`、gradient clipなし
- laterality-safe horizontal flip `p=0.5`。R2/R3のmask値・教師を同時交換
- vertical flipとtransposeは禁止

正式runはλ/β校正、5構成resource profile、6構成config、source/dependency/input/fold hashを含む
`frozen_experiment_manifest.json`が揃わない限り開始できません。smokeはcheckpointとinner検証まで実行し、
outer推論を行いません。

## 実行

このproject専用のentry pointから起動します。configはproject内の`shared_core.yaml`へ
固定済みで、armは1つなので`--arm`は省略できます。

```bash
uv run python -m fracture_detection.baseline0.cli train \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

`--outer-fold`を省くと`parallel.gpu_ids`へ5 foldを並列割当します。resource profileは
`profile`、W&B同期は`sync-wandb` subcommandです。実装は共通の`fracture_detection/core`と
`fracture_detection/cli`にあり、entry pointは委譲のみを行います。

```bash
uv run python -m fracture_detection.baseline0.cli train --resume
uv run python -m fracture_detection.baseline0.cli profile \
  --output fracture_detection/profiling/outputs/<phase>/baseline0.json
```

共通CLIを直接叩く旧来の形式も同じ実装へ到達します。

```bash
uv run python -m fracture_detection.cli.train \
  --config fracture_detection/baseline0/config/shared_core.yaml \
  --outer-fold 0 --gpu-id 0 --smoke-steps 1
```

正式runでは`--smoke-steps`を外します。run開始時とresume時にfrozen manifestを照合し、hash driftを拒否します。

W&Bは全共有アームで既定ONです。foldごとに`{experiment.name}-outer{k}`を作成し、
`history.csv`と同じepoch指標を記録します。過去にW&B無効で実行したrunは、再学習せず同期できます。

```bash
uv run python -m fracture_detection.baseline0.cli sync-wandb
```

`sync-wandb`はconfigから成果物rootを解決します。任意のディレクトリを指定する場合は
共通CLIを使います。

```bash
uv run python -m fracture_detection.cli.sync_wandb \
  --experiment-dir fracture_detection/baseline0/outputs/08_19/baseline0_shared_core
```

同期前に`uv run wandb login`でW&Bへログインしておく必要があります。

## 成果物

`fracture_detection/baseline0/outputs/{phase}/{name}/outer{k}/`へ次を保存します。

- `effective_config.yaml`
- `best_model.pt` / `best_val_prauc_model.pt` / `last_checkpoint.pt`
- `history.csv`
- `outer_predictions.csv` / `outer_predictions_prauc_checkpoint.csv`（正式runのみ）
- `summary.json`
- `wandb_run_id.txt`（W&B有効時）

正式outer予測には参照したfrozen manifestのSHA256を埋め込み、pooled OOF解析で全foldの一致を再検証します。

## 検証

```bash
uv run pytest fracture_detection -q
uv run ruff format --check fracture_detection
uv run ruff check fracture_detection
```

## 注目領域の可視化

正式OOFの各bagを、そのbagがouter評価されたfoldの`best_model.pt`でGrad-CAM解析できます。
最終CNN空間特徴`encoder.bn2`からbag確率へ逆伝播し、症例別15面図に加えて、4解剖領域内の
CAM質量比と領域面積で補正したCAM密度比を保存します。既定ではTP/FPの各fold・各椎体レベルから
最高スコアを1件ずつ選ぶため、特定foldやC2だけに偏りません。

```bash
uv run python -m fracture_detection.baseline0.cli.attention \
  --device cpu
```

少なくとも1 runに4領域アノテーションがある268 bagをすべて解析する場合は、次を実行します。

```bash
uv run python -m fracture_detection.baseline0.cli.attention \
  --selection annotated --device cpu
```

出力先は正式run直下の`gradcam_attention/`で、`--selection annotated`では
`gradcam_annotated/`です。`attention_metrics.csv`、
`region_summary.csv`、`annotated_target_summary.csv`、`region_summary.png`、および代表症例の
`cases/*.png`を保存します。椎体内CAMは画像内の椎体面積も併記し、面積補正密度を計算します。
`annotated_localization_metrics.csv`には領域教師の陽性・陰性密度差、患者cluster bootstrap区間、
AUROC/AP、および椎体レベル内順位による感度解析を保存します。
run coverageも自動照合し、未注釈runが残るbagでは記録済みの陽性だけを有効とし、0はunknownとして
領域別指標から除外します。現在は268 bag中235 bagが全run確認済みで、33 bagに36未注釈runがあります。
`annotation_coverage.csv`と各領域の`n_unknown`で除外状況を確認できます。
Grad-CAMは7x7最終特徴からの事後説明であり、骨折部位の画素単位教師や因果的根拠ではありません。

## 疑似ラベル教師信号の生成前監査（`cli.cam_audit`）

`memo/進捗/研究計画書_2026-08-21.md` の疑似ラベルMTLは、このBaseline 0のGrad-CAM領域密度を
4領域教師の種として使います。生成に着手する前に、その信号が使い物になるかを判定する監査です。
**学習は一切行わず、疑似ラベルも書き出しません。**

```bash
uv run python -m fracture_detection.baseline0.cli.cam_audit \
  --device cuda:0 --batch-size 8
```

268 bagすべてを5つのcheckpoint全部で採点します。凍結nested protocolは
`outer=k / inner=(k+1)%5 / train=残り3 fold`なので、同じbagが3つのteacherにとって
in-sample（`train`）、1つにとってcheckpoint選択のみ（`inner`）、1つにとって完全未見（`outer`）
になります。この3役割の比較が監査の中身です。

判定する2つのゲート（根拠: `.claude/docs/codex/20260823-pseudo-label-mtl-design.md`）:

| ゲート | 内容 | 中止条件 |
|---|---|---|
| teacher memorization | `train`役割と`outer`役割のCAM領域AUROC差 | 2領域以上で差 >0.05、または条件付きscore SMD >0.25 |
| mask境界感度 | 4領域maskを収縮/膨張/平行移動した際の領域スコア順位 | Spearman <0.80、またはargmax変化率 >10% |

mask摂動は erode/dilate `2/4/8 px`、shift `±4/±8 px` を全領域へ適用します。入力は0.4 mm/pxなので
4 pxは1.6 mmにあたり、これが妥当なセグメンテーション誤差としてゲート対象です。8 px以上は
感度曲線として記録するだけで判定には使いません。膨張・平行移動後のmaskは椎体全体maskで
切り取ります（椎体の内訳を誤っても組織が椎体の外へ出ることはないため）。

水平反転TTAは`outer`役割のteacherだけで計算し、**判定には使いません**。Baseline 0は
`horizontal_flip_probability=0.5`で学習済みですが、個々のbagのスコアは反転で大きく振れるため、
この指標はCAM集計の安定性ではなくモデル自体の反転安定性を測っています。

左右弁別（R2 xor R3の椎体で正しい側のスコアが高い割合）も`outer`役割で算出し、
`cam_audit_verdict.json`の`laterality`へ保存します。R2/R3は鏡像の同種構造なので、
片側だけが陽性の椎体はCAMが正しい側を指すかの直接の検定になり、椎骨動脈損傷リスクという
臨床動機に最も近い指標です。

出力先は既定で正式run直下の`cam_generation_audit/`です。

| ファイル | 内容 |
|---|---|
| `cam_audit_scores.csv` | bag × teacher × TTA × mask変種の全スコア（長形式） |
| `cam_audit_localization.csv` | `outer`役割の領域別AUROC・患者cluster CI・AP（teacher信号そのもの） |
| `cam_audit_memorization.csv` | 役割別AUROC、paired差の患者cluster bootstrap CI、SMD |
| `cam_audit_mask_perturbation.csv` | mask変種ごとのSpearman・argmax変化率・AUROC |
| `cam_audit_tta.csv` | 水平反転TTAの順位安定性（記述的） |
| `cam_audit_verdict.json` | ゲート判定、左右弁別win rate、`proceed_to_pseudo_label_generation` |

所要時間はA6000 1枚で約20〜25分です。`--limit-bags`でsmoke実行できます。


## 疑似ラベル生成（`cli.generate_pseudo_labels`）

生成段階監査（`cli.cam_audit`）のゲート逸脱を承認して継続する決定（詳細は
`.claude/docs/work-logs/2026-08/2026-08-23-pseudo-label-mtl-design-review-and-cam-gate-audit.md`
第7節）に基づき、実際に学生モデルが使う疑似ラベルの元スコアを生成します。事前登録上、
監査全体を「ゲート合格」とは扱いません。

**教師割当は fold-matched in-sample `Teacher_k` に確定**（memorizationゲートで水増しが
検出されなかったため）。outer fold `k`の学生は、`Teacher_k`（そのouter foldを学習していない
Baseline 0 checkpoint）で、`Teacher_k`自身の学習3 fold全bag（品質除外済み全13,432 bag中の
約8,050 bag）を採点します。学生の学習集合と教師のスコア元集合が完全一致するため、
比較ペアは常に同一teacher・同一スケール内で作れます。

```bash
uv run python -m fracture_detection.baseline0.cli.generate_pseudo_labels \
  --device cuda:0 --batch-size 16
```

出力は既定で`fracture_detection/baseline0/outputs/08_19/pseudo_labels/`。

| ファイル | 内容 |
|---|---|
| `pseudo_label_scores.csv` | bag × teacher_outer_foldの領域密度スコア（面積補正CAM density）とteacher ID・訓練fold・checkpoint SHA256 |
| `pseudo_label_temperatures.csv` | outer fold × 領域ごとの温度`T_r`（骨折陽性bagのCAM分布だけを使う固定規則） |
| `pseudo_label_generation_metadata.json` | 生成設定、teacher割当、score・temperature成果物のSHA256 |

### ペアワイズ順位蒸留（`analysis/pseudo_label.py`）

CAMが実証したのは順位情報（AUROC）だけで確率校正は未実証、という
Codexレビューの結論に基づき、疑似ラベルは絶対確率ではなく**同一領域のbag間ペア比較の
確信度**として構成します。

```
u_ijr = sigmoid( (log C_ir - log C_jr) / T_r )
L_P   = BCEWithLogits(z_ir - z_jr, u_ijr)
```

`T_r`はグリッドサーチではなく、その教師・その領域の訓練foldに含まれる骨折陽性bagの
CAMスコア差のIQR（四分位範囲）で固定します（`region_temperature`、seed固定で決定論的）。差が
典型的なばらつきより大きいペアほど確信度が1に近づき、僅差のペアは0.5付近の弱い教師に
なります。`pairwise_ranking_loss`は教師スコアへ勾配を流しません（`.detach()`）。

`build_region_pair_batch`は同一teacherの骨折陽性bagを領域ごとに固定seedで並べ替え、
自己ペアのない循環ペアを作ります。未定義スコアは除外し、`human+pseudo`では
`human_target_valid`のセルも除外します。同点・僅差は除外せず、0.5付近の弱いsoft targetとして
残します。`region_balanced_pairwise_ranking_loss`は各領域内で平均してからactive領域を平均します。

### 未実装（次フェーズ）

- 論理的0込みアームに追加するexact-negative損失のsource-balanced weighting
- 4領域maskでのmask-normalized pooling、Baseline 0からのBiLSTM転移を含むモデル構成
- λ/α勾配校正、MTL trainerへの統合、実際の学習run
