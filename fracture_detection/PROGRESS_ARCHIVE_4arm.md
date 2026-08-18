# fracture_detection 作業進捗（アーカイブ: 旧4アーム計画・matched学習）

> **このファイルは2026-08-17の設計転換で廃止された旧計画の記録です。**
> 現行の計画・進捗は `PROGRESS.md` を参照してください。
>
> 旧計画は「領域アノテーション268 bagを軸に、一部症例（matched 2,655 bag）で学習する
> 4アーム比較」でした。fold 0診断runで過学習が実測されたため、
> 全13,928 bagを使うMulti-Task Learningへ全面的に切り替えています。
> **やっていることが根本的に異なるため、履歴を分離しました。**
>
> ここに書かれた設計判断・数値・次タスクは**現行計画には適用されません**。
> ただし以下の基盤成果は現在も有効で、`PROGRESS.md` へ引き継がれています。
>
> - `folds/outputs/folds.csv`（患者単位層別5-fold、seed 20260807、凍結・再生成禁止）
> - `common/outputs/input_manifest.csv`（13,928 bag / 2,010 study、SHA256固定）
> - 領域ラベルのOR集約規則と確定値（268 bag / 160 study / R1 78 / R2 59 / R3 72 / R4 158）
> - 15面固定・flip/transpose不使用・回転±40°などの入力仕様

---

## 旧全体像（2026-08-07〜2026-08-14）

4アームの比較実験（当時の `memo/計画書/提案手法.md`）:

| アーム | 内容 | 領域ラベル268の扱い |
|---|---|---|
| Baseline 1 | CT+全体mask → 椎体分類（CNN+LSTM） | 不使用 |
| Baseline 2 | 4領域独立モデル | 教師あり |
| 提案A | 半教師あり（teacher → pseudo-label → student） | 教師あり + pseudo-label |
| 提案B | 弱教師あり（y_whole = OR(y_regions)） | 評価専用 |

## 失効した前提（現行計画では不採用）

- Baseline 1 matched / Baseline 2 は同じ固定2,655 bag・1,498患者を使用する。
  内訳は領域アノテーション済み陽性268 bag・160患者 + 椎体陰性2,387 bag。
  陽性率10.094%は`full`の10.095%と一致し、陰性のfold別level別件数も`full`の陰性分布に比例させる
- Baseline 1は`matched`（2,655 bag・1,498患者）と`full`（13,928 bag・2,010患者）を切替可能にする
- Baseline 2の損失は通常BCE（P/H/N重み・pos_weight・focalなし）
- **matchedのbackboneは `tf_efficientnetv2_b0` が主解析**、`tf_efficientnetv2_s` は感度分析
- **モデル選択は val AUROC の early stopping**（2026-08-11ユーザー決定。
  Codex推奨の固定epoch+EMAは却下）。OOFの楽観バイアスは登録済み限界
  → 2026-08-17に廃止。outer foldは評価専用となり、cyclic single-inner-foldへ移行
- **ステージングは `full` のみ**。`matched`（2.8 GB）はページキャッシュに乗るため直読み
- 提案Aの teacher・pseudo-label は outer fold 内で完結させる
- 「6ch入力」は2-stem（image stem / mask stem）で吸収する
  → 2026-08-17に廃止。単純early fusionへ

## 旧プロジェクト一覧（2026-08-14時点）

| プロジェクト | ディレクトリ | 状態 | メモ |
|---|---|---|---|
| fold定義 | `folds/` | 完了(検証済) | folds.csv凍結（seed 20260807）。**現行も有効** |
| 共通基盤 | `common/` | 完了(検証済) | manifest / dataset / 標準BCE / 評価。**現行も有効** |
| matched cohort | `cohorts/` | 完了(検証済) | 2,655 bag・1,498患者を凍結。SHA256 `91de42ca0475b570efb9392218c9aca0b43ce05373ecf9ec761f8527c99c6bb1`。**現行では学習不使用** |
| Baseline 1 | `baseline1/` | 改訂fold 0完了 | ReduceLROnPlateau runはepoch 22停止、best epoch 7、AUROC 0.738、AP 0.310 |
| Baseline 2 | （未作成） | 未着手 | 4独立モデル / 通常BCE / 固定2,655 bag |
| 教師ありマルチタスク | （未作成） | 未着手 | 提案AのStudent骨格 兼 Teacher |
| 提案A | （未作成） | 未着手 | pseudo-label詳細は眼底論文読解を参照 |
| 提案B | （未作成） | 未着手 | smooth-max主 / max・noisy-ORアブレーション |

---

## 進捗ログ

### 2026-08-07

- 実装計画を確定（4アーム、フェーズ順: fold定義 → 共通基盤 → B1 → B2 → 教師ありMTL → A → B）
- ユーザー決定: 15面固定 / Baseline 2 は 268+椎体陰性bag / 実装場所は `fracture_detection/`
- **Phase 0 完了**（`folds/`）:
  - `check_dataset.py`: 268 annotated bag 全読込PASS（形状・dtype・mask非空・R2/R3陽性のmaskクラス存在）。
    SHA256指紋を `outputs/annotated_bag_manifest.csv` に記録（mask版数pin）
  - 当時のblind viewの全bag棚卸し: 14,054 bag / 2,012 study。**126 bagがregion_4class.npyのみ欠落**（アノテ済みとの重複0）
    → bag母集団を完備13,928 bagに確定。train.csvの7 studyは画像データなし
  - `make_folds.py`: 患者単位・貪欲層別5-fold生成。バランス実績:
    studies 402×5 / bags 2784-2787 / 陽性椎体 281-282 / アノテstudy 31-33 /
    アノテbag 53-54 / R1 15-16 / R2 11-12 / R3 14-15 / R4 31×5
  - 再実行で同一出力を確認（決定性）。`outputs/folds.csv` は凍結（上書きガード実装済み）
  - 途中、貪欲法のコスト関数バグ（限界変化でなく絶対偏差を最小化→3 foldに崩壊）を検出し修正
- **領域ラベルdedup規則の訂正**（同日、Phase 0完了後）:
  - 当初 `run_id` を「アノテーションのやり直し」と誤解し keep last run を採用していた
  - ツール実装（`Unet/dicom_bbox_annotation_tool`）とbboxスライス範囲を確認した結果、
    run = **同一椎体内の連続bboxグループ＝別々の骨折部位**（run間の空きは5〜50スライス）と判明。
    アノテータがrunごとに画像を見て判定している
  - ユーザー確認（各runのラベルは目視判定で正しい）を受け、**OR集約に修正**
  - 影響: R1 77→78 / R3 71→72 / R4 155→158（R2 59は不変）、複数領域陽性 65→70、R2 xor R3 94→95。
    bag数268・study数160は不変
  - `load_labels.py` をOR集約に書き換え、268 bag再チェックPASS、**foldを再生成**（バランス:
    アノテbag 53-56 / R1 15-16 / R2 11-12 / R3 14-15 / R4 31-32）

### 2026-08-11

- ユーザー決定により、全アームで **flip / transpose augmentationを使用しない**方針へ変更
- R2/R3のswap処理は共通datasetに実装せず、保存済み画像の方向をそのまま維持する
- `fracture_dataset_blind/`の構造を再監査し、234 studyがbbox症例の再生成実体、残り1,778 studyが`fracture_dataset/`へのsymlinkであることを確認
- blind実体234 studyを`fracture_dataset/`へ統合。内容差分1,712ファイル（2.27 GB）を上書きし、既存側だけのbagは保持
- 全実装の入力参照を統合済み`fracture_dataset/`へ変更。統合後は14,055 bag中13,928 bagが3ファイル完備で、学習母集団13,928 bag / 2,010 studyは不変
- **Phase 1 完了**（`common/`）:
  - fold・棚卸し・椎体ラベル・OR集約領域ラベルから共通manifestを生成
  - 13,928 bag / 2,010 study / 領域アノテーション268 bagを再確認し、SHA256で固定
  - CT 5chと全体+R1〜R4 mask 5chを別テンソルで返すdatasetを実装
  - 有効な領域targetと椎体陰性の論理的0 targetへ通常BCEを適用
    （**論理的0教師は2026-08-17に不採用となったため、実装時に削除が必要**）
  - 椎体AUROC/AP・領域別AP・SideAcc balanced・患者bootstrap評価を実装
  - 単体テスト4件、ruff、mypy、実データ読込を確認
  - モデルとモデル固有損失は`common/`へ置かず、各実験プロジェクトで実装する
- **Baseline 2計画を単純化**（陰性268 bag案は後続変更で廃止）:
  - 4領域それぞれの独立モデルに通常の`BCEWithLogitsLoss`を使用
  - アノテ268 bagと、別患者から1椎体ずつ選ぶ固定陰性268 bagを使用
  - P/H/N層別重み、`pos_weight`、focal lossは使用しない
- **Baseline間のデータ数を統一**（536 bag案は後続変更で廃止）:
  - Baseline 1 matchedとBaseline 2は同じ固定536 bag・428患者を使用
  - fold別・椎体level別の陰性件数をアノテーション側と一致させる
  - Baseline 1のみ全13,928 bag・2,010患者を使う`full`設定も用意する

### 2026-08-11（続き・Baseline 1 設計確定）

- Codexへ設計相談（`.claude/docs/codex/20260811-2100-baseline1-design.md`）。
  ユーザーが7論点のうち4件を決定:
  - bag確率は**旧方式維持**（15面broadcast + mean-sigmoid）。Codexのbag-level log-mean-exp案は却下
  - モデル選択は**旧方式維持**（val AUROC early stopping）。Codexの固定epoch+EMA案は却下
  - matchedのbackboneは**B0が主・V2-Sを感度分析**
  - fold分割は現状のまま（held-out test を作らない）
- 陰性プールを**骨折なし患者のみ**に確定。全35セル充足を実測で確認
- Codexが採用された推奨: 回転±10-12°、distortion/cutout/mixup OFF、
  matchedの3段階LR・drop 0.1/0.1/0.4・grad clip 1.0、maskはnearest-neighbor+強度変換なし
- **母数の誤りを修正**: 確証的評価の母数 14,133 / 陽性1,444 → **13,928 / 1,406**
- Codex CLIが `--full-auto` により `--sandbox read-only` を上書きして `DESIGN.md` を無断編集。
  該当セクションと changelog 1行を削除。以後 `--full-auto` は使わない

### 2026-08-11（Baseline 1 初回実装完了・旧50:50コホート）

- `cohorts/make_matched_cohort.py` により、`patient_overall == 0` を二重確認した固定matched cohortを生成
  - 536 bag / 428患者、annotated 268 + negative 268
  - fold×level件数一致、negative 1患者1椎体、annotated患者との重複0
  - frozen CSV SHA256: `b120cc7593e439ae58c44d4b8eb607505cb4b4a64120a951c32ab6feab058cb4`
- `baseline1/` にCT 5ch + whole-mask 1chの6ch adapter、同期augmentation、timm EfficientNetV2 + BiLSTM、
  15面broadcast BCE / mean-sigmoid、matched/fullの固定LR schedule、checkpoint resumeを実装
- `Unet/line_only`準拠のexperiment管理を実装
  - `experiment.phase/name`でローカル出力を分離し、実効configを保存
  - 1 fold = 1 W&B run。epoch BCE/AUROC/AP/LR/grad normとbest/final summaryを記録
  - checkpoint・OOF predictionはローカル正本で、W&B artifactへアップロードしない
- full用のstagingはinput-manifest SHA256単位の共有`/dev/shm` cacheとし、同時foldが再利用する
- OOF評価は入力ID、fold、target、score範囲、checkpointのfold設定を検証してからpoolし、
  椎体AUROC/APと患者cluster bootstrap CIを計算する
- 検証: related pytest **27 passed**、ruff format/check、mypy PASS、実データ1 bagのforward/loss/backward PASS

### 2026-08-11（matched cohort自然分布化）

- ユーザー決定により、陽性・陰性を268件ずつにした人工的な50:50分布を廃止
- アノテーション済み陽性268 bagを維持し、`full`の陽性1,406 / 13,928に対応する陰性2,387 bagを決定的に抽出
- 陰性のfold×level分布を`full`の陰性分布に比例させ、同一患者の複数椎体を許可
- 固定コホートは2,655 bag・1,498患者、陽性率10.094%、SHA256 `91de42ca0475b570efb9392218c9aca0b43ce05373ecf9ec761f8527c99c6bb1`
- 旧536 bag・428患者の固定成果物は学習未使用のため廃止し、新しい成果物へ置換

### 2026-08-11（Baseline 1 pos_weight追加）

- ユーザー決定により、Baseline 1の全設定へ固定`pos_weight=2.0`を追加
- 15面へ複製したターゲットに同じ重みを適用し、学習・検証BCEの両方で使用
- `config.py`は2.0以外や欠落を拒否し、3つのYAMLとW&B実効設定へ値を保存

### 2026-08-11（Baseline 1 fold実行範囲の設定化）

- 3つのYAMLへ`data.start_fold`と`data.end_fold`を追加し、包含範囲で学習対象foldを指定
- `data.n_folds=5`は固定済み分割の総数、`start_fold`/`end_fold`は今回実行する範囲として分離
- CLIの`--start-fold`/`--end-fold`は既定値を持たず、YAMLを一時的に上書きする場合だけ使用
- 範囲外、逆順、bool、欠落したfold範囲を設定検証で拒否

### 2026-08-12（Baseline 1 学習スクリプトの直接実行対応）

- リポジトリルートから`uv run python fracture_detection/baseline1/train.py`で起動可能に変更
- 直接実行時だけ不足するプロジェクトルートをimport検索パスへ追加し、`-m`実行も維持
- `--help`を使うsubprocess回帰テストを追加し、学習を開始せず起動経路を検証

### 2026-08-12（Baseline 1 学習進捗の可視化）

- マニフェスト読込、fold分割、DataLoader、モデル、W&Bの各初期化段階を即時flushで表示
- 各epoch開始時に、初回batchではworker起動とprefetchが発生することを表示
- train・validation・最良checkpoint再評価へbatch単位の進捗バーと実行中平均BCEを追加
- tiny学習で標準出力と進捗バーを検証する回帰テストを追加

### 2026-08-12（Baseline 1 matched学習スケジュール再検討へ）

- 旧536 bag前提の10 epoch凍結・200 epochスケジュールを、2,655 bagへそのまま適用した不整合を確認
- fold 0部分runはepoch 7でval AUROC 0.697・AP 0.225が最高、epoch 9 checkpointでは0.614・0.166へ低下
- epoch 9の平均scoreは陰性0.218・陽性0.254で分離が弱く、このrunは診断用smokeとして結果から除外
- 次セッションで凍結期間、backbone/head LR、warmup、cosine期間、最大epoch、min epoch、patience、総update数をまとめて再設計する
- スケジュール変更後は現在のfold 0 checkpointをresumeせず、新しい実験名で最初から学習する

### 2026-08-14（Baseline 1 matched学習スケジュール改訂）

- fold 0診断runはepoch 51まで完走し、train BCEは0.619から0.064へ低下したため「未学習」ではなかった
- val AUROCはepoch 21の0.737が最高、val BCEは同epoch 0.633からepoch 51の1.781へ悪化し、主因を過学習と判定
- 10 epoch backbone freezeを廃止し、全層を2 epochで0.1倍LRからwarmupする
- matchedはbackbone LR `1e-4`、head LR `3e-4`から開始し、`val_bce`停滞時に`ReduceLROnPlateau`で0.5倍へ下げる
- scheduler patience 4、relative threshold 0.1%、cooldown 1、minimum LRはbackbone `1e-6` / head `3e-6`
- 最大100 epochは安全上限とし、epoch 1からcheckpoint選択、AUROC patience 15、global gradient clip 5.0を使用
- scheduler stateをcheckpointへ保存し、resume時にoptimizerと一緒に復元する
- clip率とvalidationの陽性・陰性平均score / score gapをhistory・log・W&Bへ追加
- 改訂B0は`outputs/test-2/matched_b0_v2`へ新規出力し、旧checkpointはresumeしない
- 改訂fold 0はepoch 22でearly stoppingし、best epoch 7、val AUROC 0.738、AP 0.310、BCE 0.522
- 終了時tracebackはNFS上の`TMPDIR=.tmp`をDataLoader workerが削除できない後処理エラーで、checkpoint・履歴・予測は正常保存済み
- Baseline 1起動時に`TMPDIR` / `TEMP` / `TMP`と`tempfile.tempdir`を`/tmp/vai-baseline1-{uid}`へ切り替える

## 旧「次のタスク」（2026-08-14時点・全て失効）

1. 新しい実験名でBaseline 1 matched B0のfold 0をepoch 1から再学習し、改訂scheduleを確認
2. fold 0で改善を確認後、matched B0の残り4-fold学習とpooled OOF評価
3. Baseline 1 matched V2-Sの5-fold感度分析とpooled OOF評価
4. Baseline 1 full V2-Sのstage確認後の5-fold学習とpooled OOF評価
5. 近道の床と検出力の再計算（→ 現行計画へ引き継ぎ済み）

**2026-08-17の設計転換により1〜4は全て不要になりました。** 5のみ `PROGRESS.md` へ継続。
