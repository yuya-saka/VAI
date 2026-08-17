# 2026-08-11 骨折検出共通基盤とBaselineデータ計画の確定

> **後続変更:** 50:50の536 bag matched cohort設計は、同日中に自然陽性率の
> 2,655 bag（陽性268・陰性2,387）へ置き換えた。以下の536 bag・1患者1椎体に関する記載は
> 当時の履歴であり、現行仕様ではない。現行仕様は
> `2026-08-11-baseline1-implementation-plan.md`を参照する。

## 0. セッション状態

- 状態: **Phase 1共通基盤完了・Baseline実装開始可能**
- 次回セッションは固定matched cohortの生成後、Baseline 1実装へ進む
- モデルはまだ実装していない。`fracture_detection/common/`にもモデルは置かない
- 現在の変更は未commit
- active設計:
  - `memo/計画書/提案手法.md`
  - `fracture_detection/PROGRESS.md`
  - `.claude/docs/DESIGN.md`

---

## 1. 本日のユーザー決定

### 1.1 コード内の日本語化

- `fracture_detection/`内のコメント、docstring、実行時メッセージは日本語にする
- 変数名、関数名、CSV/JSONキーなどの機械的インターフェースは英語のまま維持する

### 1.2 augmentation

- **flip / transpose augmentationは使用しない**
- 保存済み画像の向きをそのまま使用する
- R2/R3のswap処理はdatasetへ実装しない
- 将来augmentationを追加する場合も反転を伴わないものに限定し、全比較アームで統一する

### 1.3 `common/`の責務

- `fracture_detection/common/`は全アームで不変の基盤だけを置く
  - 固定値
  - 共通manifest
  - dataset読込
  - 汎用BCE
  - 評価指標
  - 単体テスト
- モデルとモデル固有損失は各実験プロジェクトに置く
- 一度作成した`common/model.py`とマルチタスク専用loss wrapperは削除済み

### 1.4 Baseline 2の単純化

- 4領域それぞれに独立したCNN+LSTMモデルを作る
- 損失は通常の重みなし`BCEWithLogitsLoss`
- P/H/N層別重み、`pos_weight`、focal lossは使用しない
- 骨折陽性だが対象領域陰性の例は、アノテーション済みcohort内の通常の陰性targetとして扱う
- 領域APは従来どおりアノテーション済み268 bagだけで計算し、追加陰性を混ぜない

### 1.5 Baseline 1 / Baseline 2のデータ数統一

- Baseline 1の`matched`設定とBaseline 2は**同じ固定536 bag・428患者**を使用する
- 内訳:
  - 領域アノテーション済み: 268 bag・160患者
  - 椎体陰性: 268 bag・268患者
- 椎体陰性cohortの固定条件:
  - アノテーション患者との重複なし
  - 1患者につき1椎体
  - fold別・C1〜C7別の件数をアノテーション側と完全一致
  - exact IDをmanifestへ保存し、両baselineで共有
- Baseline 1はデータ設定を切り替え可能にする
  - `matched`: 536 bag・428患者。Baseline 2との直接比較用
  - `full`: 13,928 bag・2,010患者。全数学習用の別実験

### 1.6 入力データセットの統合

- `fracture_dataset_blind/`は独立した全データではなく、次の複合viewだった
  - bbox症例をラベル非依存の面割り当てで再生成した実体234 study
  - `fracture_dataset/`へ戻るsymlink 1,778 study
- 実体234 studyはすべて`fracture_dataset/`側にも旧版が存在した
- blind実体を正として、内容が異なる1,712ファイル（2.27 GB）を`fracture_dataset/`へ上書き統合した
- blind側にない既存bagは削除していない
- 今後の実行時入力は`data/rsna_data/fracture_dataset/`だけを使用する
- 統合後は14,055 bag / 2,012 study。`region_4class.npy`欠損127 bagを除く、13,928 bag / 2,010 studyが学習可能

---

## 2. matched cohortの実測人数

陰性候補は11,761 bag・1,850患者あり、上記条件で268患者を重複なく選べることを確認した。
fold別・椎体level別の完全一致も成立する。

| validation fold | val bag | val患者 | train bag | train患者 |
|---:|---:|---:|---:|---:|
| 0 | 112 | 88 | 424 | 340 |
| 1 | 106 | 83 | 430 | 345 |
| 2 | 106 | 86 | 430 | 342 |
| 3 | 106 | 85 | 430 | 343 |
| 4 | 106 | 86 | 430 | 342 |

`full`設定は各foldでvalidation 402患者、training 1,608患者となる。

| validation fold | val bag | train bag |
|---:|---:|---:|
| 0 | 2,786 | 11,142 |
| 1 | 2,785 | 11,143 |
| 2 | 2,784 | 11,144 |
| 3 | 2,787 | 11,141 |
| 4 | 2,786 | 11,142 |

---

## 3. 本日の実装

### 3.1 foldコードの日本語化

- `fracture_detection/folds/check_dataset.py`
- `fracture_detection/folds/load_labels.py`
- `fracture_detection/folds/make_folds.py`

コメント、docstring、実行時メッセージを日本語化した。識別子と成果物schemaは変更していない。

### 3.2 Phase 1共通基盤

作成先: `fracture_detection/common/`

- `constants.py`: パス、形状、領域順などの固定契約
- `manifest.py`: fold・棚卸し・椎体ラベル・OR集約領域ラベルを結合
- `dataset.py`: CT 5chと全体+R1〜R4 mask 5chを別テンソルで返す
- `losses.py`: 有効領域targetと椎体陰性の論理的0 targetへ通常BCEを適用
- `metrics.py`: 椎体AUROC/AP、領域別AP、SideAcc balanced、患者cluster bootstrap
- `tests/`: manifest、向き保持、loss、metricsの単体テスト

生成済み共通manifest:

- `fracture_detection/common/outputs/input_manifest.csv`
- `fracture_detection/common/outputs/input_manifest_meta.json`
- 13,928 bag・2,010患者・領域アノテーション268 bag
- SHA256: `39d46a6c6d2ddbeb1f0eb6df0a94cb5de09461f68deb8761fd7723c4f58675a3`

---

## 4. 検証結果

- `ruff format --check fracture_detection/common`: PASS
- `ruff check fracture_detection/common`: PASS
- `mypy fracture_detection/common --exclude tests --ignore-missing-imports`: PASS
- `python3 -m pytest fracture_detection/common/tests -q`: **4 passed**
- 実データの先頭・末尾bag読込: PASS
- 共通manifest件数・患者数・領域アノテーション数: 契約どおり
- `common/`内にモデル依存がないことを検索で確認
- `git diff --check`: PASS
- blind実体234 studyと統合先の内容ハッシュ一致: PASS

---

## 5. 次回セッションの開始手順

### Step 1: 固定matched cohort manifestを生成

- 268 annotated bagをそのまま採用
- 非annotated患者から椎体陰性268 bagを決定的に選択
- 1患者1椎体、fold別・level別件数一致をassertする
- annotated患者との重複0をassertする
- exact IDとSHA256を保存する
- Baseline 1 matched / Baseline 2が同じmanifestを読むテストを追加する

### Step 2: Baseline 1を実装

- 作成先: `fracture_detection/baseline1/`
- モデル、loss、trainer、config、testsをこのプロジェクト内に置く
- 入力はCT 5ch + 椎体全体mask 1ch
- flip / transposeなし
- 通常BCE
- `data.mode: matched | full`を必須設定にする
- matched/fullで出力ディレクトリと実効configを分離する
- 凍結済み5-foldを使用する
- 旧Stage1 OOF AUROC 0.921は参考値であり、新実装の成功条件そのものにはしない

### Step 3: Baseline 1検証後にBaseline 2へ進む

- 作成先: `fracture_detection/baseline2/`
- 4領域独立モデル
- Baseline 1 matchedと同一cohort・fold・学習予算を使用
- 通常BCEのみ

---

## 6. 次回も維持する不変条件

- fold seed `20260807`、`folds.csv`は凍結済み
- 15面固定、入力は統合済み`fracture_dataset/`に固定
- flip / transposeなし
- `common/`にモデルを置かない
- Baseline 2へ独自lossを追加しない
- Baseline間のmatched cohort exact IDを一致させる
- 領域APへ追加陰性を混ぜない
- 提案Aのteacher / pseudo-labelはouter fold内で完結させる
- 旧ラベル基準のshortcut floor・検出力は事前登録前に再計算する
