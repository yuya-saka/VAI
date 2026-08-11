# fracture_detection 作業進捗

> 頸椎骨折の4領域検出研究（`memo/計画書/提案手法.md`）の実装ディレクトリ。
> 学習モデル・fold定義などは**プロジェクト単位のサブディレクトリ**に分けて作っていく。
> このファイルは各プロジェクトの状態を一覧する進捗台帳。詳細な経緯は
> `.claude/docs/work-logs/2026-08/` と `.claude/docs/DESIGN.md` を参照。

---

## 全体像

4アームの比較実験（詳細は `memo/計画書/提案手法.md`）:

| アーム | 内容 | 領域ラベル268の扱い |
|---|---|---|
| Baseline 1 | CT+全体mask → 椎体分類（CNN+LSTM） | 不使用 |
| Baseline 2 | 4領域独立モデル | 教師あり |
| 提案A | 半教師あり（teacher → pseudo-label → student） | 教師あり + pseudo-label |
| 提案B | 弱教師あり（y_whole = OR(y_regions)） | 評価専用 |

## 確定済みの前提（2026-08-11更新）

- 入力は統合済み `data/rsna_data/fracture_dataset/`（2.5D、**15面固定**、全アーム共通・変更不可）
- **bag母集団は3ファイル完備の13,928 bag / 2,010 study**（region_4class.npyのみ欠落の127 bagを全アームから除外。うち陽性椎体29。完備bagが0件の2 studyはfold外）
- 領域ラベルCSVは同一椎体の全runを **OR集約**する。`folds/load_labels.py` が唯一の実装
- 「6ch入力」は実データでは各面 5CT ch + 5mask ch = 10ch。2-stemで吸収
- Baseline 1 matched / Baseline 2 は同じ固定536 bag・428患者を使用する。
  内訳は領域アノテ268 bag・160患者 + 椎体陰性268 bag・268患者。
  陰性はアノテ患者と重複なし・1患者1椎体・fold別level件数一致で固定する
- Baseline 1は`matched`（536 bag・428患者）と`full`（13,928 bag・2,010患者）を切替可能にする
- **matchedの陰性268 bagは「骨折なし患者」からのみ取る**（候補 1,063患者・7,375 bag。
  fold×level 全35セルで充足可能、最小マージン195患者）
- Baseline 2の損失は通常BCE（P/H/N重み・pos_weight・focalなし）
- **bag確率は 15面 broadcast + 面ごとBCE + mean-sigmoid**（2026-08-11ユーザー決定。
  Codex推奨のbag-level log-mean-exp は却下）。対応する単一尤度が存在しない点は登録済み限界
- **モデル選択は val AUROC の early stopping**（2026-08-11ユーザー決定。
  Codex推奨の固定epoch+EMAは却下）。OOFの楽観バイアスは登録済み限界
- **held-out test は作らない**。5-fold OOFのみ（268という領域評価の母数を削れないため）
- **matchedのbackboneは `tf_efficientnetv2_b0` が主解析**、`tf_efficientnetv2_s` は感度分析。
  fullは `tf_efficientnetv2_s`
- **回転augmentationは ±40°まで**（2026-08-11ユーザー決定。Codex推奨の±10-12°は不採用）。
  境界は constant fill（反射境界は四隅に鏡像解剖を作るため使わない）。
  distortion / cutout / mixup は全設定OFF
- **ステージングは `full` のみ**。`matched`（2.8 GB）はページキャッシュに乗るため直読み
- **各プロジェクトに `README.md` を置き、モデル内容を記載する。仕様変更のたびに更新する**
- **領域ラベルは run をまたいだ OR 集約**。run = 同一椎体内で連続するbboxのかたまり＝別々の骨折部位
  （17椎体が複数run、うち6椎体は別部位が別領域に及ぶ）。アノテータ確認済み（2026-08-07）で各runのラベルは正しい。
  確定値 **268 bag / 160 study / R1 78 / R2 59 / R3 72 / R4 158**、複数領域陽性70、R2 xor R3 = 95
- **R2/R3 は横突孔**（椎骨動脈が通る孔）。アノテーションツールのUI文言が「椎間孔」と誤っていたが、
  ラベル自体は横突孔として判定されていることをアノテータが確認（2026-08-07）。文言のみ修正済み
- **R2/R3 の「右」「左」は画像基準**。class2は画像右（平均x=155、class3は66）＝患者の左。
  ラベル・マスク・クラス番号は相互整合しており学習/評価に影響なし。臨床的な左右の記述時のみ反転が必要
- 評価: 椎体AUROC（**13,928 bag / 陽性1,406**、確証的）/ 領域AP（268のみ、床ゲート）/
  SideAcc balanced（95、ゲート0.65）
  ⚠️ **床（R1 0.59 / R2 0.37 / R3 0.45 / R4 0.72）と検出力は旧ラベル（77/71/155）で算出されたもの。
  事前登録前に補正ラベルで再計算が必要**
  ⚠️ 旧記載の母数 14,133 / 陽性1,444 は誤り（2026-08-11修正）。確証的評価の分母・陽性数・検出力は
  凍結manifest `common/outputs/input_manifest.csv` から導出すること
- fold / seed / 入力manifest / 集約規則 / 学習予算は全アームで統一
- 提案Aの teacher・pseudo-label は outer fold 内で完結させる
- **flip / transpose augmentation は使用しない**。格納済みの画像方向とR2/R3対応を全アームで維持する

## プロジェクト一覧

| プロジェクト | ディレクトリ | 状態 | メモ |
|---|---|---|---|
| fold定義 | `folds/` | **完了(検証済)** | folds.csv凍結（seed 20260807）。再生成禁止 |
| 共通基盤 | `common/` | **完了(検証済)** | manifest / dataset / 標準BCE / 評価。モデルなし |
| matched cohort | （未作成） | 未着手 | 536 bag固定・凍結。B1 matched / B2 が共有 |
| Baseline 1 | （未作成） | **設計確定** | 仕様は `work-logs/2026-08/2026-08-11-baseline1-design.md`。参照値: 旧Stage1 OOF AUROC 0.921（再現目標ではない） |
| Baseline 2 | （未作成） | 未着手 | 4独立モデル / 通常BCE / 固定536 bag |
| 教師ありマルチタスク | （未作成） | 未着手 | 提案AのStudent骨格 兼 Teacher |
| 提案A | （未作成） | 未着手 | pseudo-label詳細は眼底論文読解を参照 |
| 提案B | （未作成） | 未着手 | smooth-max主 / max・noisy-ORアブレーション |

状態は 未着手 / 実装中 / 学習中 / 完了(検証済) / 保留 のいずれかで更新する。

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
  - 椎体AUROC/AP・領域別AP・SideAcc balanced・患者bootstrap評価を実装
  - 単体テスト4件、ruff、mypy、実データ読込を確認
  - モデルとモデル固有損失は`common/`へ置かず、各実験プロジェクトで実装する
- **Baseline 2計画を単純化**:
  - 4領域それぞれの独立モデルに通常の`BCEWithLogitsLoss`を使用
  - アノテ268 bagと、別患者から1椎体ずつ選ぶ固定陰性268 bagを使用
  - P/H/N層別重み、`pos_weight`、focal lossは使用しない
- **Baseline間のデータ数を統一**:
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

## 次のタスク

1. matched cohort 生成（`cohorts/make_matched_cohort.py`、536 bag固定・凍結）
2. Baseline 1 実装（`baseline1/`。仕様は `work-logs/2026-08/2026-08-11-baseline1-design.md`）
3. **近道の床と検出力の再計算**（旧ラベル基準のため。事前登録ゲートを固定する前に必須。1-2と並行可）
