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

## 確定済みの前提（2026-08-07）

- 入力は既存 `data/rsna_data/fracture_dataset_blind/`（2.5D、**15面固定**、全アーム共通・変更不可）
- **bag母集団は3ファイル完備の13,928 bag / 2,010 study**（region_4class.npyのみ欠落の126 bagを全アームから除外。うち陽性椎体28。全bag欠落の2 studyはfold外）
- 領域ラベルCSVのdedup規則は **keep last run**（最新run採用）。`folds/load_labels.py` が唯一の実装
- 「6ch入力」は実データでは各面 5CT ch + 5mask ch = 10ch。2-stemで吸収
- Baseline 2 の学習データは 268 bag + 椎体陰性bag（entailed negatives）
- **領域ラベルは run をまたいだ OR 集約**。run = 同一椎体内で連続するbboxのかたまり＝別々の骨折部位
  （17椎体が複数run、うち6椎体は別部位が別領域に及ぶ）。アノテータ確認済み（2026-08-07）で各runのラベルは正しい。
  確定値 **268 bag / 160 study / R1 78 / R2 59 / R3 72 / R4 158**、複数領域陽性70、R2 xor R3 = 95
- **R2/R3 は横突孔**（椎骨動脈が通る孔）。アノテーションツールのUI文言が「椎間孔」と誤っていたが、
  ラベル自体は横突孔として判定されていることをアノテータが確認（2026-08-07）。文言のみ修正済み
- **R2/R3 の「右」「左」は画像基準**。class2は画像右（平均x=155、class3は66）＝患者の左。
  ラベル・マスク・クラス番号は相互整合しており学習/評価に影響なし。臨床的な左右の記述時のみ反転が必要
- 評価: 椎体AUROC（14,133、確証的）/ 領域AP（268のみ、床ゲート）/ SideAcc balanced（95、ゲート0.65）
  ⚠️ **床（R1 0.59 / R2 0.37 / R3 0.45 / R4 0.72）と検出力は旧ラベル（77/71/155）で算出されたもの。
  事前登録前に補正ラベルで再計算が必要**
- fold / seed / 入力manifest / 集約規則 / 学習予算は全アームで統一
- 提案Aの teacher・pseudo-label は outer fold 内で完結させる
- flip augmentation は R2/R3 の target・maskチャンネル・validity を同時swap

## プロジェクト一覧

| プロジェクト | ディレクトリ | 状態 | メモ |
|---|---|---|---|
| fold定義 | `folds/` | **完了(検証済)** | folds.csv凍結（seed 20260807）。再生成禁止 |
| 共通基盤 | （未作成） | 未着手 | dataset読込 / 評価ハーネス / 損失。各プロジェクトから共用 |
| Baseline 1 | （未作成） | 未着手 | 参照値: 旧Stage1 OOF AUROC 0.921 |
| Baseline 2 | （未作成） | 未着手 | P/H/N層別損失 |
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
  - 全bag棚卸し: 14,054 bag / 2,012 study。**126 bagがregion_4class.npyのみ欠落**（アノテ済みとの重複0）
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

## 次のタスク

1. **近道の床と検出力の再計算**（旧ラベル基準のため。事前登録ゲートを固定する前に必須）
2. 共通基盤プロジェクト: dataset（10ch構成・flip時R2/R3同時swap）/ 評価ハーネス。flip swapの単体テストを最初に書く
3. Baseline 1 実装（参照値: 旧Stage1 OOF AUROC 0.921）
