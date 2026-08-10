# 2026-08-07 骨折検出4アーム実験の実装計画確定

## 0. セッション状態

- 状態: **実装計画確定・コード未着手**
- 根拠文書: `memo/計画書/提案手法.md`（本セッションからこれが研究のactive定義）
- DESIGN.md の Active セクションを提案手法.md ベースに更新済み（Changelog 2026-08-07）
- 参照すべき既存Codex分析:
  - `.claude/docs/codex/20260806-2100-region4-architecture-design.md`（損失設計・集約・負例層別）
  - `.claude/docs/codex/20260806-fundus-semisup-multitask-paper-read.md`（提案Aの参考論文）
  - `.claude/docs/codex/20260804-fresh-4region-model-design.md`（入力幾何の経緯）

---

## 1. 本日のユーザー決定（AskUserQuestionで確定）

1. **入力面数: 15面固定で全実験**（既存 `fracture_dataset_blind` をそのまま使う）
   - full-z移行・可変長対応の選択肢を提示した上で15面固定を選択
   - 受容した制約: SIカバー median 77.5%、終板骨折の系統的欠落、R1が構造的に不利、
     提案Bの OR 制約が範囲外骨折でラベルノイズ化
   - 全アーム共通の前処理契約。比較開始後は変更不可
2. **Baseline 2 の学習データ: 268 bag + 椎体陰性bag（entailed negatives）を追加**
   - 計画書の「領域アノテ症例のみ」から拡張。領域APの評価母集団は従来通り268のみ（陰性混入なし）

---

## 2. 4アーム構成（提案手法.md）

共通: 2.5D CNN + LSTM（RSNA 2022 1位 type-1系）。提案モデルは
CT 5ch + mask 5ch（全体+4領域）= 10ch/面 を 2-stem（image stem / mask stem）で
受けて融合 → 共有CNN → LSTM → 全体head + 4領域head。
（計画書の「6ch」は2.5Dでは実質10ch。2-stemで吸収）

| アーム | 構成 | 領域ラベルの使い方 |
|---|---|---|
| Baseline 1 | CT+全体mask → 椎体分類 | 使わない |
| Baseline 2 | 4領域独立モデル | 268で教師あり |
| 提案A | 半教師あり: teacher→pseudo-label→student マルチタスク | 268 + pseudo-label |
| 提案B | 弱教師あり: y_whole=OR(y_regions) | **評価専用**（学習に不使用） |

---

## 3. 実装フェーズ（承認済み計画）

置き場所: `fracture_detection/`（リポジトリ直下、新規。2026-08-07ユーザー指定。
今後の学習モデル・fold定義等は全てこの下に作る。`Unet/` 配下ではないので
コメント・docstringは既定ルール通り英語）。構成は固定パッケージ構造でなく
**プロジェクト単位のサブディレクトリ**で分ける（ユーザー指示）。
進捗台帳: `fracture_detection/PROGRESS.md`

| Phase | 内容 | 検証 |
|---|---|---|
| 0 | データ契約固定: 患者単位層別5-fold新規作成、268読込チェック、mask版数pin | 全数読込 |
| 1 | 共通基盤: dataset(10ch構成・flip時R2/R3同時swap) / 2-stemモデル / losses / eval | 1バッチ過学習・swap単体テスト |
| 2 | Baseline 1 | 旧Stage1 OOF AUROC 0.921 が参照値 |
| 3 | Baseline 2（268+陰性、P/H/N層別損失） | 領域AP vs 床（R2 0.37が最易） |
| 4 | 教師ありマルチタスク（AのStudent骨格 兼 Teacher） | |
| 5 | 提案A: fold内teacher → pseudo-label → student | リーク検査 |
| 6 | 提案B: smooth-max主、max/noisy-ORアブレーション | |
| 7 | 統一評価・4アーム比較 | 事前登録ゲート |

## 4. 実装の不変条件（過去分析より、コードに焼き込む）

- 全アームで fold / seed / 入力manifest / 集約規則 / 学習予算を統一
- flip時に R2/R3 の target・maskチャンネル・validity を**同時**swap（過去に事故あり）
- 領域AP母集団に陰性を混ぜない。床ゲート: R1 0.59 / R2 0.37 / R3 0.45 / R4 0.72
- SideAcc は balanced accuracy、94 bag、単群ゲート 0.65
- 提案Aの teacher / pseudo-label は outer fold 内で完結（全データteacherはOOF全滅）
- 領域損失は P（領域陽性）/H（骨折ありだが他領域）/N（椎体陰性）の層別平均。
  pos_weight≈215 は禁止。H が本当の localization hard negative
- noisy-OR は主方式にしない（R4飽和で全枝の正勾配が消える）

## 5. 未決（実装中に決める）

- 椎体レベルラベル14,133を領域タスクと併用するか（全アームで統一必須）
- 提案Bの集約主方式の最終確定（3方式比較後、事前登録前に）
- 提案Aのpseudo-label詳細（hard/soft、閾値、重み）→ 眼底論文読解を参照

## 6. Phase 0 実施結果（同日）

`fracture_detection/folds/` に実装・実行済み。詳細は `fracture_detection/PROGRESS.md`。

- **【重要な訂正】領域ラベルの集約規則は OR**（当初 keep last run としたのは誤り。同日中に訂正）
  - `run_id` を「アノテーションのやり直し」と誤解したのが原因。実際は
    `Unet/dicom_bbox_annotation_tool` が**同一椎体内で連続するbbox行のかたまり**を run に分けており、
    複数runは**別々の骨折部位**を意味する（README「DICOM系列上で連続するbbox行を1つの run_XX として表示」）。
    実測でrun間の空きは5〜50スライス。アノテータはrunごとにその範囲の画像を見て領域を判定している
  - 17椎体が複数run。うち6椎体でrun間のラベルが異なるが、これは矛盾ではなく
    「部位Aは椎体、部位Bは後方要素」という別部位の別領域波及
  - ユーザー確認（2026-08-07）「目視でアノテーションしたので各runのラベルは正しい」を受けOR集約を確定
  - **影響**: R1 77→**78** / R3 71→**72** / R4 155→**158**（R2 59は不変）、
    複数領域陽性 65→70、R2 xor R3（SideAcc母集団）94→**95**。bag 268 / study 160 は不変
  - keep last は実在する骨折部位を1つ捨てていた。`load_labels.py` にOR集約を検証付きで実装
  - **未処理**: 近道の床（R1 0.59/R2 0.37/R3 0.45/R4 0.72）、level-only macro-AP 0.451、
    検出力MDEは全て旧ラベル基準。事前登録前に再計算が必要（DESIGN.mdにも明記済み）

- **4領域ラベルの健全性検証**（OR集約後）:
  - all-zero bag 0件、値は全て0/1、領域数分布 単一198/2領域45/3領域21/4領域4
  - CSV `region_N` ↔ マスククラス値 N の対応をコードで確認
    （`constants.py: REGION_NAMES = (background, body, right_foramen, left_foramen, posterior)`）
  - 268 bag 全てで4クラスがマスクに存在。「陽性ラベルだがマスククラス無し」は0件
  - 面単位の存在率: 椎体89.8% / 右83.3% / 左78.6% / 後方93.8%（存在時平均面積 934/270/243/2431 px）

- **【用語訂正】R2/R3 は横突孔であって椎間孔ではない**
  - 両アノテーションツールのUIとコードコメントが「椎間孔」と表記していた（別の解剖構造）。
    研究側の文書は一貫して「横突孔」。アノテーション画面には4領域の境界線は表示されず、
    アノテータの解剖学的判断で領域が決まるため、UI文言の誤りは実質的リスクだった
  - ユーザー確認（2026-08-07）「横突孔です。ツールは名前が間違っているだけで実は間違えていない」
    → **ラベル変更は不要、文言のみ修正**
  - 修正: `dicom_bbox_annotation_tool/index.html`, `fracture_annotation_tool/{index.html,server.py}`,
    `line_only/{utils/region_eval.py,rsna_4region_segmentation/{__init__.py,apply_sdf_segmentation.py}}`,
    `debug/viz_region_seg.py`, `data_preprocessing/rsna_pipeline/fracture_region_annotation.py`
  - 英語識別子（`right_foramen` 等）は辞書キー・レポート名に使われており、かつ誤りではないため据え置き
  - 併せて両ツールの凡例に「右/左は画像基準＝患者基準では逆」を明記（過去に左右反転事故があったため）
- **268 bag 全読込チェック PASS**: 形状(15,5,224,224)uint8・mask非空・region値域・
  R2/R3陽性bagのmaskクラス存在まで全て正常。SHA256指紋で版数pin
- **新事実: 126 bagがregion_4class.npyのみ欠落**（94 study、アノテ済みと重複0、陽性椎体28含む）
  → bag母集団を「3ファイル完備の13,928 bag / 2,010 study」に確定（全アーム同一母集団の要請）。
  全bag欠落の2 study（24673, 8362）はfold対象外。train.csvの7 studyは画像なし
- **5-fold凍結**: 貪欲層別（限界コスト最小化、seed 20260807）。OR集約ラベルで再生成後のバランス:
  studies 402×5 / 陽性椎体 281-282 / アノテbag 53-56 / R1 15-16 / R2 11-12 / R3 14-15 / R4 31-32。
  再実行同一・上書きガードあり
- 途中、貪欲法が「割当後の絶対偏差」を最小化するバグで3 foldに崩壊 → 限界変化最小化に修正
- ruff check/format PASS。ty は環境未導入のためスキップ

## 7. 次回タスク

1. **近道の床・検出力の再計算**（OR集約ラベル基準。事前登録ゲート固定前に必須）
2. 共通基盤: dataset（10ch構成・flip時R2/R3同時swap）/ 評価ハーネス
3. flip swap の単体テストを最初に書く（事故歴があるため）
4. Baseline 1 実装（参照値: 旧Stage1 OOF AUROC 0.921）
