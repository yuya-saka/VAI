# 疑似ラベルMTL：セッション通し進捗チェックポイント

**日付**: 2026-08-23
**状態**: 設計確定・生成段階監査PASS（1指標のみ形式的FAIL、内容的には無関係と訂正済み）・
疑似ラベル生成の第一段実装完了。**ペア構成・モデル構成・実学習は未着手**。

このファイルはセッション全体を通しで追える1本の要約。技術的な数値の全詳細は
`.claude/docs/work-logs/2026-08/2026-08-23-pseudo-label-mtl-design-review-and-cam-gate-audit.md`
（同日、逐次追記形式）を参照。Codexとの相談全文は`.claude/docs/codex/`配下の該当ファイル。

---

## 0. 出発点

`memo/進捗/研究計画書_2026-08-21.md` を読み、疑似ラベルMTLへの転換方針
（[[project_pseudo_label_mtl_direction]]）を実装フェーズへ進めるセッション。

前提: 268 bag（約2%）の4領域人手ラベルだけで学習すると過学習し、椎体ラベルと併用する
MTLも損失の重み設計が破綻した（ρ≥0.97で4領域scoreが崩壊）。そこで、4領域教師も
4領域mask入力も使わないBaseline 0（椎体単位モデル）のGrad-CAM領域密度を、疑似ラベルの
種として使う方針へ転換した。

---

## 1. Codex設計レビュー（既存計画書のどこに反対したか）

全文: `.claude/docs/codex/20260823-pseudo-label-mtl-design.md`

計画書は「疑似ラベルが268から**全データ規模（13,432）へ広がる**」としていたが、
Codexは以下2点で明確に反対した。

1. **陰性12,100 bagへの論理的0投入に反対**。「どの領域かの情報を全く持たない」ため
   4出力が「骨折の有無」を学ぶ最短経路になり、過去の崩壊を別原因で再現しうる
2. **CAM値をBCEの確率教師にすることに反対**。CAMが実証したのは順位情報（AUROC）だけで
   確率校正は未実証

推奨した設計の骨格: fold-matched teacher、7×7 Grad-CAM維持、density enrichment維持、
BiLSTM共有＋Baseline 0転移、成功判定はhuman-only armへの増分。

## 2. ユーザー決定（1回目）

| 論点 | 決定 |
|---|---|
| 領域損失の母集団 | **両方を独立アームで比較**（陽性1,332のみ / 論理的0込み全13,432） |
| 疑似ラベルの形 | **bag間ペアワイズ順位蒸留**（Codex推奨）。二値化・ソフト確率とも不採用 |
| 正式6アーム実験 | **保留**（`baseline1_b`はouter0〜3で停止のまま） |
| 残りの監査 | 実装より先に実施 |
| teacher割当 | 監査結果を見てから決める |

## 3. 生成段階監査の実装と実行

`fracture_detection/baseline0/cli/cam_audit.py`を新規実装（+ `analysis/cam_audit.py`、
`analysis/cam_audit_report.py`、21テスト）。学習は行わず、268 bagを5つのBaseline 0
checkpoint全部で採点し、teacher記憶（memorization）とmask境界感度の2ゲートを判定する。

本番run（268 bag×5 teacher、GPU 0、約26分）の結果:

- **memorizationゲート: 完全PASS**（AUROC差は最大でも+0.041、95% CI全て0を跨ぐ。
  teacherが訓練bagを記憶して局在精度を水増ししている証拠なし）
- **mask境界感度ゲート: 形式上FAIL**。ゲート対象6変種（erode/dilate/shift×4方向、
  各1.6mm）のうち、領域内Spearman順位相関は**全変種・全領域で0.80超**（最低0.829）。
  「4領域中どれが最大か」（argmax）だけ、侵食方向のみ全領域で20%変化しゲート抵触
  （膨張・平行移動は5〜8%でPASS）
- 左右弁別win rate 0.854 [0.769, 0.929]、領域別AUROC 0.736〜0.798（既存値と整合）

## 4. Codexへの解釈相談（2回目）とユーザーによる訂正

全文: `.claude/docs/codex/20260823-cam-audit-gate-interpretation.md`

Codexの回答: 「進めてよいが、事前登録上はゲート合格と書けない。ゲート逸脱を承認して
継続、と明記すること」。argmax失敗は採用済みのペアワイズ順位蒸留には無関係、と説明。

**ここでユーザーが訂正**: argmax（4領域中どれが1位か）は「今回の方式が使わないだけ」
ではなく、**2026-08-17の設計時点から本研究のどの段階でも使う予定のない量**である。
268例中70例が複数領域陽性で4領域は互いに排他ではなく、各領域は独立に骨折の有無を
判定する設計（Codex Q3dで「softmax・rank正規化とも不採用」と既に確認済み）。

→ 進行判断の根拠は「たまたま無関係」から「そもそも測る必要のない指標だった」へ強化。
書面に残す限定事項は1点（268 annotated bagsの内部検証であり、未注釈対象集団への
一般化は未保証）に整理。

## 5. ユーザー承認、teacher割当の確定

- teacher割当: **fold-matched in-sample `Teacher_k`に確定**（Codex推奨、memorization
  水増し未検出のため）。比較ペアは同一teacher・同一領域内で作る
- 疑似ラベルの仕組み（「ペア比較の確信度のソフト化」）を対話で説明し、ユーザー承認
  → 「じゃあ、それでいきましょう」

## 6. 疑似ラベル生成の第一段実装

**実装したのは「教師スコアの生成」と「スコア差を確信度に変換する損失」の2点のみ。**
ペアの作り方、モデル構成、実学習は未着手。

### 新規ファイル

| ファイル | 内容 |
|---|---|
| `baseline0/analysis/pseudo_label.py` | `log_score`、`region_temperature`（ラベルフリーIQR温度）、`pairwise_confidence`、`pairwise_ranking_loss` |
| `baseline0/cli/generate_pseudo_labels.py` | fold-matched `Teacher_k`で学習3 fold全bagを採点するCLI |
| `baseline0/tests/test_pseudo_label.py` | 17テスト |

### 核となる式

```
u_ijr = sigmoid( (log C_ir - log C_jr) / T_r )
L_P   = BCEWithLogits(z_ir - z_jr, u_ijr)
```

`C`はteacherのCAM密度（面積補正済み）、`z`はstudentのregion logit、`T_r`は
「その教師・その領域の訓練foldにおけるCAMスコア差のIQR」から固定seedで決定論的に
計算する温度（ラベルを一切使わない）。`pairwise_ranking_loss`は教師スコアへ
勾配を流さない（`.detach()`）。

### 検証

- 20 unit tests。`fracture_detection`全体で**198 tests passed**
- Ruff check/format、mypy（新規2ファイルは既存stub不足以外クリーン）
- 実データsmoke: 5 outer fold × 40 bag、200行・81 bag分のスコアと温度表を生成、
  `teacher_checkpoint_sha256`によるprovenanceを確認
- 本番run完了: 5 teacherで40,296行、13,432 bag（各bagは対応する3 teacherに1行ずつ）。
  温度20件、骨折陽性スコアの未定義は2件。fold/provenance不変条件と成果物SHA256を検証済み

```bash
uv run python -m fracture_detection.baseline0.cli.generate_pseudo_labels \
  --device cuda:0 --batch-size 16
```

---

## 7. 現在地点と次に決めること

bagごとの領域CAM密度スコア、温度表、順位ペア構成は実装済み。
ここから先、以下は**まだ決めていない**。

1. **論理的0込みアームのexact-negative重み**。CAMペアは両アームとも骨折陽性内だけに固定し、
   負例同士・陽性負例間のCAMペアは作らない
2. **モデル構成**: 4領域maskでのmask-normalized pooling、Baseline 0からのBiLSTM転移を
   含む学生モデルのアーキテクチャ（Codex推奨: 4領域で共有BiLSTM＋小さなregion固有head）
3. **λ/α勾配校正**の実装、MTL trainerへの統合、provenance（teacher ID・fold・
   checkpoint hash）を学習ループでどう記録するか
4. **実際の学習run**（正式6アームと同様、ユーザーの手動起動が必要。自動連鎖しない）

## 8. 関連メモリ・ファイル索引

- 研究計画書: `memo/進捗/研究計画書_2026-08-21.md`
- 現行方針の記憶: `project_pseudo_label_mtl_direction`（要更新: 本セッションの実装進捗を反映）
- 進捗台帳: `fracture_detection/PROGRESS.md`（2026-08-23節）
- Baseline 0 README: `fracture_detection/baseline0/README.md`
  （「疑似ラベル教師信号の生成前監査」「疑似ラベル生成」の2節を追加済み）
- Codex相談: `.claude/docs/codex/20260823-pseudo-label-mtl-design.md`、
  `.claude/docs/codex/20260823-cam-audit-gate-interpretation.md`
- 詳細worklog（本チェックポイントの元データ）:
  `.claude/docs/work-logs/2026-08/2026-08-23-pseudo-label-mtl-design-review-and-cam-gate-audit.md`
- 正式6アーム実験（保留中）: `baseline1_b`はouter0〜3完了・outer4以降未着手。
  `control_b`・Proposed 3構成は未着手。GPU 0/1/2は空き
