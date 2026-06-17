# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-06-17 -->

## line_only 最新結果（line_20260616, sig4.0_ALL）

| 実験 | angle mean | angle p90 | worst | outlier ≥10° |
|------|:---:|:---:|:---:|:---:|
| blob追加（CC なし） | 5.076° | 10.52° | 88.79° | 11.3% |
| CC-metrics（旧ckpt + CC） | 4.804° | 10.193° | 38.65° | 10.5% |
| **CC適用（新規学習 + CC）** | **4.926°** | **10.45°** | **54.1°** | **10.9%** |

CC フィルタ（連結成分でピーク周辺のみ残す）により外れ値を大幅削減。  
CC適用 vs CC-metrics の差は training variability によるもの（CC 自体の問題ではない）。

## スライス分析による知見（2026-06-17）

- **同部位に非外れ値スライスが必ず存在**（angle ≥10°・rho ≥8px ともに 100%）
- スライス選択または ensemble で外れ値を原理的に排除できる
- 失敗パターン: 特定チャンネルが端スライスで局所崩壊（他チャンネルは正常）
- 最良スライス選択後: mean 4.93° → **3.39°**、outlier 10.9% → **1.1%**

## 精度比較（5-fold CV 平均）

| モデル | perp (px) | angle (°) | rho (px) | seg_mIoU | fg_mdice |
|--------|:-:|:-:|:-:|:-:|:-:|
| line_only sig3.5 | 11.55 | 5.53 | 3.29 | — | — |
| seg_only_v1 | — | — | — | 0.906 | 0.936 |
| multitask_v1 (α=0.03) | 11.56 | 6.07 | 3.54 | 0.910 | 0.939 |
| multitask α=0.02 | **11.34** | **5.30** | **3.46** | 0.909 | 0.938 |
| **multitask_v2 (椎体条件付け)** | 11.45 | **5.47** | 3.42 | 0.907 | 0.937 |

- v2: C1 angle ▼1.1°、C2 angle ▼1.4° — 形状特異な上位椎体で顕著な改善

## multitask_v2 椎体別 angle_error（v1 比）

| C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 8.30→**7.17** | 8.33→**6.90** | 4.89→**4.10** | 4.32→4.25 | 4.21≈4.22 | 4.59→4.77 | 7.20→**7.01** |

## 実装状況

- `line_only/src/trainer.py` を責務分割済み（784行→351行）
  - `evaluation.py`: validation 指標
  - `inference.py`: test 推論・直線評価・JSON/画像出力
  - `example_writer.py`: GT/予測ヒートマップ保存
  - `experiment.py`: 出力パス・wandb・ログ表示
- `line_only/test/` は全36テスト pass
- `multitask/` + `seg_only/` 両方に vertebra conditioning 実装済み（config で on/off）
- `multitask/` に per_class (body/right/left/posterior) Dice/IoU の記録追加済み
- `multitask/` に `decoder_type` 切り替え実装済み（`dual_decoder` / `shared_decoder`）
- テスト: multitask 19/19、seg_only 14/14 pass


## aug変換修正 実験結果（line_only, 2026-04-15）

| 実験 | peak_dist [px] ↓ | angle_err [°] ↓ | rho_err [px] ↓ |
|------|:----------------:|:----------------:|:--------------:|
| aug/sig2.0（旧ベース） | 22.21 | 6.75 | 3.71 |
| reg/sig3.5 | 20.76 | 6.26 | 3.47 |
| reg/sig4.0 | 20.68 | 6.14 | 3.54 |
| **aug変換修正/sig3.5** | 21.45 | **5.87** | **3.20** |

- peak_dist は reg/sig4.0 が最良、angle/rho は aug変換修正 が全実験中最良
- reg + aug変換修正 の組み合わせが次の候補

## 領域分割評価（2026-06-17, sig4.0_ALL(CC適用)）

予測線 → z 伝播 → 領域マスク → GT との 3D Dice 評価パイプラインを構築。

| | 3D Dice mean | c1 椎体 | c2 右孔 | c3 左孔 | c4 後方 |
|---|:---:|:---:|:---:|:---:|:---:|
| 全由来（284 椎骨） | **0.829** | 0.898 | 0.740 | 0.749 | 0.929 |
| 外挿のみ | 0.798 | 0.887 | 0.683 | 0.700 | 0.920 |

- per-slice Dice (0.750) より 3D Dice (0.829) が高い → 端スライスの GT 極小が原因
- 椎間孔(c2/c3)低精度 = junction ズレへの敏感さ + GT 自体が小さい（端スライス）
- 実装: `Unet/line_only/utils/region_eval.py` / `Unet/debug/eval_region_3d.py`
- 出力: `outputs/…/sig4.0_ALL(CC適用)/region_eval_3d/` (summary.json / details/ / viz/)

## 保留タスク

- スライス ensemble / 信頼度ベース選択の実装（M00 を信頼度スコアに使用）
- multitask/ で aug 変換修正の実験実施
- `compute_perpendicular_distance` Y符号バグ修正（4ファイル共通）

---

## コード構成

実行: `uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml`
テスト: `uv run pytest -o pythonpath=Unet Unet/line_only/test -v`（36/36 pass）

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
| [2026-06-17](work-logs/2026-06-17.md) | CC フィルタ実装・全パス統一（val/test）・angle 5.076°→4.804°・worst 88°→38° |
| [2026-06-16](work-logs/2026-06-16.md) | config移動・best epoch→angle_error_deg・Blob IoU追加・LR/BS増量 |
| [2026-06-15](work-logs/2026-06-15.md) | line_only trainer 責務分割・公開API互換維持・テスト収集整理 |
| [2026-04-20](work-logs/2026-04-20.md) | パイプライン正当性確認・perp_dist Y符号バグ発見・V字型仮説否定 |
| [2026-04-16](work-logs/2026-04-16.md) | multitask/ に aug 変換修正を移植（19/19 pass） |
| [2026-04-15](work-logs/2026-04-15.md) | aug変換修正（ポリライン再生成）実装・line_only 実験比較 |
| [2026-04-14](work-logs/2026-04-14.md) | heatmap multitask 打ち止め・distance map 回帰への方向転換決定 |
| [2026-04-12](work-logs/2026-04-12.md) | vertebra conditioning 実装（multitask/seg_only）・v2 結果確認・per_class メトリクス追加 |
| [2026-04-10](work-logs/2026-04-10.md) | multitask vs seg_only 比較・per-image ハードケース分析 |
| [2026-04-09](work-logs/2026-04-09.md) | seg_only/ プロジェクト新規作成（11/11テスト pass）・background_weight/gamma_dice 設計方針確認 |
| [2026-04-08](work-logs/2026-04-08.md) | Phase 5完了(13/13 pass)・fold 0実験: angle 5.42deg / mIoU 0.911 |
| [2026-04-07](work-logs/2026-04-07.md) | Phase 7完了: dataset.py に gt_masks 読込統合・bad_slices フォーマット修正 |
| [2026-04-06](work-logs/2026-04-06.md) | 半平面分割実装・CT-maskアフィンズレ修正・全件バッチ保存(906/909)・QCスコアリング完了 |
| [2026-04-05](work-logs/2026-04-05.md) | 半平面分割プロトタイプ検証(99.9%)・方針決定 |
| [2026-04-02](work-logs/2026-04-02.md) | region-mask 設計ミス判明・再設計候補整理 |
| [2026-04-01](work-logs/2026-04-01.md) | eval_error_viz.py 実装 |
| [2026-03-31](work-logs/2026-03-31.md) | sigma確定(3.5)、threshold sweep、GT品質確認 |
