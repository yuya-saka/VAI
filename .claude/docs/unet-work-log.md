# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-15 -->

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

## 保留タスク

- multitask/ で aug 変換修正の実験実施（line_only の改善効果確認）
- reg + aug変換修正 の組み合わせ実験
- 評価コードに部位別 angle/rho 記録を追加
- seg_only + 椎体条件付け実験

---

## コード構成

実行: `uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml`
テスト: `uv run pytest Unet/multitask/test/ Unet/seg_only/test/ -v`（30/30 pass）

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
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
