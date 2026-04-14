# Unet/ 作業サマリー

<!-- ルール: 現在地・次アクションのみ。完了詳細は work-logs/YYYY-MM-DD.md へ。上限60行 -->
<!-- 最終更新: 2026-04-14 -->

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

## 方針転換（2026-04-14）

line heatmap ベースの multitask 改善は打ち止め。**distance map 回帰** を新たな補助タスクとして検討中。

**理由**: heatmap (σ=3.5) は線近傍の数ピクセルにしか勾配信号がなく、seg への情報伝搬が限定的だった。

**distance map 回帰のコンセプト**:
- 各ピクセルが境界線からどれくらい離れているかを予測
- 全ピクセルに監視信号 → 距離急変点＝境界 → seg boundary sharpening に直結

**次セッションで設計を詰める**:
1. 距離の定義（ユークリッド or 符号付き）
2. タスク構成（`L_seg + β·L_dist`）
3. GT 生成方法（distance_transform_edt 等）

## 保留タスク

- shared_decoder 実験（コード実装済み・実験未実施。distance map 方針と組み合わせる可能性あり）
- seg_only + 椎体条件付け実験
- left クラス悪化の原因調査

---

## コード構成

実行: `uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml`
テスト: `uv run pytest Unet/multitask/test/ Unet/seg_only/test/ -v`（30/30 pass）

---

## 過去ログ

| 日付 | 主な内容 |
|------|---------|
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
