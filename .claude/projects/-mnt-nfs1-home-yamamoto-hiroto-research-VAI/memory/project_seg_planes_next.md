---
name: project-seg-planes-next
description: 4領域分割segプレーン実装完了・次フェーズの方針
metadata:
  type: project
---

4領域分割の実現可能性検証が完了（2026-06-27）。仕様確定。

**Why:** 最大断面積スライス中心 ±2スライスの5枚を使って、各椎体の4領域分割モデルを訓練するため。

**確定した仕様:**
- 入力: `seg_ct.npy` (5,224,224) uint8 + `seg_vertebra_mask.npy` (5,224,224) uint8
- 全13,944レベル・フィルタなしで使用可（成功率99.8%）
- C1/C2/C7 の line_2/line_4 の角度外れ値（最大22%）は ±180°符号あいまいさが原因で構造破綻なし → 除外不要
- 学習データ規模: ~69,700スライス（既存42患者1,610スライスの約43倍）

**How to apply:** 次セッションは z 軸外挿法の設計から始める。

**次セッション TODO:**
1. **z軸方向の線外挿法の検討**
   - 5枚の分割結果を z 軸方向に外挿してライン推定する手法の設計
   - `Unet/outputs/angle_stats/angles_raw.csv` に全データの角度があり参照可

**検証スクリプト（参照用）:**
- `Unet/line_only/verify_seg_4region.py` — 4領域分割の成功率・面積統計
- `Unet/line_only/collect_angle_stats.py` — 線角度統計（椎体レベル別）
- `Unet/line_only/visualize_angle_outliers.py` — 角度分布ヒストグラム・外れ値可視化

**実装済みデータ準備ファイル:**
- `data_preprocessing/rsna_pipeline/segmentation_plane_sampling.py`
- `data_preprocessing/rsna_pipeline/add_segmentation_planes.py`
- `data_preprocessing/rsna_pipeline/process_study.py` / `process_dataset.py`

**データ状況:**
- 保存完了: 1,939件 / 失敗0件
- 除外: study 1件 + level 134件（excluded_studies.csv / excluded_levels.csv）
