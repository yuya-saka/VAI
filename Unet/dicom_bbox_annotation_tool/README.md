# 元DICOM bboxアノテーションツール

`data/rsna_data/train_bounding_boxes.csv` の矩形を、対応する元DICOMスライスの
ピクセル座標へリサイズ・再投影せず直接描画する独立ツールです。

椎体レベル (`C1`–`C7`) はCSVから推定せず、
`data/rsna_data/processing_metadata/*.json` の
`assigned_bbox_slice_numbers` に保存された前処理時の確定割当を使用します。
DICOM系列上で連続するbbox行を1つの `run_XX` として表示します。
割当椎体の処理済み3D maskも元DICOM平面へ最近傍投影し、水色で重ねます。

## 起動

```bash
uv run python Unet/dicom_bbox_annotation_tool/server.py
```

ブラウザを自動起動しない場合:

```bash
uv run python Unet/dicom_bbox_annotation_tool/server.py --no-browser
```

既定URLは `http://localhost:8767` です。

## 保存先

判定結果は既存ツールのラベルを上書きせず、次へ保存します。

```text
data/rsna_data/fracture_region_labels_dicom.csv
```

別の保存先を使う場合は `--label-csv` を指定してください。

## 表示仕様

- CT表示のみ固定bone window (`W=2000`, `L=400`) を適用
- 画像の行列サイズ、向き、bbox座標は元DICOMのまま
- 水色の半透明領域が、割り当てられた `C1`–`C7` の椎体mask
- 赤い矩形と半透明塗りがCSVの `x`, `y`, `width`, `height`
- 各カードをクリックすると元解像度に近い拡大表示
- 表示画像はメモリ上で生成し、画像ファイルとしては保存しない

## 骨折椎体の見落とし対策

`assigned_bbox_slice_numbers` はDICOM z範囲だけで椎体を割り当てるため、
隣接椎体のz範囲が重なる箇所でbbox行が別椎体へ誤って割り当てられ、
`train.csv` で骨折ラベルがある椎体にrunが1つも生成されないことがある
(全235 studyの検証で25件確認)。起動時に以下の2段階で対処する。

- **被覆率から復元 (`coverage_recovered`)**: bbox矩形内で全7椎体のmask
  被覆率を比較し、最も被覆する椎体へrunを追加する（サイドバーに黄色
  「補完」バッジ）。9件該当。
- **bbox欠損の警告 (`bbox_missing`)**: 上記でも該当bboxが1行も見つから
  ない場合、画像を持たない警告専用runとして残す（赤「bboxなし」バッジ、
  保存不可）。RSNA元データにその椎体のbbox自体が存在しないケースで、
  ツール側では復元できない。16件該当。

いずれも `--train-csv`（既定 `data/rsna_data/train.csv`）の骨折ラベルを
参照する。既存の前処理割当runは変更せず追加のみ行うため、保存済み
ラベルには影響しない。
