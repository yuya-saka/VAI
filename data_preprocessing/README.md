# data_preprocessing — 前処理パイプライン

頸椎CTの全前処理をここに集約。**用途別にサブフォルダを分割**している。
DICOM（生データ）から、学習用の2Dデータセットまでを一本のパイプラインで生成する。

> RSNAデータ（`rsna_data/`）は外部Kaggleデータセットで、**本パイプラインの対象外**（別系統）。

---

## パイプライン全体図

```
data/sampleX/dicom/  (生DICOM・入力)
        │  ① dicom_to_nifti/
        ▼
   NIfTI + nifti_list.csv
        │  ② volume_prep/
        ├──────────────► segmentations/      (椎体C1-C7マスク / TotalSegmentator)
        ├──────────────► fracture_labels/     (骨折GTボリューム)
        ▼
   predata_simple/  (椎体ごと固定サイズcrop・中間)
        │  auto_tilt + add_fracture_labels
        ▼
   annotation_data/ (傾き補正 + ランドマーク + fracture.nii.gz・中間)
        │
        ├─ ③ segmentation_dataset/ ─► dataset/        ──► 領域分割・直線検出モデル
        │                                                  (Unet/{line_only,multitask,seg_only,seg_sdf})
        │
        └─ ④ learning_dataset/    ─► dataset_zprop/   ──► 骨折検出学習
                                                           (learning/)
```

---

## サブフォルダと責務

### ① `dicom_to_nifti/` — DICOM → NIfTI 変換（入口）

| スクリプト | 役割 | 入力 → 出力 |
|---|---|---|
| `recognize_data.py` | `data/` のフォルダ名整理（`sampleX`化・`DICOMSAVE`→`dicom/`） | `data/` |
| `output_nifti.py` | dcm2niixでNIfTI化＋骨折範囲をniftiインデックスへマップ | `data/.../dicom/` → NIfTI |
| `create_csv.py` | `input_list.csv` → `nifti_list.csv`（座標変換） | csv → csv |
| `input_list.csv` / `nifti_list.csv` | 入出力リスト（gitignore対象） | — |

> ⚠️ **注意**: `recognize_data.py` 等に別マシンの絶対パス（`/home/yuya/...`）がハードコードされている。実行前に各自の環境パスへ要変更。NIfTIのstaging先名（`nifti_out` / `nifti_output`）が②と不統一なので合わせること（[既知の課題](#既知の課題)参照）。

### ② `volume_prep/` — 3Dボリューム準備

| スクリプト | 役割 | 出力dir |
|---|---|---|
| `segmentation_spine.py` | TotalSegmentatorで椎体C1-C7マスク生成 | `segmentations/` |
| `phase0_generate_labels.py` | 骨折GTボリューム生成 | `fracture_labels/` |
| `generate_pre.py` | NIfTI＋segを椎体ごと固定サイズ(256²)でcrop | `predata_simple/` |
| `auto_tilt.py` | axial断面マスク面積最大の傾きを自動探索し補正 | `annotation_data/` |
| `add_fracture_labels.py` | 骨折ラベルを椎骨座標系へ変換し付与 | `annotation_data/.../fracture.nii.gz` |

### ③ `segmentation_dataset/` — 領域分割学習用2D（→ `dataset/`）

| スクリプト | 役割 |
|---|---|
| `convert_to_png.py` | NIfTI → 224×224 PNG（images / masks / overlays / lines.json） |
| `extract_slices.py` | 斜断面サンプリングで代表スライス抽出（補間ぼけ回避） |
| `generate_region_mask.py` | 4領域（body / 左右foramen / posterior）マスク生成（TLS直線ベース） |
| `pilot_region_mask.py` / `preprocess_all.py` | 全サンプルへ領域マスク適用＋`bad_slices`収集 |
| `qc_score.py` | スライス品質スコア → `qc_scores.json`（keep / downweight / exclude） |
| `rebuild_annotated_region_masks.py` | 手動補正済み領域マスクを再構築 → `region_mask_evaluation/` |

### ④ `learning_dataset/` — 骨折検出学習用2D（→ `dataset_zprop/`）

| スクリプト | 役割 |
|---|---|
| `propagate_lines_z.py` | アノテ済みスライスの直線をz方向へ伝播し、全スライスにラベル付与（`line_provenance.json` 付） |

### `visualization/` — 可視化ツール

`visualize_gt_masks.py` / `visualize_polylines.py` / `visualize_region_masks.py`
→ 出力は `data_preprocessing/visualization/output/`

### `tests/`

`test_generate_region_mask.py`（6件パス） / `test_propagate_lines_z.py`（[既知の課題](#既知の課題)参照）

---

## 実行順序

```bash
# ① DICOM → NIfTI（別環境/パス要調整）
uv run python data_preprocessing/dicom_to_nifti/recognize_data.py
uv run python data_preprocessing/dicom_to_nifti/output_nifti.py
uv run python data_preprocessing/dicom_to_nifti/create_csv.py

# ② ボリューム準備
uv run python data_preprocessing/volume_prep/segmentation_spine.py
uv run python data_preprocessing/volume_prep/phase0_generate_labels.py
uv run python data_preprocessing/volume_prep/generate_pre.py
uv run python data_preprocessing/volume_prep/auto_tilt.py
uv run python data_preprocessing/volume_prep/add_fracture_labels.py

# ③ 領域分割データセット
uv run python data_preprocessing/segmentation_dataset/preprocess_all.py   # 領域マスク + bad_slices
uv run python data_preprocessing/segmentation_dataset/qc_score.py

# ④ 骨折検出データセット
uv run python data_preprocessing/learning_dataset/propagate_lines_z.py
```

---

## データディレクトリ用語集（プロジェクトルート直下）

| ディレクトリ | 内容 | 用途 |
|---|---|---|
| `data/` | 生DICOM | **入力** |
| `segmentations/` | 椎体C1-C7マスク | 中間（②） |
| `fracture_labels/` | 骨折GTボリューム | 中間（②） |
| `predata_simple/` | 椎体crop（ct/mask） | 中間（②） |
| `annotation_data/` | 傾き補正＋ランドマーク＋骨折 | 中間（②） |
| `dataset/` | 2D PNGスライス＋領域マスク | **領域分割学習** |
| `dataset_zprop/` | 2D PNG＋z伝播直線 | **骨折検出学習** |
| `region_mask_evaluation/` | 領域マスク手動補正の評価 | 評価/QA |
| `rsna_data/` | RSNA 2022 Kaggle | **外部・パイプライン外** |

---

## 既知の課題

1. **NIfTI staging名の不統一**: `volume_prep/` の各スクリプトは `<root>/nifti_output` を参照するが、`dicom_to_nifti/output_nifti.py` は `./nifti_out` へ出力する。実行前にstaging先を揃える必要がある（現状 `nifti_output` は未作成）。
2. **dms2niixのハードコードパス**: `dicom_to_nifti/recognize_data.py` に `/home/yuya/...` の別マシン絶対パスが残る。
3. **`tests/test_propagate_lines_z.py` は既存破損**: 削除済み関数 `align_polyline_direction` をimportしており収集エラーになる（本リファクタ以前から壊れている）。テスト側の更新が必要。

---

## 履歴

- 2026-06-16: 散在していた前処理（旧 `data_preprocessing/` 直下 ＋ `Unet/preprocessing/`）を本ディレクトリに用途別統合。旧3Dボリューム版（`spine_data`系）は `trash/old_3d_pipeline/` へ退避。
