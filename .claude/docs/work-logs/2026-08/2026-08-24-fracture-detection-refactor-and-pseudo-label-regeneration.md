# fracture_detection整理と疑似ラベル再生成 worklog

**日付**: 2026-08-24  
**状態**: 完了。コード整理、疑似ラベル機能復元、全量再生成、成果物監査まで実施済み。

## 1. 依頼内容

1. `fracture_detection/` 配下の過剰なファイル・ディレクトリを削減する。
2. 失敗した過去手法を現行treeから除外する。
3. `fracture_detection/baseline0/` 直下のコードを責務別に整理する。
4. `PROGRESS.md` を現行方針へほぼ初期化する。
5. 今後多用する疑似ラベル生成とCAM監査を主要機能として維持する。

## 2. Repository整理

- 失敗したMTL、Proposed、Type2、凍結multi-arm基盤、旧診断run、tree内archiveを削除した。
- Baseline 0を `cli/`、`config/`、`data/`、`modeling/`、`training/`、`evaluation/`、`resources/`、`tests/` へ責務分割した。
- 正式5-fold教師checkpointは `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/` に保持した。
- `fracture_detection/PROGRESS.md` を現行実装と保守方針だけへ縮小した。
- 新しい手法ごとにトップレベルdirectoryを増やさず、採用済み機能をBaseline 0配下へ責務別配置する方針にした。

## 3. 疑似ラベル機能の復元

整理時に疑似ラベル生成とCAM監査を失敗手法と誤認し、コードと生成物を削除した。疑似ラベルは今後の主要機能であるため、この判断を訂正した。

- Git commit `4a32907` から疑似ラベル、Grad-CAM、CAM監査、監査reportを復元した。
- 実装を `fracture_detection/baseline0/pseudo_labeling/` へ集約した。
  - `scoring.py`: log score、温度推定、pairwise confidence、ranking loss、pair構築
  - `gradcam.py`: checkpoint読込、Grad-CAM、領域集約
  - `cam_audit.py`: teacher role、mask摂動、density enrichment
  - `report.py`: memorization、摂動、TTA、局在監査table
- `baseline0/cli/attention.py`、`cam_audit.py`、`generate_pseudo_labels.py` を復元した。
- `region_validity.py` を現行 `baseline0/data/` へ移した。
- 削除済み `common/`、`core/`、旧 `analysis/` へのimportを現行Baseline 0 APIへ置換した。
- `PROGRESS.md`、Baseline 0 README、`.claude/docs/DESIGN.md` を訂正し、疑似ラベル生成とCAM監査をfirst-class componentとして明記した。

## 4. 検証結果

- `uv run pytest -q fracture_detection/baseline0/tests`: **101 passed**
- `uv run ruff check fracture_detection/baseline0`: **passed**
- `uv run ruff format --check fracture_detection/baseline0`: **passed**
- `python -m fracture_detection.baseline0.cli.cam_audit --help`: **passed**
- `python -m fracture_detection.baseline0.cli.generate_pseudo_labels --help`: **passed**

## 5. 疑似ラベル全量再生成

### 実行契約

- 教師: `baseline0_shared_core/outer0`〜`outer4` の全5 `best_model.pt`
- 割当: fold-matched in-sample `Teacher_k`
- 処理行数: `[8074, 8055, 8056, 8048, 8063]`、合計 **40,296行**
- 一意bag数: **13,432 bag**（各bagは対応する3 teacherで1回ずつ採点）
- device: GPU 0 (`cuda:0`, NVIDIA RTX A6000)
- batch size: 16
- 出力先: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/`
- 実行PID: `526388`（`.tmp/pseudo_label_generation.pid`）
- 完了watcher: tmux session `pseudo_label_watcher_20260824`

### 2026-08-24 17:54 JST時点

- 元の生成processはGPU 0を約37.6 GiB使用して継続中。
- process I/O `rchar=72,298,812,485` bytes。1 bagで読む3配列の実サイズ約5.27 MBを基準にすると、約13,700 / 40,296 bag相当、概算34%地点。
- `generate_pseudo_labels.py` は全foldのrowをmemoryへ集約後にCSVを書き出すため、完了前は出力directoryが空で正常。
- launcher PIDが消えたため一度だけ重複detached起動を試みたが、元processがGPU上で生存していたため重複側だけCUDA OOMで終了した。元processは中断していない。
- 重複側がruntime logを上書きしたため、以後はhost PID、GPU utilization、completion marker、最終成果物で監視する。

### 完了結果

- 開始: 2026-08-24 17:29:29 JST
- 生成完了: 2026-08-24 18:53:09 JST
- 所要時間: 約1時間24分
- runtime log最終行: `wrote 40296 rows for 13432 bags`
- `pseudo_label_scores.csv`: 9,142,411 bytes
- `pseudo_label_temperatures.csv`: 1,179 bytes
- `pseudo_label_generation_metadata.json`: 2,629 bytes

### 完了監査

1. `pseudo_label_scores.csv`: **40,296行でPASS**
2. 一意な `study_id × level`: **13,432 bagでPASS**
3. 各bagのteacher行数: **全bagが3行でPASS**
4. teacher別行数: **outer0=8,074 / outer1=8,055 / outer2=8,056 / outer3=8,048 / outer4=8,063でPASS**
5. fold provenance: **全行がmetadataの `teacher_train_folds` 内でPASS**
6. teacher ID・checkpoint hash: **全行と5実checkpointの再計算SHA-256が一致してPASS**
7. temperature table: **5 teacher × 4領域 = 20行、全temperatureが有限かつ正でPASS**
8. scores SHA-256: `2a78aededc11b3231aaf906cbf907e2104486c6db18205d6a0d79f212bfea22f`
9. temperatures SHA-256: `ed91915d0b30e0a83a0c2f72898ec10d6b9da402c4362249998c262ba426559b`
10. metadata SHA-256: `bf65e5f27c954cb26ced3ab1d4b3f8775c3d70d1c94e0b5ecbca1b2b85eec7e6`

`region_3_score` には同一bag（study `1.2.826.0.1.3680043.30177`、C6）のteacher 0・1で値0が2セルある。他の領域を含む残り161,182セルは全て正値。これは生成失敗ではなくCAM massが0の未定義順位であり、temperature表の `n_defined` と一致し、pair構築時に除外される。
