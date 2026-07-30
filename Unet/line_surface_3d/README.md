# Line Surface 3D

連続する頸椎CTスライスの4本線を、z方向に1次のリボンとして学習する独立プロジェクト。
既存コードは `Unet/line_only/` だけを設計参照元とし、実行時依存は持たない。

## 入出力

- 密画像: `data/dataset_zprop/sample*/C*/images/`
- 密椎体mask: `data/dataset_zprop/sample*/C*/masks/`
- 手動教師: `data/dataset/sample*/C*/lines.json`
- 初期窓: `N=15`
- 入力: `(30, 224, 224)` の slice-major `[CT_z, mask_z]`
- 出力: `(15, 4, 224, 224)` の線heatmap

`dataset_zprop` の `lines.json` は読み込まない。

## 学習

Heatmap-only baseline:

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  Unet/line_surface_3d/train.py \
  --config Unet/line_surface_3d/config/baseline.yaml \
  --start_fold 0 --end_fold 0
```

リボン損失あり:

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  Unet/line_surface_3d/train.py \
  --config Unet/line_surface_3d/config/ribbon.yaml \
  --start_fold 0 --end_fold 0
```

W&Bの有効・無効は各configの `wandb.enabled` で切り替える。成果物は
`Unet/outputs/line_surface_3d/{experiment_name}/` に保存する。
DataLoaderは現在 `num_workers=8`。実測では `0` の約281秒/epochから
約37秒/epochまで短縮した。大きなslab batchを過剰に先読みしないよう
`prefetch_factor=1` とし、epochごとのworker再生成を避けるため
`persistent_workers=true` としている。

## 全高推論

Test split:

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  Unet/line_surface_3d/predict.py \
  --config Unet/line_surface_3d/config/ribbon.yaml \
  --fold 0 --split test
```

全sample:

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run python \
  Unet/line_surface_3d/predict.py \
  --config Unet/line_surface_3d/config/ribbon.yaml \
  --fold 0 --split all
```

推論は重複窓の重心とdoubled-angleを平均し、不一致を信頼度指標として保存する。
同時に4領域欠損率、距離別欠損率、z平滑性、冠状断・矢状断図を生成する。

## 検証

学習時は、line_onlyと同じ定義・名前で次の共通指標を記録する。

- `val_loss_mse`
- `peak_dist_mean`
- `blob_iou`
- `angle_error_deg`
- `rho_error_px`
- `val_outlier_angle_rate`
- `val_outlier_rho_rate`
- `per_vertebra`

共通指標は手動GTポリライン由来の `(phi, rho)` と、適応閾値・
connected-component filter後の予測momentを比較する。
checkpointは `angle_error_deg`、schedulerとearly stoppingは
`val_loss_mse` を監視する。
この評価契約を `line_surface_3d_v2` checkpoint protocolとし、旧v1は推論で拒否する。

3D固有指標として、`surface_raw_*` / `surface_fitted_*` の角度・重心誤差、
surface loss成分、検出率、推論時の重複窓不一致・領域欠損率・z平滑性を追加で記録する。

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline pytest \
  -o pythonpath=Unet Unet/line_surface_3d/test -q
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline ruff check \
  Unet/line_surface_3d
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline mypy \
  Unet/line_surface_3d --ignore-missing-imports
```
