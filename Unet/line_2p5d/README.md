# line_2p5d

前後2スライスを含む5枚のCT・椎体maskから、各画像の4本線heatmapを予測する独立プロジェクト。
`line_only/` と `line_surface_3d/` は変更せず、平面・傾き・外挿は扱わない。

## モデル

```text
(B, 5, 2, H, W)
    -> 全スライス共有2D Encoder
    -> bottleneckのz方向Residual Conv
    -> 全スライス共有2D Decoder
    -> (B, 5, 4, H, W)
```

- 手動線1画像につき1学習sampleを作る。
- heatmap教師は中心画像だけに適用する。
- 周辺4画像の出力は局所幾何整合性に使用する。
- 評価も中心画像だけを使い、重複窓平均で誤差を隠さない。

## 損失

```text
L = L_mse
  + schedule * (lambda_angle * L_angle + lambda_position * L_position)
```

- `L_angle`: 隣接画像間の180度周期の線角度変化を抑える。
- `L_position`: 3画像間の線重心の法線方向二階差分を抑える。
- `L_mse`: `line_only/` と同じ通常のmean MSEを中心画像へ適用する。
- 一階差分を固定しないため、線がz方向へ自然に移動することは許容する。
- `loss.geometry.enabled` で幾何損失の使用有無を切り替える。
- `start_epoch` まではheatmap損失だけで学習し、その後 `ramp_epochs` で線形に立ち上げる。

## 実行

heatmap-only対照群:

```bash
uv run python Unet/line_2p5d/train.py \
  --config Unet/line_2p5d/config/baseline.yaml
```

段階的な局所幾何整合性あり:

```bash
uv run python Unet/line_2p5d/train.py \
  --config Unet/line_2p5d/config/geometry.yaml
```

実行foldは各configの `folds.start` / `folds.end` で指定する。CLIの
`--start_fold` / `--end_fold` を渡した場合だけconfig値を上書きする。

`wandb.enabled: true`の場合はfoldごとにW&B runを作成し、epochの学習・検証指標、
best checkpointの指標、test指標を記録する。`project: null`では
`unet-{experiment.phase}-{experiment.name}`、`run_name: null`では`foldN`を使う。

主評価は画像単位の `line_angle_error_deg`、`line_rho_error_px`、
`line_combined_error_px`、`heatmap_collapse_rate`。評価時の線抽出は
`heatmap_threshold.mode: adaptive` を使い、角度・rho外れ値率も保存する。

学習終了後は `line_only/` と同じ形式で以下を保存する。

- VAL/TEST各`evaluation.visualization_samples`件のGT・予測heatmap grid/overlay
- TEST全件のGT/予測直線比較画像とheatmap・予測線・GT線の3列画像
- TEST全件の予測端点・誤差JSON

実験は`line_only/config/config.yaml`と同じく`experiment.phase`と
`experiment.name`で管理する。既定の保存先は
`Unet/outputs/<experiment.phase>/<experiment.name>/vis/foldN/`。予測線長はGT線長へ
`evaluation.line_extend_ratio`を掛けて描画する。`visualization_dir`を指定すると
可視化だけを別directoryへ保存できる。
