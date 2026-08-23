# mtl_type2（RSNA Type2型 branch分離MTL・探索project）

2026-08-20、RSNA 1st place solutionの`stage2-type2.ipynb`（患者単位・7椎骨×15スライスを
1バッチとして学習し、`lstm`/`head`と`lstm2`/`head2`を分離するmulti-task構成）を
参考にした修正方針の実装。既存の正式パイプライン（`baseline0/` `mtl/` `proposed/`、
`core/` `cli/` `config/`）は一切変更していない。学習で効果を確認できたら
正式パイプラインへ取り込む方針（現時点では未統合）。

## 何を変えたか（既存`mtl/`アームとの差分）

| 項目 | 既存`mtl/`（`EarlyFusionMtlModel`） | `mtl_type2`（`BranchedMtlModel`） |
|---|---|---|
| LSTM | whole/regionで共有1本 | whole用・region用を完全に分離（RSNAのtype2と同じ`lstm`/`lstm2`） |
| detail batch size | 1 bag | 4 bag |
| detail更新頻度 | natural stepの**毎回**（annotated 160 bagを3周以上） | **1 epoch=annotated datasetを1周**（約40回/epoch）。
  `training.schedule.region_step_schedule`でnatural step列へ等間隔・決定論的に配置 |
| λ（region loss重み） | 勾配比校正（target_ratio 0.5） | **1.0固定**（学習頻度自体を大きく変えたため校正は行わない） |
| 対象アーム | 6構成（Proposed含む） | **2構成のみ**: `control_type2`（6ch）/ `baseline1_type2`（10ch）。
  Proposed（PMGAN式attention）は延期 |
| 評価 | val AUROC/AP・領域別AP | 上記に加え、**train/val双方のregion APを毎epoch記録**（診断用）。
  `best_region.pt`（val region AP最良）を**診断専用**checkpointとして追加保存
  （outer推論には使わない。outer推論はこれまで通りval AUROC-bestの`best_model.pt`のみ） |

変えていないもの: 15面入力・15面broadcast loss・推論時15面平均、CT5ch＋mask、
detail batchは`L_region`のみに寄与（whole/attentionへは寄与させない）、
natural→annotated逐次backward・optimizer step 1回、fold分割（凍結`folds.csv`）、
pos_weight=2.0、augmentation設定。

## ディレクトリ構成

```
mtl_type2/
├── modeling/model.py       BranchedMtlModel（共有CNN + whole/region独立BiLSTM）
├── training/
│   ├── schedule.py         region_step_schedule（低頻度detail stepの配置）
│   ├── steps.py            train_step（annotated_batch=Noneのstepはregion backwardしない）
│   ├── diagnostics.py      region_average_precision（train/val診断用、bootstrapなし）
│   ├── experiment.py       成果物パス（core.experimentのpackage制限を避けるため独立実装）
│   └── trainer.py          train_fold/_train_epoch（core.trainer.evaluate等は直接再利用）
├── config/
│   ├── schema.py           凍結config契約（PROTOCOL_VERSION="mtl-type2-v1"）
│   ├── control_type2.yaml
│   └── baseline1_type2.yaml
├── cli/train.py            学習CLI（fold並列launcherは持たない。1プロセス=1 outer範囲）
└── tests/                  schedule/model/steps/trainer/configの単体テスト（19件、CPU）
```

## 実行方法

outer foldごとに手動で1プロセス起動する（`feedback-manual-execution-trigger`の方針どおり、
自動連鎖はしない。複数GPUで並行させたい場合は`--gpu-id`を変えて別プロセスを手動起動する）。

```bash
uv run python -m fracture_detection.mtl_type2.cli \
  --arm baseline1_type2 --outer-fold 0 --gpu-id 2

uv run python -m fracture_detection.mtl_type2.cli \
  --arm control_type2 --outer-fold 0 --gpu-id 3
```

`--resume`でepoch境界から再開できる（outer推論済みならskip）。

## 見るべき指標

`outputs/08_20/<arm>/outer<N>/history.csv`の

- `train_region_ap_macro` / `val_region_ap_macro` — train側が低ければregion branch自体が
  未学習（更新回数不足の疑い）、trainだけ高くvalが低ければ過学習
- `val_auroc` — 既存`mtl/`のBaseline 1–B（`fracture_detection/mtl/outputs/08_19/baseline1_b/`）と
  比較し、wholeタスクが悪化していないか確認する
- `train_region_optimizer_steps` — 想定通り約40（natural stepの毎回ではないこと）になっているか

## 未確定・今回スコープ外

- Proposed（PMGAN式attention、H2検定）は延期。3構成で先に確認してから戻す
- λは1.0固定のみを実行。0.5/2の探索は結果を見てから判断（2026-08-20ユーザー決定）
- `best_region.pt`はouter推論に使わない（診断専用、同ユーザー決定）
- 正式パイプラインへの統合（`core/`側の`_train_epoch`の低頻度化、6構成への反映）は
  この探索で効果を確認した後の別タスク
