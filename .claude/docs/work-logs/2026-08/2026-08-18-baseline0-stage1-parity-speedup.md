# 2026-08-18: Baseline 0 Stage1条件統一・高速化

## 目的

`fracture_detection/baseline0/`を`train_models/stage1/`の学習条件へ揃えつつ、flip系augmentationを除外し、品質除外・loss定義・評価ログ・入力pipelineを正式run前に固定した。

## 現在の正式設定

- protocol: `baseline0-nested-v7`
- experiment: `08_18/v4`
- GPU: `cuda:1`
- nested split: `outer=k`、`val=(k+1)%5`、残り3 foldsで学習
- batch size: 16
- DataLoader workers: 8
- epoch上限: 75
- early stopping: `val_bce`、patience 15
- checkpoint: val AUROC最大をprimary、val PR-AUC最大をsecondary
- scheduler: `2.3e-4 -> 2.3e-5`の75 epoch単一cosine

## 品質除外と共通manifest

Stage1と同じ`data/rsna_data/excluded_studies.csv` / `excluded_levels.csv`を共通manifestへ適用した。既存`folds.csv`は再生成せず、品質除外後も同じpatient-fold割当を使う。

| 項目 | 値 |
|---|---:|
| 品質除外前 | 13,928 bag |
| 除外bag | 496 bag |
| 品質除外後 | 13,432 bag |
| study | 2,009 |
| whole陽性 | 1,332 |
| region annotation | 268 bag（不変） |

- manifest SHA256: `9bc0b8b91a5ff719519a63a3b2a7aa7f14476b45fade5582efb58a258ef21ac3`
- excluded studies SHA256: `45830a8eb448a49f61af434f066a40342fc2189dbd754decb6e31ab1d5a6e6ab`
- excluded levels SHA256: `3afc3e004a90d8065895eba5360f3f2c9bee15526fefc042d8d04617f64c7d4d`

## Stage1準拠augmentation

R2/R3の向きを維持するためhorizontal flip、vertical flip、transposeだけは禁止した。それ以外はStage1へ揃えた。

| 変換 | 設定 |
|---|---|
| brightness | limit 0.1、`p=0.7`、contrast 0 |
| Affine | shift 0.3、scale 0.7–1.3、rotate ±45°、`BORDER_REFLECT_101`、`p=0.7` |
| blur/noise | Motion / Median / Gaussian / GaussNoiseのOneOf、`p=0.5` |
| distortion | Optical / GridのOneOf、`p=0.5` |
| cutout | 112×112、1 hole、`p=0.05` |
| mixup | batch単位`p=0.2`、`lambda ~ U(0,1)` |

15面×5 CT channelを75 channelへ、15面whole maskを15 channelへstackし、Albumentationsは1 bagにつき1回だけ呼ぶ。これにより全15面・全channel・maskへ同じ幾何変換を適用する。intensity系とcutoutはCTだけに作用する。val・outerではaugmentationとmixupを使わない。

## Loss定義

旧実装のPyTorch `pos_weight`付き要素数平均を廃止し、Stage1と同じ定義へ変更した。

```text
element_loss = BCEWithLogits(logit, target, reduction="none")
weight = 2.0 if target > 0 else 1.0
loss = sum(element_loss * weight) / sum(weight)
```

- 15面へwhole targetをbroadcastして計算する。
- train/valのepoch lossはStage1と同じbatch loss平均。
- mixup時は`lambda * loss(target_a) + (1-lambda) * loss(target_b)`。
- v5以前とはlossの数値定義が異なるため旧checkpointをresumeしない。

## 評価ログと正式閾値

epochごとに次をconsole、`history.csv`、W&Bへ保存する。

- val BCE / AUROC / PR-AUC
- 固定閾値0.5のprecision / recall / F1
- val内でF1を最大化する閾値
- その閾値でのprecision / recall / F1

F1最大閾値はval scoreの各ユニーク値を候補にし、同率なら最も高い閾値を選ぶ。`val_recall_at_f1_optimal`はrecall自体の最大値ではなく、F1最大閾値における`TP/(TP+FN)`である。正式評価ではAUROC-bestとPR-AUC-bestの各checkpointが自身のvalで閾値を決め、その閾値を対応するouterへ固定適用する。outerでは閾値を再最適化しない。

## Staging対応

- 新manifest cache容量: 65.9 GiB / 40,296 files
- cache path: `/dev/shm/vai-fracture-dataset/9bc0b8b91a5ff719519a63a3b2a7aa7f14476b45fade5582efb58a258ef21ac3`
- 旧manifest `39d46...75a3`の69 GiB cacheが残り空き不足になったため、旧cacheだけを削除した。
- 新manifestの`READY.json`作成は完了済み。同じmanifestでは以降NFSから再copyせず再利用できる。
- sandbox内とホスト側で`/dev/shm`のmount namespaceが異なるため、容量確認・削除はホスト側で行う必要がある。

## 入力pipeline高速化

v6は面・CT channelごとにReplayを呼び、1 bagにつき75回Albumentationsを実行していた。Stage1実装を再確認し、v7では全面stack後の1回呼出しへ変更した。

さらに以下を適用した。

- Baseline 0で不要な`region_4class.npy`を学習時に読み込まない。
- CTとwhole maskを`uint8`のままDataLoaderからnonblocking GPU転送する。
- whole maskは転送前に0/255へ変換し、GPU上で全6chを255除算することで従来のmask入力0/1を維持する。
- 固定224×224 shape向けに`torch.backends.cudnn.benchmark = True`を有効化する。

ローカル計測結果:

| 計測 | v6旧方式 | v7新方式 | 改善 |
|---|---:|---:|---:|
| augmentation / bag中央値 | 0.440秒 | 0.145秒 | 約3.0倍 |
| 実bag item生成中央値 | 0.434秒 | 0.109秒 | 約4.0倍 |
| DataLoader payload / item | 17.23 MiB | 4.31 MiB | 4分の1 |

v6診断runのouter 0 epoch 1では`data_wait=162.24秒`、`compute=226.77秒`、epoch全体`423.95秒`だった。入力待ちが十分大きいため、v7の一括augmentationとuint8転送は実学習時間にも寄与すると判断した。GPU計算そのもの約227秒はモデル・batch size・GPU数を変えない限り大幅には短縮しない。

## v6診断成果物の扱い

`fracture_detection/baseline0/outputs/08_18/v3/`はv6でstaging後に開始した診断runで、少なくともouter 0 epoch 1の成果物が存在する。v7高速化コードは起動済みprocessへ反映されないため、ユーザーが旧processを停止してv7を新規起動する。v6 checkpointはv7へresumeせず、成果物は診断記録として保持する。

## 検証

- common + Baseline 0: 66 tests passed
- Ruff format: passed
- Ruff check: passed
- mypy: 27 source files passed
- 実bagのuint8・一括augmentation pathが正常に`[15,6,224,224]`を返すことを確認
- augmentationが1 bagにつき1回だけ呼ばれる回帰テストを追加
- GPU転送後にuint8入力がfloat32の0〜1へ正規化される回帰テストを追加

## 次の確認

1. 旧v6 processを完全に停止する。
2. `baseline0-nested-v7` / `08_18/v4`を`--resume`なしで起動する。
3. 新cacheが`READY.json`から再利用され、copyが発生しないことを確認する。
4. v7 epoch 1の`train_data_wait_seconds` / `train_compute_seconds` / `epoch_seconds`をv6診断値と比較する。
5. v7以外の旧checkpoint・旧effective configを正式結果へ混在させない。
