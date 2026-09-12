# fracture_detection 進捗

最終更新: 2026-09-10

## 現在の主軸

`fracture_detection/weak/` の region-MIL モデル（4領域logit → 固定noisy-ORでwhole）を実装済み。
設計は `REGION_MIL_DESIGN.md`（2026-09-10採用）。CAM疑似ラベル・校正フェーズ・collapse自動停止を
すべて廃し、GTなし陽性はnoisy-ORの弱教師としてのみ使う3群(N/A/U) mixed supervision。

- architecture: fine-tuneするEfficientNetV2-S trunk（Baseline 0からencoderのみ転送）、
  stride-4 FPN + mask-normalized pooling + 有効面のみpackした共有region BiLSTM +
  共有Linear head。whole専用経路は存在しない
- 損失: N=4領域陰性BCE和、A=4セルBCE和、U=positive noisy-OR（beta倍）。数値は
  float32のlog-survival/log1mexpで安定化
- unit test 84件（合成データ、実manifestでの件数再現含む）で確認済み
- 実装はArm B（提案モデル）のみ。Arm A/C切替機構は未実装
- **outer0のみ学習完走済み（2026-09-11）。** region_macro_ap=0.778、whole AP=0.703
  （baseline0と同一母集団で-0.069。陰性bagのp_whole平均が4倍高いことが原因で、
  design上想定済みの代償。バグではない）。outer1〜4は未実行。
  N/U比率変更の追加比較は要否未決定（ユーザー回答待ち）
- 詳細は `weak/README.md` と
  `.claude/docs/work-logs/2026-09/2026-09-11-weak-arm-b-outer0-training.md` を参照

## 過去の主軸1（維持）

`fracture_detection/region_branch/` の4領域骨折検出モデル（統合4領域 + 単一領域4本）を実装済み。
`baseline0/` は疑似ラベルの供給元・whole path architectureの参照元として維持する。

- architecture: shared EfficientNetV2-S trunk、whole path（Baseline 0とbit-exact）、
  region path（FPN + mask-normalized pooling + shared region BiLSTM + 領域別head）
- 損失: `L = L_whole + lambda_k*(L_exact + alpha_k*L_rank)`。`alpha_k`/`lambda_k`は
  outer foldごとに学習前の決定的な64 batchで一度だけ勾配ノルム校正する
- unit test 35件（合成データ）+ 実データでのGPU/CPU smoke testで一通り確認済み
- 学習・評価は未実施（校正・学習・OOF評価・比較はすべてユーザーが手動起動）
- 詳細は `region_branch/README.md` を参照

## 過去の主軸2（維持）

`fracture_detection/baseline0/` の椎体単位骨折分類と、その教師モデルを使う疑似ラベル生成を主軸として扱う。

- 入力: 15面固定の2.5D CT 5ch + 椎体全体mask 1ch
- モデル: EfficientNetV2-S + BiLSTM
- 出力: 面ごとの骨折logitをmean-sigmoidでbag確率へ集約
- 評価: patient-grouped nested 5-fold
- データ: 品質除外済み13,432 bag
- 疑似ラベル: fold対応教師のGrad-CAMを4解剖領域へ集約し、CAM監査を通して生成

学習済み5-fold成果物は
`fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/` に保持する。

疑似ラベルは2026-08-24に全量再生成済み。

- 出力: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/`
- score: 40,296行、13,432一意bag、各bag 3 teacher
- temperature: 5 teacher × 4領域 = 20行
- provenance・checkpoint hash・出力hashは独立再計算で確認済み

## 現在の構成

```text
fracture_detection/
├── PROGRESS.md
├── baseline0/
│   ├── cli/          # train / evaluate / attention / CAM audit / pseudo-label generation
│   ├── config/       # schema / YAML
│   ├── data/         # dataset / staging / split / sampling / constants
│   ├── modeling/     # model / loss
│   ├── training/     # trainer / optimizer / experiment management
│   ├── evaluation/   # metrics
│   ├── pseudo_labeling/ # Grad-CAM / CAM audit / score / report
│   ├── resources/
│   └── tests/
├── region_branch/
│   ├── README.md
│   ├── cli/          # train / calibrate / evaluate / compare_arms
│   ├── config/       # schema / YAML（統合1 + 単一4）
│   ├── data_pipeline/ # dataset / pseudo_labels / sources / sampling / loaders / constants
│   ├── modeling/     # model / pooling(FPN) / losses
│   ├── training/     # trainer / calibration / monitoring(collapse) / experiment
│   ├── evaluation/   # metrics（validity mask付き領域別AP/AUROC）
│   └── tests/
└── weak/
    ├── README.md
    ├── REGION_MIL_DESIGN.md（リポジトリ直下）参照
    ├── cli/          # inventory / train / evaluate
    ├── config/       # schema / YAML
    ├── data_pipeline/ # dataset / groups(N/A/U) / sampling(GT-pass) / augmentation / loaders
    ├── modeling/     # model / pooling(FPN) / losses(noisy-OR) / initialization
    ├── training/     # trainer / optimization / monitoring(診断のみ) / experiment
    ├── evaluation/   # metrics（3母集団分離 + Brier/ECE）
    └── tests/
```

## 整理方針

- 失敗した MTL、Proposed、Type2 は再利用しない。
- 疑似ラベル生成とCAM監査はbaseline0の主要機能として維持する。
- 新しい手法ごとにトップレベルdirectoryを増やさない（region_branchは4領域検出という
  独立した検討単位のため例外）。
- コードは責務別directoryへ置き、各project直下へ実装fileを増やさない。
- 各directory内では、役割が1ファイルで収まる限り過剰に階層化しない。
- 過去の実装や判断が必要な場合はGit履歴を参照し、現行treeへarchiveを置かない。

## 次の作業

### 0. weak/ の学習実行（実装済み・未実行）

`uv run python -m fracture_detection.weak.cli.inventory` で件数・4領域mask被覆を
確認してから、`uv run python -m fracture_detection.weak.cli.train` を手動起動する。
校正フェーズは存在しない（design docの方針どおり不要）。5 fold完走後に
`uv run python -m fracture_detection.weak.cli.evaluate` でOOF評価する。
VRAMはランダム入力・実寸で実測済み（1 step 18.42 GiB、region_branchの19.00 GiBより小さい。
`weak/README.md`の既知の制約を参照）。実データでのsmoke testは未実施。

### 1. Baseline 0 fine-tuning化（実装済み）

outer foldごとにfold-matched Baseline 0 checkpointからencoder / whole BiLSTM / whole headを
読み込み、FPN / region BiLSTM / region headsをランダム初期化する。全層を学習対象とし、
学習済み部分は`2.3e-5`、新規region部分は`2.3e-4`からcosine減衰する。
freeze/warmupは使わない。初期化監査情報は`initialization.json`へ保存する。

### 2. 実行順（すべてユーザーが手動起動）

校正（`cli/calibrate.py`）→学習（`cli/train.py`）→評価（`cli/evaluate.py`）→
比較（`cli/compare_arms.py`）。統合4領域modelを先に5 fold終えてから、単一領域4本の
要否・優先順位を判断する（計算量目安は`region_branch/README.md`参照）。

校正結果は各configの`calibration.version`で選択する。現在は`v1`で、保存先は
`region_branch/outputs/calibration/v1/outer{k}/`。学習条件を変更して再校正する場合は
全5 configのversionを同じ新番号へ更新する。
