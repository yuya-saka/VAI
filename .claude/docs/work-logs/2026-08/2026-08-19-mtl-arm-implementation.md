# MTL 6構成共有実装 worklog

**日付**: 2026-08-19  
**状態**: 実装・1-step smoke完了。正式preflightと学習は未開始。

## 実装

- `fracture_detection/core/`へarm非依存trainer、loss、optimizer、RNG、checkpoint、immutable artifact、
  fold-process並列launcherを集約し、Baseline 0を含む6経路を同じ契約へ統合した。
- canonical 10ch datasetと単一augmentation replayを導入し、Control/Baseline 0は6ch、Baseline 1以降は
  10chをadapterで選択する。horizontal flipはR2/R3のmask値とlabelを同時交換する。
- annotated streamはBN moduleだけeval、Dropoutはtrainのままとし、natural backward後に独立RNGで
  annotated forward/backwardを行ってoptimizer stepを1回だけ実行する。
- Proposed branchを`blocks[5] -> conv_head -> bn2 -> GAP -> BiLSTM`の独立4複製として実装し、
  `L_att`をβ=0でも計算する3構成を追加した。
- λ/β校正、loss-weight結合、resource profile、frozen manifest、正式run guard、pooled OOFと固定順序検定を
  CLIまで実装した。smokeはinner validationまでで停止し、outer予測を生成しない。

## 実機検証での修正

- 1-step smokeがouter推論まで実行していたため、smoke専用の`run_outer_inference=False`経路へ修正した。
- 2 GPU resumeで`torch.load(map_location=cuda)`が保存済みCPU RNG stateをCUDAへ移し、
  `torch.set_rng_state`が失敗する問題を再現した。RNG復元境界ですべてのtorch RNG stateを連続な
  CPU `uint8` tensorへ正規化し、CUDA checkpoint loadとの両立を保証した。

## 検証結果

- `uv run pytest fracture_detection -q`: **111 passed**。
- mypy: shared implementationの**31 source files**でerrorなし。
- Ruff: **90 files formatted**、check通過。
- 実データ1-step GPU smoke: Baseline 0、Control–B、Baseline 1–B、Proposed–B、
  Proposed–max β>0、Proposed–max β=0が完走。
- Control–Bの2 GPU × 2 fold smokeとresumeが完走。GPU割当は`outer0 -> GPU 0`、`outer1 -> GPU 1`、
  fold別成果物に衝突はなく、outer prediction fileは0件だった。

## 残る運用工程

1. outerごとのλ/β 64-batch正式校正を実行する。
2. 5構成の20-step resource profileと2 GPU × 2 fold 20-step preflightを実行する。
3. 全hashとfold-to-GPU計画をfrozen manifestへ固定する。
4. manifest照合下で正式full trainingとouter 1回推論を開始する。

これらは未実装ではなく、設計どおり実装完了後に実測artifactを作る運用工程である。

## 正式pipeline開始

- A6000 3枚を`[0, 1, 2]`として全6 configへ固定し、最大3 foldを並列実行する。
- `.tmp/run_formal_fracture_pipeline.sh`をdetach起動した。λ/β校正後に校正結合、5構成profile、
  2 GPU × 2 fold preflight、manifest凍結、5構成正式学習を順番に自動実行する。
- 初回校正でwhole-model eval下のcuDNN LSTM backward失敗とProposed–B FP32 OOMを検出したため、
  BN-only eval、BF16 autocast、CPU state snapshot、RNG復元へ修正して再開した。
- 監視rootは`fracture_detection/experiments/logs/formal_20260819/`。

## 2026-08-20 進捗スナップショット

### Baseline 0正式学習

- `fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/`で5 foldの学習成果物を確認した。
- outer 0は50 epoch、outer 1は55 epoch、outer 2は48 epochでearly stoppingまで完了し、
  AUROC-best / PR-AUC-bestのouter予測と`summary.json`を保存済み。
- outer 3は38 epoch、outer 4は35 epochまで`history.csv`とcheckpointを保存済み。
  この2 foldはouter予測が未生成であり、正式完了にはresumeとouter推論が必要。

### W&B追跡の修正とバックフィル

- 共有configのW&B既定値を`enabled: true`へ修正した。
- 共有trainerへfold単位のresumable W&B runを追加し、`history.csv`へ永続化する全epoch指標を
  同じstepで送信するようにした。run IDは各foldの`wandb_run_id.txt`へ保存する。
- 旧設定でW&B無効だった既存run向けに、再学習なしで`history.csv`を送る
  `fracture_detection.cli.sync_wandb`を追加した。
- 2026-08-20に既存5 foldをW&Bへバックフィルした。同期時点では合計225 epochで、projectは
  `https://wandb.ai/yuya00-university-of-hyogo/fracture-08_19-baseline0_shared_core`。
- W&B修正の対象テストは13件通過し、対象ファイルのRuff check / format checkも通過した。

### 次の作業

1. outer 3/4の正式resume前に、W&Bのみのsource/config差分を既存frozen manifestと安全に両立させる。
2. outer 3/4をearly stoppingまで継続し、各checkpointのouter推論を1回だけ実行する。
3. 5 foldのpooled OOF解析を実行し、Baseline 0の正式精度を確定する。
