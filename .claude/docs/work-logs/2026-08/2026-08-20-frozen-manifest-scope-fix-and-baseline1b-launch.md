# frozen manifestの凍結範囲修正とBaseline 1–B起動 worklog

**日付**: 2026-08-20
**状態**: 実装・再凍結完了。Baseline 1–Bが2GPUで正式学習中（outer0/1完了、outer2進行中）。

## 発端

08-19夜の実装（core/統合・W&B対応）後、baseline0のouter3/4完走を待つ間に
`.tmp/run_formal_fracture_pipeline.sh`がbaseline1_b等を自動連鎖起動する設計だったため、
ユーザーから「自動で始まるのはだめ。実行をたたくのは私」と明示指摘。
→ パイプラインscript(PID)のみをkillし、走行中のbaseline0学習には影響を与えず停止
（学習プロセスはPPID=1で独立継続していたため安全）。[[feedback-manual-execution-trigger]]

## 実装1: 各projectから起動できるCLI再編

- `fracture_detection/cli/project_entry.py`を新設。`ProjectCli`（project名・arm→config表・
  校正kind）を渡すとsubcommand(`train`/`profile`/`calibrate`/`sync-wandb`)を組み立てて
  既存の共通CLI実装（`cli/train.py`の`run_cli`、`cli/calibrate.py`の`run_calibration`、
  `cli/resource_profile.py`の`run_resource_profile`）へ委譲する薄い層。
- `baseline0/cli/__main__.py`・`mtl/cli/__main__.py`・`proposed/cli/__main__.py`を追加。
  各20行弱、arm→config対応表の宣言のみ。
- `python -m fracture_detection.{project}.cli train --arm <arm>`で各project配下から起動可能に。
  旧`fracture_detection.cli.train --config <path>`形式も同じ実装へ到達するため後方互換。
- calibrateの参照arm不一致（λはbaseline1_b、βはproposed_b固定）を拒否するガードを追加
  （従来は他armのconfigで校正しても検出できなかった）。
- 旧baseline0専用パイプライン（`baseline0/training/`, `baseline0/config/schema.py`,
  `baseline0/cli/train.py`, `baseline0/cli/evaluate.py`）はユーザー判断で温存。
  現行の学習経路（`core/`統合実装）からは一切importされていない、実質デッドコード。
- テスト12件追加（`cli/tests/test_project_entry.py`）、README 3本更新。

## 実装2: frozen manifestの凍結範囲を「科学的に意味のある設定」だけに限定

**問題**: `verify_frozen_manifest`が実験名・GPU割当・fold範囲・W&B設定まで含めて
1つのtree hashで凍結していたため、運用上の値を変えるだけで再凍結（校正・profile再実行）
が必要になっていた。ユーザー指摘「実験名やfold回すかとかは凍結すべきでない」を受けて修正。

- `core/artifacts.py: normalized_config()` — `experiment`・`parallel`・`wandb`セクション全体、
  `data.start_outer_fold`/`end_outer_fold`、`training.gpu_id`を凍結比較から除外。
  モデル・損失・データ経路・乱数系列に影響する設定（`model`/`augmentation`/`data.random_seed`/
  学習ハイパラ）のみ`config_sha256`の対象に残す。
- `fold_to_gpu`を凍結対象から完全に削除（GPU割当は運用設定であり、統計的妥当性に影響しない
  ため）。旧実装は「6アーム全部が同一fold-to-GPU割当であること」を`create_frozen_manifest`で
  強制していたが、これはbaseline1_bを2GPU、他を3GPUのように**アームごとに違うGPU構成へ変更
  できない**という副作用があった。ユーザーから「2GPUで実行したい」の要望が出た際に発覚。
- `source_tree_sha256`の対象を`.py`のみに変更（`.yaml`除外）。armのconfig YAMLを編集すると
  `config_sha256`だけでなく`source_tree_sha256`まで変わってしまい、`config_sha256`側の
  正規化が意味を失っていたバグを修正。
- テスト追加: `test_operational_settings_are_not_frozen`（運用設定変更で拒否されないことを確認）、
  `test_scientific_settings_are_still_frozen`（seed/batch/lr/backbone/augmentation変更は
  引き続き拒否されることを確認）、`test_source_tree_sha256_ignores_arm_config_yaml_edits`。

## 再凍結の実行

08-19夜以降の複数回の実装変更（W&B対応・CLI再編・上記manifest scope修正）でsource_tree_sha256が
ドリフトしたため、λ/β校正・5構成resource profile・frozen manifestを再実行。
旧artifact（08-19 19:2x時点のもの）は`fracture_detection/experiments/archive/stale_20260819/`
へ退避（削除はしていない）。

- λ校正（baseline1_b, GPU0）・β校正（proposed_b, GPU1）並列実行、両方成功
- 校正結合 → `calibration/outputs/loss_weights.json`
- 5構成resource profile（GPU2）→ `profiling/outputs/08_20/`
- `freeze_experiment` → `experiments/frozen_experiment_manifest.json`（新規）
- 6アーム全部で`verify_frozen_manifest`が通ることを直接呼び出しで確認（学習は起動せず）

実測されたλ/β（outer別、baseline1_b/proposed_b基準）:

```
outer0: λ=0.476  β=0.438
outer1: λ=0.319  β=0.087
outer2: λ=0.336  β=0.306
outer3: λ=0.332  β=0.209
outer4: λ=0.427  β=0.894
```

## Baseline 0 正式学習: 完了

`fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/`で5 fold全て
`outer_inference_complete: true`。

| fold | best_epoch | best_prauc_epoch | stopped_epoch |
|---|---|---|---|
| outer0 | 44 | 47 | 50 |
| outer1 | 55 | 55 | 55 |
| outer2 | 48 | 43 | 48 |
| outer3 | 53 | 55 | 59 |
| outer4 | 40 | 44 | 46 |

## Baseline 1–B 正式学習: 進行中

`fracture_detection/mtl/outputs/08_19/baseline1_b/`。`parallel.gpu_ids: [0, 1]`,
`max_concurrent_folds: 2`で実行中（09:46開始）。

| fold | 状態 | best_epoch | stopped_epoch |
|---|---|---|---|
| outer0 | 完了 | 41 | 43 |
| outer1 | 完了 | 39 | 59 |
| outer2 | 進行中(epoch16〜) | — | — |
| outer3, 4 | 未着手 | — | — |

outer0のheld-out実測（`common/metrics.evaluate_prediction_frame`で直接計算、n=2671、陽性262）:

- AUROC-best checkpoint: AUROC=0.883 [0.857, 0.908]、PR-AUC=0.687
- PR-AUC-best checkpoint: AUROC=0.879 [0.851, 0.904]、PR-AUC=0.689
- region別AUROC/PR-AUC（annotated 56 bagのみ）: region_1 0.90/0.83、region_2 0.77/0.57、
  region_3 **0.60/0.34**（弱い）、region_4 0.81〜0.86/0.81〜0.85
- n=56/foldと小さいため、region_2/3の弱さ・非対称性は1fold単独では確定できない

## 分析: val_lossの振れ幅とregion精度の関係（未対応の懸念事項）

- baseline1_bのval_loss epoch間変化幅は、region無効のBaseline 0と比べ**3〜4.5倍**大きい
  （|Δval_loss|平均: Baseline0=0.012 vs Baseline1_b outer0=0.053 / outer1=0.036）
- 主因と推定: 各optimizer stepでnatural batch(16件)のwhole勾配とannotated 1 bagのregion勾配が
  同じbackboneへ合算される設計（`mtl/README.md`記載どおり）。batch=1の勾配は本質的に高分散
- 同じ現象がregion_2/3（全体でも59/72件と少数）の精度不足とも整合すると推定（未検証、仮説段階）
- λの`target_ratio=0.5`は2026-08-18確定のPI仕様で、**grid探索なし・結果を見て変更しない**と
  PROGRESS.mdに明記。今回「λ強すぎるのでは」という懸念が出たが、この確定ルールに従い
  **変更していない**。見直す場合は次の凍結サイクルで正式な意思決定プロセスに載せる想定

## 比較: train_models/stage3との設計差（別プロジェクト、参考）

`train_models/stage3`（RSNAコード、fracture_detectionとは別系統）はOOF AUROC=0.925と、
現状のbaseline1_b（0.85〜0.89帯）より高い。設計を確認したところ、region maskを
**pooling範囲を決める構造的prior**として使うのみで、region labelを教師信号に一切使っていない
（`stage3_loss`はvertebra_logitのみ教師）。fracture_detectionのbaseline1_bは268 bagの
region labelを直接教師にする設計のため、annotated batch=1由来のノイズが原理的に発生する。
この設計差は`memo/計画書/提案手法.md`のコア設計判断に関わるため、λ調整より大きい論点として
将来検討の余地がある（今回は変更していない）。

## 次の作業

1. baseline1_bのouter2〜4完走を待つ（max_concurrent_folds=2、GPU2は空き）
2. 5 fold揃ったらpooled OOF解析（`cli/analyze.py`）でH1判定に必要な数値を確定
3. Control–Bをまだ起動していない（H1の比較相手）。ユーザー指示があれば
   `uv run python -m fracture_detection.mtl.cli train --arm control_b`
4. proposed_b / proposed_max / proposed_max_beta0 も未着手
5. λの`target_ratio=0.5`見直しは今回結論を出していない。ユーザー判断待ち
6. `core/`・`cli/`・`config/`・`calibration/`・`profiling/`・`evaluation/`のREADMEは未作成のまま
