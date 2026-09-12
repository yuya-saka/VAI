# weak/ Arm B outer0 学習・分析

日付: 2026-09-11

状態: `fracture_detection/weak/` の実装（2026-09-10完了）に対し、outer0のみ学習を実行。
完走・分析済み。outer1〜4は未実行。

## 実行内容

```bash
uv run python -m fracture_detection.weak.cli.train \
  --config fracture_detection/weak/config/weak.yaml
```

config は `experiment.name: test_v1`、`data.end_outer_fold: 0`。GPU 0で実行、
`/dev/shm` へのステージング（66GB、13,432 bag）を含めて完走。

pass45で早期停止（`patience_gt_passes=10`）。best_gt_pass=35。クラッシュではなく
design通りの正常終了。結果は
`fracture_detection/weak/outputs/09_10/test_v1/outer0/fold_metrics.json` に保存済み。

## 実測結果

2026-09-11再監査で訂正: `fold_metrics.json.best_metrics` はinner指標であり、
outer指標ではなかった。以下は予測CSVから再計算したouter値。
詳細は `.claude/docs/experiments/2026-09-11-weak-test-v1/analysis.md`。

条件付き局在（outerのGTあり陽性56bag）: region_macro_ap **0.7766**。
innerのGTあり陽性53bagでは **0.7780**。

whole判定をbaseline0と同一母集団（outer0テスト、2,671bag、陽性262）で比較:

| | AUROC | AP |
|---|---:|---:|
| baseline0 | 0.9186 | 0.7726 |
| weak/ | 0.9084 | 0.7284 |

**AP低下(-0.044)の機序は未確定:** 陰性bagのp_whole平均が0.291（baseline0は0.068、4倍以上）。
p90でも0.703 vs 0.174。AUROCがほぼ変わらずAPだけ下がるのは「上位ランクへの陰性の
紛れ込み」パターンと整合する（precision@top262が0.657 vs 0.706、13件差）。
4/4/8サンプリングのbatch内陽性75%はスコア上昇と整合するが、単調な校正変化だけでは
順位に依存するAPは変わらない。新規region経路の陰性提示不足、固定noisy-ORの集約、
whole専用headの置き換え、局在によるcheckpoint選択も同時に変わっており、
サンプリングを単独の原因と断定した以前の解釈は撤回する。

N（陰性）とU（GTなし陽性）の分離: outer held-outでAUROC=0.9035, AP=0.6972。
U群のp_whole中央値0.975はA群(GT既知)の0.973とほぼ同じ。ただし**この分離が弱教師
(β×OR項)自体の効果か、A群だけの直接教師からのCNN特徴の一般化かは区別できていない**。
今回の実装はArm Bのみ（β=1固定）で、Arm A比較（β=0）を実施していないため。

augmentation由来のdropped_bags（Affine/Cutoutで小領域が15面全部から消える）を
再現実験で確認: 500試行中1.0%（R2/R3のみ発生、R1/R4は面積が大きく消えない）。
バグではなく許容範囲内と判断。

## 未決定: サンプリング比率の見直し

ユーザーから「陰性を増やすと局在が学習しなくなりそう」という指摘があり、
これは`compute_weak_losses`の平均演算（分母=batch内実bag数）の力学として正しい。
単純に陰性(N)を増やすとGTの相対比重が薄まり局在学習が鈍る。
GT比重を変えずに調整する選択肢は「弱陽性(U)を減らして陰性を増やす」
（batch総数は16のまま、A=4は固定、N/U比率だけ変える）。

**2026-09-11時点で結論は出していない。** ユーザーに3つの選択肢
（①今の結果をそのまま受け入れる、②N/U比率だけ変えた比較を追加実施、
③ここで一旦停止）を提示し、回答待ちの状態で本ログを保存した。

## 次回への申し送り

- outer1〜4の学習は未実行。5 fold OOF評価（`cli/evaluate.py`）は全fold揃うまで動かない。
- N/U比率変更の比較実験を行う場合は、config を複製し `experiment.name` を変え、
  `sampling.negative_bags_per_batch`/`weak_bags_per_batch` だけ変更する
  （`annotated_bags_per_batch`は触らない）。既存の4/4/8結果（outer0のみ）と比較する。
- Arm A（β=0）比較を行う場合は `loss.beta: 0` のconfigを複製するだけで既存コードのまま実行できる。
- 詳細な実測値・数式的な力学は
  `/mnt/nfs1/home/yamamoto-hiroto/.claude/projects/-mnt-nfs1-home-yamamoto-hiroto-research-VAI/memory/project_weak_region_mil_result.md`
  にも保存済み。
