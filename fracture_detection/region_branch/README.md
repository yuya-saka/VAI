# region_branch

`fracture_detection/` の4領域（R1 椎体・R2 右横突孔・R3 左横突孔・R4 後方要素）骨折検出モデル。
fold-matched Baseline 0 checkpointからwhole pathを初期化し、FPNで作ったregion pathを
shared CNN trunkへ接続して全層を微調整する。疑似ラベル
（`baseline0/pseudo_labeling/`生成物）はregion headの学習に使う。

設計の詳細と根拠は以下を参照する。

- `.claude/docs/work-logs/2026-08/2026-08-25-region-branch-design.md`
- `.claude/docs/work-logs/2026-08/2026-08-25-region-loss-final-decisions.md`
- `.claude/docs/research/20260825-region-loss-balancing.md`

## モデル

```text
Shared EfficientNetV2-S trunk
    ├─ whole path（Baseline 0とbit-exact。encoder(x) -> whole BiLSTM -> whole head）
    └─ region path（FPNでstage1-4をstride 4 (56x56) 256chへ統合
                     -> mask-normalized pooling -> shared region BiLSTM
                     -> 領域別head ×|active_regions|）
```

共有されるのはCNN trunkのみ。`conv_head`/`bn2`/whole BiLSTM/whole headはwhole lossから、
FPN/region BiLSTM/region head群はregion lossからのみ勾配を受ける。
`active_regions`一つで統合4領域モデル（`region_branch_all.yaml`）と単一領域モデル
（`region_branch_r1〜r4.yaml`）を切り替える。単一領域モデルは統合modelと同じ式・
同じcalibration係数を使うが、パラメータは独立に持つ（4本共有しない）。

## 初期化とfine-tuning

student outer fold `k`は
`baseline0/outputs/08_19/baseline0_shared_core/outer{k}/best_model.pt`から初期化する。
checkpointのnested fold設定と`checkpoint_role=best_val_auroc`を読込時に検証する。

- `encoder` → `encoder`
- `lstm` → `whole_lstm`
- `head` → `whole_head`
- `fpn` / `region_lstm` / `region_heads`はseed固定でランダム初期化

全parameterを学習対象とし、freeze/warmupは使わない。Baseline 0由来部分の初期LRは
`2.3e-5`、新規region pathは`2.3e-4`とし、各々をcosineで10分の1まで減衰する。
checkpoint hash、読込key数、ランダム初期化key数はfoldごとの`initialization.json`へ保存する。
`alpha_k`・`lambda_k`校正もこの初期状態からやり直す。

## 損失

```text
L = L_whole + lambda_k * (L_exact + alpha_k * L_rank)

L_whole = broadcast_bce_loss(whole_plane_logits, vertebra_target, pos_weight=2.0)  # natural streamのみ
L_exact = 0.5 * L_H + 0.5 * L_N   # human-annotated / whole-negative、source-balanced
L_rank  = region_balanced_pairwise_ranking_loss(...)                              # pseudo-positiveのみ
```

補助region batch（batch size 16固定）は human 4 / whole-negative 4 / pseudo-positive 8を
persistent queue（`AnnotatedCycleSampler`の再利用）で循環させる。natural stream（whole学習、
mixupあり）は16 bag。1 stepは32 bagを2回forwardし、wholeとregionを順にbackwardして
勾配を蓄積した後、1回だけoptimizerを更新する。これにより目的関数を変えず、2本の
EfficientNet計算グラフを同時にVRAMへ保持しない。

CUDA学習では`torch.compile(mode="default", dynamic=False)`をmodelへ適用する。RTX A6000の
同一step実測でeager比1.70倍、peak allocated VRAM 27.43 GiB→18.26 GiBだった。
初回compileは数分かかるが約1.3 epochで償却する。compile cacheはNFSを避け、CLIが
`/tmp/vai-region-branch-$UID/torchinductor-cache`へ配置する。校正とCPU実行はcompileしない。

`alpha_k`・`lambda_k`はouter foldごとに、学習前の決定的な64 batchで一度だけ勾配ノルムを
測って決める（`training/calibration.py`）。統合4領域modelで測った値を、同じouter foldの
単一領域4modelがそのまま読む。性能を見た再調整・model別/region別の再校正は行わない。

校正結果はconfigの`calibration.version`で指定した
`outputs/calibration/<version>/outer{k}/calibration.json`へ置く。
`experiment.phase`/`name`には依存しないため、出力先だけを変えた実行では同じ校正を参照する。
統合1本+単一4本の計5 configには同じversionを指定する。

```yaml
calibration:
  version: v1
```

model、loss、sampling、augmentationなどの学習条件を変更して再校正するときは、全configの
versionを`v2`のように更新する。artifactには校正関連configのSHA-256 fingerprintを保存し、
同じversionのまま条件が変わった場合や、別versionを誤って参照した場合は学習開始前に拒否する。
既存のcompatibleなfoldは校正CLIがskipするため、途中から安全に再開できる。

## 実行

**1. 統合4領域modelの校正**（`active_regions`が4領域全てのconfigでのみ実行できる。
`region_branch_r1〜r4.yaml`はこの結果を読むだけで、自分では校正しない）

```bash
uv run python -m fracture_detection.region_branch.cli.calibrate \
  --config fracture_detection/region_branch/config/region_branch_all.yaml \
  --outer-fold 0
```

`--outer-fold`を省略すると`config.data.start_outer_fold`〜`end_outer_fold`を順に校正する。
校正中はfoldごとに`alpha校正`と`lambda校正`の進捗を表示する。CLI起動時に
multiprocessing一時領域をローカル`/tmp/vai-region-branch-$UID`へ切り替えるため、
中断時もNFS上の`.tmp/pymp-*` cleanup errorを発生させない。

**2. 学習**（統合1本 + 単一4本 × 5 outer fold = 計25 run。すべて手動で起動する）

```bash
uv run python -m fracture_detection.region_branch.cli.train \
  --config fracture_detection/region_branch/config/region_branch_all.yaml \
  --start-outer-fold 0 --end-outer-fold 4 --gpu-id 0
```

**3. OOF評価**

```bash
uv run python -m fracture_detection.region_branch.cli.evaluate \
  --config fracture_detection/region_branch/config/region_branch_all.yaml
```

**4. 統合modelと単一4modelの領域別比較**（5本すべてのOOF評価が揃ってから実行する）

```bash
uv run python -m fracture_detection.region_branch.cli.compare_arms
```

4本ensembleはprimary endpointにしない。統合modelの各領域scoreと、対応する単一領域model
のscoreを領域ごとに1:1で比較する。

## Collapse監視

固定diagnostic subset（train foldのpseudo pool、pseudo-positiveかつ非annotated、
`diagnostic_subset_size`件、seed固定）で毎epoch、領域間・領域-whole Spearman相関と
標準化4-logit行列の第1主成分説明率を記録する（`fold_dir/diagnostics.csv`）。
領域間中央値・領域-whole中央値がともに0.95以上の状態が3epoch連続したら、事前定義した
失敗として学習を停止する（`fold_dir/collapse_alarm.json`）。alarm後に係数を調整して
再実行しない。

## 評価の注意

region sigmoidはsource-balanced samplingとranking supervisionで学習されており、
population prevalenceを表す較正済み確率ではない。`region_r_target_valid`が真の
セルだけを使い、scoreとしてAP/AUROCで評価する（`evaluation/metrics.py`）。

## 検証

```bash
uv run pytest fracture_detection/region_branch/tests -q
uv run ruff check fracture_detection/region_branch
uv run ruff format --check fracture_detection/region_branch
```

## 既知の制約

- Baseline 0の`data/staging.py`（`/dev/shm`へのローカルステージング）は移植していない。
  NFS I/Oが律速する場合は`data.dataset_dir`をローカルコピー先へ上書きする
- 計算量の目安: 1 stepが32 bag（Baseline 0の2倍）。統合5 fold ≒ 40 GPU時間、
  単一4アーム×5 fold ≒ 160 GPU時間、合計 約200 GPU時間
