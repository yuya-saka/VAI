# weak

`fracture_detection/` の4領域骨折検出モデル。設計の正本は
`fracture_detection/REGION_MIL_DESIGN.md`（2026-09-10 ユーザー承認）。
配置先は設計書では `region_mil/` と書かれているが、ユーザー指定によりこの
`weak/` を正式な配置先とする。

`region_branch/` が使っていた CAM 疑似ラベル・校正フェーズ・collapse 自動停止を
すべて廃し、GT なし陽性椎体を「4領域のうち少なくとも1つが骨折」という
1-bit の弱教師として使う。v2の既定集約は正規化logit-LSE（τ=0.5）。

## モデル

```text
15面 × (2.5D CT 5ch + 椎体mask 1ch)
    ↓ fine-tuneするEfficientNetV2-S trunk（Baseline 0からencoderのみ転送）
    ↓ stride-4 FPN (256ch)
    ↓ 4領域label mapでmask-normalized pooling（面ごと・領域ごと）
    ↓ 有効面だけを元の順序でpack_padded_sequenceへ
    ↓ 共有1層双方向BiLSTM（hidden 128、4領域で重み共有・入力は独立）
    ↓ 有効面でのBiLSTM特徴のmasked mean
    ↓ 共有Linear(256,1)を各領域へ個別適用
    ↓ z_1..z_4（4領域logit）
    ↓ sigmoid → q_1..q_4
    ↓ 固定の正規化logit-LSE（学習可能parameterなし）
    z_whole = τ [logsumexp(z_r / τ) - log(4)]
    ↓ sigmoid
    p_whole
```

whole 専用の BiLSTM・head・global-feature 迂回路は存在しない。推論に GT は不要で、
1 checkpoint が `q_1..q_4` と `p_whole` の両方を同じ parameter から出す。

**転送されるのは `encoder.*` だけ。** Baseline 0 の `lstm.*`（whole BiLSTM）・
`head.*`（whole head）は接続先がなく捨てる。FPN・region BiLSTM・region head は
ランダム初期化。BatchNorm の running mean/variance は転送後も凍結し、conv 重みと
BN affine parameter だけを学習する（`WeakRegionMilModel.train()` が
`freeze_encoder_bn_stats=True` のとき BN だけ eval に戻す。`requires_grad` は
一切変更しないので backbone 凍結とは無関係）。

`forward()` は `encoder.forward_intermediates(..., indices=(1,2,3,4),
intermediates_only=True)` しか呼ばない。stem と全 block stage は実行されるが、
`bn2`/`conv_head`（Baseline 0 の whole path 専用の global pooled feature を
作る層）には到達しない。これら2層は checkpoint 転送はされるが**このモデルでは
一切勾配を受けない**（`tests/test_model.py::test_bn_running_stats_unchanged_while_affine_grads_exist`
で確認済み）。バグではないが、checkpoint の「読み込んだ key 数」に含まれる点は
留意する。

## 損失

3群 mixed supervision（`fracture_detection/REGION_MIL_DESIGN.md` §4-6）。

| bag の状態 | 群 | 損失 |
|---|---|---|
| whole 陰性 | N | `Σ_r BCEWithLogits(z_r, 0)` |
| whole 陽性・region GT あり | A | `Σ_r BCEWithLogits(z_r, t_r)`（4セル全部、追加OR項なし） |
| whole 陽性・region GT なし | U | `softplus(-z_whole)`（beta 倍） |

`L = (Σ_N + Σ_A + beta·Σ_U) / b`。`b` は loss 対象になった実 bag 数。
N の損失は「4領域陰性BCEの和」。LSE whole BCEとは同値ではないが、既知の4個の
陰性教師をすべて使うため維持し、追加のwhole陰性BCEは重ねない
（`tests/test_losses.py::test_negative_loss_matches_four_region_bce_value_and_gradient`
が値・勾配の両方の一致を検証）。

LSEはlogitに適用し、`loss.whole_aggregation: normalized_logsumexp` と
`loss.lse_temperature` で設定する。`-log(4)` によって、4領域logitがすべて同じなら
whole logitも同じ値になる。計算はfloat32で行う。`whole_aggregation: noisy_or` と
`lse_temperature: null` を指定すればv2でも旧集約を比較でき、旧v1 configは
noisy-ORとして読み込む。

`beta` は既定 1.0。`beta=0` を指定すれば U bag は forward されたまま弱教師項だけ
0 になるので、Arm A（弱教師なし対照）が必要になった場合もコード追加なしで
とれる。**今回の実装は Arm B（提案モデル）のみで、Arm 切替の config スイッチは
作っていない。**

## サンプリング: GT-pass

batch 16 = 陰性(N) 8 + GTあり陽性(A) 4 + GTなし陽性(U) 4
（`sampling.negative_bags_per_batch` などで変更可能）。

- **1 GT-pass** = A 群を1周。`baseline0.data.sampling.EpochShuffleSampler` と
  同じ「seed+pass_indexで完全再現できるshuffle」をA群だけに適用し、重複なく
  1回ずつ提示する。
- N・U 群は `baseline0.data.sampling.AnnotatedCycleSampler` をそのまま流用した
  無限ストリーム（プールを1周したときだけ再shuffle）で、pass をまたいで
  カーソルを引き継ぐ。
- outer 0 train は A=159, U=643, N=7,272 → 39 step が 8/4/4、最後の1 stepが
  8/3/4（batch 15）になる。短い最終batchは実bag数で平均し、重複やimportance
  補正はしない。
- サンプラの状態は `(seed, pass_index)` だけで完全に決まるため、resume は
  `pass_index` を1つ復元するだけでよい
  （`tests/test_sampling.py::test_sampler_state_is_reproducible_from_pass_index`）。

## augmentation

**baseline0 の `augmentation:` 設定をそのまま使う**
（flip/transpose + affine + brightness + blur/noise + distortion + Cutout）。
CT・椎体mask・4領域label map は `data_pipeline/augmentation.py` の
`apply_bag_transform_with_regions` が1回の `ReplayCompose` 呼び出しで同期変換する。
mask 補間は最近傍（`cv2.INTER_NEAREST`）で、変換後の label map は
`clip(round(x), 0, 4)` に再量子化するため、領域 label が混ざることはない。

**MixUp は構造的に使えないため常に無効。** `training.mixup_probability` は
`FORBIDDEN_CONFIG_KEYS` にあり設定できない。region_branch は mixup step で
region loss を丸ごと skip して回避していたが、`weak/` には region 経路以外の
loss が存在しないため、skip すると損失が 0 になってしまう。加えて2 bag を
混ぜると (1) 4領域GTの意味が失われ、(2) U群の「少なくとも1領域」という
OR制約も定義できず、(3) `region_4class.npy` の label map 自体が混合不能になる。

**Cutout はbaseline0の設定 (`cutout_probability=0.05`) のまま残している**が、
`albumentations.CoarseDropout` は既定でmaskを書き換えないため、U bag唯一の
骨折所見をCTから消してもOR教師は「どこかに骨折がある」と主張し続ける
偽教師要因になり得る。5%の発生率なので初期実験を止める理由にはしないが、
疑わしい場合は `cutout_probability: 0.0` の1値変更で切れる。

## 初期化・最適化

- CNN trunk: Baseline 0 の fold 対応 `best_model.pt` から `encoder.*` のみ転送。
  `lstm.*`/`head.*` は捨てる（`modeling/initialization.py`）。
- 2 param group: `transferred`（encoder, LR 2.3e-5→2.3e-6）、
  `new`（FPN/region BiLSTM/region head, LR 2.3e-4→2.3e-5）。
  `baseline0.training.optimization.create_cosine_scheduler` は2groupで
  min LRを同一に強制するため使えず、`training/optimization.py` に
  `region_branch` と同じ per-group `LambdaLR` cosine を実装している。
- AdamW, weight decay 1e-4, bf16 autocast（CUDAのみ）, GradScalerなし。
- 60 GT-pass上限、最短10。inner region macro APのpatienceはconfigで指定し、
  現行v2実験では15。
- `torch.compile` は初期実装では使わない
  （bag×領域ごとに有効面長が変わり再compileが多発するため）。

## 検証曲線（GT-passごと）

毎GT-passの後に inner fold を augmentation なし・全件（自然分布）で1回推論し、
`history.csv` に `train_*`/`val_*` として記録する（W&Bにも同じ値を送る）。

| 列 | 母集団 | 内容 |
|---|---|---|
| `val_region_macro_ap`, `val_region_ap_r1..r4` | inner の GTあり陽性（A） | 領域AP。macro が checkpoint 選択と早期停止の指標 |
| `val_region_bce_annotated`（`_r1..r4`）, `val_region_brier_annotated`, `val_region_ece_annotated` | 同上の4セル（GTのみ） | 局在確率の過信の診断 |
| `val_whole_average_precision`, `val_whole_auroc`, `val_whole_bce`, `val_whole_brier`, `val_whole_ece` | inner の全bag | `p_whole` の判定性能と確率の質 |
| `train_/val_negative_loss`, `train_/val_annotated_loss`, `train_/val_weak_loss` | 各群 | 群ごとの1 bagあたり損失（βを掛ける前）。weak は GTなし陽性の `-log(p_whole)` |
| `train_objective`, `val_objective` | 3群 | 群平均を学習と同じ 8/4/4 と β で合成した値 |

`*_objective` は、8/4/4 の batch に対する `compute_weak_losses().total` と一致するように
定義しており、train と val を同じ尺度で並べるための列。inner を自然分布のまま平均すると
約9割が陰性 bag になり、GTなし陽性の項が埋もれるため、この重み付けにしている。
train 側は augmentation ありで、pass の途中でもパラメータが動くので、train/val の差には
過学習以外の要因も含まれる。train 側にはほかに `train_dropped_bags`、`train_grad_norm`、
`train_cnn_grad_norm`、val 側には `val_n_dropped_unobserved` がある。

同じ情報を別の形で持つ列がある。`val_annotated_loss` は `val_region_bce_annotated` の4倍
（4セルの和と平均の違いで、丸め差を除く）。v2のLSEでは`val_negative_loss`は
4領域陰性BCEの和であり、集約後のwhole BCEとは一致しない。

inner の A は53 bag（outer4 のみ56）で、領域別の陽性は R1 約15、R2 約12、R3 約14、
R4 約32。選択指標は少数の陽性で決まるため、曲線の細かい上下は雑音として読む。

`diagnostics.csv` には同じ inner 予測から、領域ごとの q の平均と標準偏差、`sum(q)`、
全領域 q≥0.8 の割合、領域間 Spearman、標準化4-logitの第1主成分寄与率、陽性bag内の
argmax 領域分布、陰性bagの領域別偽陽性率（q≥0.5）を記録する。自動停止には使わない。
予測CSV（`outer_predictions.csv`）には q と併せて生の logit（`z_1..z_4`）も保存する。

## 評価

`fracture_detection/REGION_MIL_DESIGN.md` §8の4母集団を分離して報告する
（母集団を混同すると過去に誤った比較をした教訓から。
`feedback_verify_metric_population_before_comparing`）。

| 評価 | 母集団 | 指標 |
|---|---|---|
| 条件付き局在 | region GTあり陽性(A) のみ | 領域別AP/AUROC + macro AP |
| 全椎体の骨折判定 | 全bag（自然分布） | `p_whole` のAP/AUROC |
| 領域検出 | A ∪ 陰性(N) | 領域別AP/AUROC（GTなし陽性(U)は陰性扱いしない） |
| 確率の質 | 上記それぞれ | BCE/Brier/ECE |

いずれの指標辞書にも `n`/`positives`/`prevalence` を必ず含める。
8/4/4 で訓練分布を意図的に変えているため、`q`/`p_whole` を較正済み確率とは
呼ばない。閾値付き指標を報告する場合の閾値は inner の自然分布で決め、outer では
変えない。

4領域のうち1つでも「全15面で観測不能」な bag は `region_observed_all=False`
として region 系の評価（条件付き局在・領域検出）から除外し、件数を
`n_dropped_unobserved` として報告する（q=0で黙って埋めない）。
whole 検出は `p_whole` が well-defined なので全 bag を使う。

正規化LSEはwhole検出用のsurrogateで、4領域の厳密な和事象確率ではない。
温度が小さいほどmaxに近づき、大きいほど平均logitに近づく。初期評価は連続スコアの
AP/AUROCで行う。

## 実行

すべて手動起動。段階を自動連鎖させるスクリプトは作っていない。

```bash
# 学習前1回: N/A/U件数と4領域maskの全数被覆スキャン
uv run python -m fracture_detection.weak.cli.inventory

# 学習（parallel.mode: fold で GPU 0/1 へ最大2 fold自動配分）
uv run python -m fracture_detection.weak.cli.train \
  --config fracture_detection/weak/config/weak.yaml

# 1 foldだけ直接起動する場合
uv run python -m fracture_detection.weak.cli.train \
  --config fracture_detection/weak/config/weak.yaml --outer-fold 0 --gpu-id 0

# 5 fold完走後のOOF評価
uv run python -m fracture_detection.weak.cli.evaluate \
  --config fracture_detection/weak/config/weak.yaml
```

## 検証

```bash
uv run pytest fracture_detection/weak/tests -q
uv run ruff check fracture_detection/weak
uv run ruff format --check fracture_detection/weak
uv run mypy fracture_detection/weak --exclude tests
```

## baseline0 からの再利用（import。コピーしない）

- fold分割: `baseline0.data.splits.{resolve_nested_folds, split_nested_manifest}`
- manifest読込・augmentation構築: `baseline0.data.dataset.{load_manifest,
  default_augmentation, augment_from_config, build_train_transform}`
- sampler: `baseline0.data.sampling.{EpochShuffleSampler, AnnotatedCycleSampler}`
- /dev/shm ステージング: `baseline0.data.staging.{stage_dataset, manifest_sha256}`
- DataLoaderの決定性: `baseline0.training.trainer.{seed_worker, set_seed,
  create_data_loader}`
- GPU並列起動: `baseline0.training.parallel.launch_fold_processes`
  （`module_name` 引数を `fracture_detection.weak.cli.train` にするだけで動く）
- 指標: `baseline0.evaluation.metrics.{safe_auroc, safe_average_precision,
  binary_metrics, cluster_bootstrap_interval, region_metrics,
  evaluate_vertebra_prediction_frame}`

`baseline0/modeling/`・`baseline0/data/region_validity.py` は使わない
（後者は本モデルが明示的に破棄した旧 validity 意味論）。

`region_branch/` への依存は作っていない。`RegionFpn`/`mask_normalized_pool`
（自己完結・依存なしのため）と 2-group optimizer/scheduler のパターンは
`region_branch` から着想を得て `weak/` 内に自前実装した。これにより実験が
失敗した場合のロールバック単位を `weak/` だけに閉じられる。

## 既知の制約とリスク

- **VRAM（実測済み・対策済み）**: 16 bag×15面、224×224、bf16、forward+backward
  1 step のピーク（RTX A6000、ランダム入力、eager、optimizer状態は含まない）。

  | 構成 | ピークVRAM | 1 step |
  |---|---:|---:|
  | baseline0 | 17.90 GiB | 0.40s |
  | region_branch（陽性2 bag） | 19.00 GiB | 0.45s |
  | region_branch（陽性4 bag） | 20.14 GiB | 0.47s |
  | weak（FPN一括・対策前） | 26.73 GiB | 0.55s |
  | weak（FPNを1 bagずつcheckpoint・現行） | 18.42 GiB | 0.67s |

  FPNのstride-4特徴は backward 用に1 bagあたり約0.57 GiBを占める。region_branch は
  これを whole 陽性 bag だけに流すが、weak は全 bag に流す必要がある。そこで
  FPN+mask pooling を1 bag（15面）ずつ activation checkpointing し、pooling 後の特徴
  だけを残して backward 時に FPN だけを再計算する（encoder は再計算しない）。
  計算結果は分割しない場合と一致する
  （`tests/test_model.py::test_chunked_fpn_matches_unchunked_logits_and_gradients`）。
  代償は1 stepあたり約20%の時間増。残りの大半は encoder 本体（baseline0 と同じ
  17.9 GiB）なので、これ以上減らす場合は encoder 側の対策になる。
- **MixUp不可・Cutout残存**は上記「augmentation」節を参照。
- **`encoder.bn2`/`conv_head` は転送されるが未使用**（上記「モデル」節参照）。
- **LSE whole scoreは較正済みの和事象確率ではない**。温度0.5は初期実験値であり、
  椎体・局在性能をinnerで検証する必要がある。
- Baseline 0・region_branch・既存 checkpoint・manifest は一切変更していない。
