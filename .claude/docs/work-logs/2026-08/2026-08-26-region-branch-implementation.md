# region_branch 実装セッション worklog

作成日: 2026-08-26

2026-08-25 に確定した設計（`2026-08-25-region-branch-design.md`、
`2026-08-25-region-loss-final-decisions.md`）を `fracture_detection/region_branch/` として
実装した。**コードは一通り完成し、実データ・実configでのGPU動作も確認済み。
学習・評価の本番runは未実行。**

> **重要（セッション末にユーザーが指摘）:** 本実装は timm の ImageNet 事前学習重みから
> スクラッチで学習を始める構成になっている。しかしこのMTLは **Baseline 0 の学習済み
> パラメータから開始するファインチューニングにすべき**、というのがユーザーの見立て。
> 次セッションでこれを検討する。詳細は §6。

---

## 1. 実装前に確定した3点

実装計画時にユーザーへ確認し、いずれも推奨案で確定した。

| 論点 | 決定 | 理由 |
|---|---|---|
| horizontal flip の R2/R3 | **mask追従・ラベル入れ替えなし** | maskは画像と一緒に反転するので mask値2は反転後も解剖学的な右横突孔を覆う。`head r = 解剖学的領域 r` が統合1本でも単一4本でも成立し、アーム間で拡張規約が揃う。`baseline0/data/dataset.py:48-51` の「ラベルとマスク値を同時入れ替え」コメントはこの規約と異なるため、region_branch側のdocstringで明示的に上書きした |
| 1 epoch の長さ | **natural stream基準**（train 3 folds 約8,059 bag ÷ 16 ≒ 505 batch/epoch） | Baseline 0 の whole 露出・LRスケジュール・early stopping cadence が完全一致する（決定文書§6「whole exposureをBaseline 0と揃える」に忠実）。補助queueは human 約12周 / pseudo 約6周 / negative 約0.3周 per epoch |
| checkpoint選択・early stopping | **val (L_whole + λ·L_exact) の最小化** | val の `L_rank` は構造上計算不能（教師kは自分の学習foldしか採点していない）。inner fold の人手bagは約54件しかないため、`L_whole` を含めた方が指標が安定する |

---

## 2. 実装した構成

```text
fracture_detection/region_branch/   # 34 python files, 実装3,657行 + test 1,138行
├── README.md
├── config/     schema.py + yaml 5本（統合1 + 単一4）
├── data_pipeline/ constants / dataset / pseudo_labels / sources / sampling / loaders
├── modeling/   pooling(FPN) / model / losses
├── training/   experiment / calibration / monitoring / trainer
├── evaluation/ metrics（validity mask付き領域別AP/AUROC）
├── cli/        calibrate / train / evaluate / compare_arms
└── tests/      35 test
```

### Architecture

```text
Shared EfficientNetV2-S trunk
    ├─ whole path（Baseline 0 と bit-exact）
    │    encoder.forward_head(final) → whole BiLSTM(1280→256×2) → whole head
    └─ region path
         stage1-4 を 1×1 conv で 256ch へ → stride 4 (56×56) へ bilinear 統合
         → concat → 1×1 conv → GroupNorm → SiLU
         → mask-normalized pooling → shared region BiLSTM → 領域別head ×|R|
         → 有効面のみで logit(mean sigmoid) → bag logit
```

**whole path の bit-exact 性は実測で確認済み。**
`encoder.forward_intermediates(indices=(1,2,3,4), intermediates_only=False)` の第1返り値へ
`forward_head` を適用した結果が `encoder(x)` と maxdiff 0.0 で一致する（timm 1.0.22）。
これにより「Baseline 0 と同一の whole path」を保ったまま中間特徴を取り出せる。
`test_model.py::test_whole_path_matches_baseline0_bit_exact` で回帰検証している。

### 流用した既存資産（新規実装していない）

- pairwise ranking 損失・ペア構築: `baseline0/pseudo_labeling/scoring.py`
- whole 損失・bag確率: `baseline0/modeling/losses.py`
- fold分割: `baseline0/data/splits.py`
- persistent cycle sampler: `baseline0/data/sampling.py:AnnotatedCycleSampler`
- optimizer / LR制御: `baseline0/training/optimization.py`
- 指標・patient cluster bootstrap: `baseline0/evaluation/metrics.py`

---

## 3. セッション中に変更した設計

実装計画時にはなかったが、実際に動かす過程でユーザー指摘により変更した2点。

### 3.1 校正結果の置き場所を experiment 出力先から独立させた

**変更前:** `region.calibration_path` を config に持ち、統合modelは `self`（自分の
`outputs/<phase>/<name>/`）へ書き、単一4modelはその path を文字列で書き写して読む。

**問題:** `experiment.phase`/`name` は試行のたびに変えるので、そのたびに単一4本の
yaml のpathを書き換える必要があり運用に耐えない。

**変更後:** `outputs/calibration/outer{k}/calibration.json` という固定位置にした。
`region.calibration_path` 設定項目自体を削除（5つのyamlからも除去）。
`resolve_calibration_path(outer_fold)` は config を受け取らなくなった。

同じouter foldでは統合1本+単一4本の計5 configが必ず同じ校正結果を使う、という
制約（`.claude/docs/research/20260825-region-loss-balancing.md`）を、path解決の
レベルで保証する形にした。

### 3.2 `calibrate.py --outer-fold` を任意引数にした

`train.py` は `config.data.start_outer_fold`〜`end_outer_fold` をループするのに、
`calibrate.py` だけ `--outer-fold` 必須で config を無視していた非対称を解消。
省略時は config の range を順に校正する。単一fold指定も引き続き可能。

---

## 4. 実データGPU実行で発見・修正したバグ

**CPUのunit testでは全て通っていたが、GPU実機で初めて落ちた。**

### 4.1 `model.eval()` で cuDNN LSTM の backward ができない

```
RuntimeError: cudnn RNN backward can only be called in training mode
```

校正コードは勾配ノルムを再現可能にするため `model.eval()` にしていたが、
GPU上の cuDNN LSTM 実装は `eval()` 状態での backward をサポートしない。
CPU実装にはこの制約がなく、testがCPUのみだったため見逃していた。

**修正:** `model.train()` に変更。Dropoutの確率性は `_measure_alpha_norms` /
`_measure_lambda_norms` の直前で `torch.manual_seed(seed)` を張って再現性を維持。
実際の学習も `train()` で行うので、勾配の実測としてもこちらが忠実。

### 4.2 GPUメモリ枯渇（47.5GB枠に対しforward 1回で42GB）

4.1の修正後に顕在化。原因は2つ。

1. **校正コードが bf16 autocast を使っていなかった。** `trainer.py` の学習ループは
   元々 `torch.autocast(bfloat16)` を使っていたので同じ問題が出ていなかった。
   校正だけ fp32 のままで、forward 1回の活性化が実測 42GB になっていた
   （autocast適用後は 22GB）
2. **各batchの中間結果が次batchのforward開始まで解放されていなかった。**
   ループ本体のローカル変数（`output` / `bag_logits` / `l_exact` 等）が
   次のiterationの `model(...)` 実行時点でもまだ生きており、ピークがほぼ2倍になっていた

**修正:**
- 3箇所の forward を `torch.autocast(device_type, bfloat16, enabled=cuda)` で包んだ
- 1 batch分の測定を関数（`_alpha_norms_for_batch` / `_whole_norm_for_batch` /
  `_region_norm_for_batch`）へ切り出し、scope終了で活性化が解放されるようにした
- `_grad_norm` に `retain_graph` 引数を追加。同じforward graphを他のlossと共有しない
  最後の呼び出しでは `retain_graph=False` を渡してpeakを下げる

**修正後の実測（本番設定 `n_batches=64`、outer fold 0、GPU実機）:**

```
CALIBRATE OK, time= 81.08 s
alpha 0.4369045072575934  lambda 0.5398769018476762
peak mem allocated 28.82 GB / reserved 33.10 GB
```

47.5GB枠に対しピーク33GBで収まった。**なおこの alpha/lambda は動作確認のための
実行値であり、正式な校正成果物としては保存していない**（`outputs/` 未作成）。

### 4.3 `GroupNorm(32, channels)` が小さいchannel数で落ちる

testで `fpn_channels=8` を使ったところ `num_channels (8) must be divisible by
num_groups (32)` で失敗。本番configは `fpn_channels=256` なので実害はないが、
小さいbackboneでのtestができない。`_group_norm_groups()` で channels を割り切る
最大のgroup数を選ぶようにした（256なら常に32を返す）。

---

## 5. 検証状況

- unit test 35件（region_branch）+ 101件（baseline0、回帰なし）= **136件 all pass**
- `ruff check` / `ruff format --check`: clean
- `mypy`: region_branch固有の新規エラーは解消済み。残る22件は `pandas-stubs` 等の
  stub未インストールと `Sampler[int]` 型注釈の食い違いで、いずれも baseline0 側にも
  存在する既存の技術的負債
- 実データGPU smoke: loader構築 / 1 step学習 / evaluate / diagnostics収集 /
  calibrate(n_batches=64) / `train_fold` の2 epoch run

**注意: `poe typecheck` は `mypy` であり `ty` ではない。**
`.claude/rules/dev-environment.md` は `uv run ty check` と書いているが `ty` は
このリポジトリに入っておらず、`pyproject.toml` の実体は `mypy . --exclude tests`。
ルール側の記述が実態と乖離している。

---

## 6. 次セッションの主題: Baseline 0 からのファインチューニング化

### 現状の実装

`RegionBranchModel` は timm の `pretrained=True`（ImageNet事前学習重み）から
初期化しており、**Baseline 0 の学習済みパラメータを読む経路が存在しない。**
つまり whole path も region path も、骨折検出タスクとしてはスクラッチ学習になる。

### ユーザーの指摘（2026-08-26）

参考論文
`memo/research_paper/眼底画像の解釈可能な品質評価のための半教師ありマルチタスク学習.pdf`
にあるとおり、**単一タスク学習モデル（本プロジェクトでは Baseline 0）から得られた
パラメータで開始し、微調整していく学習にすべき**。

### 検討時に使える事実（本セッションで確認済み）

Baseline 0 checkpoint の構造:

```
fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/outer{k}/best_model.pt
  model の top-level module prefix: ['encoder', 'head', 'lstm']   （tensor 805個）
  checkpoint_role: best_val_auroc
```

`RegionBranchModel` との module 名の対応:

| Baseline 0 | RegionBranchModel | 備考 |
|---|---|---|
| `encoder` | `encoder` | shared CNN trunk。形状完全一致 |
| `lstm` | `whole_lstm` | 名前だけ違う。hidden/layers は同一設定 |
| `head` | `whole_head` | 名前だけ違う。構造同一 |
| — | `fpn` | 対応なし。新規初期化が必要 |
| — | `region_lstm` | 対応なし。新規初期化が必要 |
| — | `region_heads` | 対応なし。新規初期化が必要 |

whole path 側は 1:1 で載せ替え可能（`test_model.py` で bit-exact 性を検証済みなので
重みを移せば挙動も一致する）。region path 側は対応する重みが存在しないため、
必然的に「一部が学習済み・一部がランダム初期化」の混在状態から始まることになる。

### 次セッションで詰めるべき論点（Claudeの提案であり未決定）

1. **fold対応をどう取るか。** student outer fold k は Teacher_k（= `outer{k}/best_model.pt`）
   と学習foldが完全一致する。疑似ラベルと同じ fold-matched 対応にするのが自然に見えるが、
   「同じ重みを初期値にし、かつその重み由来のCAMを教師にする」ことの妥当性は要検討
2. **LR設計。** 学習済みのwhole path と ランダム初期化のregion path で適切なLRが
   違う可能性がある。現在は `backbone_learning_rate` / `head_learning_rate` の2群で、
   region path は head 群に入っている（`head_parameters()` が FPN・region_lstm・
   region_heads を含む）。3群に分けるか、region path の warmup を入れるか
3. **λ/α 校正への影響。** 校正は「学習前の初期状態での勾配ノルム比」を測る手続きなので、
   初期状態が変われば校正値も変わる。ファインチューニング構成に変えたら
   **校正はやり直しになる**（本セッションで出た alpha 0.437 / lambda 0.540 は無効）
4. **比較実験の公平性。** 統合4領域modelと単一領域4modelの両方を同じ初期化規約に
   揃える必要がある（これは既存の「同じλ/αを使う」制約と同じ趣旨）
5. **参考論文の該当箇所の確認。** 上記PDFで、事前学習モデルのどの部分をどう
   引き継いでいるか（全層か、encoderのみか、LRをどう設定しているか）を読む

### 実装として必要になりそうなもの（未着手・未承認）

- `config.model` に「Baseline 0 checkpoint から初期化する」設定項目
- checkpoint の `lstm`→`whole_lstm`、`head`→`whole_head` へのキー変換ローダ
- 読み込んだキーと読み込まなかったキーを両方 log / artifact に残す仕組み
  （どこがランダム初期化のままかを後から監査できるようにする）

---

## 7. 現在の状態

- **本番runは一切実行していない。** 校正・学習・評価すべて未実施
- `fracture_detection/region_branch/outputs/` は存在しない（成果物ゼロ）
- 作業ツリー上、region_branch は未 commit（`git status` で untracked）
- `fracture_detection/PROGRESS.md` を更新済み
- memory `project_region_branch_implementation.md` を新規作成、
  `project_pseudo_label_mtl_direction.md` に「実装状態は更新済み」の追記を入れた

### `.gitignore` について

新設の5 yaml は `*.yaml` 除外により未追跡のまま。baseline0 と同様に
`!fracture_detection/region_branch/config/*.yaml` を足せば追跡できるが、
`feedback_no_config_tests_tracking`（tests/・*.yaml の除外は解除しない）に従い
**変更していない**。追跡が必要ならユーザー判断で。

---

## 8. 次にやること

1. fine-tuning初期状態でλ/α校正を全foldやり直す
2. 統合4領域model の 5 fold 学習 → OOF評価
3. 結果を見てから単一領域4本の要否・優先順位を判断する
   （計算量目安: 統合5 fold ≒ 40 GPU時間、単一4アーム×5 fold ≒ 160 GPU時間）

## 9. Fine-tuning方針の確定と実装（2026-08-26追記）

§6の未決事項は次で確定した。

- student outer fold `k`はfold-matched Baseline 0 `outer{k}/best_model.pt`から初期化する
- `encoder`、`lstm`、`head`をregion modelのwhole pathへ移し、新規region pathのみrandom初期化する
- 全parameterを学習し、freeze/warmupは使わない
- 学習済み部分の初期LRを`2.3e-5`、新規region pathを`2.3e-4`とする
- checkpoint role・nested fold・state keyを厳密検証し、hashと初期化内訳をartifactへ保存する
- `alpha_k`・`lambda_k`はfine-tuning初期状態から全fold再校正する

## 10. Calibration中断時のNFS multiprocessing error修正

校正は1 foldあたりalpha 64 batchとlambda 64 batchの勾配測定を行うが、進捗表示がなく、
停止して見えたため手動中断された。表示された大量の`OSError: [Errno 16]`は計算失敗ではなく、
32 DataLoader workersがNFS上の`.tmp/pymp-*`を同時削除した際の終了処理errorだった。

- regionのcalibrate/train CLIは起動直後に`TMPDIR`/`TEMP`/`TMP`をローカル`/tmp`へ上書きする
- alpha/lambda校正ループへfold別progress barを追加する
- 本体tracebackの`KeyboardInterrupt`とworker cleanup noiseを分離して診断できるようにする

## 11. Calibration artifactのversion管理

- configへ`calibration.version: v1`を追加した
- 保存先を`outputs/calibration/<version>/outer{k}/calibration.json`へ変更した
- artifactへversionと校正関連configのSHA-256 fingerprintを保存する
- output phase/name、GPU、統合/単一active regionの差は同じ校正を共有できる
- その他の学習条件が変わった場合はcompatibility errorにして、新versionでの再校正を要求する
- 完了済みouter0は内容を再計算せず`calibration/v1/outer0`へ移行した

## 12. 学習stepのpeak VRAM削減

natural 16 bagのwhole graphを保持したまま補助16 bagのregion graphを構築していたため、
15断面×合計32 bagのEfficientNet活性値が同時に残り、RTX A6000のVRAMを約48 GB使用した。

- `L_whole`をbackwardしてgraphを解放してからregion forward/backwardを行う
- 両backwardの勾配を蓄積後、gradient clippingとoptimizer更新は従来どおり1回だけ行う
- 合成目的関数のgradientと逐次backwardのgradientが一致するunit testを追加する
- model、batch、loss係数、optimizer step数は不変なので`calibration/v1`の再校正は不要

## 13. RTX A6000での学習高速化調査

本番と同じEfficientNetV2-S、natural 16 bag + region 16 bag、各15断面、224×224、
BF16 autocast、whole/region逐次backwardという条件で、GPU上に固定したsynthetic batchを
3 warmup + 8 measured step実行した。DataLoader時間は含まないが、eagerの998 ms/stepは
本番runの約1.08 s/stepに近く、model計算の比較として妥当な結果だった。

| variant | median ms/step | eager比 | peak allocated | 505 step換算 |
|---|---:|---:|---:|---:|
| eager | 998.27 | 1.00x | 27.43 GiB | 8.40 min |
| fused AdamW | 999.97 | 1.00x | 27.43 GiB | 8.42 min |
| TF32 (`high`) | 997.57 | 1.00x | 27.43 GiB | 8.40 min |
| channels-last | 668.85 | 1.49x | 28.33 GiB | 5.63 min |
| `torch.compile(mode="default")` | 585.75 | 1.70x | 18.26 GiB | 4.93 min |
| default compile + channels-last | 585.79 | 1.70x | 18.80 GiB | 4.93 min |

- default compileの初回warmupは約285秒だが、eagerとの差は約413 ms/stepなので約657 step
  （1.30 epoch）で償却する
- compile時はchannels-lastを明示しても速度の上積みがないため、追加しない方が単純
- fused AdamWとTF32は測定noise範囲で、採用価値なし
- `reduce-overhead`は2回forward/逐次backward間でCUDA Graph出力が上書きされRuntimeErrorになる
- `max-autotune-no-cudagraphs`はdepthwise convolution候補の探索が極端に遅く、数分経過後も
  初回whole graphをcompileできなかったため中止した
- production codeへのcompile採用は未実施。採用する場合は`mode="default"`だけを候補とする

## 14. TorchInductor default modeのproduction採用

§13の実測結果を受け、CUDA学習に限り`torch.compile(mode="default", dynamic=False)`を
有効化した。CPU testとcalibrationはeagerのまま維持する。compile cacheはNFS上のhomeを
避け、`/tmp/vai-region-branch-$UID/torchinductor-cache`へ固定する。

これはmodel、loss、batch、optimizer、校正係数を変えない実行最適化なので、既存の
`calibration/v1`はそのまま使用する。config fingerprintも変更しない。

## 15. region data packageのrename

repository共通の`.gitignore`にある`data/`規則が
`fracture_detection/region_branch/data/`へも適用され、Python sourceが追跡対象外に
なっていたため、packageを`data_pipeline/`へrenameした。全import、test、構成図を更新し、
データ実体を置くrepository rootの`data/`とは用途と追跡方針を明確に分離した。
