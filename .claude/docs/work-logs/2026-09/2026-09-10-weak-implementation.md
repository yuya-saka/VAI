# weak/ 実装完了

日付: 2026-09-10

状態: `fracture_detection/weak/` を実装完了（Arm B のみ）。学習・評価は未実行。

本ログは `2026-09-10-region-mil-implementation-handoff.md` の引き継ぎに対する
実装結果である。設計・引き継ぎログに矛盾が見つからなかったため、
アーキテクチャの再検討はしていない。

## セッション内で確定した3点

ユーザーからの指示:

1. **baseline0 を基本部品として import する。それ以外は `weak/` 内に実装する。**
   `region_branch` への依存は作らない（実験パッケージのロールバック単位を
   `weak/` に閉じるため）。
2. **augmentation は baseline0 の凍結設定をそのまま使う**（flip/transpose/
   affine/brightness/blur・noise/distortion/Cutout）。
3. **最初の実装は Arm B（提案モデル）のみ。** Arm A/C 切替機構は作らない。

## 実装中に見つけた非自明な事実

- **`fracture_detection/weak/data/` は`.gitignore`の`data/`パターンと衝突する。**
  トップレベルの`data/`除外（実データ用）は非スラッシュ始まりのため、
  同名の任意深さのサブディレクトリにもマッチする。`baseline0/data/`は
  個別negation entryで回避しているが、ユーザーの「tests/・*.yamlの除外解除は
  絶対禁止」指示との整合を優先し、negationを増やす代わりに
  `region_branch`と同じ`data_pipeline/`という名前へ変更して衝突を避けた。
  `.gitignore`は一切変更していない。
- **`log1mexp`の`torch.where`にNaN勾配の罠があった。** 両分岐とも入力全体で
  評価されるため、選択されなかった分岐が特異点（`log1p(-1)=-inf`相当）で
  評価されると、その局所勾配がNaNになり`0*NaN=NaN`で漏れる。
  bf16の全logit=-50のテストで発見・修正済み（各分岐の入力を安全な値へ
  事前clampしてから選択する標準パターンに直した）。
- **`encoder.bn2`/`conv_head`は転送されるが一切勾配を受けない。**
  `forward_intermediates(..., indices=(1,2,3,4), intermediates_only=True)`は
  stemと全block stageは実行するが、Baseline 0のwhole path専用の
  global-pooled feature層(`bn2`/`conv_head`)には到達しない。バグではないが
  `README.md`に明記した。
- **`baseline0.data.sampling.AnnotatedCycleSampler`（既存コードで未使用だった
  死んだコード）が、N/U群のGT-passまたぎqueueにそのまま使えた。**
  「pool を1周したときだけ再shuffleする無限stream」という実装がそのまま
  設計書の要求と一致したため、新規実装せず import で済ませた。

## 実装した内容（フェーズ順）

1. **data契約**: `data_pipeline/groups.py`（N/A/U判定・inventory）、
   `data_pipeline/augmentation.py`（CT+椎体mask+4領域label mapの同期変換）、
   `data_pipeline/dataset.py`（`WeakBagDataset`）、
   `data_pipeline/sampling.py`（`GtPassBatchSampler`）、
   `data_pipeline/batching.py`、`data_pipeline/loaders.py`。
2. **モデル部品**: `modeling/pooling.py`（`RegionFpn`+`mask_normalized_pool`、
   region_branchから着想を得た自己完結実装）、
   `modeling/model.py`（`WeakRegionMilModel`: 有効面だけを
   `pack_padded_sequence`で共有BiLSTMへ通し、masked meanしてから共有Linear）。
3. **損失**: `modeling/losses.py`（float32 log-survival/log1mexpによる
   数値安定なnoisy-OR、N/A/U3群の集約）。
4. **初期化・最適化・trainer**: `modeling/initialization.py`
   （`encoder.*`のみ転送）、`training/optimization.py`（2 param group
   cosine）、`training/monitoring.py`（診断のみ・自動停止なし）、
   `training/experiment.py`、`training/trainer.py`（GT-passループ、
   checkpoint 1本、resume）。
5. **評価・CLI・README**: `evaluation/metrics.py`（3母集団分離+Brier/ECE）、
   `config/schema.py`+`config/weak.yaml`、
   `cli/{inventory,train,evaluate}.py`、`README.md`。
6. **設計文書の追随**: 本ログ、`REGION_MIL_DESIGN.md`の配置先修正、
   `PROGRESS.md`、`.claude/docs/DESIGN.md`。

## 検証結果

- `uv run pytest fracture_detection/weak/tests -q` → 84件全成功（FPN分割の等価性テスト2件、
  学習目的関数の合成テスト2件を含む）。
  実manifest（13,432 bag）でN=12,100/A=268/U=1,064/教師セル1,072を再現。
  合成データでのGT-passトレーナーend-to-end、意図的なクラッシュからの
  resume決定性（config一致による誤検知ではなく、`_train_pass`を
  monkeypatchして本物の中断を模擬）を検証済み。
- `uv run ruff check` / `ruff format --check` → 全ファイル通過。
- `uv run mypy fracture_detection/weak --exclude tests` → 新規の型エラー0件。
  残るエラーは全てpandas/albumentations/sklearn/scipy/tqdmのstub欠落
  （baseline0/region_branchにも同じ既知の事前差分がある）と、
  `LambdaLR(lr_lambda=list[...])`のmypy既知の癖（region_branchの同じ実装にも
  同一のエラーが出ることを確認済み）のみ。
- `uv run pytest fracture_detection/baseline0/tests fracture_detection/region_branch/tests -q`
  → 256/257成功。1件の失敗(`test_pseudo_label.py::test_run_generation_end_to_end_smoke_writes_the_three_new_artifacts`)は
  `git stash`でこのセッションの変更を全て退避した状態でも再現した
  既存の不具合であり、今回の変更とは無関係（未修正のまま）。

## 未実施・次回への申し送り

- **実データでの学習は未実行。** GPU上でのsmoke testもまだ。
- **VRAMは実測済み（対策済み）。** 当初READMEに「region_branchの4〜8倍」と
  書いたのは誤りで、FPN部分だけの比を全体の比のように書いていた。実測
  （RTX A6000、ランダム入力、eager、bf16、16 bag×15面）ではFPN一括計算の
  weakが26.73 GiBで、region_branch（陽性2 bag）の19.00 GiBの約1.4倍だった。
  FPN+poolingを1 bagずつactivation checkpointingする実装に変え、18.42 GiB
  （region_branchより小さい）に下げた。代償は1 stepあたり約20%の時間増。
  計算結果は分割しない場合と一致することをテストで確認済み。
- `cli/inventory.py`の4領域mask全数被覆スキャン（13,432 bag）は未実行。
  学習開始前に一度実行して結果を確認すること。
- Arm A/C は本セッションでは実装していない。必要になった場合、
  `loss.beta=0`をconfigで指定すればArm Aと同じ効果（弱教師なし・
  同じforward経路）が既存コードのまま得られる。Arm Cが必要な場合は
  `compute_weak_losses`にA群をORだけで学習する分岐の追加が必要。

## 検証曲線の追加（同日・ユーザー指示）

実装後の確認で、検証曲線について次の不足が見つかったため修正した。

- 旧 `val_loss` は学習と同じN/A/U混合損失を自然分布で平均しただけの値で、
  約9割が陰性bagのため実質「陰性bagの4領域BCE」だった。設計書§8の
  「whole BCEは自然分布のbag BCE、region BCEは既知セル平均」とも違い、
  train lossとも比較できなかった。`history.csv`にも入っていなかった。
- 評価関数が返すBrier/ECE/BCEを記録せず捨てていた。
- `training/monitoring.py`の診断がtrainerから呼ばれておらず、
  `diagnostics.csv`が出力されていなかった（docstringの記述と実装が不一致）。

修正後は、領域系をinnerのGTあり陽性（A）の4セルだけ、whole系をinnerの全bagで
分けて記録する。ユーザーの指示で、GTなし陽性を含む損失も観測できるよう、
群別の1 bagあたり損失（N/A/U）と、それを学習と同じ4/4/8・βで合成した
`train_objective`/`val_objective`を追加した。A∪Nをまとめた領域BCEは、
セルの約98%が陰性bagになり、その部分がwhole BCEと恒等的に一致するため記録していない。
`diagnostics.csv`を毎GT-pass書き出し、予測CSVに生logit（`z_1..z_4`）を追加した。
