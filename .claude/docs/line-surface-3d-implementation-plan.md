# Unet 3D Line Surface 実装計画

## 0. ステータス

- 作成日: 2026-07-30
- 対象仕様: `.claude/docs/work-logs/2026-07/2026-07-30-line-surface-3d.md`
- 状態: Phase 1-5 実装完了、fold 0実験待ち
- 新規プロジェクト名: `Unet/line_surface_3d/`

## 1. 実装境界

### 必須制約

- 既存コードの参照元は `Unet/line_only/` のみに限定する。
- `Unet/multitask/`、`Unet/seg_only/`、`Unet/debug/` からコードを流用しない。
- 新規プロジェクトは `Unet/line_surface_3d/` に独立して作り、既存
  `Unet/line_only/` は変更しない。
- モデルは `line_only/src/model.py` の `TinyUNet` を基準にする。
- コメントと docstring は `Unet/` の規約どおり日本語で記述する。
- 入力は `dataset_zprop` の密な画像と椎体マスク、教師は `dataset` の
  手動 `lines.json` だけを使う。`dataset_zprop` の擬似線ラベルは使わない。

### 初期スコープ外

- 3D convolution
- 新規の直接回帰ヘッド
- 横突孔検出器
- 追加アノテーション
- 線分長の z 回帰
- `line_only` の既存チェックポイント読み込み

## 2. 基本設計

### データ単位

- 初期値は奇数窓 `N=15`、stride 1。
- 1スラブは同一 `sample/vertebra` の連続した N スライスだけで構成する。
- 学習スラブは有効な手動ラベルを3枚以上含むものだけ採用する。
- Fold 分割はスラブ作成前に sample 単位で行い、同一被験者の窓が
  train/validation/test をまたがないようにする。
- `bad_slices_all.json` と各椎体の `qc_scores.json` による除外規則は
  `line_only` と同じ意味で適用する。

### テンソル契約

| 名前 | 形状 | 内容 |
|---|---|---|
| `image` | `(2N, H, W)` | slice-major の `[CT_z, mask_z]` |
| `heatmaps` | `(N, 4, H, W)` | ラベル付きスライスだけGTを保持 |
| `label_mask` | `(N, 4)` | 教師が有効な要素 |
| `slice_indices` | `(N,)` | 元データの z index |
| model logits | `(4N, H, W)` | slice-major に reshape して `(N, 4, H, W)` |

`model.in_channels` と `model.out_channels` は設定に重複記載せず、
`slab_size` からそれぞれ `2N`、`4N` を導出する。これにより設定不整合を防ぐ。

### モデル

- `line_only` の `TinyUNet` 構造を新規パッケージへ必要最小限で移植する。
- 初期モデルは `in_ch=30`、`out_ch=60`。
- C1-C7 の bottleneck one-hot conditioning は `line_only` と同じ方式を
  config で切り替え可能にする。
- 新規プロジェクトの独立性を保つため、実行時に `line_only` を import しない。

### Augmentation

- Albumentations `ReplayCompose` を使う。
- 先頭スライスで決まった幾何・輝度変換を、残り N-1 枚へ replay する。
- ポリラインは各スライスで同一 replay を適用した後にヒートマップ化する。
- 初期 baseline では水平反転を無効にする。
- 水平反転を有効化する場合は4線の意味的 channel swap をテストで保証してから行う。

### リボン表現

各線について、中心化した `dz` 上で次を1次近似する。

```text
cx(z)     = cx0 + u  * dz
cy(z)     = cy0 + v  * dz
cos2a(z)  = p0  + p1 * dz
sin2a(z)  = q0  + q1 * dz
```

- ヒートマップからの `cx, cy, nx, ny` 抽出は `line_only/utils/losses.py` の
  `_compute_moments_batch` を基準に移植する。
- `cos2a = nx^2 - ny^2`、`sin2a = 2*nx*ny` とし、0/180度境界を回避する。
- N が固定かつ `dz` が中心化されるため、切片は平均、傾きは
  `sum(dz*y) / sum(dz^2)` の閉形式で計算する。
- フィット後の doubled-angle はスライスごとに正規化する。
- 新しい learnable head は追加せず、全処理を微分可能な tensor 演算で実装する。

### 損失

```text
L_total =
    L_heatmap
  + w_angle   * L_angle
  + w_centroid * L_centroid
  + w_ribbon  * L_ribbon
```

- `L_heatmap`: `label_mask` が真の要素だけを使う masked MSE。
- `L_angle`: ラベル位置におけるフィット後 doubled-angle と
  GTヒートマップ由来 doubled-angle の距離。
- `L_centroid`: ラベル位置におけるフィット後重心と
  GTヒートマップ由来重心の pixel 距離。
- `L_ribbon`: 全 N スライスで、生の予測 moment とフィット値の残差。
- 損失重みと warmup は config 化する。
- `ribbon.enabled=false` で heatmap-only baseline に完全に戻せるようにする。

## 3. 予定ディレクトリ

```text
Unet/line_surface_3d/
├── __init__.py
├── train.py
├── predict.py
├── config/
│   ├── baseline.yaml
│   └── ribbon.yaml
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── inference.py
│   ├── model.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   ├── detection.py
│   ├── losses.py
│   ├── metrics.py
│   ├── ribbon.py
│   ├── region_eval.py
│   └── visualization.py
└── test/
    ├── __init__.py
    ├── test_dataset.py
    ├── test_inference.py
    ├── test_losses.py
    ├── test_model.py
    └── test_ribbon.py
```

## 4. 実装フェーズ

### Phase 1: 独立 scaffold とデータ契約

優先度: 最優先

実装:

1. パッケージ、config、CLI entry point を作る。
2. `dataset` と `dataset_zprop` を照合して椎体単位の manifest を作る。
3. 連続 N スライスかつラベル3枚以上の学習窓を列挙する。
4. sample 単位の既存5-fold分割を移植する。
5. tensor shape、channel ordering、label mask を固定する。

主な対象:

- `Unet/line_surface_3d/config/*.yaml`
- `Unet/line_surface_3d/src/dataset.py`
- `Unet/line_surface_3d/src/data_utils.py`
- `Unet/line_surface_3d/test/test_dataset.py`

完了条件:

- sample/vertebra をまたぐ窓が0件。
- 全窓で z index が連続。
- `dataset_zprop/lines.json` を読んでいないことをテストで確認。
- `N=15` の1サンプルが期待 shape で DataLoader から取得できる。

### Phase 2: 共有 augmentation と heatmap-only baseline

優先度: 最優先

実装:

1. `line_only` の `ReplayCompose` とポリライン後生成方式を移植する。
2. 同じ replay をスラブ全体へ適用する。
3. `TinyUNet` を `2N -> 4N` に拡張する。
4. masked heatmap MSE のみで学習・検証・checkpoint 保存を実装する。
5. ラベル付きスライスの angle/centroid を validation 指標として保存する。

主な対象:

- `Unet/line_surface_3d/src/dataset.py`
- `Unet/line_surface_3d/src/model.py`
- `Unet/line_surface_3d/src/trainer.py`
- `Unet/line_surface_3d/src/evaluation.py`
- `Unet/line_surface_3d/utils/losses.py`
- `Unet/line_surface_3d/test/test_model.py`
- `Unet/line_surface_3d/test/test_losses.py`

完了条件:

- synthetic slab の全スライスで同一幾何変換になる。
- model の出力が `(B, N, 4, H, W)` に正しく復元できる。
- 1 batch の forward/backward で有限勾配が得られる。
- 小規模データを overfit できる。
- fold 0 の heatmap-only baseline が最後まで実行できる。

### Phase 3: 微分可能リボン fit と面損失

優先度: 高

実装:

1. moment 抽出を `(B,N,4,H,W)` に対応させる。
2. doubled-angle 化と閉形式1次 fit を実装する。
3. angle、centroid、ribbon residual の各損失を実装する。
4. warmup と損失別 W&B/ローカルログを追加する。
5. raw prediction と fitted prediction の両方を validation で比較する。

主な対象:

- `Unet/line_surface_3d/utils/ribbon.py`
- `Unet/line_surface_3d/utils/losses.py`
- `Unet/line_surface_3d/utils/metrics.py`
- `Unet/line_surface_3d/src/trainer.py`
- `Unet/line_surface_3d/src/evaluation.py`
- `Unet/line_surface_3d/test/test_ribbon.py`
- `Unet/line_surface_3d/test/test_losses.py`

完了条件:

- 完全な1次 synthetic ribbon の residual が数値誤差範囲で0。
- 0/180度をまたぐ角度列を正しく fit できる。
- 欠損ラベルが損失へ混入しない。
- 一様・ゼロ近傍ヒートマップでも NaN/Inf を出さない。
- 全損失を通じて有限勾配が model logits まで到達する。
- `ribbon.enabled=false` の結果が Phase 2 と一致する。

### Phase 4: 全高 sliding-window 推論

優先度: 高

実装:

1. 同一椎体の全 z を stride 1-3 で走査する推論 Dataset を作る。
2. 各窓の fitted centroid と doubled-angle を global z へ戻す。
3. 重複窓の予測を平均し、分散を disagreement として保存する。
4. 重複窓の raw heatmap を平均し、線分長だけを推定する。
5. 最終 centroid/angle と推定長から4本線の endpoints を再構成する。
6. `sample/vertebra/z/line` 単位の JSON と可視化を保存する。

主な対象:

- `Unet/line_surface_3d/predict.py`
- `Unet/line_surface_3d/src/inference.py`
- `Unet/line_surface_3d/utils/detection.py`
- `Unet/line_surface_3d/utils/visualization.py`
- `Unet/line_surface_3d/test/test_inference.py`

完了条件:

- 椎体の先頭・末尾を含む全スライスが1回以上予測される。
- synthetic ribbon で overlap 集約後も元の面を復元できる。
- 角度平均を degree の直接平均ではなく doubled-angle で行う。
- disagreement が一致窓で0、意図的な不一致で増加する。
- 推論 JSON から全スライスの4線を再現できる。

### Phase 5: 領域形成と比較評価

優先度: 高

実装:

1. `line_only/utils/region_eval.py` の呼び出し方を基準に、
   予測4線と椎体マスクから4領域を生成する。
2. 全域、帯内、帯外、アンカー帯からの距離 bin ごとに領域欠損率を集計する。
3. right/left foramen の欠損率を独立に記録する。
4. 隣接 z の角度・重心変化と2階差分を集計する。
5. 冠状断・矢状断リフォーマットを新規プロジェクト内で実装する。
6. heatmap-only と ribbon の fold 0 結果を同じ評価スクリプトで比較する。

主な対象:

- `Unet/line_surface_3d/utils/region_eval.py`
- `Unet/line_surface_3d/utils/visualization.py`
- `Unet/line_surface_3d/src/experiment.py`
- `Unet/line_surface_3d/test/test_inference.py`

主要評価:

- 帯外4領域欠損率: 現行基準 14.4%
- 帯外 right/left foramen 欠損率
- 9.6-12.4 mm 地点の欠損率: 現行基準 28.8%
- overlap disagreement
- 隣接スライスの最大 angle/centroid 変化
- ラベル帯内の angle/centroid error

## 5. 実験順序

1. `baseline.yaml`、fold 0、N=15、stride 1、heatmap-only。
2. `ribbon.yaml`、fold 0、同じ split/seed/optimizer 条件。
3. fold 0 の領域欠損率と disagreement を比較。
4. 有望なら N=9/15/21 と inference stride 1/3 を小規模 ablation。
5. N を確定後に5-fold実行。

最初から5-foldを回さず、Phase 2-5 の各契約を fold 0 で通してから拡大する。

## 6. 検証コマンド

実装後は対象を狭くしてから広げる。

```bash
uv run pytest -o pythonpath=Unet Unet/line_surface_3d/test -v
uv run pytest -o pythonpath=Unet \
  Unet/line_only/test/test_line_losses.py \
  Unet/line_only/test/test_moment_extraction.py -v
uv run ruff check Unet/line_surface_3d
uv run ruff format --check Unet/line_surface_3d
uv run python Unet/line_surface_3d/train.py \
  --config Unet/line_surface_3d/config/baseline.yaml \
  --start_fold 0 --end_fold 0
uv run python Unet/line_surface_3d/predict.py \
  --config Unet/line_surface_3d/config/baseline.yaml \
  --fold 0
```

## 7. 依存関係

- Phase 2 は Phase 1 の manifest と tensor 契約に依存する。
- Phase 3 は Phase 2 の heatmap baseline と moment 抽出に依存する。
- Phase 4 は Phase 3 の fitted ribbon API に依存する。
- Phase 5 は Phase 4 の全高4線 JSON に依存する。
- region mask 生成は、`line_only` が既に利用している共通
  `data_preprocessing.segmentation_dataset.generate_region_mask` を同じ契約で呼ぶ。

## 8. リスクと対策

| リスク | 対策 |
|---|---|
| 30入力/60出力 channel によるGPUメモリ増加 | batch sizeを小さく開始し、AMPとgradient accumulationは必要時だけ追加 |
| 重複窓が多く実効サンプル多様性が低い | sample単位splitを厳守し、窓数ではなく椎体単位の指標も保存 |
| 未学習の上下端で domain shift | disagreement と領域欠損率を位置別に保存し、失敗範囲を可視化 |
| ribbon loss が初期学習を阻害 | heatmap-only checkpoint、warmup、config無効化を用意 |
| 線分長をfitしないため endpoints が不安定 | overlap平均heatmapから長さだけ推定し、角度・重心はribbon値を使う |
| 0/180度付近で角度平均が破綻 | doubled-angle のみでfit・集約し、degree直接平均を禁止 |

## 9. ロールバック

- 新規プロジェクトだけを追加し、`line_only` と既存データは変更しない。
- すべての出力を `Unet/outputs/line_surface_3d/...` に分離する。
- checkpoint に config、slab size、channel order、data manifest hash を保存し、
  不一致 checkpoint は読み込まない。
- ribbon に問題があれば `ribbon.enabled=false` で Phase 2 baseline へ戻す。
- sliding inference が現行 zprop より悪い場合も、既存 zprop 成果物は上書きせず保持する。
- Phase 2 の baseline checkpoint を Phase 3 以降とは別名で永続化する。

## 10. 実装開始時の最初の作業

最初の実装ターンでは Phase 1 のみを行う。

1. `Unet/line_surface_3d/` の最小 scaffold を作成する。
2. dataset manifest と slab index を実装する。
3. Dataset の単体テストを作成する。
4. 実データで N=15 の窓数と shape を確認する。
5. Phase 1 のレビュー後に model/training へ進む。

## 11. 実装結果（2026-07-30）

- `Unet/line_surface_3d/` を独立パッケージとして実装した。
- N=15の実データ学習窓は計画値どおり4,370件。
  - train: 2,551
  - validation: 944
  - test: 875
- 実データ1スラブで `(30,224,224) -> (60,224,224)` の
  forward/backwardと有限勾配を確認した。
- Ribbon有効時も実データ1スラブで全損失から有限勾配を確認した。
- 新規テスト16件と `line_only` moment回帰テスト27件がpassした。
- `ruff check`、`ruff format --check`、`mypy`、`compileall`がpassした。
- W&Bは既定無効とし、認証なしで学習・推論できる。
