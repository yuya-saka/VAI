# 2026-08-04 line_surface_3d 平面パラメータ化への書き換え

## 0. セッション状態

- 状態: **実装完了・5fold実験は未実施**
- モデルは変更していない（TinyUNet 505,740 parameter のまま、追加head なし）
- 前提となる監査: `.claude/docs/research/line-surface-3d-training-audit.md`
- 傾きの学習可能性検討: `.claude/docs/research/line-surface-plane-tilt-design.md`

---

## 1. ユーザー指示

1. GT線からまず平面をGTとして用意する。
2. パラメータは **角度・線の重心位置・平面方向の傾き** の3つ。原則これらだけが重要。
3. GTから明確な傾きを算出できない場合は垂直方向として扱う。
4. 傾き方向のパラメータも学習の制約に入れる。
5. モデルを複雑にしすぎない。機能追加でごまかさない。

---

## 2. 実装した構成

```text
15-slice CT+mask slab
    -> TinyUNet（変更なし）
    -> per-slice 4線 heatmap
    -> fit_plane: 微分可能な厳密平面射影
    -> (phi, rho_0, k) の3パラメータ
    -> 3項の幾何損失 + heatmap損失
```

追加headは作っていない。傾きは heatmap から射影で取り出し、その損失の勾配が
heatmap まで逆伝播する。`test_tilt_loss_backpropagates_into_heatmap` で、
教師のない上下スライスにも勾配が届くことを確認済み。

### 2.1 GT平面

`utils/plane.py::build_surface_plane` が椎体単位・面単位で1枚の平面をfitする。
窓内の数枚ではなく、利用可能な中央帯**全体**を使う（窓内3枚では符号が安定しない）。

信頼できない傾きは `k=0`（垂直平面）にする。判定条件:

- スライス5枚以上、z幅4スライス以上
- 点残差RMS 2.0px以下、角度残差RMS 5.0度以下
- 帯全体の移動量1px以上
- leave-one-out の符号一致率0.8以上
- かつ「移動量2px以上」または「t検定有意 かつ 奇偶分割の符号一致」

実データでは 700面中 269面（38.4%）がreliable。40 sample全体・各line 60〜75面に分散。

### 2.2 予測側の平面射影

`utils/plane.py::fit_plane`:

1. 各スライスのheatmapから重心・主軸・confidenceを出す
2. confidence加重の doubled-angle 平均で**共有法線を1つだけ**決め、上半平面へ正規化
3. その共有法線から各スライスの `rho_i = n . c_i` を作る
4. `rho` を z に対してridge付き重み付き回帰 → `rho_0`, `k`

重要な点:

- スライスごとに法線をcanonicalizeしない（符号反転で `k` が壊れる）
- 重心x,yを独立に回帰しない（線方向のドリフトが傾きへ混入する）
- confidenceは `detach` する（heatmapを平坦化して重みを下げる逃げ道を塞ぐ）
- ridgeは有効スライス数が3未満のときだけ効かせる（定数を足すと傾きが一律減衰する）

### 2.3 損失

```text
L = L_heatmap
  + 0.0001 * L_angle
  + 0.0020 * L_rho
  + 0.0300 * L_tilt
```

幾何は3項だけ。平面のパラメータが3つだからである。
3項ともpx単位（線の位置ずれ）へ換算して同じHuberに通す。

- angle: doubled-angleの外積 `sin(2θ)` に腕の長さ `image_size/4` を掛ける
- rho: 予測平面の交線と教師線のオフセット差
- tilt: `v = k*n` にスラブ端までの距離を掛ける（符号表現の反転に不変）

垂直fallback面は `fallback_weight: 0.25` で弱く0へ引く。完全なゼロ回帰は課さない。

---

## 3. 重みの決め方（重要）

**重みは直感で決めてはいけない。** 3項をpx単位に揃えても、heatmapまで遡る勾配は
項ごとに3桁違う。最初に全部1.0にしたところ heatmap 損失が完全に停滞した
（0.2442 → 0.2444、幾何項が heatmap の40倍）。

`training_audit.py --check loss-balance` で実測する。heatmap項だけで少し学習させてから
（warmup終了相当）、10バッチ平均で勾配ノルムを比較する:

| 項 | 重み1.0での勾配ノルム | heatmapの50%にする重み |
|---|---:|---:|
| heatmap | 0.0255 | — |
| angle | 35.2 | 0.00012 |
| rho | 1.99 | 0.00214 |
| tilt | 0.157 | 0.0272 |

reliable面は疎なので tilt の勾配は1バッチだとばらつく。必ず複数バッチで平均する。
**損失の定義を変えたら必ず測り直すこと。**

---

## 4. 削除したもの

| 削除 | 理由 |
|---|---|
| `utils/ribbon.py` | 平面では角度はz方向に定数。ねじれリボンは要件と矛盾する |
| `utils/detection.py` | `extract_gt_line_params` 以外すべて未使用になった。同関数は `plane.py` へ移動 |
| `config/ribbon.yaml` | ねじれリボン実験の設定。要件が否定した |
| `peak_dist` 指標 | リッジ形状の教師ではargmaxが線上で任意。線の精度を測っていない |
| 線長推定・`line_from_ribbon` | 平面は無限に伸びる。推論は画像境界で切る |

`config/baseline.yaml` は `loss.plane.enabled: false` の対照群にした。
`plane.yaml` との差はその1行だけ。

---

## 5. 副産物として直したバグ

`qc_scores.json` はスライスindexをキーとする dict だが、`load_manual_labels` が
list として走査していたため、**exclude 指定7件が一度も除外されていなかった**。
`_load_qc_excluded` で dict/list 両形式に対応させた。

---

## 6. 検証状態

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline pytest -o pythonpath=Unet -q Unet/line_surface_3d/test
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline ruff check Unet/line_surface_3d/
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline ruff format --check Unet/line_surface_3d/
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline python -m Unet.line_surface_3d.analysis.training_audit --check plane-fit
```

- tests: 31 passed（うち `test_plane.py` 9件が幾何の回帰確認）
- ruff / mypy: 変更ファイルはクリーン（`model.py`・`experiment.py` の指摘は変更前から存在）
- `fit_plane` の傾き回復: 相対誤差 1.7e-5（旧 `fit_ribbon` は16倍減衰）
- 実データend-to-end 25 epoch: heatmap 0.251 → 0.027、平面角度 60.7度 → 22.7度

**5fold実験は未実施。精度の主張は一切していない。**

---

## 6.5 評価・選定指標の再設計（本セッション追加分）

test-v1（GPU実行、fold0）を見返した結果、`selection_metric` が無根拠だった。

### 発見した問題

1. **`selection_metric: plane_rho_error_px` に根拠がなかった。** 旧baseline-v1が
   `angle_error_deg` で選んでいたのを、rho指標を作ったからと機械的に差し替えただけ。
2. **`evaluate()` の角度・rho指標が面単位の集約 vs 集約比較だった。** これは「平面
   という仮定が内部的に整合しているか」しか測らない。ユーザーの要件は「角度・線の
   重心位置」＝**各画像（各アノテーション済みスライス）**での精度であり、集約同士
   を比べるとスライスごとのズレが平均で相殺されて隠れる。損失側の `L_rho` は既に
   各スライス単位で比較していたのに、評価指標だけ集約単位になっていて設計が矛盾していた。

### 実装した修正

`evaluate()` を「予測平面（複数窓を集約した1枚）が、実際にアノテーションされた
**各スライスの生の線**をどれだけ説明するか」で測るよう作り直した。

- `line_angle_error_deg` / `line_rho_error_px`（各画像単位、主指標）を新設。
  予測平面の交線 `rho(z) = rho_0 + k*(z - z_ref)` と、生GT（`build_surface_plane`
  で集約する前の `line_params_gt`）を各アノテーション済みglobal zで直接比較する。
  窓の重複は`(sample, vertebra, line, global_z)`でdedupし、Finding3の重複カウント
  バグを再導入しないようにした。
- `plane_angle_error_deg` / `plane_rho_error_px`（面単位、集約 vs 集約）は診断用に残す。
  「平面仮定自体の整合性」を見るには依然有用。
- `plane_combined_error_px` を新設し、これを `selection_metric` にした。
  `line_rho_error_px + ANGLE_ARM_PX*radians(line_angle_error_deg) + TILT_ARM*tilt_error_px_per_slice`
  （角度・傾きを損失と同じ「線の位置ずれpx」換算で合算。Huberでは潰さない生の平均値）。
  heatmap損失は選定基準にしない（手段であって目的の量ではないため）。
- 回帰テスト `test_line_metrics_detect_per_slice_angle_noise_that_aggregate_hides` を追加。
  面単位では平均されて0度近くに見えるのに、各画像単位では実際の10度ズレが検出できることを確認。

### 未修正のまま残っている問題

角度損失の `sin(2Δφ)` 非単調性バグ（誤差90°で損失0になる）はこのセッションでは
まだ直していない。今回のfold0では90°付近への張り付きは実測上起きていなかったが、
別foldや学習が進んだ場合に顕在化しうる潜在的欠陥。

## 6.6 test-v2実行結果と、傾きの過学習の発見

`plane.yaml`（selection/early-stopping修正後）でfold0を再実行（実行名 `test-v2`、
GPU、別window実行、本文書作成時点でepoch102まで進行中、未完了）。

### 良い点

- heatmap損失(`val_loss_mse`)は0.0022台まで安定して下がり続けた（前回early stopした
  0.0028より明確に改善）。
- `line_angle_error_deg` はepoch80台で5.7〜6.0°まで到達。旧baseline-v1のCC閾値フィルタ
  版(4.963°)にかなり近い。各画像単位の正しい指標でこの水準なので実質的な改善と見てよい。
- `selection_metric: plane_combined_error_px` は正しく機能し、10〜12px台で推移。
- **複数スライス間の内部整合性は良好。** 個々のスライスの生予測（heatmapからその場で
  取り出した向き・位置）と、窓全体を集約した共通平面との残差は、教師ありスライスで
  角度中央値1.43°/rho中央値0.39px、教師なしスライスでも1.62°/0.42pxとほぼ同水準。
  「集約が個々のスライスのバラバラな予測を平均でごまかしている」という懸念は否定された。
  （ただし内部整合性は「正しさ」の証明ではない。全スライスが揃って間違っていても
  この指標は0になる。）

### 悪い点: tilt損失が過学習している

`geometry_weight=1.0`になった以降（epoch21〜94）で比較:

| | epoch21 | epoch94 | 変化 |
|---|---:|---:|---:|
| `train_tilt_loss` | 0.249 | 0.047 | -81%（大きく低下） |
| `val_tilt_loss` | 0.239 | 0.265 | **+11%（悪化）** |
| `tilt_error_px_per_slice`(val) | 0.239 | 0.268 | +12%（悪化） |
| `tilt_sign_accuracy`(val) | 0.51 | 0.55 | 横ばい、事前分布60.2%を超えず |

trainは下がり続けるのにvalは頭打ち〜悪化という典型的な過学習パターン。
1foldの学習データで傾きreliableな面は150〜165面程度しかなく、505,740パラメータの
モデルには個別事例を記憶できてしまう規模だと推測される。

**ユーザー判断: 傾きの問題は一旦棚上げし、次セッションでモデルの構造自体を検討する。**

## 7. 次にやること

### 最優先（次セッション）

**モデルの構造を再検討する。** 現在のTinyUNetは15スライスをchannel方向へ積んで
1回のforward passで処理する2D U-Net（`in_channels=2*15, out_channels=4*15`）。
傾きの過学習を踏まえて、この構造が適切かを検討する。検討観点の例（未確定、次回議論）:

- 傾きreliable面（150〜165/fold）に対してモデル容量(505,740param)が過剰なのか
- channel方向にスライスを積む現構造が、tilt推定に適した帰納バイアスを持つか
- 複数スライス間の内部整合性は良好だった（角度中央値1.4〜1.6°）ので、
  「集約方法」ではなく「モデルの容量・正則化・構造」側に問題がある可能性が高い
- baseline.yaml（plane制約なし）との比較で、tiltの過学習が plane loss 由来か
  モデル構造由来かを切り分けられるか

### その他

1. `plane.yaml` で5fold学習を回す。
2. 対照群として `baseline.yaml`（平面制約なし）も回し、中央線精度が劣化していないか確認する。
3. 傾き指標（`tilt_sign_accuracy`）を、実測した事前分布ベースライン **60.2%** と比較する。
   `.claude/docs/research/line-surface-plane-tilt-design.md` §4 参照。
4. 統計はsample単位クラスタのbootstrapで出す。foldの平均5点にt検定をかけない。

### 未着手の判断事項

- heatmap損失を BCE-with-logits にするか（監査の所見5）。baseline無効化と5fold再実行を伴うので、
  上記1〜2の結果を見てから決める。
- checkpoint選択指標は `plane_rho_error_px` にしたが、傾きを反映していない。
  傾きが学習できると分かってから見直す。
