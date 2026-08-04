# 2026-08-03 line_surface_3d 平面学習検討

> **[CORRECTED 2026-08-03]** 直近の実行方針は、新しい3Dモデルの実装ではない。
> 完了済み `baseline-v1` の2.5Dモデルを使い、strict-plane projection、
> 傾斜GT/fallback、傾斜Loss、評価指標が成立するかを先に検証する。
> 本文中の3D encoder-decoder案は将来候補であり、現在の実装方針として扱わない。

## 0. セッション状態

- 状態: **2.5D検証設計中。新しい3D学習実装は保留**
- 次回目的: 完了済み2.5D baseline-v1で、現在の中央アノテーションからz傾斜精度を学習・評価できるか深く検討する
- 参照優先度: 本work-logを、`2026-08-02-line-surface-seg3d-requirements.md` の無効化された§2〜3より優先する
- 既存比較対象: `Unet/line_surface_3d/baseline-v1`（2.5D、5-fold完了）

---

## 1. ユーザーが明確に確定した要件

1. 境界は**曲面ではなく平面**である。
2. z方向に平面がどちらへ傾くかという**符号付き方向**が重要である。
3. 境界を目視できるのは、横突孔などが見える椎体中央部である。
4. 不確かな上下スライスへ手動線を追加することは目的ではない。
5. 最重要な臨床的失敗は、中央部で境界が横突孔を横切って領域分割することである。
6. 直近の検証は、完了済み**2.5D baseline-v1を再利用**して行う。
7. 3Dモデルは難易度とデータ量リスクが高いため、平面GT・傾斜Loss・評価が2.5Dで成立するまで保留する。
8. z傾斜方向が明確に作れないGTは、中央線を変化させずz方向へ伸ばす**垂直平面 (`k=0`)** とする。
9. 平面の傾き精度を、学習制約または明示的な評価項目へ必ず入れる。

---

## 2. 現在合意している2.5D検証の骨格

```text
15-slice 2.5D CT+mask slab
    -> completed baseline-v1 TinyUNet
    -> per-slice four-line heatmaps
    -> differentiable strict-plane projection
    -> four plane parameters and reconstructed slab lines
    -> tilt-aware loss and evaluation
```

### 2.1 baseline-v1再利用

- 現在の `2N -> 4N` heatmap出力と中央手動ポリライン教師を維持する。
- `baseline-v1` checkpointを初期値としてfine-tuningする案を第一候補にする。
- 手動GTがある中央zだけでraw heatmap lossを計算する。
- 未注釈の上下スライスを背景として扱わない。
- `dataset_zprop/lines.json` の外挿結果を真の教師として使わない。

### 2.2 Strict-plane projection

- 各スライスheatmapから線角度・位置・confidenceを取り出す。
- 1面について、画像内角度はz方向で一定、線位置だけがz方向へ符号付き線形移動する平面へ射影する。
- 検証対象はraw線だけでなく、射影後の厳密な平面とする。
- 平面射影を推論後処理だけにせず、学習グラフへ入れて中央GTから勾配を流す。

### 2.3 GT plane fallback

- 利用可能な中央アノテーション全体から1面をfitする。
- 符号付きz傾斜が信頼できる面では、その傾斜をGTに保持する。
- 傾斜方向が曖昧なら `k=0` とし、中央の共通線をz方向へ垂直押し出しする。
- 元のfit値、fallback flag、confidence/QC根拠を保存する。
- fallbackは境界面ごとに判定する案を現在採用しているが、最終確定は次回行う。

---

## 3. 本セッションで実施した幾何検証

解析コード:

- `Unet/line_surface_3d/analysis/plane_feasibility.py`
- `Unet/line_surface_3d/test/test_plane_feasibility.py`

詳細報告:

- `.claude/docs/research/line-surface-plane-feasibility.md`

成果物:

- `Unet/outputs/line_surface_3d/plane_feasibility/summary.json`
- `Unet/outputs/line_surface_3d/plane_feasibility/surfaces.csv`
- `Unet/outputs/line_surface_3d/plane_feasibility/held_out_predictions.csv`

### 3.1 単一平面への適合

対象は、QC後に5枚以上の中央手動線を持つ175椎体、700面、4,868線観測。

| 指標 | median | p90 | p95 |
|---|---:|---:|---:|
| 共通角度からのRMS残差 | 1.908° | 4.121° | 4.637° |
| ポリライン点から平面交線までのRMS距離 | 0.994 px | 1.790 px | 2.015 px |
| 中央手動帯全体での絶対z移動量 | 1.296 px | 3.546 px | 4.429 px |

- 96.9%の面で角度RMS残差が5°以下。
- 94.7%の面で点距離RMSが2 px以下。
- 現在の中央手動線を「1枚の平面GT」にまとめること自体は可能。

### 3.2 z傾斜符号の内部安定性

中央手動帯全体で1 px以上移動する422面（60.3%）では:

- leave-one-outの全fitで傾斜符号一致: 93.6%
- 奇数/偶数スライス分割で符号一致: 94.8%

2 px以上移動する215面では両方96.7%。1 px未満では符号がアノテーション変動に埋もれるため、垂直fallback候補となる。

### 3.3 必要アノテーション枚数

中央手動帯の利用可能なz幅全体へ分散して選んだ場合、1 px以上移動する面の傾斜符号一致率は:

| 枚数 | 符号一致率 | 未使用中央線への点誤差median |
|---:|---:|---:|
| 2 | 97.6% | 1.375 px |
| 3 | 98.6% | 1.167 px |
| 4 | 99.1% | 1.125 px |
| 5 | 100.0% | 1.041 px |

ただし2枚では外れアノテーションを検出できないため、実用上は5枚を推奨した。

中央の狭い範囲で連続5枚だけを使う場合、同じ符号一致率は90.0%。これは「見えない上下を無理にannotateするべき」という意味ではない。見える中央帯で利用可能な線をすべて使い、曖昧な傾斜を垂直fallbackへ送る。

### 3.4 検証結果の解釈上の注意

今回確認したのは、**中央手動線から内部的に安定した平面教師を作れるか**である。

未確認事項:

- その傾斜が椎体上下端における真の解剖学的境界と一致するか
- 3D画像から学習モデルがその符号付き傾斜を予測できるか

帯外の真の手動GTは存在しないため、この2点を確認済みと扱ってはならない。

---

## 4. 現在のデータ量監査

| 条件 | 独立sample | 椎体 | 面 | 線観測 |
|---|---:|---:|---:|---:|
| 何らかの有効中央線あり | 41 | 284 | 1,136 | 6,444 |
| 3枚以上 | 41 | 276 | 1,104 | 6,380 |
| 4枚以上 | 41 | 250 | 1,000 | 6,068 |
| 5枚以上 | 40 | 175 | 700 | 4,868 |

既存sample-level 5-foldでは、5枚以上の平面教師を使える学習データは各fold:

- 独立sample: 24
- 椎体: 約102〜108
- validation: 8 sample / 約33〜38椎体
- test: 8 sample / 約33〜38椎体

判断:

- 大規模な通常の3D U-Netを一から学習する量ではない。
- まず既存の小型・全C1-C7共通2.5D baselineで検証する。
- 3D feasibility modelは、この検証が成功した後の候補として保留する。
- C1-C7別モデルは禁止。sample単位分割を維持する。
- 175椎体をplane-projected supervisionへ使う。
- 残りの中央線付き椎体をraw sparse surface lossだけに使う案は有力だが、未確定。

---

## 5. 2.5D baseline-v1の比較基準

> **[INVALIDATED 2026-08-03 後半セッション]** 本節の数値の一部は信頼できない。
> 監査結果: `.claude/docs/research/line-surface-3d-training-audit.md`
> 検証スクリプト: `Unet/line_surface_3d/analysis/training_audit.py`
>
> - `surface_*` 指標（fitted angle/centroid）は**全て無効**。`fit_ribbon` が
>   未アノテーションスライスの全ゼロheatmap（centroid=(0,0)）を等重みで回帰に含めるため、
>   傾きが**16倍減衰**する。`valid` フラグは計算されているが使われていない。
> - `peak_dist_mean` 18〜22 px は**指標自体が無効**。リッジ形状の教師ではargmaxが線上で任意
>   （最大値近傍に111画素）。モデルの失敗を示すものではない。
> - `angle 4.963° / rho 3.116 px` は**窓単位・集約前**の数値。評価は同一スライスを平均13.2回
>   重複カウントし、`inference.py` が行う重複窓集約を行っていない。
>   実運用の集約後精度は未測定で、おそらくこれより良い。
>   → §3のSNR議論はσ_ρ=3.116を使っているため**悲観側にずれている**。
> - `blob_iou 0.685` はconfigのadaptive閾値ではなくハードコードされた0.1で計算されている。

- モデルparameter: 505,740
- 各foldのtraining window: 2,551〜2,691
- 5-fold平均:
  - line angle error: 4.963°
  - rho error: 3.116 px
  - Blob IoU: 0.685

解釈:

- 現データで中央線の2.5D学習が可能なことは実証済み。
- ただし中央手動帯全体のz移動量median 1.296 pxよりrho誤差3.116 pxの方が大きいため、baseline-v1を固定して後処理fitするだけでは傾斜方向が不安定な可能性が高い。
- 新3D方式は中央精度をbaseline-v1と比較しながら、傾斜指標を追加して評価する。

---

## 6. 傾き精度を入れる候補Loss（未決定）

### 6.1 Raw central heatmap loss

- 注釈zだけでbaseline-v1のraw line heatmapとGT line heatmapを比較。
- 現行の中央線性能を維持する基本項。

### 6.2 Projected central loss

- 予測3D evidenceをstrict planeへ射影し、その中央交線を手動GTと比較。
- 最終平面から中央解剖へ直接勾配を流す。

### 6.3 Tilt-vector loss（第一候補）

- 平面内法線を `n=(nx, ny)`、zあたりの符号付き移動を `k` とし、`v=k*n=(dx/dz, dy/dz)` を比較する。
- scalar `k` より法線符号の影響を受けにくく、zが増えると画像内のどちらへ動くかを直接表す。
- 垂直fallback GTは `v=(0,0)`。

### 6.4 Virtual extrapolation loss（第一候補）

- GT平面と予測平面を中央から仮想的に `±5 mm` などへ延長し、その位置での線位置差を計算する。
- 小さい傾斜誤差を外挿距離によって増幅し、実際の用途に近い形で制約できる。
- 真の帯外GTではなく、生成したplane GT間の比較であることを明示する。

### 6.5 3D plane-normal loss

- `1 - |N_pred・N_gt|` で3D法線角度を制約。
- 位置誤差は別Lossが必要。

### 6.6 Full-volume SDF/surface loss

- 予測平面とGT平面のSDFを3D ROI内で比較。
- 強いが、生成GTへの依存が大きいため初期導入には慎重。

現時点の有力な最小構成:

```text
L_total = L_sparse
        + lambda_projected * L_projected
        + lambda_tilt * L_tilt_vector
        + lambda_extrap * L_virtual_extrapolation
```

初期epochはsparse項中心とし、plane関連項をwarmup/rampする案。ただし係数・開始時期は未決定。

---

## 7. 傾き評価候補（未決定）

最低限の候補:

1. `plane_normal_error_deg`
2. `tilt_vector_error_px_per_slice` または物理単位版
3. `tilt_direction_accuracy`（reliable-tilt GTだけ）
4. `virtual_extrapolated_line_error_5mm`
5. `vertebra_edge_plane_error`（GT plane同士の代理評価）
6. `vertical_fallback_predicted_tilt`
7. `central_point_error`
8. `transverse_foramen_crossing_rate`
9. 既存の中央 `angle_error_deg`, `rho_error_px`, `blob_iou`

必ず以下を別集計する:

- reliable signed-tilt GT
- vertical-fallback GT
- 全体
- C1〜C7別
- line_1〜line_4別
- fold別

帯外指標は真の外側GTではなく、生成plane GTまたは解剖学的代理指標であることを明示する。

---

## 8. 次回セッションで優先して決めること

> **[UPDATED 2026-08-03 後半セッション]** §8の多くはCodex相談と追加実測で回答済み。
> 提案（未承認）: `.claude/docs/research/line-surface-plane-tilt-design.md`
> Codex完全回答: `.claude/docs/codex/20260803-2211-plane-tilt-loss-eval.md`
> 実測スクリプト: `Unet/line_surface_3d/analysis/tilt_identifiability.py`
>
> 特に以下は前提が変わった:
> - §6/§7の後付け平面fit前提: SNR 0.30〜0.94で**ノイズ以下**。傾斜は明示的な学習対象にする必要がある。
> - `fit_ribbon()` の重心x,y独立fit: 線方向ドリフトが信号の**2.32倍**混入。ρ=n·cへの射影が必須。
> - §7-8 `transverse_foramen_crossing_rate`: 現行maskでは**算出不可**（利用可能スライス約10%）。
>   代替は `gt_masks` の4領域ラベルに対するIoU。
> - 符号の対照ベースラインは実測 **60.2%**（level+line prior 59.2%はそれ以下）。

### 最優先

1. baseline-v1出力に適用するstrict-plane projectionの具体式と微分安定性
2. GT傾斜をreliableとする判定規則
   - 候補: 中央帯の総移動量1 px以上
   - 候補: leave-one-outで傾斜符号一致
   - 正確なAND/OR規則は未確定
3. fallbackを境界面単位にするか、4面まとめて椎体単位にするか
4. 既存checkpointをfine-tuningするか、同じ2.5D構造を再学習するか
5. 2.5D検証の成功条件と、3Dへ進むgo/no-go条件

### 学習

6. 2〜4枚annotationの椎体をraw sparse lossへ含めるか
7. Lossの最小構成と重み
8. plane lossのwarmup/ramp
9. baseline-v1の中央線精度を守るcheckpoint選択方法
10. 2.5D sliding windowから椎体ごとに1枚の平面をどう集約するか

### 評価

11. checkpoint選択指標
12. 横突孔横断率の計算定義
13. virtual extrapolation距離（±5 mm等）
14. 帯外についてどこまでを代理評価として許容するか

---

## 9. 次回開始時の推奨手順

1. 本work-logを読む。
2. `.claude/docs/research/line-surface-plane-feasibility.md` を読む。
3. `Unet/line_surface_3d/analysis/plane_feasibility.py` のGT定義と検証範囲を確認する。
4. §8の最優先5項だけを先に決める。
5. 2.5D検証が終わるまで新しい3Dモデル設計・実装へ進まない。

---

## 10. 検証状態

実行済み:

```bash
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline pytest \
  -o pythonpath=Unet -q Unet/line_surface_3d/test
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline ruff check \
  Unet/line_surface_3d/analysis/plane_feasibility.py \
  Unet/line_surface_3d/test/test_plane_feasibility.py
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline ruff format --check \
  Unet/line_surface_3d/analysis/plane_feasibility.py \
  Unet/line_surface_3d/test/test_plane_feasibility.py
UV_CACHE_DIR=/tmp/vai-uv-cache uv run --offline mypy \
  Unet/line_surface_3d/analysis/plane_feasibility.py --ignore-missing-imports
```

結果:

- `line_surface_3d` tests: 20 passed
- ruff check: passed
- ruff format check: passed
- mypy: passed
