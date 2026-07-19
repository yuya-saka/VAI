# bbox由来4領域ソフトラベル設計

> **スコープ分離**: bboxの軸補正・形状変換・表示は、本書の前処理ではない。
> 先に [`bbox軸補正後投影設計.md`](bbox軸補正後投影設計.md) に従って
> canonicalな `bbox_corrected_occupancy.npy` / `bbox_corrected_contours.json` を生成・QCする。
> 本書は、その確定済みbbox geometryから4領域soft labelを作る後段だけを扱う。
> bbox補正段階ではcenter weightingやregion overlapを使用しない。

## 目的

RSNAの粗いaxial bboxを、補正後の4解剖学的領域
（`body`, `right_foramen`, `left_foramen`, `posterior`）に割り当てる。
ただし、bboxを領域別の確定骨折ラベルとはみなさず、
**骨折陽性であることを条件としたbbox証拠の領域分布**として扱う。

## 現行方式の問題

`build_fracture_bbox_planes.py` は、native DICOM bboxの4隅を補正後planeへ投影し、
投影四辺形を軸平行な `row_min/col_min/row_max/col_max` に変換したうえで、
bbox中心に最も近い1枚のclassifier planeへ割り当てている。

この方式には2つの情報損失がある。

1. 傾いた投影四辺形を外接矩形へ潰すため、他領域との偽の重なりが増える。
2. native sliceと補正後planeが非平行でも、bbox全体を1枚へ押し潰すため、
   本来は補正後z方向へ分布する証拠を失う。

全7,217 bbox行で確認すると、投影されたbbox四隅の補正後plane法線方向の広がりは
中央値6.87 mm、95 percentile 24.81 mm、最大48.19 mmだった。
外接矩形面積は投影四辺形に対して95 percentileで1.13倍、最大1.53倍だった。
したがって、外接矩形だけを四領域maskへ重ねる方法は使用しない。

## 基本方針

```text
native DICOM bbox
  -> native DICOM格子上のsoft fracture evidence
  -> 患者LPS物理座標
  -> CT/maskと同一の補正後classifier planeへ再サンプリング
  -> softな4領域membershipとの重なりを積分
  -> 条件付き4領域分布 alpha[4] + supervision weight
```

座標変換のsingle source of truthは、各classifier planeに保存済みの
`center_lps_mm`, `row_basis_lps`, `column_basis_lps`, `normal_lps` とする。
ラベル生成では `fracture_bbox_planes.csv` の外接矩形を使用しない。
同CSVは可視化互換用に残せるが、必要なら投影4隅も別列で保存する。

## 1. native bboxをsoft evidenceへ変換

canonicalな `assigned_bbox_slice_numbers` に従い、bbox行を `(study, level)` へ割り当てる。
各bboxは、そのbboxが付いたnative DICOM sliceのrow/column格子上に描画する。

bbox内を一様な1にせず、中心を高く、辺を低くしたsoft rectangleとする。
たとえば、bbox内だけで定義した楕円Gaussianと一様成分の混合を用いる。

```text
e_xy = inside_bbox * (uniform_floor + (1 - uniform_floor) * center_gaussian)
```

複数bboxが同じvoxelを支持する場合は加算せず `maximum` で統合する。
これにより、連続sliceに付いた同一病変のbbox行数がラベル強度を不当に増やさない。

native DICOM格子上のevidence volumeは、CTと同じ物理サンプラを使って
15枚の補正後classifier planeへ線形再サンプリングする。
この結果を `bbox_soft_evidence.npy`、shape `(15, 224, 224)`、range `[0, 1]`
として保存する。軸補正によりbbox証拠が斜めになる場合、その斜め形状と
補正後z方向への広がりをそのまま保持する。

### geometry例外

初期実装は `qc.geometry_mode == "native_dicom"` のstudyだけをsoft-label教師に使う。
`repaired_nifti_affine` のstudyは、bbox pixelを修復affineで解釈してはならないため除外する。
将来含める場合は、bbox行ごとの元DICOM
`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing` から連続LPS evidenceを
直接評価し、classifier planeと交差させるslice-wise geometry経路を実装する。

## 2. 4領域maskをsoft membershipへ変換

`region_4class.npy` のhard one-hotを各plane内で軽くGaussian smoothingし、
各pixelで4領域の和が1になるよう再正規化する。
椎骨mask外は全領域0とする。

```text
A[p, r, y, x] in [0, 1]
sum_r A[p, r, y, x] = 1  (inside vertebra)
```

これにより、線検出誤差の影響が大きい領域境界付近を、どちらか一方へhardに確定しない。
領域内部では従来どおりほぼone-hotになる。

## 3. 4領域ソフトラベル

classifier plane位置はbbox強制挿入により完全な等間隔ではないため、
各planeには補正後normal方向の局所Voronoi幅 `delta_p` を付ける。

```text
mass[p, r] = delta_p * sum_yx(E[p, y, x] * A[p, r, y, x])
mass[r]    = sum_p mass[p, r]
alpha[r]   = mass[r] / sum_r mass[r]
```

`alpha` は4領域の独立した骨折確率ではなく、
**椎体が骨折陽性であるとき、bbox証拠が各領域に属する比率**であり、和は1になる。
たとえばbody/right境界上のbboxは `[0.6, 0.4, 0.0, 0.0]` のように保持し、
4領域を広く囲む情報量の低いbboxは一様分布に近づく。

## 4. supervision weight

各soft targetには、ラベルの確定度ではなく**局在教師としての情報量**を表す
`label_weight` を付ける。

- `inside_fraction`: evidenceのうち椎骨mask内に入った割合
- `informativeness`: `1 - entropy(alpha) / log(4)`
- `stability`: bbox辺とregion境界を小さく摂動したときの分布の安定度
- `geometry_qc`: native geometry、region QC、FOV QCを通過したか

```text
label_weight = geometry_qc * inside_fraction * informativeness * stability
```

全領域に広がるbboxは `alpha` を無理にhard化せず、`informativeness` により
局在lossへの寄与を自動的に小さくする。椎体骨折BCEには引き続き使用できる。

## 5. 保存形式

`fracture_region_soft_labels.csv` に最低限以下を保存する。

```text
study_id, level,
body_share, right_foramen_share, left_foramen_share, posterior_share,
label_weight, inside_fraction, normalized_entropy, stability,
bbox_row_count, geometry_mode, label_schema_version
```

生成物には入力bbox CSV、processing metadata、region mask manifestのhashを保存し、
古い座標変換やregion maskから作ったラベルとの混在を防ぐ。

## 6. Stage3への導入

Stage3の既存 `region_evidence_logits` `(B, 4)` に対し、bbox付き陽性だけ
confidence-weighted soft cross entropyを追加する。

```text
pi = softmax(valid_region_evidence_logits)
L_bbox = weighted_mean(-sum_r alpha[r] * log(pi[r]), label_weight)
L_total = L_stage3 + lambda_bbox * L_bbox
```

独立BCEは使わない。`alpha` は条件付き領域分布であり、各領域の絶対骨折確率ではないためである。
椎体陽性/陰性は従来のbag BCE、bboxなし陽性は従来のMIL制約、陰性instanceは
従来のnegative regularizationで学習する。

MixUp時はbag lossと同様に、source A/Bそれぞれの `L_bbox` を
`mixup_lambda` と `1 - mixup_lambda` で重み付けする。
領域IDを解剖学的に戻す既存horizontal-flip remapを使う限り、level単位の `alpha` はswapしない。

## 7. 検証

### 幾何テスト

- 補正角0度でnative bbox evidenceと一致する。
- 既知の15度、30度、35度tiltで、期待する斜め断面になる。
- bbox中心のLPS点が補正後格子の期待pixelへ写る。
- `repaired_nifti_affine` を誤って教師対象に含めない。
- bbox evidenceをnearest 1 planeや外接矩形へ再変換していない。

### データQC

- `alpha` の和が1、全値が有限、`label_weight` が `[0, 1]`。
- bbox evidence、CT、椎骨mask、4領域membershipのoverlayを保存する。
- hardなtop-1領域だけでなく、entropy、top-2 mass、摂動時Jensen-Shannon divergenceを確認する。

### 学習ablation

1. 現行Stage3（bbox lossなし）
2. `lambda_bbox > 0`、confidence weightなし
3. `lambda_bbox > 0`、confidence weightあり
4. 座標整合を壊したAABB/nearest-plane control

主要評価は椎体AUROC/AUPRCを維持したうえで、bbox保持testに対するsoft CE、
Jensen-Shannon divergence、bbox evidenceと予測region evidenceの順位相関とする。
bbox由来soft target自身を真の領域診断とみなしてregion AUCを報告しない。
