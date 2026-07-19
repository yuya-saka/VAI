# bbox軸補正後投影設計

## スコープ

本設計は、RSNA native DICOM上の連続slice bboxを、軸補正後classifier CTへ
幾何学的に正しく変換して表示できる状態にするところまでを扱う。
4領域maskとのoverlap、center weighting、soft label生成、学習lossは扱わない。

生成するものは、軸補正後15断面上のcanonical bbox geometryである。

```text
bbox_corrected_occupancy.npy   # (15, 224, 224), float32 [0, 1]
bbox_corrected_contours.json  # planeごとのpolygon/component
```

`occupancy` の中間値は再サンプリング時のpartial-volumeであり、
4領域骨折ラベルのsoftnessを意味しない。

## 現行実装の問題

現行 `build_fracture_bbox_planes.py` は、bbox中心に最も近いclassifier planeを1枚選び、
native bboxの4隅をそのplane基底へ正射影した後、投影四辺形を軸平行外接矩形へ変換する。

この処理では次を失う。

- native planeと補正後planeの傾きによる補正後z方向への広がり
- 投影後の傾いた輪郭
- 連続slice間で変化するbbox位置・幅・高さ
- 1つのbbox runが複数classifier planeと交差する事実

したがって `row_min/col_min/row_max/col_max` はcanonical bbox geometryには使わない。

## canonicalな入力

各bbox行について、bbox CSVの `x`, `y`, `width`, `height`, `slice_number` と、
**その行が属する元DICOM slice自身**の以下のheaderを使う。

- `ImagePositionPatient`
- `ImageOrientationPatient`
- `PixelSpacing`

study共通の近似basisだけでbboxを変換しない。
特に `geometry_mode == "repaired_nifti_affine"` でも、bbox pixelのLPS変換には
修復affineではなく元DICOM sliceのheaderを使う。

## 1. bbox四隅をLPSへ変換

各native bboxの4隅をsubpixel座標のままLPS mmへ変換する。

```text
P(x, y) = ImagePositionPatient
          + x * column_spacing * row_direction
          + y * row_spacing * column_direction
```

pixel center/edgeの `0.5 px` conventionは、軸補正なしのidentity overlay testで確定し、
全経路で統一する。整数丸めしてからLPSへ変換しない。

## 2. bboxを連続runへ分割

bbox sliceは `slice_number` の数値差ではなく、DICOM series内の実indexで並べる。
隣接indexが1の行だけを同じrunとして接続する。

離れたrun間には3D bbox geometryを作らない。
別病変またはannotation gapの可能性があり、接続すると未注釈範囲を捏造するためである。

## 3. 連続sliceから3D bbox envelopeを作る

隣接するbbox rectangleの対応する4隅をLPS空間で接続し、
slice間のbbox envelopeを作る。

```text
native slice k      : P00, P10, P11, P01
native slice k + 1  : Q00, Q10, Q11, Q01

P00-P10-P11-P01 と Q00-Q10-Q11-Q01 を接続した3D cell
```

これをrun内の全隣接slice間で連結する。
最初と最後のbboxは、隣接DICOM sliceとの物理的中間面まで端部supportを持たせる。
single-slice runもlocal slice spacingの半分ずつを持つslabとして扱う。

この3D envelopeは骨折segmentationではなく、元の連続2D bbox annotationを
軸補正後断面へ変換するための幾何表現である。

## 4. 補正後classifier planeと交差させる

各classifier planeは既存metadataの以下で定義する。

- `center_lps_mm`
- `row_basis_lps`
- `column_basis_lps`
- `normal_lps`
- output spacing `0.4 mm`
- output size `224 x 224`

各3D bbox cellとclassifier planeの交差polygonを求め、交点をplaneのrow/column基底へ
投影して224px座標へ変換する。複数cell・複数runのpolygonはplane内でunionする。

軸補正により、出力polygonは次のようになりうる。

- 回転した四辺形
- 台形または多角形
- 複数component
- 複数classifier planeへの連続した出現

外接矩形へ戻さず、polygonとrasterized occupancyの両方を保存する。

## 5. occupancyを表示する

表示では、軸補正後CTのcenter channelへ `bbox_corrected_contours.json` のpolygonを描画する。
面として確認する場合は `bbox_corrected_occupancy.npy` を透過表示する。

```text
contour: 元bbox geometryの軸補正後輪郭
fill:    再サンプリングによるpartial-volume occupancy
```

region maskやregion labelはこの段階では表示・計算に必要ない。

## 6. bbox中心の15断面を再生成する

bbox付きデータでは、既存の「椎骨robust range全体へ15枚を配置し、bbox planeを強制挿入」する
方式を使わない。canonicalな3D bbox envelopeを作った後、そのbbox構造を中心とする
別の15断面viewを生成する。

### bbox中心位置

各contiguous bbox runを独立したsampling targetとする。
補正後normal方向の位置を `t`、その位置で3D bbox envelopeを切った面積を `A(t)` とする。

1. `A(t)` を密な物理間隔で評価する
2. `A(t)` の累積面積が50%になるvolume-median位置 `t50` を求める
3. `A(t) > 0` の候補のうち `t50` に最も近い位置を `t_center` とする
4. 同距離なら交差面積が大きい候補を選ぶ

これにより、単純なbbox supportの両端中点や離れたrun間のgapではなく、
実際にbbox構造が存在し、かつrunの中央に近い断面を選べる。

### 15断面の範囲

`t_center` を必ずsequence index `7` に置き、前後7枚を対称に配置する。

```text
t[i] = t_center + (i - 7) * plane_spacing_mm,  i = 0..14
```

必要なhalf extentは、bbox run全体と椎骨contextの両方から決める。

```text
bbox_half_extent = max(t_center - bbox_low, bbox_high - t_center)
context_half_extent = max(
    bbox_half_extent + context_margin_mm,
    vertebra_robust_span_mm / 2,
)
plane_spacing_mm = context_half_extent / 7
```

非常に長いrunで `plane_spacing_mm` が許容上限を超える場合は、解像度を落として1viewへ
押し込まず、runを複数のoverlap windowへ分割する。各windowでも中央planeにはbbox構造が存在する。

### 同時に再生成するデータ

決定した15枚の `PhysicalPlane` をsingle source of truthとして、以下を同時に再サンプリングする。

- center planeを基準にした5-channel CT
- vertebra mask
- 4-region mask生成に必要なline/SDF geometry
- corrected bbox polygon/occupancy

CTだけを新しい位置で切り、region maskやbboxを古いplane indexから流用してはならない。

### 出力分離

既存のfull-vertebra `fracture_dataset` はStage1/2/3比較用に保持する。
bbox中心viewは別ディレクトリへ出力する。

```text
bbox_centered_dataset/{study_id}/{level}/run_{run_id}/
```

最低限、以下をmetadataへ保存する。

- `sampling_mode = "bbox_centered"`
- `bbox_run_id`
- `source_bbox_slice_numbers`
- `bbox_support_range_mm`
- `bbox_center_position_mm`
- `plane_spacing_mm`
- `bbox_occupancy_area_by_plane`
- `center_plane_index = 7`

同一study/levelに複数runがある場合も、train/validation/test splitは必ずstudy単位のままにする。
また、bbox陽性だけsampling方法が異なるviewをprimary分類へそのまま混ぜるとsampling shortcutに
なるため、まずはbbox geometry確認とbbox由来局在教師の補助viewとして使用する。

## 実装上の簡略化

`native_dicom` の平行seriesでは、native格子にbbox occupancy volumeを作り、
CTと同じ `sample_physical_planes()` で再サンプリングしてもよい。
ただし以下を満たす必要がある。

- bboxのsubpixel edgeを整数丸めで失わない
- CTと同じLPS geometryを使う
- nearest-plane割り当てをしない
- polygon/occupancyがLPS cell-intersection実装と一致することをテストする

非平行または修復geometryでは、元DICOM sliceごとのLPS cell-intersection経路をcanonicalとする。
ただし、修復affineで再サンプリングしたCT自体は元DICOM slice geometryの近似であるため、
両者の完全なoverlay一致は保証できない。初期production生成は `native_dicom` のみに限定し、
修復geometryは明示的なoverrideを指定したQC用途に限る。

## 検証

### identity test

軸補正角0度、classifier planeがnative sliceと一致する条件で、
補正後bbox contourが元bboxとsubpixel精度で一致することを確認する。

### synthetic tilt test

一定サイズのbbox stackを15度、30度、35度のplaneで切り、
理論的な交差polygon、plane数、位置と一致することを確認する。

### varying bbox test

連続sliceでbboxの中心・幅・高さを変化させ、補正後polygonがその3D変化を反映すること、
gapをまたいだpolygonが生成されないことを確認する。

### real-data QC

軸補正角、geometry mode、bbox run数で層別し、native bbox、補正後CT、
補正後bbox contourを並べたoverlayを保存する。
現行AABB表示との差分面積とcentroid差も記録する。

さらにbbox中心viewでは、index `7` のbbox occupancyが必ず非zeroであること、
bbox run全体のcoverage、15枚中bboxが見えるplane数、context内のvertebra mask coverageを記録する。
