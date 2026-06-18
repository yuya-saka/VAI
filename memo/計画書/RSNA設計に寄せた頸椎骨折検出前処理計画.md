# RSNA設計に寄せた頸椎骨折検出前処理計画

## 目的

本計画では、RSNA 2022 Cervical Spine Fracture Detection 1位解法の考え方に寄せて、頸椎CT画像から椎体単位の骨折検出用データセットを作成する。

ただし、本研究では椎骨maskの学習は行わない。  
RSNA 1位ではStage1として椎骨segmentation modelを学習していたが、本研究ではTotalSegmentatorによってC1〜C7の椎骨maskを取得するため、Stage1は不要とする。

その代わり、TotalSegmentatorで得た椎骨maskを用いて、椎体ごとのROI作成、向き補正、224×224スライス画像生成、線検出モデルによる4領域分割、骨折分類用データセット作成を行う。

---

## 基本方針

本研究の前処理方針は以下の通りである。

```text
DICOM
↓
NIfTI化
↓
TotalSegmentatorでC1〜C7 mask作成
↓
椎体ごとのROI作成
↓
等方性サンプリングは行わない
↓
椎体ごとの向き補正は行う
↓
補正後スライスを224×224で出力
↓
既存の線検出U-Netを適用
↓
4領域maskを作成
↓
RSNA式の2D CNN + RNN / LSTM分類へ入力
```

---

## RSNA 1位解法との対応

RSNA 1位解法では、以下のような構成だった。

```text
Stage1:
DICOM CT
↓
3D segmentation model
↓
C1〜C7の椎骨mask予測

Stage2:
予測maskで椎体ごとにcrop
↓
各椎体から固定枚数のスライスを作成
↓
2D CNN + RNN / LSTMで骨折分類
```

本研究では、Stage1をTotalSegmentatorで置き換える。

```text
RSNA 1位:
椎骨segmentation modelでmask予測

本研究:
TotalSegmentatorでC1〜C7 mask取得
```

したがって、学習対象は椎骨maskではなく、椎体ごとに切り出した画像を用いた骨折分類である。

---

## 前処理全体の流れ

### 1. DICOMからNIfTIを作成する

まず、元のDICOM CT seriesをNIfTI形式に変換する。

```text
DICOM series
↓
dcm2niix
↓
ct.nii.gz
```

この段階では、等方性リサンプリングは行わない。  
元CTのspacingを保持する。

```text
x spacing: 元CTのまま
y spacing: 元CTのまま
z spacing: 元CTのまま
```

NIfTI化する理由は、TotalSegmentatorの入力・出力を扱いやすくするためである。

---

### 2. TotalSegmentatorでC1〜C7 maskを作成する

NIfTI化したCT画像に対してTotalSegmentatorを適用し、C1〜C7の椎骨maskを取得する。

出力例：

```text
segmentations/sample_id/
├── vertebrae_C1.nii.gz
├── vertebrae_C2.nii.gz
├── vertebrae_C3.nii.gz
├── vertebrae_C4.nii.gz
├── vertebrae_C5.nii.gz
├── vertebrae_C6.nii.gz
└── vertebrae_C7.nii.gz
```

ここで得たmaskは、分類モデルへの直接入力というより、主に以下の目的で用いる。

```text
1. 椎体ごとのROI決定
2. z方向の椎体存在範囲の取得
3. xy方向のcrop範囲の決定
4. 向き補正の基準
5. 線検出モデル入力画像の作成補助
```

---

### 3. 椎骨maskのcleaningを行う

TotalSegmentatorの出力maskには、小さなノイズや孤立領域が含まれる可能性がある。  
そのため、各椎体maskについて最大連結成分のみを残す。

```text
vertebrae_C3 mask
↓
connected component labeling
↓
largest connected component
↓
cleaned C3 mask
```

これにより、ROI決定や向き補正が安定する。

---

### 4. 椎体ごとのROIを作成する

C1〜C7それぞれについて、maskの非ゼロ領域からbboxを計算する。

```text
C1 mask → C1 ROI
C2 mask → C2 ROI
...
C7 mask → C7 ROI
```

ROIは、CT画像そのものではなくmaskから決定する。  
ただし、最終的に切り出す画像はmask画像ではなく、元CT画像である。

```text
maskでROI範囲を決める
↓
元CTからその範囲をcropする
```

marginは必ず付ける。

推奨margin：

```text
x, y方向: 10〜20 mm
z方向: 数sliceまたは数mm
```

横突孔や後方要素を評価するため、左右方向のmarginは狭くしすぎない。

---

## 等方性サンプリングについて

本計画では、CT全体を1mm isotropicや0.4mm isotropicに再構成する処理は行わない。

行わない処理：

```text
CT全体を1mm × 1mm × 1mmに変換
CT全体を0.4mm isotropicに変換
z方向に新しいスライスを大量生成
```

理由は以下である。

```text
1. RSNA 1位の学習設計に近づけるため
2. 骨折線が補間でぼけることを避けるため
3. 元CTのaxial slice情報をできるだけ保持するため
4. 3D CNNではなく2D CNN + RNN設計を想定しているため
```

---

## 向き補正について

等方性サンプリングは行わないが、椎体ごとの向き補正は行う。

ここでいう向き補正とは、CT全体を等方性3D volumeに変換することではなく、椎体ROIに対して、線検出モデルが学習した断面に近い2D axial断面を作る処理である。

```text
椎体ROI
↓
maskから椎体の傾きを推定
↓
補正後のaxial断面を作成
↓
224×224画像として出力
```

これは、4領域分割を安定させるために必要である。

特に本研究では、以下の4領域を扱う。

```text
1. 左横突孔領域
2. 右横突孔領域
3. 椎体中心部
4. 後方要素
```

通常のaxial sliceのままだと、頸椎の傾きにより横突孔や後方要素が安定して同一断面に現れない可能性がある。  
そのため、椎体ごとの向き補正を行う。

---

## 向き補正時の注意点

等方性サンプリングをしない場合、voxel spacingは方向ごとに異なる。

例：

```text
x spacing = 0.4 mm
y spacing = 0.4 mm
z spacing = 1.0 mm
```

この状態で単純にNumPy配列上で回転すると、物理空間上では歪んだ回転になる可能性がある。

したがって、向き補正はvoxel index空間だけではなく、物理座標mmを意識して行う。

考慮すべき情報：

```text
spacing
origin
direction
affine
CTとmaskの対応
```

補間方法は以下のように分ける。

```text
CT画像: linear interpolation
mask画像: nearest neighbor interpolation
```

maskにlinear interpolationを使うと、0/1のラベルが壊れるため使用しない。

---

## 224×224画像の作成

向き補正後、既存の線検出U-Netに入力するために、各スライスを224×224画像として出力する。

```text
補正後CT slice
↓
windowing
↓
224×224
↓
線検出U-Netへ入力
```

ここで重要なのは、画像サイズだけでなく、線検出モデルの学習時と画像の見え方をそろえることである。

そろえるべき条件：

```text
1. 画像サイズ: 224×224
2. Window Level / Window Width
3. 椎体の中心位置
4. 椎体の大きさ
5. 上下左右の向き
6. crop範囲
7. 正規化方法
```

線検出モデルを再学習せずに使う場合、入力分布のズレが性能低下につながるため、前処理条件をできるだけ学習時と一致させる。

---

## Windowing

骨折検出および線検出用のCT画像にはbone windowを用いる。

初期設定案：

```text
Window Level = 400
Window Width = 2000
HU range = -600 ~ 1400
```

処理例：

```text
HU値
↓
clip(-600, 1400)
↓
0〜1に正規化
↓
0〜255 uint8 PNG または float tensor
```

---

## 線検出モデルの適用

224×224に整形した補正後スライスに対して、既存の線検出U-Netを適用する。

```text
224×224 CT slice
↓
線検出U-Net
↓
4本の線heatmap
↓
線座標へ変換
↓
4領域mask作成
```

線検出モデルの出力から、以下の4領域maskを生成する。

```text
1. 左横突孔領域
2. 右横突孔領域
3. 椎体中心部
4. 後方要素
```

分類モデルへの入力として使用する場合は、one-hot maskとして扱う。

例：

```text
Channel 1: CT
Channel 2: 左横突孔領域mask
Channel 3: 右横突孔領域mask
Channel 4: 椎体中心部mask
Channel 5: 後方要素mask
```

---

## 骨折分類用データセット作成

RSNA 1位に寄せるため、まずは椎体ごとに固定枚数のスライスを作成する。

基本設計：

```text
1 sample = 1 vertebra
input = 15 slices × C channels × 224 × 224
label = fracture / non-fracture
```

例：

```text
sample001_C1.npy
sample001_C2.npy
...
sample001_C7.npy
```

各ファイルの中身：

```text
15 × C × 224 × 224
```

初期段階では、Cは1でよい。

```text
C = 1
Channel 1: CT
```

次に、mask情報を追加する。

```text
C = 2
Channel 1: CT
Channel 2: vertebra mask
```

最終的には4領域maskを追加する。

```text
C = 5
Channel 1: CT
Channel 2: 左横突孔領域mask
Channel 3: 右横突孔領域mask
Channel 4: 椎体中心部mask
Channel 5: 後方要素mask
```

---

## スライス選択方法

各椎体maskが存在するz範囲を求め、その範囲から固定枚数のスライスを選択する。

```text
C3 maskのz_min〜z_max
↓
np.linspace(z_min, z_max, 15)
↓
15枚のslice indexを取得
```

この方法では、各椎体を相対位置で15分割することになる。

```text
0枚目: 椎体上端側
7枚目: 椎体中央付近
14枚目: 椎体下端側
```

RSNA 1位のstage2に近い設計であり、2D CNN + LSTMに入力しやすい。

---

## 学習設計

### Baseline 1: CTのみ

最初は、最も単純な入力で椎体単位の骨折分類を行う。

```text
input:
15 × 1 × 224 × 224

label:
椎体単位の骨折有無
```

モデル：

```text
2D CNN
↓
slice feature
↓
LSTM / GRU / attention pooling
↓
vertebra-level fracture probability
```

---

### Baseline 2: CT + vertebra mask

次に、椎体maskを入力チャンネルとして追加する。

```text
input:
15 × 2 × 224 × 224

Channel 1: CT
Channel 2: vertebra mask
```

目的は、モデルに「どこが椎体か」を明示的に与えることである。

---

### Proposed: CT + 4領域mask

最終的には、線検出モデルで生成した4領域maskを入力する。

```text
input:
15 × 5 × 224 × 224

Channel 1: CT
Channel 2: 左横突孔領域mask
Channel 3: 右横突孔領域mask
Channel 4: 椎体中心部mask
Channel 5: 後方要素mask
```

出力は、まずは椎体単位の骨折有無とする。  
その後、bboxや弱ラベルを用いて領域別骨折ラベルを作れる場合は、領域別分類へ拡張する。

---

## 領域別骨折検出への拡張

最終目的は、単なる椎体単位の骨折分類ではなく、4領域ごとの骨折検出である。

出力例：

```text
C3:
- 左横突孔領域: fracture probability
- 右横突孔領域: fracture probability
- 椎体中心部: fracture probability
- 後方要素: fracture probability
```

RSNAデータのbbox付き症例を使う場合、bboxと4領域maskを重ねることで、骨折bboxがどの領域に属するかを割り当てる。

```text
fracture bbox
↓
4領域maskとoverlap計算
↓
最も重なる領域に骨折ラベル付与
```

これにより、領域別骨折ラベルを作成できる。

---

## QC項目

骨折分類に進む前に、前処理結果の品質確認を行う。

確認すべき項目：

```text
1. TotalSegmentator maskがC1〜C7に正しく対応しているか
2. mask cleaning後に椎体が消えていないか
3. ROI cropで横突孔・後方要素が切れていないか
4. 向き補正後の断面が線モデルの学習時と近いか
5. 224×224画像で椎体が中央にあるか
6. 上下左右の向きが正しいか
7. 線検出U-Netの出力が妥当か
8. 4領域maskが解剖学的に破綻していないか
9. 15枚抽出したとき、椎体の上端〜下端を十分にカバーしているか
10. 骨折スライスが抽出対象から漏れていないか
```

特に、以下の可視化を保存する。

```text
CT image
CT + vertebra mask overlay
CT + predicted lines overlay
CT + 4-region mask overlay
```

---

## 保存形式

前処理後のデータは、以下のような構造で保存する。

```text
processed_rsna_style/
├── sample001/
│   ├── C1/
│   │   ├── ct_slices.npy
│   │   ├── vertebra_mask.npy
│   │   ├── region_mask.npy
│   │   ├── overlay/
│   │   └── metadata.json
│   ├── C2/
│   ├── ...
│   └── C7/
├── sample002/
└── labels.csv
```

ct_slices.npy:

```text
15 × 1 × 224 × 224
```

region_mask.npy:

```text
15 × 4 × 224 × 224
```

metadata.jsonには以下を保存する。

```text
sample_id
vertebra_level
original_spacing
selected_slice_indices
crop_bbox
orientation_correction_params
window_level
window_width
line_model_version
qc_flags
```

---

## 実験順序

最初から4領域maskを入れるのではなく、段階的に実験する。

### Step 1: RSNA風baseline

```text
TotalSegmentator mask
↓
椎体ごとROI
↓
向き補正
↓
224×224 CT
↓
15 slices
↓
CTのみで分類
```

目的：

```text
椎体単位分類が成立するか確認する
```

---

### Step 2: CT + vertebra mask

```text
CT + 椎体mask
↓
2D CNN + LSTM
```

目的：

```text
maskを入力に加えることで精度が変わるか確認する
```

---

### Step 3: CT + 4領域mask

```text
224×224 CT
↓
線検出U-Net
↓
4領域mask
↓
CT + 4領域maskで分類
```

目的：

```text
4領域情報が骨折分類に寄与するか確認する
```

---

### Step 4: 領域別分類

```text
bbox付きRSNA症例
↓
4領域maskとbboxを対応
↓
領域別骨折ラベル作成
↓
領域別骨折分類
```

目的：

```text
VAIリスク評価につながる領域別骨折検出を行う
```

---

## まとめ

本計画では、RSNA 1位解法の「椎骨maskでcropし、2D CNN + RNNで骨折分類する」という設計を踏襲する。

ただし、椎骨maskの学習は行わず、TotalSegmentatorでC1〜C7 maskを取得する。

また、等方性サンプリングは行わない。  
元CTの情報をできるだけ保持しつつ、椎体ごとの向き補正のみを行い、既存の224×224線検出モデルを適用する。

最終的には、以下の流れで骨折検出用データセットを作成する。

```text
DICOM
↓
NIfTI化
↓
TotalSegmentator
↓
C1〜C7 mask
↓
椎体ごとROI
↓
向き補正
↓
224×224 CT slice
↓
線検出U-Net
↓
4領域mask
↓
15 slices / vertebra
↓
2D CNN + LSTM
↓
椎体単位または領域単位の骨折分類
```

この設計により、RSNAの学習設計に近づけながら、本研究独自の4領域分割とVAIリスク評価に接続できる。