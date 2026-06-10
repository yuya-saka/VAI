# 椎体骨折分類ベースライン 確定設計 (end-to-end 2.5D + DMIL)

確定日: 2026-06-11
参考: [[dmil-slice-aggregation]] (Hibi et al. 2023 WSAD), Codex検証 (.claude/docs/codex/20260611-loss-aug-schedule-validation.md)

## タスク
- メイン: 椎体単位の骨折分類 (vertebra fracture y/n)。副産物: slice 異常スコア (局在化用)。
- 弱教師: 椎体ラベルのみで学習。slice ラベルは評価専用。

## データ (確定)
- 入力: `dataset_zprop/{sample}/{Cx}/images/slice_XXX.png` (224x224 uint8)
- 椎体マスク: 同 `masks/slice_XXX.png` (0/255、全bag完備)
- 椎体ラベル: `Unet/preprocessing/add_fracture_labels.py` の `FRACTURE_VERTEBRAE`
- slice ラベル(評価のみ): `annotation_data/{sample}/{Cx}/fracture.nii.gz`、`slice_XXX`番号=z-index で整合
- 規模: 41症例 / 284 bag / 陽性62・陰性222 (21.8%) / slice 44-138 (中央値71)

## 前処理 (per-slice)
1. PNG → float[0,1]
2. mask外抑制: img * (mask>0) (hard, 背景=他椎体/軟部を除去)
3. 2.5D: [z-1,z,z+1] 3ch スタック (端は端複製パディング)
4. ImageNet正規化 (ResNet18 pretrained)
5. Augment (学習時、下記)

## モデル
- ResNet18 (ImageNet pretrained), conv1=3ch 流用
- C1-C7を7次元one-hot化し、512次元の画像特徴と結合して分類headへ入力
- 分類head: dropout 0.5 + Linear(512+7, 1)
- per-slice logit z_ij → s_ij=sigmoid(z_ij)
- 1 bag の全 slice を CNN ミニバッチで一括 forward → grad accum 4 bag (実効 bag-batch=4), AMP fp16

## 損失 (DMIL + center loss) ※Codex検証反映
- top-k 選択 (config化、αを後から調整可能):
  - default (capped): k_i = clip(round(0.10 * t_i), 3, 8)
  - 切替 (ratio):     k_i = ceil(t_i / alpha)   ← α を sweep する時に使用
- DMIL (top-k の logit に BCEWithLogits、sigmoid二重適用厳禁):
  - I_i = TopK(z_i, k_i)
  - L_DMIL^i = (1/k_i) Σ_{j∈I_i} BCEWithLogits(z_ij, y_i)
- center loss (陰性 bag のみ、自己 detached mean への分散縮小):
  - μ_i = sg(mean_j s_ij)
  - L_c^i = 1[y_i=0] * (1/t_i) Σ_j (s_ij - μ_i)^2
- 総損失:
  - L = mean_i L_DMIL^i + β * (Σ_i L_c^i / max(1, #neg in batch))
  - β=5 (default)、最初3epで 0→5 線形warmup、sweep {2,5,10}
- 椎体スコア(評価) = mean(top-k_i s_ij)  ← 学習と同じ k

## 不均衡対策
- WeightedRandomSampler(replacement=True) で bag を ~50:50 sampling
- epoch あたり sampled bag 数 = len(train_set)
- augmentation は両クラス同一ポリシー (陽性だけ強い拡張は禁止=リーク)
- balanced sampler 使用時、loss 側に追加 class weight は重ねない

## Augmentation (両クラス共通、3slice+mask 同一パラメータ、理想は bag 内全 slice 同一)
- Spatial: Affine のみ p=0.7 — rot ±7°, trans ±4%, scale 0.95-1.05, shear 0
- flip / elastic / cutout / mixup: 不使用 (左右非対称性を将来利用)
- Intensity (stack/bag 同一): brightness ±0.08, contrast ±0.12, gamma 0.9-1.1,
  Gaussian noise σ0-0.02 p0.25, blur σ0.3-0.7 p0.1
- 補間: image=bilinear, mask=nearest。変換後 mask 再適用で背景0

## 学習スケジュール (cosine 不採用) ※Codex検証反映
- Optimizer: AdamW, wd 1e-4, grad clip 1.0
- Differential LR: backbone 5e-5 / head 2e-4
- Warmup 2ep (0.2x→1.0x 線形)、以後 constant
- Max 35ep、early stop on val 椎体AUPRC patience 8
- gradual unfreeze 不要 (train all from start)
- 全284 bagをStratifiedGroupKFold(5)で患者単位分割（患者リーク厳禁）
- 各bagは1回だけvalになり、全foldのval予測を結合してOOF評価
- fold単体の結果は`fold{n}/fold_metrics.json`、全体結果は`metrics.json`へ保存

## 評価
- 主: OOF pooled 椎体 AUROC/AUPRC (fold平均でなく OOF集約)、有病率21.8%併記、患者クラスタ bootstrap CI
- レベル別: C1-C7を個別表示し、各 n_pos を併記（C3-7 pooledは出力しない）
- slice: fracture.nii.gz から slice-AUC (評価のみ、副指標)

## 必須ユニットテスト
- DMIL: y=1,k=2,logits=[2,0,-1] → grad=[-0.0596,-0.25,0]、top-2のみ流れ、未選択=0
- center: scores=[0.1,0.4,0.7] → grad=[-0.2,0,0.2] (分散縮小のみ、mean を0に引かない)
- 最重要 pitfall: DMIL で logit/probability 混在 (sigmoid二重適用)

## コード構成
Unet/fracture_baseline/
├── train.py
├── config/config.yaml
├── src/  model.py / dataset.py / data_utils.py / trainer.py
├── utils/ losses.py(DMIL+center) / metrics.py / sampler.py / augment.py
└── test/ test_losses.py / test_dataset.py
