# 論文要約: Part-Aware Mask-Guided Attention Network (PMGAN)

**原題**: Part-Aware Mask-Guided Attention for Thorax Disease Classification
**出典**: Zhang et al., Entropy 2021, 23, 653 (MDPI)
**ファイル**: `memo/research_paper/胸部疾患分類のための部位認識型マスク誘導型アテンション.pdf`
**読解方法**: Claude が PDF を直接読解（Codex は未使用）

---

## 1. 課題設定

- タスク: ChestX-ray14（胸部X線14疾患のマルチラベル分類、112,120枚）
- 動機: 疾患ごとに関連する臓器が異なる（例: atelectasis→肺、cardiomegaly→心臓）。
  全体画像だけの学習ではlocalな病変手がかりが埋もれる。

## 2. 全体アーキテクチャ（4分岐構造）

```
CXR画像 → Conv1 → Block I → [SA] → Block II → [SA] → Block III
                                                          │
                                        ┌─────────────┬───┴───┬─────────────┐
                                        │             │       │             │
                                   M0(全体mask)  M1(左肺)  M2(右肺)   M3(心臓)
                                        │             │       │             │
                                      [MA]          [MA]    [MA]          [MA]
                                        │             │       │             │
                                    Block IV-0    Block IV-1 Block IV-2  Block IV-3
                                    （各分岐独立パラメータ）
                                        │             │       │             │
                                      [SA]→GAP→FC   [SA]→GAP→FC (×3, localブランチ)
                                        │             └───────┴───────┘
                                      L_ce^1                  max score
                                                                  │
                                                                L_ce^2
```

- **バックボーン**: ResNet50（Block I〜IV）
- **Block I, II**: 全分岐共有。Soft Attention (SA) で逐次的に特徴を精製
- **Block III出力**: ここで4分岐に分岐（全体・左肺・右肺・心臓）。分岐点はablationで検証済み（後述）
- **Block IV**: 各分岐が独立パラメータを持つ（4本の並列ブロック）
- 各分岐末尾で GAP → FC → 疾患確率。localの3分岐は **max score** で統合してから2つ目のBCE損失

## 3. Soft Attention (SA) の構造

- 空間方向attention（spatial-wise, h×w×1）とチャネル方向attention（channel-wise, 1×1×c）を**独立サブネットに分離**して計算量削減
- 空間: encoder-decoder構造（stride-2 conv複数層→対称deconv）で多重解像度を統合
- チャネル: GAP → 2層conv（reduction factor r=16）
- 結合: `m_i = Conv(s_i × t_i)` → sigmoid → `f̂_i = (1 + m_i) ⊗ f_i`（残差attention、m_i≈0で元特徴を保存）

## 4. Mask-Guided Attention (MA) — 核心アイデア

- **構造はSAと全く同じ**。唯一の違いは **セグメンテーション制約で正則化される**点
- 4つの臓器マスク（全体・左肺・右肺・心臓）は**オフザシェルフのセグメンテーションモデルで事前生成**（訓練時のみ使用、推論時には一切使わない＝推論コストが増えない）
- 制約は空間attentionマップ `s_3^b` とマスク `M^b` のRMSE:
  ```
  L_att^b = RMSE(M^b, s_3^b)
  L_att = L_att^0 + β * Σ_{b=1}^{3} L_att^b   (β=1.0が最良、Table参照)
  ```
- **マスクは「ハードにcropして入力する」のではなく「attentionマップを教師する」形で使う**。
  これによりマスク自体の誤差（セグメンテーション精度）が致命傷になりにくい設計思想。

## 5. 損失関数の全体像

```
L_ce^1 = BCE(全体分岐の予測, ラベル)                    # global branch
L_ce^2 = BCE(local 3分岐max統合の予測, ラベル)          # local branches (同じラベルを再利用)
L_ce = L_ce^1 + α * L_ce^2                              # α=0.5が最良
L_att = L_att^0 + β * Σ L_att^b                          # β=1.0が最良
L_total = L_ce + L_att
```

- **重要**: globalとlocalは**同じ疾患ラベルに対して独立した2つのBCE損失**（マルチタスクというより「同一タスクを2つの視点から二重に教師する」設計）。疾患ごとに「どの臓器が正解か」というラベルは不要（弱ラベルなしで機能する）。

## 6. Ablation studyの結果（何が効いたか）

| 追加要素 | 平均AUC | 差分 |
|---|---|---|
| Baseline (ResNet50のみ) | 84.18 | - |
| + Soft Attention (SA) | 85.40 | +1.22 |
| + Mask-Guided Attention (MA) | 86.51 | **+1.11** |

- **MA分岐点はBlock III出力が最良**（Block I/II/IVより高い）。理由: 浅い層で分岐すると過学習・パラメータ増、深い層すぎると分岐の恩恵が減る。中間層分岐が最適という設計知見。
- **multi-task（2つの独立BCE損失）はsingle lossより有意に良い**（85.70→86.51）。localブランチを別損失で独立最適化する方が、単純に統合してから1つの損失にするより良い。
- **α=0.5, β=1.0が最適**（グリッドサーチ、ピーク型の応答曲線）。
- 計算コストは4分岐でもBlock I〜IIIを共有するため**FLOPsは2.1倍で済む**（4倍にはならない）。
- セグメンテーション制約の損失関数（BCE/Dice/RMSE比較）は**RMSEが僅かに最良**だが差は小さい。

## 7. VAI（4領域骨折検出）への転用ポイント（メモ・未検証）

以下は論文内容の忠実な要約であり、VAIプロジェクトへの適用可否はまだ検討していない。次セッションで `2026-08-04-region4-direction.md` の6.2節（学習アーキテクチャの形）と突き合わせて設計すること。

- **入力チャネルへの直接結合ではなく、attentionをマスクで教師する**というPMGANの設計は、
  region4-directionメモ6.2節で既に検討していた「入力チャンネルとしてmask結合＋mask drop」路線とは異なる代替案。
  マスク精度（4領域マスクは線検出由来で誤差を含む）への頑健性という点で比較検討の価値がある。
- **分岐点をどこに置くか**（浅い共有→深い分岐）はPMGANでは中間層が最良。VAIの2.5Dモデルでも
  バックボーンのどの深さで4領域に分岐するかは実験パラメータになりうる。
- **local分岐の統合方法**: PMGANは疾患ラベル（≒椎体レベル骨折ラベル）を4分岐共通で再利用し、
  max scoreで統合後に2本目の損失をかける。VAIのregion4-directionメモが検討していた
  「特徴レベルattention集約 vs ロジットmax/LSE」の議論と直接対応する。PMGANはロジット統合寄り
  （max score）だが、椎体レベルラベルをlocal分岐にも流用できる点は弱ラベル問題の一つの解法になりうる。
- **マスクは訓練時のみ使用し推論時コストゼロ**という設計は、VAIの4領域マスク生成コスト
  （線検出→椎体マスク切り）を推論時に回避できる可能性を示唆する。
- 疾患横断の「臓器ごとの弱教師なし」という前提は、VAIの「領域ラベルの推定（擬似ラベル）」課題とは
  性質が異なる（PMGANは臓器マスクの位置は既知、疾患と臓器の対応は未知。VAIは領域マスクの位置も
  ラベルの対応関係も両方不確か）。この違いは設計時に要注意。

---

## 関連ファイル

- 眼底画像論文の精読結果（Codex担当）: `.claude/docs/codex/20260806-fundus-semisup-multitask-paper-read.md`
- VAI側の現状設計: `.claude/docs/work-logs/2026-08/2026-08-04-region4-direction.md`
