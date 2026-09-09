# Research: region_branch の少数ラベル問題への対処

Date: 2026-08-27
Context: outer fold 0 の学習完了（epoch 24 early stop, best epoch 4）を受けた原因調査と文献調査。
Gemini CLI はクォータ超過（本日 13:18 JST 頃から）のため WebSearch と手元 PDF で実施。

## 1. 測定で確定した事実（outer fold 0, n=1 fold）

| 事実 | 数値 | 出所 |
|---|---|---|
| val_exact は epoch 1 が最小 | 改善幅 0.0000 | history.csv |
| val_human は倍増 | 0.4909 → 1.0303 (rho=+0.953, p=7e-13) | history.csv |
| ベースレート予測器の BCE | 0.6123（epoch24 はこれより悪い） | 実測 |
| region AUROC 平均 | 0.8305 → 0.8155 (rho=+0.039, p=0.86) 横ばい | history.csv |
| region AP 平均 | 0.8420 → 0.7754 (rho=-0.522, p=0.009) 低下 | history.csv |
| teacher 一致 | 0.7467 → 0.8599 (rho=+0.923, p=1e-10) 上昇 | diagnostics.csv |
| whole AP | 0.6854 → 0.7188 (rho=+0.925, p=1e-10) 上昇 | history.csv |
| human プール周回数 | 12.7 周/epoch（159 bag に 2020 抽出） | sampling.py:37 |
| CAM teacher の対人手GT AUROC | 0.789〜0.831（4領域） | 実測 |
| CAM を 0/1 化した場合のラベル誤り率 | 22.3%（人手GTで閾値最適化した最良ケース） | 実測 |
| 反実仮想: region 項の悪化を止めた場合 | val_total 0.4090→0.3736 (rho=-0.612, p=0.0015) | 実測 |

### コードで確認した構造的欠陥

1. **人手 GT が ranking 損失に入っていない**
   `loaders.py:67` が `attach_teacher_scores` を pseudo にのみ適用 → `dataset.py:158` で human/negative は
   `region_scores = NaN` → `scoring.py:165` の `eligible = positive & isfinite(scores) & scores > 0` で除外。
   唯一の正解ラベルが BCE のみを訓練し、主評価（AP/AUROC=順位）を訓練していない。

2. **陽性側の論理制約が未使用**
   `losses.py` / `pooling.py` / `trainer.py` に領域方向の max / MIL 制約が存在しない（grep 済）。

   | 論理制約 | bag 数 | 状態 |
   |---|---:|---|
   | 骨折なし ⟹ 全4領域 0 | 7,272 | ✅ `L_N` で使用 |
   | 骨折あり ⟹ 最低1領域 1 | **802** | ❌ **未使用** |

   検証: 人手GT陽性 bag 159件の 100% で成立。1 bag あたり陽性領域 1.40/4（スパース＝有効な制約）。

3. **hard 0/1 BCE に有限最適点がない**
   勾配 `sigmoid(z) - y` は正解例でもゼロにならない。設計書 §6.4 は `L_exact` に「絶対 scale の固定」を
   担わせているが、hard target BCE はその役割を果たせない。

## 2. 手元論文

### 眼底画像 半教師ありMTL (Telesco et al., Computers in Biology and Medicine, 2025)
- Task A（品質詳細）GT 802枚で Teacher-A を学習 → EyeQ **12,543枚**に擬似ラベル → Student
- `L = λA·LBCE(ŷᴬ, fA) + λB·LCE(yᴮ, fB)`
- **少数GTは Teacher の学習にのみ使い、Student には直接与えない**（過学習経路が存在しない）
- **モデル選択は主タスク(B)の F1**。λA・λB も同基準でチューニング。補助損失は選択に使わない
- teacher の過学習とノイズを明示的に許容:
  "Because S is small, the resulting model may overfit and generalize poorly. Nevertheless, its
  predictions can provide a useful–albeit noisy–training signal"
- 擬似ラベルの 0/1 化は記法が曖昧（`yᴬ ∈ {0,1}ᵏ` かつ `fⱼ` を "predicted probability" と記述）。断定不可

### 脳CT WSAD (Automated screening of CT using weakly supervised anomaly detection)
- scan 単位ラベルのみ、**スライス注釈ゼロ**で MIL → スライス AUC **0.89**
- dynamic MIL loss + center loss、AR-Net ベース
- 注釈数を 97.1% 削減

### A Weakly Supervised Fine Label Classifier Enhanced by Coarse Supervision (ICCV, Taherkhani et al.)
- 細ラベル 10% で +2.97%、20% で +2.52%、50% で +2.11%（細ラベルが少ないほど粗ラベルの恩恵大）
- 機構は低ランク/スパース自己表現層。細クラスは排他的（本件は多ラベル）なので機構転用は限定的
- 本件の細ラベル比率は 159/8074 = **2.0%** とさらに極端

### PMGAN: Part-Aware Mask-Guided Attention for Thorax Disease Classification
- 臓器マスクでアテンションを正則化し、**領域ラベル無しで**領域認識特徴を獲得
- 推論時の追加計算なし。本件は `region_4class.npy` のマスクを保持済みで機構的に近い

## 3. Web 調査

### MIL の基礎定義と医用画像での位置づけ
- 陽性 bag は最低1つの陽性インスタンスを含み、陰性 bag は全て陰性 — MIL の基礎定義
- 局所注釈が高コスト・不可能で、全体ラベルのみ入手可能な場合が MIL の想定シナリオ（医用画像で標準）
- WSI 分類では bag 分類と instance 判別の2タスクが、臨床診断と腫瘍局在に対応

### プーリング関数の選択（重要）
Wang et al., "Comparing the Max and Noisy-Or Pooling Functions in MIL for Weakly Supervised
Sequence Learning Tasks" (Interspeech 2018, arXiv:1804.01146):
- **max pooling は局在に成功、noisy-OR は失敗**
- noisy-OR の失敗機構: 多数のインスタンスにわたる小さな確率（0.02程度）が bag 予測を高くしてしまい、
  インスタンス予測が弱いままでも「既に検出済み」とモデルが判断する
- LSE は max の平滑版。弱教師あり物体局在ではあまり使われてこなかった
- 集約方法の最適解は未解決問題（max / mean / LSE が併用されている）

注: 本件は領域が 4 個のみなので、noisy-OR の破綻は数百フレームの場合ほど深刻ではない可能性がある。
ただし文献の推奨は max / LSE。

### Label smoothing
- 医用画像の過信対策として標準的。hard one-hot を一様分布との加重平均に置換し、
  ピーク確率の出力を抑制する正則化
- 医用画像では「疾患を強く示唆するが確定的でない」所見に対し懐疑を保持させる意義がある
- 注意点: label smoothing は selective classification を劣化させるという報告あり (arXiv:2403.14715)
- 発展形として Online Label Smoothing (arXiv:2510.20011)

### 小標本での閾値選択
- 小標本では分割ごとに最適閾値が大きくばらつくことが文献で報告されている
- 推奨: 閾値専用のホールドアウトを設ける（臨床応用例では開発 90-95% / 閾値選定 5-10%）
- CV は「閾値移動を含むパイプラインの性能推定」に使うべきで、最終モデルの閾値決定には使わない
- 本件の実測とも整合: val 53 件で選んだ閾値は転移せず（region_3/region_4 はむしろ悪化）

## 4. 対処案の整理

| 案 | 出所 | ノイズ | 追加データ | 評価 |
|---|---|---|---|---|
| MIL 陽性制約（max/LSE） | WSAD, MIL 標準 | **ゼロ** | **802 bag** | 有力。teacher 不要 |
| sampling 比の是正 | 測定した直接原因 | — | — | 有力。設定1行 |
| label smoothing / soft target | 医用画像で標準 | — | — | 有力。損失1行 |
| 人手GTを ranking にも入れる | 構造的欠陥の修正 | ゼロ | — | 主評価に直結 |
| モデル選択を主タスク指標へ | 眼底論文の実践 | — | — | 論文の先例あり |
| 閾値専用ホールドアウト | 閾値選択の文献 | — | — | F値を出すなら必須 |
| 擬似ラベル 0/1 化 | 眼底論文 | **22%** | 643 bag | 天井が下がる。非推奨 |
| region teacher 作り直し | 眼底論文 | — | — | **却下**（159件では過学習） |
| 順位への全面移行 | — | — | — | **撤回**（医用画像で非標準） |

## 5. 未確認事項

- 本件の測定は outer fold 0 のみ（n=1）。設計書は 5 fold × 5 model = 25 run を想定
- 設計書の検証課題1（統合4領域 vs 単一4本）は未着手
- epoch 24 checkpoint での logit 飽和の実測は未実施（best は epoch 4 のみ outer 評価済）
- Gemini による網羅的文献調査は未実施（クォータ回復後に再試行の余地）

## Sources

- [Comparing the Max and Noisy-Or Pooling Functions in MIL](https://arxiv.org/abs/1804.01146)
- [Improving Predictive Confidence in Medical Imaging via Online Label Smoothing](https://arxiv.org/abs/2510.20011)
- [Towards Understanding Why Label Smoothing Degrades Selective Classification](https://arxiv.org/pdf/2403.14715)
- [Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical image analysis](https://arxiv.org/pdf/1804.06353)
- [Label-Efficient Deep Learning in Medical Image Analysis](https://arxiv.org/html/2303.12484v5)
- [GHOST: Adjusting the Decision Threshold to Handle Imbalanced Data](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160)
- [Weakly labeled fine-grained classification with hierarchy relationship of fine and coarse labels](https://www.sciencedirect.com/science/article/abs/pii/S1047320319302056)
