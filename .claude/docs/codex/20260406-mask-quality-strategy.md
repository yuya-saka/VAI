# Codex Analysis: TotalSegmentator Mask Quality Strategy
Date: 2026-04-06

## Question
In a medical image segmentation pipeline for cervical vertebrae in CT, we generate region masks within vertebra boundaries from TotalSegmentator. Inner region boundaries are from human-annotated polylines using half-plane partition. Problem: TotalSegmentator masks are sometimes distorted.

What strategies best handle distorted TotalSegmentator masks? Evaluate:
1. Quality filtering (detect bad masks automatically)
2. Algorithm robustness improvements
3. Training-time strategies (augmentation, curriculum learning, loss weighting)
4. More direct use of line annotations (reduce reliance on vertebra mask boundary)
5. Post-processing

For each: implementation effort (low/medium/high), expected impact, and risks.
Recommend priority order for a research team with limited time.

## Codex Response

前提を明確にすると、この問題は「inner boundary 推定」の問題というより、「外側 support に使っている TotalSegmentator vertebra mask の upstream noise」が region generation に伝播している問題です。したがって、限られた時間なら「bad mask を見つけて切る」「アルゴリズムが外周歪みに過敏にならないようにする」が最優先です。

| Strategy | 実装工数 | 期待効果 | 主なリスク |
|---|---|---|---|
| 1. Quality filtering | `Low-Medium` | `High`。壊れた pseudo-label を早く除外できるので、最も即効性が高い | 閾値調整が必要。false reject で有効データを捨てる可能性 |
| 2. Algorithm robustness improvements | `Medium` | `High`。mask distortion への依存自体を下げられるので本命 | 境界を無視しすぎると leakage や anatomy drift が出る |
| 3. Training-time strategies | `Medium` | `Medium`。軽度ノイズには効くが、gross failure には弱い | ノイズを「学習で吸収」しようとして逆にバイアスを固定化する |
| 4. More direct use of line annotations | `High` | `Medium-High`。長期的には最も筋が良い可能性 | 研究テーマが一段重くなる。再設計と検証コストが大きい |
| 5. Post-processing | `Low-Medium` | `Low-Medium`。軽い穴・ギザギザ・小リークには有効 | 根本故障を隠すだけになりやすい。トポロジー破綻は直しにくい |

### 各項目の実務評価

**1. Quality filtering**

bad mask 検出は最初に入れるべきです。
見るべき特徴は、`area/extent/solidity/eccentricity`、mask 境界の roughness、隣接 slice との面積・重心・主軸の連続性、そして line annotation との整合性です。特にこの課題では「4 region が非空か」「line endpoint/junction が mask 内の妥当位置にあるか」「region area ratio が破綻していないか」が効きます。
推奨は hard reject だけでなく `confidence score` 化です。除外、低 weight、要再確認を分けられます。

**2. Algorithm robustness improvements**

ここが本命です。
方針は「vertebra mask を正確な boundary と見なさず、粗い outer support としてだけ使う」ことです。具体的には、region partition は line/junction 主体で作り、mask は `crop` と `final clip` 程度に後退させます。さらに raw boundary ではなく、`eroded mask` や interior distance-weighted core 上で region を決めると歪みの影響が減ります。
今の half-plane 系なら、boundary 交点依存を減らし、line から決まる内部分類を優先し、外周近傍は低信頼扱いにするのがよいです。

**3. Training-time strategies**

有効ですが、優先度は 1, 2 より下です。
やるなら、`boundary distortion augmentation`、`sample reweighting by QC score`、`curriculum` が現実的です。最初は high-confidence mask だけで学習し、後から低-confidence を混ぜるのが無難です。loss も boundary 近傍の重みを下げ、line-consistent interior を重くするのが合っています。
ただし、壊れた outer mask が多い場合、augmentation や robust loss だけでは直りません。

**4. More direct use of line annotations**

研究としてはかなり筋が良いです。
理想は、human line annotation から separator/barrier や signed distance map を直接作り、vertebra mask は ROI 制約に留めることです。つまり「region boundary の真の情報源は line」であり、TotalSegmentator は補助に下げます。
ただしこれは pipeline redesign です。短期の安定化より、中期の研究テーマ向けです。

**5. Post-processing**

軽い defect には使えますが、主役ではありません。
`largest connected component`、`hole filling`、軽い `closing/opening`、slice 間平滑化はありです。けれど、mask が大きく食い込む、左右が潰れる、posterior が消える、のような failure は post-process だけでは直しにくいです。

### 限られた時間での優先順位

1. `Quality filtering`
2. `Algorithm robustness improvements`
3. `Training-time strategies` with QC score
4. `Post-processing`
5. `More direct use of line annotations`

補足すると、もしチームの目標が「短期で安定化」なら上の順です。
もし目標が「次の論文化テーマ」なら、`4` を `3` より前に上げてもいいです。

### 短期の具体策

- まず 50-100 例を監査して bad mask taxonomy を作る
- QC score を実装して `exclude / downweight / keep` に分ける
- region generation を `boundary-driven` から `line-driven + conservative mask clip` に寄せる
- その上で confidence-weighted training を入れる

### 参考リンク

- TotalSegmentator 論文: https://pubmed.ncbi.nlm.nih.gov/37795137/
- TotalSegmentator 公式実装: https://github.com/wasserth/TotalSegmentator
- Scribble supervision の医用 segmentation 例: https://conferences.miccai.org/2022/papers/441-Paper0453.html
- 構造的不確実性推定の例: https://pubmed.ncbi.nlm.nih.gov/40311302/
