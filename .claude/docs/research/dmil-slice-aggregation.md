# 参考論文: スライスレベル弱教師あり集約 (DMIL + center loss)

出典: Hibi et al. 2023, "Automated screening of computed tomography using weakly
supervised anomaly detection", IJCARS. (memo/research_paper/脳CTからのスライスレベルの弱教師あり学習.pdf)
GitHub: https://github.com/hibiat/wsad

## 課題設定
- scan-level (= 本研究では vertebra-level) のラベルのみで学習し、slice-level の異常スコアを出力。
- 動画異常検知 AR-Net (Wan et al. 2020) を CT に応用。
- 論文は 2-stage: 凍結 pretrained CNN/ViT で特徴抽出 → 小さな AR-Net (3 conv + avgpool + 1 FC) → sigmoid。
- 本研究は end-to-end を採用 (損失だけ借用)。

## 記法
- scan x_i, slice 数 t_i, slice j の異常スコア s_i^j ∈ [0,1]
- scan ラベル y_i ∈ {0,1} (0=正常, 1=異常)

## DMIL loss (Dynamic Multiple-Instance Learning)
- 動的 top-k: k_i = ceil(t_i / α)   (α はハイパラ, 論文 α=2 → 上位50%)
- 降順ソートして上位 k_i スコア tilde_s_i を選択
- L_DMIL = -(1/k_i) Σ_{tilde_s_i^j} { y_i log(tilde_s_i^j) + (1-y_i) log(1-tilde_s_i^j) }
- 陽性 bag: 上位 k_i を 1 に近づける (最も異常らしい slice を引き上げ)
- 陰性 bag: 上位 k_i を 0 に近づける (最も異常らしく見える slice すら抑制 → 全体が下がる)
- → slice ラベル不要で scan ラベルのみから slice 教師を生成。

## center loss (陰性 bag のみ)
- L_c = (1/t_i) Σ_j || s_i^j - c_i ||^2   (y_i = 0 のときのみ)
- c_i = 正常 scan の異常スコアの中心 (mean)
- 全 slice を 0 に強制せず中心へ集める「緩和版」→ specificity 改善

## 総損失
- 論文: L = λ L_DMIL + L_c   (λ=512)
- 本研究の再パラメータ化案: L = L_DMIL + β L_c  (β を小さく, 推奨 0.5 付近)

## 論文のハイパラ・結果
- α=2, λ=512, 200 epoch, batch 60, lr 1e-4, wd 5e-3
- 特徴抽出: ResNet50 / ViT-large penultimate (凍結)
- RSNA brain hemorrhage: scan-AUC 0.92 (ViT), slice-AUC 0.89。教師量 97.1% 削減。
- COVID-CTset (258 scan の小規模) でも >90% 異常 scan 同定 → 小規模データに頑健と主張。

## 本研究への適用 (確定事項)
- scan = vertebra bag, slice = axial slice (dataset_zprop)。
- end-to-end 2.5D CNN を per-slice scorer にして DMIL+center loss を乗せる。
- α は要調整: 陽性椎体の陽性 slice 比率は高い (66-89%) ので α=2 (上位50%) を default、{2,3,4} を sweep。
- center loss は陰性椎体 222 bag に効く想定 (specificity 向上)。
- slice ラベル (fracture.nii.gz, slice_XXX 番号=z-index で整合) は学習に不要、評価 (slice-AUC) に使用。
