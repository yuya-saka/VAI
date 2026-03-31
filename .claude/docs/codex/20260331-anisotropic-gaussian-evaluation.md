# Codex Analysis: Anisotropic Gaussian Proposal Evaluation
Date: 2026-03-31

## Question
Critical evaluation of anisotropic Gaussian GT heatmap proposal for improving angle estimation in U-Net vertebral boundary line detection.

Context:
- U-Net predicts 4-channel heatmaps for vertebral boundary lines in CT images
- Current GT: distance transform from polyline -> exp(-dist^2 / (2*sigma^2))
- Inference: threshold -> moment method (mu20, mu02, mu11) -> principal axis angle
- Current best: angle_error ~6.3 deg, rho_error ~3.5 px (224x224 images)
- Proposal: replace isotropic with anisotropic Gaussian (sigma_parallel >> sigma_perp)

## Codex Response

結論から言うと、あなたの前提評価はかなり正しいです。
「GT heatmap が方向情報を欠いている」が主因、という仮説は **ほぼ誤り** です。異方性Gaussianは主に「新情報の追加」ではなく「学習・推定の重み付け変更」です。

### 1. 提案は新情報を足しているか

ほぼ足していません。現在の
`H(x,y)=exp(-d_line(x,y)^2/(2σ^2))`
は、`d_line` が線への垂直距離なので、GT自体が既に細長い支持領域を持ち、方向は2次モーメントに強く入っています。
理想化すると `λ_parallel ≈ L^2/12`, `λ_perp ≈ σ^2` なので、`L` が十分長ければ異方性は既に大きいです。
異方性Gaussianが変えるのは主に「どの領域をどれだけ強く罰するか（勾配配分）」です。

### 2. モーメント法への効き方

効く可能性はありますが、一次原因でない可能性が高いです。
角度誤差は、GT不足よりも以下で増えやすいです。
- 予測の低強度ノイズ/ブロブ
- 閾値処理依存（特に固定閾値）
- オフセットした漏れ質量の `d^2` 増幅で共分散が回る

異方性を強めると `λ_parallel-λ_perp` を広げて角度安定化する余地はありますが、予測側が崩れていれば改善は限定的です。

### 3. この設定での実際の角度誤差要因（優先度）

| 優先度 | 要因 |
|--------|------|
| 高 | 推定器のロバスト性不足（閾値/ゲーティング、crosstalk、境界切れ） |
| 高 | 学習目標ミスマッチ（MSEは角度最適化を直接しない） |
| 中高 | データ規模（~150例）と注釈ばらつき |
| 中 | 境界近傍で線が切れてモーメントが偏る |
| 中 | モデル容量/表現力（ただし容量増だけで解決しないことが多い） |

### 4. より有効な代替案（優先順）

1. Heatmapは維持しつつ、`angle/rho` の幾何ロスを補助追加（MSE + geometry）
2. 推定後処理を強化（soft-weighted moments、適応ゲート、必要ならRANSAC/ODR）
3. 失敗ケース特化Augmentation（回転、境界付近、低コントラスト、マスク形状ゆらぎ）
4. ハイブリッド化（heatmap + `(phi,rho)` 補助ヘッド）。完全な直接回帰一本化より安全
5. 容量増加は最後（過学習リスクとROIの見極めが必要）

### 5. 異方性GTを実装するなら注意点

- `σ_parallel/σ_perp` の比率チューニングが難しい
- `σ_perp` を小さくしすぎると勾配が細り学習不安定
- polylineの折れ点・端点・境界で局所方向定義が不連続
- train/infer の処理差（閾値や正規化）で逆効果になりやすい
- 線を長手方向に広げる実装はチャネル間crosstalkを増やし得る
- 「異方性のみ変更」の単独アブレーションを必ず取るべき

### 推奨実験設計

`Baseline` / `+anisotropic GTのみ` / `+robust postprocのみ` / `+geometry lossのみ` / `併用` の5条件を同一CVで比較し、平均だけでなく中央値・p90・失敗率（例 `>15°`）を見るのが最短です。
