# line-only 実装方針まとめ

## 方針
まずは **line-only のまま**進める。  
やることは以下の順。

1. **評価指標を見直す**
2. **現状の MSE ベースラインを再現**
3. **角度損失を追加**
4. **ρ損失を追加**

現状の基本構成は、  
- 入力: **CT + 椎体マスク**
- 出力: **4ch の境界ヒートマップ**
- 推論: **モーメント法で重心・主軸を求めて直線化**  
である。 :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## 1. 固定する前提
- 課題は **4本の境界線検出**
- モデル構造は **当面そのまま**
- 出力は **4ch heatmap**
- 推論時の直線復元も **現状どおり**でよい

---

## 2. GT直線の定義
アノテーションは実質 **2点を結ぶ直線** とみなして扱う。

### GT の作り方
- アノテーションの **始点・終点の2点** を使う
- この2点から GT 直線を定義する

### 直線表現
以下の法線形式を使う。

\[
x\cos\phi + y\sin\phi - \rho = 0
\]

### 実装上のルール
- 原点は **画像中心**
- \(\rho\) は **画像対角長で正規化**
- GT と pred で **同じ座標系** を使う

---

## 3. 学習の段階
### Phase 1: ベースライン
まずは現状のまま。

\[
L = L_{\text{heatmap}}
\]

- ヒートマップ回帰のみ
- まずはこれで再学習・再評価する

---

### Phase 2: 角度損失を追加
\[
L
=
L_{\text{heatmap}}
+
w(t)\lambda_\theta \left(1-|\cos(\phi_{\text{pred}}-\phi_{\text{gt}})|\right)
\]

- 最初に追加するのは **角度損失のみ**
- 初期学習では不安定なので、**warmup後に重みを徐々に上げる**
- いきなり大きくしない

---

### Phase 3: ρ損失を追加
\[
L
=
L_{\text{heatmap}}
+
w(t)\left[
\lambda_\theta \left(1-|\cos(\phi_{\text{pred}}-\phi_{\text{gt}})|\right)
+
\lambda_\rho \,\mathrm{SmoothL1}\left(\frac{\rho_{\text{pred}}-\rho_{\text{gt}}}{D}\right)
\right]
\]

- \(D\): 画像対角長
- まずは **angle を主**
- **ρ は弱め** に入れる

---

## 4. pred 側の \((\phi,\rho)\) の出し方
予測ヒートマップから以下を計算する。

1. **重み付き重心**
2. **重み付き共分散**
3. **主軸**
4. 主軸から法線ベクトル \(n=(\cos\phi,\sin\phi)\) を作る
5. \(\rho = n^\top c\) で求める

ここで \(c\) は重心。

---

## 5. 実装時の注意
### 符号合わせが必要
同じ直線でも

- \((\phi,\rho)\)
- \((\phi+\pi,-\rho)\)

は同じ意味になる。

そのため、loss 計算前に向きをそろえる。

### 例
- 法線ベクトルの内積を見る
- 内積が負なら
  - \(n_{\text{pred}} \leftarrow -n_{\text{pred}}\)
  - \(\rho_{\text{pred}} \leftarrow -\rho_{\text{pred}}\)

としてから比較する

### その他
- 共分散計算には **eps** を入れる
- 学習初期は line loss を **0 か極小**
- GT / pred の座標系は必ず一致させる

---

## 6. 評価指標
現状は以下が使われている。 :contentReference[oaicite:2]{index=2}

- MSE
- Peak distance
- Success@10px
- Line centroid distance
- Line angle difference

ただし、line-only の主評価としては、  
**Peak distance / Success@10px は外す候補**。

### 主評価として使うもの
#### 1. 角度誤差
\[
\Delta\theta
=
\arccos\left(|\cos(\phi_{\text{pred}}-\phi_{\text{gt}})|\right)
\]

#### 2. 法線方向距離
\[
\Delta\rho
=
|\rho_{\text{pred}}-\rho_{\text{gt}}|
\]

#### 3. GT線分から予測線への平均垂線距離
- GT線分上を複数点サンプリング
- 各点から予測線への距離を計算
- その平均を指標にする

### 副指標
- MSE は残してよい

---

## 7. 比較実験
最低限、以下の3条件で比較する。

1. **Baseline**
   - MSE のみ

2. **+ Angle loss**
   - MSE + angle loss

3. **+ Angle loss + rho loss**
   - MSE + angle loss + rho loss

---

## 8. 実装優先順位
1. **評価コードを書き換える**
2. **現状 MSE で再評価**
3. **angle loss を追加**
4. **rho loss を追加**
5. **重みと warmup を調整**

---

## 一言でまとめると
- **モデルは今のまま**
- **評価を線の幾何量ベースに変更**
- **損失は angle から段階的に追加**
- **最初は line-only をきちんと成立させる**