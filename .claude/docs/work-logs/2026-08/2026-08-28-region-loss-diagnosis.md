# region_branch val_loss不降下の原因調査と損失再設計の論点整理

作成日: 2026-08-28
状態: **調査完了・設計未確定・実装未着手**

前回`2026-08-27-region-branch-training-debug.md`のouter fold 0学習（epoch 24 early stop,
best epoch 4）を受けた原因調査。実装はまだ何も行っていない。

---

## 1. 完走したrunの実測結果

`fracture_detection/region_branch/outputs/08_26_region_branch_all/test_v1/outer0/`

| epoch | val_total | val_whole | val_human | whole AP | region AP平均 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4090 | 0.3512 | 0.4909 | 0.6854 | 0.8420(最良) |
| 4 (採用) | 0.3933(最良) | 0.3307 | 0.5423 | 0.7016 | 0.7913 |
| 24 (最終) | 0.4234 | 0.3158(最良) | 1.0303 | 0.7188(最良) | 0.7754 |

`fold_metrics.json`（outer評価）: whole AUROC 0.9063 / whole AP 0.7311。

## 2. 確定した事実（測定順）

### 2.1 val_totalが下がらない理由: 3成分の綱引き

`val_total = val_whole + λ(0.5·val_human + 0.5·val_negative)`, λ=0.2053, α=0.5202
（`outputs/calibration/v1/outer0/calibration.json`）。

- val_whole: 0.3512→0.3158 **改善**（rho=-0.612, p=0.0015、epoch24でもまだ最良更新中）
- val_negative: 飽和済み、寄与ほぼゼロ
- **val_human: 0.4909→1.0303 悪化**（rho=+0.953, p=7e-13）。ベースレート予測器0.6123より悪化

反実仮想: region項をepoch1値に凍結した場合、val_totalは0.4090→0.3736（rho=-0.612, p=0.0015で
有意に減少）。**region項の悪化を止めるだけでval_totalは減少方向になる**。region精度を上げる
必要はない。

### 2.2 human BCEが崩壊した機構

- hard 0/1 BCEは有限最適点を持たない（勾配`sigmoid(z)-y`は正解例でもゼロにならない）
- human プール159件がsteps_per_epoch(505)×4/batch=2020抽出 → **12.7周/epoch**、24epochで約305周
- 順位（region AUROC/AP/teacher一致）はこの間**悪化していない**。壊れたのは確信度のみ
  - region AUROC平均: 0.8305→0.8155 (p=0.86 横ばい)
  - region AP平均: 0.8420→0.7754 (rho=-0.522, p=0.009 **低下**)
  - student-teacher rank相関: 0.7467→0.8599 (rho=+0.923, p=1e-10 **上昇**)

### 2.3 train_rankがほぼ動かない理由

- 実際のteacher score・温度で算出したentropy floor=0.6326、trivial baseline(ln2)=0.6931
- 観測値0.7558→0.7052は**trivial baselineより悪いまま**。順位は合っているのに絶対scaleが
  teacherの1.5〜2.6倍に過大（過信）
- soft targetの中央値0.5013、42%が[0.4,0.6]に集中 → 可動域はわずか0.0605
- α=0.5202は勾配ノルム比から逆算された値（`calibration.py`: `ALPHA_TARGET=0.25`）。
  region gradient budget: whole 80% / exact 16% / **rank 4%**（設計上の意図的な劣後）
- ただし train_exact は 0.32→0.06 (80%崩壊) で勾配も縮む一方、train_rankは不動
  → **開始時1:4だった比率が学習中にrank側へドリフト**（未実測、推定）

### 2.4 コードで確認した構造的欠陥

1. **人手GTがranking損失に不参加**: `loaders.py:67`が`attach_teacher_scores`をpseudoのみに
   適用 → `dataset.py:158`でhuman/negativeは`region_scores=NaN` → `scoring.py:165`の
   `eligible = positive & isfinite(scores) & scores>0`で除外。主評価(AP/AUROC=順位)を
   訓練しているのはCAM teacherのみ。
2. **陽性側MIL制約が未使用**: 骨折陽性802 bag（human159+pseudo643）に対し「最低1領域は陽性」
   という論理的に確実な制約が、`losses.py`/`pooling.py`/`trainer.py`のどこにも実装されて
   いない（grep確認済）。100%成立、1bagあたり陽性領域平均1.40/4でスパース。
3. **L_H/L_N=0.5/0.5が任意の切片を作る**: 実効目標率17.5%（母集団0.69%とも人手35.1%とも
   不一致）。F1最適閾値が0.105〜0.636で領域間に6倍の開き、別データに転移しない
   （半分割検証でregion_3/4はむしろ悪化）。

### 2.5 CAM teacherの質は問題ではない（訂正済み）

**当初「teacher対GT AUROC 0.80はin-sampleで楽観的」と主張したが誤り。** 既存の本番監査
（2026-08-23実施、`.claude/docs/work-logs/2026-08/2026-08-23-pseudo-label-mtl-design-review-and-cam-gate-audit.md`）
が268bag×5checkpointでtrain(in-sample) vs outer(完全未見)を直接比較済み。

| 領域 | train AUROC | outer AUROC | 差 | 95%CI |
|---|---:|---:|---:|---|
| R1 | 0.8124 | 0.7980 | +0.0144 | 跨ゼロ |
| R2 | 0.8172 | 0.7856 | +0.0317 | 跨ゼロ |
| R3 | 0.8177 | 0.7882 | +0.0295 | 跨ゼロ |
| R4 | 0.7764 | 0.7356 | +0.0408 | 跨ゼロ |

memorizationゲート完全PASS。領域内Spearman順位相関も全変種で0.80超。**CAM局在信号は健全**。
問題は teacher の質ではなく、ranking損失の運用（2.3節の係数ドリフトと狭いレンジ）にある。

### 2.6 面プーリングは問題ではない（実測で確認・変更不要）

epoch4/epoch24 checkpointから面ごとのregion logitを抽出し、mean/max/LSE/top-kを事後比較
（inner・outerで独立に再現）。

| pooling | inner平均AP | outer平均AP |
|---|---:|---:|
| **mean（現行）** | **0.7902** | **0.7844** |
| topk(k=3) | 0.7631 (-0.027) | 0.7570 (-0.027) |
| lse(τ=1.0) | 0.7626 (-0.028) | 0.7573 (-0.027) |
| max | 0.7528 (-0.037) | 0.7471 (-0.037) |

局在性の直接指標（陽性/陰性セルのmax/mean比）: 陽性1.11〜1.61、陰性2.02〜2.81。
**陽性の方が平坦で陰性の方が尖る**（局在仮説と逆）。Codexの推奨(top-3 mean)も実測で棄却。
mean pooling は変更不要と確定。

### 2.7 データ構造上の追加所見（未検証の設計候補）

- 評価対象268件は**全て骨折陽性**。学習は8,074bag中7,272件が骨折陰性を含む周辺分布を解いて
  いるが、評価は条件付き分布(P(領域|骨折陽性))しか見ていない。ミスマッチの可能性（未検証）。
- R2(右横突孔)/R3(左横突孔)は解剖学的に鏡像構造だが、`region_heads`は4本の完全独立
  `nn.ModuleList`（コード確認済）。R2は陽性36件・AP0.550（4領域中最悪）、R3は陽性44件。
  重み共有すれば有効陽性数36→80。**未実装・効果量未測定**。既存augmentationの
  horizontal flip規約（`dataset.py`docstring: flip時にregion_maskの値は入れ替えない、
  head rは常に解剖学的領域rを予測）はこの共有と矛盾しない。
- level(C1〜C7)ごとに骨折部位の分布が大きく異なる（C2は椎体75%、C4/C6は後方要素73-79%）。
  level単独のOOF予測でmacro AP 0.4965。現行入力(`in_chans=6`)にlevelは含まれない。

## 3. 文献・Codex相談（参照用、詳細は各ファイル）

- 文献調査: `.claude/docs/research/20260827-region-branch-few-label-remedies.md`
- Codex相談全文: `.claude/docs/codex/20260827-2230-region-loss-redesign-f1-prauc.md`
  （`codex exec --model gpt-5.6-sol --sandbox read-only`）

Codexの推奨のうち、面プーリング変更（top-3 mean）は2.6節の実測で否定済み。それ以外の
具体的な修正案は**まだ何も決定していない**。

## 4. 私(Claude)の判断ミスの記録

このセッションで複数回、実測前に断定して後から訂正した。次回セッションで同じ轍を踏まない
よう記録する。

1. 順位への損失全面移行を提案→撤回（医用画像で非標準とユーザー指摘）
2. region teacher作り直しを提案→ユーザーが即却下（159件では過学習、既に実測済みの事実を
   見落としていた）
3. F1@0.5固定は確信度スケールで不変という事実を誤って逆に説明→訂正
4. pooling比較を最初outer foldで実施（locked evaluation汚染）→inner foldで取り直し、
   結論は変わらず再現したが手順は誤り
5. 「評価セットが183→983に5.4倍」と主張→誤り。268bag/983セルは既に5fold OOF結合で
   全部評価に使われている。183は1foldのinner検証数で別物
6. CAM teacherの対GT AUROC 0.80をin-sampleと断定→誤り。2026-08-23の既存監査
   （train vs outer AUROC差<0.05, CI跨ゼロ）を確認せず主張した
7. 「疑似ラベルの局在天井が0.80で固定される」という一般化されたceiling論を主張→
   ユーザー指摘で撤回。Noisy Student等、studentがteacherを超える機構は確立されており、
   根拠のない断定だった
8. Codexが推奨したtop-3 plane poolingを検証せず追認しそうになった→自分で実測し否定

**教訓**: 十分な実測やコード確認をする前に「これが原因」「これで解決する」と断定しない。
特に「既存の監査・設計文書を先に検索する」を怠って車輪の再発明をした（2.5節）。

## 5. 修正の論点（ユーザー確定、2026-08-28）

修正方法は一切未確定。範囲だけ次の3点に確定した。

1. **疑似ラベルの扱い方**
2. **損失の設計**
3. **batchの構成**

具体的な手法・数値・実装方針はまだ何も決めていない。
