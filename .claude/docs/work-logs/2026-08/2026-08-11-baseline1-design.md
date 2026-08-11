# 2026-08-11 Baseline 1 設計確定

> 前段: `2026-08-11-fracture-common-and-baseline-plan.md`（Phase 1 共通基盤完了）
> Codex 相談全文: `.claude/docs/codex/20260811-2100-baseline1-design.md`

## 0. セッション状態

- 状態: **Baseline 1 設計確定・実装未着手**
- 未確定: 近道の床と検出力の再計算（旧ラベル基準のまま。事前登録前に必須）

---

## 1. 本日のユーザー決定

| # | 論点 | 決定 |
|---|---|---|
| 1 | matched の陰性268 bag の抽出元 | **骨折なし患者のみ**から取る |
| 2 | bag 確率の作り方 | **旧方式を維持**（15面 broadcast + 面ごとBCE + mean-sigmoid） |
| 3 | モデル選択規則 | **旧方式**（val AUROC で early stopping） |
| 4 | matched の backbone | **B0 を主解析・V2-S を感度分析**、full は V2-S |
| 5 | fold 分割 | **現状の凍結5-fold のまま**（held-out test は作らない） |
| 6 | Codex が無断編集した DESIGN.md | 該当セクションと changelog 1行を**削除**（実施済み） |

決定2・3 は Codex 推奨（bag-level log-mean-exp / 固定epoch+EMA）を**却下**したもの。
利点・欠点を提示した上での選択であり、下記「受容した限界」に登録して進める。

---

## 2. 事実確認の結果

### 2.1 陰性プール（骨折なし患者のみ）

| 項目 | 実測 |
|---|---|
| 骨折なし study | 1,063 患者 / 7,375 bag |
| アノテ160患者との重複 | 0 |
| fold×level 全35セルの充足 | 全て可能・最小マージン 195 患者 |
| fold別 必要患者数 / 利用可能 | 53〜56 / 211〜214 |

1患者1椎体・fold別×C1〜C7別の件数一致を満たしたまま268患者を選べる。

### 2.2 matched の fold 別内訳

| fold | train bag (陽性) | val bag (陽性) | step/epoch @bs16 |
|---:|---:|---:|---:|
| 0 | 424 (212) | 112 (56) | 27 |
| 1-4 | 430 (215) | 106 (53) | 27 |

### 2.3 旧 Stage1 との差分（0.921 が再現目標にならない理由）

| 項目 | 旧 Stage1 `v1_parity` | 新 Baseline 1 |
|---|---|---|
| 学習母集団 | 10,730 bag（+ test 2,703 分離） | `full` 13,928 / `matched` 536 |
| fold | seed 42・test分離あり | seed 20260807 の凍結5-fold・test分離なし |
| 陽性率 | 10.0% | `full` 10.1% / `matched` 50.0% |
| 損失 | pos_weight 2.0 付き BCE | **通常BCE**（重み禁止） |
| augmentation | hflip/vflip/transpose あり | **全て禁止**、回転も45°→±10-12° |
| OOF AUROC | 0.9209 (CI 0.910–0.931) | — |

**0.921 は歴史的参照値であり、成功条件にしない。**

### 2.4 母数の不整合を修正

`PROGRESS.md` が確証的評価の母数として 14,133 / 陽性1,444 を引いていたが、
凍結manifestは **13,928 bag / 陽性1,406**。修正済み。検出力の再計算は未処理。

---

## 3. Baseline 1 確定仕様

### 3.1 入力

- CT 5ch + 椎体全体mask 1ch = **6ch**、15面、224×224
- `common/dataset.py` の `FractureDataset` を包み、mask 5ch のうち全体maskのみ使う
- flip / transpose / 左右入れ替えなし。`p_rand_order = 0`

### 3.2 モデル

```
(bs, 15, 6, 224, 224)
  → timm backbone (in_chans=6)         # 面をバッチ次元へ展開
  → (bs, 15, hdim)
  → BiLSTM (hidden 256, 2層, bidirectional)
  → per-plane head (Linear→BN→Dropout→LeakyReLU→Linear)
  → (bs, 15) plane logits
```

特徴次元は `timm` から動的に取得する（旧実装のハードコード表は移植しない）。

### 3.3 損失と bag 確率（決定2）

```python
labels = y_bag.unsqueeze(1).expand(-1, 15)         # 15面へ複製
loss   = BCEWithLogitsLoss()(plane_logits, labels)  # 重みなし
p_bag  = torch.sigmoid(plane_logits).mean(dim=1)    # 推論
```

`pos_weight` / focal / class-balanced sampling は**使わない**（全アーム共通の制約）。

### 3.4 学習設定

| 項目 | `matched` | `full` |
|---|---|---|
| backbone（主） | `tf_efficientnetv2_b0` | `tf_efficientnetv2_s` |
| backbone（感度） | `tf_efficientnetv2_s` | — |
| batch | 16 | 16 |
| optimizer | AdamW, wd 1e-4（bias/norm除外） | 同左 |
| LR | 3段階（下記） | 2.3e-4 → cosine → 2.3e-5 |
| max epoch | 200 | 75 |
| early stopping | val AUROC, patience 30, min_epoch 20 | val AUROC, patience 15 |
| drop_rate / drop_path / head dropout | 0.10 / 0.10 / 0.40 | 0.0 / 0.0 / 0.3 |
| grad clip | 1.0 | 1.0 |
| AMP | bf16 | bf16 |
| mixup | 0（OFF） | 0（OFF） |
| EMA | なし | なし |

`matched` の3段階スケジュール（430 bag に対する過学習抑制）:

- epoch 1-2: backbone 凍結。head LR を 1e-4 → 1e-3 に線形warmup
- epoch 3-10: backbone 凍結のまま head LR 1e-3。凍結中は backbone の BatchNorm を eval に固定
- epoch 11-15: unfreeze。backbone 3e-6 → 3e-5、head 3e-5 → 3e-4 へwarmup
- epoch 16-200: cosine decay（backbone 3e-6、head 3e-5 まで）

early stopping は `min_epoch 20`（unfreeze 完了後）から数え始める。

### 3.5 augmentation

15面 × 5CTch を1回のtransformで処理し、面間で幾何パラメータを共有する
（旧 `_augment_volume` のチャンネル積み上げパターンを踏襲）。

| | `full` | `matched` |
|---:|---:|---:|
| Affine 確率 | 0.50 | 0.70 |
| shift | ±5% | ±7% |
| scale | 0.90–1.10 | 0.88–1.12 |
| **rotate** | **±40°** | **±40°** |
| brightness / contrast | ±0.10 / ±0.10, p=0.30 | ±0.12 / ±0.15, p=0.40 |
| blur / noise OneOf | p=0.15 | p=0.20 |
| flip / transpose / distortion / cutout / mixup | OFF | OFF |

- CT は bilinear、mask は nearest-neighbor + 二値化
- 強度変換は CT のみ。mask には適用しない
- **回転は ±40°（2026-08-11ユーザー決定）**。Codexは±10-12°を推奨したが不採用
- **境界は constant fill**（CT背景値 / mask 0）。旧実装の `BORDER_REFLECT_101` は使わない。
  40°回転では反射境界が四隅へ**鏡像の解剖を作り込む**ため、
  「反転augmentationを使わない」という全アーム共通の制約と衝突する

### 3.6 評価

- 5-fold OOF を pool してから算出（fold を5反復として扱わない）
- `common/metrics.py: evaluate_prediction_frame` を使用
- 患者 cluster bootstrap で95%区間
- B1 vs B2 の椎体レベル比較は、B2 の4領域スコアを**事前登録した固定規則**で
  椎体スコアへ融合し、paired 差 + 患者 cluster bootstrap で評価する

---

## 4. 受容した限界（事前登録に記載する）

1. **15面 broadcast + mean-sigmoid**: 学習と推論が別の関数を最適化しており、
   対応する単一Bernoulli尤度が存在しない。陽性bagでも骨折が写らない面を陽性と強制する。
   提案Bの集約設計とは非整合になるため、提案B実装時に
   「B1/B2 と提案B で bag 確率の定義が異なる」ことを明示する必要がある
2. **val AUROC による early stopping**: 評価に使う fold で checkpoint を選ぶため
   OOF は楽観側に歪む。特に `matched` は val 106 bag（53陽性）で AUROC の標準誤差が
   約 0.03–0.04 あり、winner's curse が乗る
3. **held-out test なし**。5-fold OOF のみ。268 という領域評価の母数を削れないため
4. **`matched` は人工的な50%有病率**。キャリブレーション・PPV・一般集団への外挿は主張しない
5. **`matched`(B0) と `full`(V2-S) は backbone が異なる**ため、両者の差を
   データ量の効果として読めない。`matched` の V2-S 感度分析で部分的に補う
6. **15面は椎体SI方向の中央値77.5%（p5 62.5%）しか覆わない**。
   陽性bagの一部は骨折の証拠を含まない可能性がある（label/observation mismatch）
7. **患者 bootstrap は学習seedの分散を含まない**。seed選択は行わない

---

## 5. 実装計画

### Phase 2-1: matched cohort の固定

作成先: `fracture_detection/cohorts/`

- `make_matched_cohort.py`
  - 入力: `common/outputs/input_manifest.csv`
  - アノテ268 bag をそのまま採用
  - 骨折なし患者1,063から陰性268 bag を決定的に選択（seed固定）
  - assert: 1患者1椎体 / fold×level 件数一致 / アノテ患者との重複0 / 骨折なし患者由来
  - 出力: `outputs/matched_cohort.csv`（536行）+ `matched_cohort_meta.json`（SHA256）
  - `folds.csv` と同様の**上書きガード**を実装し、生成後は凍結
- `tests/`: 上記assertの単体テスト、B1/B2が同一cohortを読むテスト

### Phase 2-2: Baseline 1 実装

作成先: `fracture_detection/baseline1/`

| ファイル | 内容 |
|---|---|
| `model.py` | timm backbone + BiLSTM + per-plane head |
| `dataset.py` | `common.FractureDataset` を包み6ch化 + augmentation |
| `trainer.py` | 3段階LR / early stopping / bf16 / grad clip |
| `staging.py` | `full` 用のローカルコピー |
| `train.py` | CLI。`data.mode: matched \| full` を必須設定に |
| `config/matched_b0.yaml` `config/matched_s.yaml` `config/full_s.yaml` | 実験設定 |
| `README.md` | **モデル構成・入出力・損失・学習設定・実行方法** |
| `tests/` | 単体テスト |

出力は `outputs/{mode}_{backbone}/fold{N}/` に分離する。

### README.md 運用ルール（2026-08-11ユーザー決定・全プロジェクト共通）

- `fracture_detection/` 配下の**各プロジェクトに `README.md` を必ず置く**
- 記載するもの: モデル構成（層・次元・集約規則）、入出力の形状と意味、損失、学習設定、
  データ設定、実行方法、出力物
- **仕様を変更したら同じコミットで README.md も更新する**。設計変更が入るたびに必ず追随させる

### Phase 2-3: 学習実行

| 実験 | 概算コスト |
|---|---|
| `matched` B0（主） | 5 fold で数時間 |
| `matched` V2-S（感度） | 5 fold で数時間 |
| `full` V2-S | 5 fold で 25–30 GPU時間（A6000 3枚でfold並列なら約1/3） |

**ステージングは `full` のみ**（2026-08-11ユーザー決定）。NFS 直読みは 105 ms/bag（50 MB/s）で、
13,928 bag では毎epoch 73 GB を読むことになるため。`/dev/shm` に126 GB空きあり。

`matched` はステージングしない。536 bag = 2.8 GB は初回epoch後に
OSのページキャッシュへ丸ごと乗るので、実効I/Oコストはほぼ消える。

### Phase 2-4: OOF 評価

- `common/metrics.py` で椎体AUROC / AP / 患者cluster bootstrap CI
- 結果を `PROGRESS.md` と DESIGN.md へ記録

### 別タスク（Baseline 1 と並行可）

- 近道の床（R1/R2/R3/R4）と検出力の**補正ラベルでの再計算**。事前登録ゲート固定前に必須

---

## 6. 維持する不変条件

- fold seed `20260807`、`folds.csv` は凍結
- 15面固定、入力は統合済み `fracture_dataset/`
- flip / transpose なし
- `common/` にモデルを置かない
- 通常BCEのみ（pos_weight / focal / balanced sampling 禁止）
- matched cohort の exact ID を B1/B2 で一致させる
- 領域APへ追加陰性を混ぜない
- 提案Aの teacher / pseudo-label は outer fold 内で完結
- Codex 呼び出しから `--full-auto` を外す（`--sandbox read-only` を上書きするため）
