# 2026-08-18: 実装フェーズへの引き継ぎ

> **設計は凍結済み。未決事項ゼロ。次セッションから実装に入る。**
>
> 設計の経緯は `2026-08-17-fulldata-mtl-redesign.md`（長い）。
> **実装だけなら本ファイルと `memo/計画書/提案手法.md` を読めば足りる。**
> 進捗台帳は `fracture_detection/PROGRESS.md`。
> 旧4アーム計画は `fracture_detection/PROGRESS_ARCHIVE_4arm.md`（**現行に適用しない**）。

---

## 1. 実装する対象（6構成 / 30 run）

| # | 構成 | 入力 | head | whole出力 | β |
|---|---|---|---|---|---|
| 1 | Baseline 0 | 6ch（CT5 + whole mask1） | whole のみ | 直接 | — |
| 2 | Control–B | 6ch | whole + region(4) | 独立head | — |
| 3 | Baseline 1–B | 10ch（CT5 + mask5） | whole + region(4) | 独立head | — |
| 4 | Proposed–B | 10ch | 4 branch + whole | 独立head | >0 |
| 5 | Proposed–max | 10ch | 4 branch | max集約 | >0 |
| 6 | Proposed–max β=0 | 10ch | 4 branch | max集約 | 0 |

- **noisy-OR は実装しない**（全アーム削除済みの決定）
- Control は method B のみ。Control と Baseline 1 の差は**入力チャンネルだけ**にすること

## 2. 学習ループの確定仕様

### fold（凍結 `folds/outputs/folds.csv` を使用。再生成禁止）

```
outer fold = k、inner fold = (k+1) mod 5
  学習: 残り3 fold（8,355〜8,359 bag / annotated 159〜162）でinnerを監視しearly stopping
  推論: そのモデルでouterを1回だけ → pooled OOF
```

- 選択metricは**全アーム共通で inner の椎体AUROC**
- **outer は評価専用**。checkpoint選択・構成選択・ハイパラ調整に一切使わない
- `ReduceLROnPlateau` は inner を監視してそのまま動かす（再fitなし＝LR軌跡の記録・再生は不要）
- 1構成 5 run

### two-stream sampling と損失

```
各 optimizer step:
  W_t = outer-training 全bagからの natural stream（global batch B_W）
  A_t = annotated training bag、1 bag/step 固定（混合比は調整しない）

L_B0          = mean_{W_t} L_whole
L_Control/B1  = mean_{W_t} L_whole + λ·L_region(A_t)
L_Proposed    = mean_{W_t} L_whole + λ·L_region(A_t) + β·mean_{W_t} L_att
```

厳守事項:

- `A_t` は **`L_region` にのみ**寄与。`L_whole` にも `L_att` にも寄与させない
- annotated bag が自然に `W_t` に出たら、通常の頻度で `L_whole` / `L_att` に寄与する
- `L_att` は natural stream 上で計算（maskは全bagにある）
- **Baseline 0 も同一 natural sampler・同一 `W_t`・同一 optimizer step 数**
  （annotated stream の forward をしないだけ。ゼロ損失のstreamを流さない）
- epoch長は **natural stream の一巡**で定義（composite batch sizeで定義しない）
- `L_whole` は常に `B_W` で mean（`B_W+1` で割らない）
- 全アームで**同じ natural-stream seed / 順序**
- annotated sampler は bag 単位の shuffle-without-replacement cycle
- `pos_weight=2.0` は whole loss に全アーム固定
- **椎体陰性への論理的0教師は使わない**（region loss は268のみ）

ログ必須: `region_optimizer_steps` / `region_passes = T / N_annotated_train` /
annotated bag別 visit回数の min・median・max / epoch毎 unique annotated bag数 /
shared CNN block上の `‖∇L_whole‖`・`‖λ∇L_region‖`・`‖β∇L_att‖` とその比

### λ / β 校正（各 outer fold で1回、追加full run 0）

```
1. 共通seedで初期化
2. optimizer更新前に、3 training folds から決定論的に 64 calibration batch
3. eval mode（BN統計もparameterも更新しない）
4. 最後の shared CNN block で、重み付け前の損失ごとの gradient L2 norm（ε=1e-12）

λ_k = clip_[1e-2, 1e2]( 0.5 · exp( median_{b=1..64} log( (g_whole,b + ε)/(g_region,b + ε) ) ) )
      reference = Baseline 1–B
β_k = 同型、reference = Proposed–B、g_att を使用
```

- **同一 λ_k を、その outer fold の全アーム・全構成へ適用。arm別チューニング禁止**
- 同じ β_k を Proposed–B と Proposed–max に使い、ablation だけ β=0
- 偶数個の median は中央2つの log-ratio の平均
- 非有限 gradient が1件でも出たら **run を開始せず implementation error で停止**
- clipping到達はログするが、**結果を見て範囲や重みを変更しない**
- calibration後、parameter・optimizer・BN state が初期状態と一致することを assert

## 3. 既存コードで改修が必要な箇所（実測済み）

| ファイル | 箇所 | 必要な変更 |
|---|---|---|
| `common/losses.py` | 36-37行 `entailed_negative` | **論理的0教師を削除**。region lossは`region_target_valid`のみで判定する |
| `common/metrics.py` | 140行 `side_balanced_accuracy`、192行の返り値dict | **endpointから除外**。領域別APはR1〜R4個別に返す形へ |
| `common/tests/test_metrics.py` | 43行 `side_balanced_accuracy == 1.0` | 該当assertを削除 |
| `common/dataset.py` | `FractureDataset` | 変更不要。two-stream samplerを別途追加 |
| `baseline0/config/schema.py` | `start_outer`/`end_outer`、`early_stopping_patience` | 実装済み。inner foldを実行時に解決し、outerでのearly stoppingを廃止 |
| `baseline0/training/trainer.py` | validation loop | 実装済み。監視対象をinnerへ切替え、outer推論はbest確定後の1回だけ |

新規実装（全アーム共通なので `common/` か実験共通基盤へ）:

- two-stream sampler（natural + annotated 1 bag/step）
- nested選択のfold割当（`inner = (k+1) % 5`、train = 残り3 fold）
- λ/β校正ルーチン
- 床（cross-fitted OOF comparator）

## 4. 実装順序

1. **床とper-region MDEの再計算・凍結**（独立して進められる。他と並行可）
   - 床仕様: outer fold毎に**3 training foldsだけ**から `p̂(r,l) = (x_{r,l}+0.5)/(n_l+1)`
     （Jeffreys平滑化）→ outer bagへlevel別に割当 → 5 foldをpool
   - 評価母集団は **268陽性のみ**（whole-negativeを足さない）
   - **AP実装とtie処理を固定して明記**（level-only scoreは7水準しかなく同点が大量発生し、
     APの値がtie規約に強く依存する）
   - ⚠️ 旧記録値（R1 0.59 / R2 0.37 / R3 0.45 / R4 0.72）は**使わない**。
     暫定測定（in-sample: 0.5303 / 0.3243 / 0.4298 / 0.7259）とも一致しない
   - ⚠️ **既存MDEはmacro-AP基準で失効**。per-region MDEを補正ラベルで再計算する
2. `common/` 改修（上表）
3. nested選択の共通実装
4. λ/β校正の共通実装
5. Baseline 0（`baseline0/`、full設定・V2-S）をnested選択へ改修（実装・検証済み。学習は凍結後）
6. Control–B / Baseline 1–B
7. Proposed 3構成（B β>0 / max β>0 / max β=0）
8. **凍結してから outer 推論**

## 5. やってはいけないこと

- `folds/outputs/folds.csv` の再生成（凍結・上書きガードあり）
- outer fold を checkpoint選択・構成選択・ハイパラ調整に使うこと
- 旧床（0.59/0.37/0.45/0.72）や旧MDE（macro-AP基準）を使うこと
- 全268からfloorを作り同じ268で検定すること（**label leakage**）
- 領域AP評価にwhole-negative bagを足すこと（APが機械的に潰れる）
- λをarm別にチューニングすること（ControlとB1で違えばmask入力だけの比較でなくなる）
- macro-APへ潰すこと（R1〜R4は個別に報告する）
- SideAccを復活させること
- **複数構成の結果を見てから最終構成を選ぶこと**（outer OOFがtuning setになる）
- Codex CLI に `--full-auto` を付けること（read-onlyサンドボックスを上書きしてファイルを書き換える）
- 指示なしのcommit / push

## 6. 検定計画（結果を出す段階で使う）

```
H1: AUROC(Baseline 1–B) > AUROC(Control–B)          ← primary
H2: AUROC(Proposed–max β>0) > AUROC(Proposed–max β=0) ← H1が有意なときだけ確証的

固定順序 H1 → H2。判定は patient-cluster bootstrap の paired差 95%両側CI下限 > 0
endpointは両方とも 13,928 bag の paired pooled-OOF 椎体AUROC
```

key-secondary（確証的順序に入れない）: Control–B vs Baseline 1–B の領域別AP差 / 床ゲートfamily

床ゲート: 対象は **Proposed–B β>0 のみ**。R1〜R4の4検定に **Holm補正**、
10,000 patient-cluster bootstrap、model と floor に同一resample、bootstrap内で再fitしない。

感度解析2件（記述的、新endpointにしない）:
- level removal check（各level内でscoreをpercentile rank化して4つのAPを再計算）
- **R2/R3 swap negative control**（R2ラベルをR3 scoreで評価。
  **失敗したらlateralityの主張は無効**）

## 7. 登録済みの限界（報告時に必ず守る）

1. 結論は**登録済み60% protocol下の相対比較**。通常訓練での順位は主張しない
2. 領域AP・床は**annotated-positive population（非ランダム選択）に条件づけた内部妥当性**のみ
3. 1 seed / outer fold なので、CIに **training stochasticity は含まれない**
4. R2/R3 APが高くても「症例ごとの左右を正しく選んだ」証明にはならない
5. floorは cross-fitted OOF 必須
6. 6構成・重み・検定順序・code hash を**凍結してから outer 推論を開始**

## 8. 参照

- 設計本体: `memo/計画書/提案手法.md`
- 進捗台帳: `fracture_detection/PROGRESS.md`
- Codex回答（fold/test設計）: `.claude/docs/codex/20260817-fold-and-testset-design.md`
- Codex回答（未決4点・検定計画・λ校正・構成削減）: `.claude/docs/codex/20260818-remaining-four-decisions.md`
- 設計決定の記録: `.claude/docs/DESIGN.md`（Architecture の Active セクションと changelog）
