# region_branch 校正完了・学習デバッグ worklog

作成日: 2026-08-27

`region_branch`の校正artifact整備、学習高速化、起動時の可観測性改善、実GPU学習で発生した
数値不安定の修正まで完了した。統合4領域modelのouter fold 0を実行中だが、epoch 4以降は
train exact lossだけが継続的に低下し、validationはbestを更新していない。現設定のregion
学習率が高い可能性はあるものの、変更方針は未確定。

---

## 1. Calibration v1の完了

fine-tuning構成で全5 outer foldの校正を完了した。

```text
fracture_detection/region_branch/outputs/calibration/v1/
├── outer0/calibration.json
├── outer1/calibration.json
├── outer2/calibration.json
├── outer3/calibration.json
└── outer4/calibration.json
```

統合1本と単一4本の全configは`calibration.version: v1`を参照する。校正artifactには
学習条件のSHA-256 fingerprintが入り、GPU・出力名・`active_regions`以外の学習条件を
変更すると互換性検証で拒否される。

## 2. source packageのrename

repository共通の`.gitignore`にある`data/`規則がPython sourceにも適用されていたため、

```text
fracture_detection/region_branch/data/
```

を次へrenameした。

```text
fracture_detection/region_branch/data_pipeline/
```

全import、README、進捗文書、設計文書を更新した。新package内の7 Python filesは
`data/` ignore規則の対象外になった。

## 3. 学習速度・VRAM改善

natural 16 bagのwhole graphをbackwardして解放してから、補助16 bagのregion graphを
forward/backwardする逐次方式へ変更した。目的関数とoptimizer更新回数は変えていない。

CUDA学習には次を有効化した。

```python
model.compile(mode="default", dynamic=False)
```

RTX A6000での現在の実測は次のとおり。

- steady-state: 約`1.5〜1.55 step/s`
- 505 step/epoch: 約`354〜359秒`（約5.9分）
- VRAM: 約`21 GB / 49 GB`
- 初回epoch: compile時間を含み約`538秒`
- GPU温度: 学習中`87℃`前後（高めなので継続監視が必要）

TF32を有効化できるというTorchInductor warningが出るが、事前benchmarkでは
`torch.set_float32_matmul_precision("high")`による速度改善は測定noise範囲だったため、
現時点では採用していない。

## 4. 学習開始前の進捗ログ

起動後に処理が止まって見えたため、次の各段階へ開始・完了・失敗・経過秒数を追加した。

- config読込と実効config保存
- manifest読込
- fold準備と校正artifact検証
- 学習用、inner、outer、診断DataLoader構築
- Baseline 0 checkpointからのmodel初期化
- `train_fold`開始
- 各epoch開始
- 初回batchのDataLoader worker起動・TorchInductor compile警告

次回起動から、待機時間がどの処理に由来するか標準出力で判別できる。

## 5. epoch 4で発生した非有限region loss

最初の本番runはepoch 3完了後、epoch 4の学習中に次で停止した。

```text
FloatingPointError: weighted region lossが非有限値です
```

### 原因

region forwardはBF16 autocast内で実行されるため、`plane_logits.sigmoid()`もBF16だった。
高い確率と上限`1 - 1e-6`はBF16で厳密な`1.0`へ丸まり、bag集約後の
`torch.logit(1.0)`が`inf`になっていた。epochが進みregion出力が強くなった時点で顕在化した。

### 修正

`region_bag_logits()`のsigmoid、mask平均、clamp、logit変換をfloat32で行うようにした。

```python
probabilities = plane_logits.float().sigmoid()
```

目的関数、校正係数、batch構成は変わらず、BF16の丸めによる`inf`化だけを除去するため、
`calibration/v1`の再校正は不要と判断した。飽和したBF16 logitsでもbag logitsとgradientが
finiteである回帰testを追加した。

## 6. 修正後のouter fold 0学習

実行config:

```text
fracture_detection/region_branch/config/region_branch_all.yaml
```

主な学習条件:

```yaml
pretrained_learning_rate: 0.000023
region_learning_rate: 0.00023
pretrained_min_learning_rate: 0.0000023
region_min_learning_rate: 0.000023
early_stopping_patience: 20
```

出力先:

```text
fracture_detection/region_branch/outputs/08_26_region_branch_all/test_v1/outer0/
```

2026-08-27の記録時点ではepoch 10を実行中。epoch 9までの主要値は次のとおり。

| epoch | train total | train exact | val total | val whole | val exact | whole AUROC | best |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.3822 | 0.3188 | 0.4090 | 0.3512 | 0.2816 | 0.8998 | yes |
| 2 | 0.3707 | 0.2723 | 0.4096 | 0.3501 | 0.2901 | 0.8974 | no |
| 3 | 0.3766 | 0.2426 | 0.4239 | 0.3543 | 0.3390 | 0.8974 | no |
| 4 | 0.3486 | 0.2153 | **0.3933** | **0.3307** | 0.3050 | **0.9041** | **yes** |
| 5 | 0.3368 | 0.1917 | 0.4208 | 0.3479 | 0.3549 | 0.8919 | no |
| 6 | 0.3440 | 0.1718 | 0.4320 | 0.3558 | 0.3712 | 0.9019 | no |
| 7 | 0.3413 | 0.1560 | 0.4010 | 0.3333 | 0.3299 | 0.8960 | no |
| 8 | 0.3267 | 0.1338 | 0.4042 | 0.3274 | 0.3740 | 0.8975 | no |
| 9 | 0.3273 | 0.1228 | 0.4344 | 0.3510 | 0.4064 | 0.8953 | no |

best checkpointはepoch 4の`val_total=0.393305`。その後5 epoch連続でbestを更新しておらず、
`early_stopping_bad_epochs=5`。train exactは`0.3188 → 0.1228`へ一貫して低下する一方、
val exactはepoch 4以降に悪化しており、region側の過学習または更新幅過大が疑われる。
whole側もepoch間の変動が大きく、region学習率だけが原因とはまだ断定していない。

## 7. 現時点の判断と未決事項

現在のrunはbest checkpointを保持したまま継続中。patienceは20なので、best更新がなければ
epoch 24終了時まで学習が続く可能性がある。

次の候補としてregion学習率を半減する案がある。

```yaml
region_learning_rate: 0.000115
region_min_learning_rate: 0.0000115
```

ただし、これはまだ採用決定ではない。学習率は校正fingerprint対象なので、変更する場合は
統合・単一の全5 configを更新し、`calibration.version: v2`として全outer foldを再校正する。
`calibration/v1`を新学習率で流用してはいけない。

## 8. 検証

BF16数値安定化と起動ログ追加後の検証結果:

- region_branch test: **51 passed**
- `ruff check fracture_detection/region_branch`: pass
- `ruff format --check fracture_detection/region_branch`: pass
- 既知warning: synthetic testのconstant inputに対するSciPy `ConstantInputWarning`のみ

## 9. 次にやること

1. 現runを継続するか、epoch 10以降の結果を見て停止するか決める
2. epoch 4のbestを更新できず、train/val exactの乖離が続く場合はregion LR半減案を採否判断する
3. LRを変更する場合は全5 configを`v2`へ更新し、5 outer foldを再校正する
4. v2の統合model outer fold 0で学習曲線を再確認する
5. 統合modelの設定が安定してから残りfoldと単一領域4アームへ進む
