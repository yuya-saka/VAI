# GT優先統一target・conditional region loss実装とv3校正

作成日: 2026-09-03  
状態: **実装・疑似ラベル再生成・outer fold 0校正まで完了。学習本体は未開始**

参照する正本:

- `.claude/docs/DESIGN.md`
- `.claude/docs/REGION_MODEL_DESIGN_JA.md`
- `fracture_detection/region_branch/config/region_branch_all.yaml`

---

## 1. 今回の目的

旧`cam_soft`学習では、regionの学習損失がhard GTとpseudo targetで別々に正規化され、
inner validationにはpseudo targetが付与されていなかった。そのため、学習した混合目的と
checkpoint選択時の目的が一致せず、`val_pseudo_loss=0`のまま少数GTのmacro APだけで
region checkpointを選択していた。

今回、GTとpseudo targetを単一target tensorへ統合し、学習とinner validationで同一の
conditional-positive objectiveを使うprotocol v5へ変更した。

## 2. 実装した契約

### 2.1 GT優先の単一region target

datasetは各bagについて次を返す。

- `region_target`: human GTがvalidならhard 0/1、unknownならCAM soft target `q`
- `region_target_valid`: 統一targetが有効なcell
- `region_hard_valid`: human GTまたはwhole-negativeの論理0
- `region_pseudo_valid`: CAM soft targetを使用したcell

同じcellにGTとpseudo targetが存在する場合は、必ずGTを優先する。whole-negativeは
whole分類では使用するが、conditional region headの損失からは除外する。

### 2.2 Conditional-positive region loss

region headは`vertebra_target == 1`のbagだけで学習する。各region内でvalid cellの
plain `BCEWithLogits`を平均し、有効な4 regionをmacro平均する。region用の
`pos_weight`、GT/pseudo別係数、focal loss、source別正規化は使用しない。

```text
L_region = macro_r mean_valid_positive_bags BCEWithLogits(z_r, t_r)
```

学習勾配には通常のsoft BCEを使う。validation監視値はtarget entropyを引いた
centered lossとする。

```text
L_centered = BCEWithLogits(z, t) - H(t)
```

hard GTでは`H(t)=0`なので通常のBCEと一致する。soft targetでは予測`p=t`のとき
0となる。entropy項はモデル出力に依存しないため、学習勾配は通常BCEと同一である。

### 2.3 Validationとcheckpoint

- inner validationにもleakage-safeなpseudo targetを付与する。
- region checkpointは`val_region_centered_loss`最小で選択し、early stoppingにも使う。
- hard GTのregion AP/AUROCは別の診断指標として記録する。
- whole checkpointは従来どおりwhole validation lossで独立選択する。
- outer推論のend-to-end region scoreは、best wholeとbest regionを独立ロードし、
  `p(whole) * p(region | whole-positive)`で計算する。

### 2.4 Leakage-safe pseudo targetのfold対応

`pseudo_region_targets.csv`は`student_outer_fold`と`teacher_outer_fold`を持つ。

- studentのtraining subset: `teacher_outer_fold == student_outer_fold`
- studentのinner validation subset:
  `teacher_outer_fold == (student_outer_fold + 1) % 5`
- outer testにはpseudo targetを付与しない。

protocolは`region-branch-v5`、校正artifact versionは`v3`とした。

## 3. 疑似ラベル再生成

実行コマンド:

```bash
uv run python -m fracture_detection.baseline0.cli.generate_pseudo_labels \
  --overwrite \
  --device cuda:1
```

生成結果:

- 5教師、4-view TTAで全foldを再推論
- 5,328 rows
- 1,332 unique bags
- training用: 3,996 rows
- inner validation用: 1,332 rows
- `student_outer_fold`/`teacher_outer_fold`の許可済み対応を全行で確認
- student/teacher foldはいずれも0〜4を包含

成果物:

- `fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_region_targets.csv`
- `fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_target_calibration.csv`
- `fracture_detection/baseline0/outputs/08_19/pseudo_labels/pseudo_target_generation_metadata.json`

fold別shared calibration結果:

| student outer fold | slope | intercept | OOF AP | OOF Brier | OOF log loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.4759 | 1.0880 | 0.7082 | 0.1662 | 0.5113 |
| 1 | 1.5185 | 1.0721 | 0.7154 | 0.1594 | 0.4914 |
| 2 | 2.0502 | 1.6083 | 0.7307 | 0.1550 | 0.4702 |
| 3 | 2.4383 | 1.9760 | 0.7249 | 0.1595 | 0.4851 |
| 4 | 1.6478 | 1.2209 | 0.6637 | 0.1825 | 0.5491 |

## 4. outer fold 0のv3 lambda校正

実行コマンド:

```bash
uv run python -m fracture_detection.region_branch.cli.calibrate \
  --config fracture_detection/region_branch/config/region_branch_all.yaml \
  --outer-fold 0 \
  --gpu-id 1
```

結果:

- `lambda = 0.3156107231711445`
- raw gradient-norm ratio median: `0.2330486476271456`
- clip: `false`
- 有効校正batch: 46（region valid cellなしのbatchはskip）
- seed: `20260807`
- config fingerprint:
  `95e30a55ac0361942953b5857ce6b2a37a442b5dc8a62b2379dacc720f9e9178`
- 初期化元Baseline 0 outer0 checkpoint SHA-256:
  `779da18f02489fedbe127d18dd829d47318a85f34825a53430571f44cbdd4702`

成果物:

- `fracture_detection/region_branch/outputs/calibration/v3/outer0/calibration.json`
- `fracture_detection/region_branch/outputs/calibration/v3/outer0/initialization.json`

## 5. 検証状況

実装後の全体検証:

```bash
uv run ruff check fracture_detection/baseline0 fracture_detection/region_branch
uv run ruff format --check fracture_detection/baseline0 fracture_detection/region_branch
uv run pytest -q fracture_detection/baseline0/tests fracture_detection/region_branch/tests
```

- 245 tests passed
- warningはsynthetic test入力に対する既知の`ConstantInputWarning`のみ

最終のvalid-cell skip修正後のfocused validation:

```bash
uv run pytest -q \
  fracture_detection/region_branch/tests/test_losses.py \
  fracture_detection/region_branch/tests/test_calibration.py \
  fracture_detection/region_branch/tests/test_trainer.py
```

- 28 tests passed
- `ruff check`、`ruff format --check`、`git diff --check`も通過

## 6. 次の実行

現在のconfigは`09_03_region_branch_all/test_v3`、GPU 1、outer fold 0のみを対象とする。
学習CLIには`--outer-fold`引数はなく、foldを上書きする場合は
`--start-outer-fold`と`--end-outer-fold`を使う。

```bash
uv run python -m fracture_detection.region_branch.cli.train \
  --config fracture_detection/region_branch/config/region_branch_all.yaml \
  --start-outer-fold 0 \
  --end-outer-fold 0 \
  --gpu-id 1
```

config自体にもouter fold 0とGPU 1が設定済みなので、次でも同じ実行になる。

```bash
uv run python -m fracture_detection.region_branch.cli.train \
  --config fracture_detection/region_branch/config/region_branch_all.yaml
```

学習本体は本ログ作成時点では未開始である。
