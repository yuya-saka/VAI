# 固定matchedコホート

`make_matched_cohort.py`は、Baseline 1（`matched`設定）とBaseline 2で共有する2,655 bagのコホートを固定します。領域アノテーション済みの陽性bagを268件すべて保持し、固定済み入力マニフェストから陰性bagを2,387件選択します。

## 選択の契約

- 選択する陰性bagは、固定済み入力マニフェストで`vertebra_target == 0`である必要があります。
- コホート全体の陽性率を`full`の陽性率10.095%とほぼ一致させます。
- 陰性bagの各`fold` × 頸椎`level`件数は、`full`の陰性分布に比例させます。
- 同一studyの複数椎体を許可し、患者構成を人工的な1患者1椎体へ変えません。
- 選択にはseed `20260807`と決定的なハッシュによる同点解消を使用します。

## 出力

- `outputs/matched_cohort.csv`：共通マニフェストの列に`cohort_role`を追加した表です。
- `outputs/matched_cohort_meta.json`：入力SHA256値、コホートSHA256、seed、件数、陽性率を記録します。

出力は固定済みです。バイト列が完全に同じ内容での再実行は許可しますが、異なる内容の場合は失敗させ、削除前に手動で調査する必要があります。

## 実行方法

```bash
uv run python -m fracture_detection.cohorts.make_matched_cohort --dry-run
uv run python -m fracture_detection.cohorts.make_matched_cohort
```
