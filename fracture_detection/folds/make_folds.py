"""全実験アームで共有する患者単位の層別5-fold割り当て。

割り当て単位はstudyとする（RSNA 2022では1 studyにつき1患者）。同じstudyの全椎体bagを
同一foldへ割り当てる。study単位の件数特徴量に貪欲な反復層別化を適用し、希少な項目
（領域アノテーション陽性、最初はR2）から多い項目へ順に均衡させる。

    region_2、region_3、region_1、region_4のアノテーション陽性bag数、
    アノテーション済みbag数、骨折陽性椎体数、bag数

決定性を保つため、固定SEEDで入力を一度だけシャッフルして同点を解消する。再実行時には
バイト単位で同一のfolds.csvを生成する。foldファイルは研究契約に基づく凍結成果物であり、
実験開始後に再生成すると全比較が無効になるため、outputs/folds.csvは追記専用として扱う。

事前にcheck_dataset.pyの出力（bag_inventory.csv）が存在する必要がある。

実行方法: uv run python fracture_detection/folds/make_folds.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from load_labels import load_region_labels, load_vertebra_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO_ROOT / "data/rsna_data/train.csv"
REGION_CSV = REPO_ROOT / "data/rsna_data/fracture_region_labels_dicom.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
INVENTORY_CSV = OUTPUT_DIR / "bag_inventory.csv"

N_FOLDS = 5
SEED = 20260807
# 均衡化の優先順位は希少なものからとする（陽性59件のR2が最も厳しい制約）。
FEATURE_COLUMNS = [
    "region_2",
    "region_3",
    "region_1",
    "region_4",
    "annotated_bags",
    "positive_vertebrae",
    "bags",
]


BAG_FILE_COLUMNS = ["ct_bytes", "vertebra_mask_bytes", "region_4class_bytes"]


def build_study_features() -> pd.DataFrame:
    """層別化に使う件数特徴量をstudyごとに1行で作成する。

    3ファイルがすべて存在するbagを母集団とする。region_4class.npyだけが欠ける126 bagは
    いずれも領域アノテーション対象外であり、アーム間で母集団を統一するため全アームから除外する。
    """
    inventory = pd.read_csv(INVENTORY_CSV)
    inventory = inventory[(inventory[BAG_FILE_COLUMNS] > 0).all(axis=1)]
    vertebra_df = load_vertebra_labels(TRAIN_CSV)
    region_df = load_region_labels(REGION_CSV)

    bags = inventory.merge(vertebra_df, on=["study_id", "level"], how="left")
    if bags["fractured"].isna().any():
        raise ValueError("train.csvの椎体ラベルがないbagがあります")

    per_study = bags.groupby("study_id").agg(
        bags=("level", "size"),
        positive_vertebrae=("fractured", "sum"),
    )
    region_per_study = region_df.groupby("study_id").agg(
        annotated_bags=("level", "size"),
        region_1=("region_1", "sum"),
        region_2=("region_2", "sum"),
        region_3=("region_3", "sum"),
        region_4=("region_4", "sum"),
    )
    features = per_study.join(region_per_study).fillna(0).astype(int)

    missing_annotated = set(region_df["study_id"]) - set(features.index)
    if missing_annotated:
        raise ValueError(
            f"画像データのないアノテーション済みstudyがあります: {missing_annotated}"
        )
    return features.reset_index()


def assign_folds(features: pd.DataFrame) -> pd.Series:
    """各studyを貪欲法で均衡するfoldへ割り当てる。

    全体での希少度を重みとし、特徴量合計がfoldごとの目標値を最も下回るfoldを選ぶ。
    希少な項目を全foldに余裕があるうちに配置するため、studyは希少度の降順で処理する。
    """
    rng = np.random.default_rng(SEED)
    shuffled = features.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    global_totals = shuffled[FEATURE_COLUMNS].sum().to_numpy(dtype=float)
    weights = 1.0 / np.maximum(global_totals, 1.0)
    target = global_totals / N_FOLDS

    scarcity = shuffled[FEATURE_COLUMNS].to_numpy(dtype=float) @ weights
    order = np.argsort(-scarcity, kind="stable")

    fold_totals = np.zeros((N_FOLDS, len(FEATURE_COLUMNS)))
    fold_study_counts = np.zeros(N_FOLDS)
    assignment = np.full(len(shuffled), -1, dtype=int)
    study_count_target = len(shuffled) / N_FOLDS

    for index in order:
        study_features = shuffled.loc[index, FEATURE_COLUMNS].to_numpy(dtype=float)
        # このstudyを各foldへ加えた場合の、全体の二乗偏差目的関数の限界変化を求める。
        # 他のfoldは変化しないため、限界変化の比較は目的関数全体の比較と等価になる。
        after = ((fold_totals + study_features - target) * weights) ** 2
        before = ((fold_totals - target) * weights) ** 2
        cost = (after - before).sum(axis=1)
        # 小さな同点解消項を加え、studyの実数も均等に保つ。
        cost += 1e-4 * ((fold_study_counts + 1) / study_count_target) ** 2
        best_folds = np.flatnonzero(cost == cost.min())
        fold = int(rng.choice(best_folds))
        assignment[index] = fold
        fold_totals[fold] += study_features
        fold_study_counts[fold] += 1

    return pd.Series(assignment, index=shuffled["study_id"], name="fold")


def fold_report(features: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    merged = features.merge(folds.rename("fold"), on="study_id")
    merged["annotated_studies"] = (merged["annotated_bags"] > 0).astype(int)
    has_foramen_positive = (merged["region_2"] > 0) | (merged["region_3"] > 0)
    merged["foramen_studies"] = has_foramen_positive.astype(int)
    report = merged.groupby("fold").agg(
        studies=("study_id", "size"),
        bags=("bags", "sum"),
        positive_vertebrae=("positive_vertebrae", "sum"),
        annotated_studies=("annotated_studies", "sum"),
        annotated_bags=("annotated_bags", "sum"),
        region_1=("region_1", "sum"),
        region_2=("region_2", "sum"),
        region_3=("region_3", "sum"),
        region_4=("region_4", "sum"),
        foramen_studies=("foramen_studies", "sum"),
    )
    return report


def main() -> None:
    if not INVENTORY_CSV.exists():
        raise SystemExit(
            "先にcheck_dataset.pyを実行してください（bag_inventory.csvが必要です）"
        )
    features = build_study_features()
    folds = assign_folds(features)

    if (folds < 0).any():
        raise ValueError("fold未割り当てのstudyがあります")
    if folds.index.duplicated().any():
        raise ValueError("同じstudyが重複して割り当てられています")

    folds_csv = OUTPUT_DIR / "folds.csv"
    if folds_csv.exists():
        existing = pd.read_csv(folds_csv)
        merged = existing.merge(
            folds.reset_index(), on="study_id", suffixes=("_old", "_new")
        )
        if (
            len(existing) != len(folds)
            or (merged["fold_old"] != merged["fold_new"]).any()
        ):
            raise SystemExit(
                "内容の異なるfolds.csvがすでに存在します。fold定義は凍結済みです。"
                "まだどの実験にも使用していない場合に限り、手動で削除してください"
            )
        print("同一内容のfolds.csvがすでに存在するため、処理は不要です")
        return

    folds.reset_index().sort_values("study_id").to_csv(folds_csv, index=False)
    report = fold_report(features, folds)
    report.to_csv(OUTPUT_DIR / "fold_report.csv")
    meta = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "feature_columns": FEATURE_COLUMNS,
        "n_studies": int(len(features)),
    }
    (OUTPUT_DIR / "folds_meta.json").write_text(json.dumps(meta, indent=2))

    print(report.to_string())
    print(f"\n{folds_csv}へ書き込みました")


if __name__ == "__main__":
    main()
