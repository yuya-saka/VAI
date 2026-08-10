"""Patient-level stratified 5-fold assignment, shared by every arm.

Unit of assignment is the study (one patient per study in RSNA 2022); all
vertebra bags of a study land in the same fold. Folds are stratified with a
greedy iterative-stratification pass over per-study count features, ordered so
the scarcest quantities (region-annotated positives, R2 first) are balanced
before the abundant ones:

    region_2, region_3, region_1, region_4 annotated-positive bags,
    annotated bag count, fracture-positive vertebrae, bag count

Determinism: fixed SEED shuffles the input once for tie-breaking; rerunning
produces byte-identical folds.csv. The fold file is a frozen artifact of the
study contract - regenerating it after experiments start invalidates every
comparison, so treat outputs/folds.csv as append-only.

Requires check_dataset.py outputs (bag_inventory.csv) to exist.

Run: uv run python fracture_detection/folds/make_folds.py
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
# Balancing priority: scarcest first (R2 = 59 positives is the binding one).
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
    """One row per study with the count features used for stratification.

    The bag population is bags with all three files present. 126 bags lack
    region_4class.npy only (none of them region-annotated); they are excluded
    from every arm so the population stays identical across arms.
    """
    inventory = pd.read_csv(INVENTORY_CSV)
    inventory = inventory[(inventory[BAG_FILE_COLUMNS] > 0).all(axis=1)]
    vertebra_df = load_vertebra_labels(TRAIN_CSV)
    region_df = load_region_labels(REGION_CSV)

    bags = inventory.merge(vertebra_df, on=["study_id", "level"], how="left")
    if bags["fractured"].isna().any():
        raise ValueError("bag without a train.csv vertebra label")

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
        raise ValueError(f"annotated studies without image data: {missing_annotated}")
    return features.reset_index()


def assign_folds(features: pd.DataFrame) -> pd.Series:
    """Greedy balanced assignment: each study goes to the fold whose feature

    totals are furthest below the per-fold target, weighted by global scarcity.
    Studies are processed in descending scarcity order so rare quantities are
    placed while all folds still have room.
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
        # Marginal change of the global squared-deviation objective if the
        # study joins fold f (other folds are unchanged, so comparing the
        # marginal equals comparing the full objective).
        after = ((fold_totals + study_features - target) * weights) ** 2
        before = ((fold_totals - target) * weights) ** 2
        cost = (after - before).sum(axis=1)
        # Small tie-breaker keeps raw study counts even as well.
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
        raise SystemExit("run check_dataset.py first (needs bag_inventory.csv)")
    features = build_study_features()
    folds = assign_folds(features)

    if (folds < 0).any():
        raise ValueError("unassigned study")
    if folds.index.duplicated().any():
        raise ValueError("study assigned twice")

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
                "folds.csv already exists with different content - the fold "
                "definition is frozen; delete it manually only if no experiment "
                "has used it yet"
            )
        print("folds.csv already exists and is identical; nothing to do")
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
    print(f"\nwrote {folds_csv}")


if __name__ == "__main__":
    main()
