"""Phase 0 data-contract check for fracture_dataset_blind.

Verifies, before any model code exists:
1. Every one of the 268 region-annotated bags loads with the expected shapes
   and dtypes, and gets a sha256 fingerprint (mask/CT version pin).
2. A full inventory of all bags (file presence + sizes, no array loads).
3. Cross-checks against train.csv (studies without image data, and the
   vertebra-label population actually available for training).

Outputs (under fracture_detection/folds/outputs/):
- annotated_bag_manifest.csv  268 rows: labels, shapes, sha256 fingerprints
- bag_inventory.csv           one row per (study, level) directory
- check_report.json           summary counts for the work log

Run: uv run python fracture_detection/folds/check_dataset.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from load_labels import LEVELS, load_region_labels, load_vertebra_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data/rsna_data/fracture_dataset_blind"
TRAIN_CSV = REPO_ROOT / "data/rsna_data/train.csv"
REGION_CSV = REPO_ROOT / "data/rsna_data/fracture_region_labels_dicom.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BAG_FILES = ["ct.npy", "vertebra_mask.npy", "region_4class.npy"]
EXPECTED_CT_SHAPE = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE = (15, 224, 224)
EXPECTED_CT_DTYPE = np.uint8
REGION_MASK_ALLOWED_VALUES = {0, 1, 2, 3, 4}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_dataset() -> pd.DataFrame:
    """List every (study, level) directory with file presence and sizes."""
    rows: list[dict[str, object]] = []
    for study_dir in sorted(DATASET_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        for level in LEVELS:
            bag_dir = study_dir / level
            if not bag_dir.is_dir():
                continue
            row: dict[str, object] = {"study_id": study_dir.name, "level": level}
            for file_name in BAG_FILES:
                file_path = bag_dir / file_name
                key = file_name.removesuffix(".npy")
                row[f"{key}_bytes"] = (
                    file_path.stat().st_size if file_path.exists() else -1
                )
            rows.append(row)
    return pd.DataFrame(rows)


def check_annotated_bag(bag_dir: Path) -> dict[str, object]:
    """Load one annotated bag fully and return shape checks + fingerprints."""
    result: dict[str, object] = {}

    ct = np.load(bag_dir / "ct.npy")
    vertebra_mask = np.load(bag_dir / "vertebra_mask.npy")
    region_mask = np.load(bag_dir / "region_4class.npy")

    result["ct_shape_ok"] = ct.shape == EXPECTED_CT_SHAPE
    result["ct_dtype_ok"] = ct.dtype == EXPECTED_CT_DTYPE
    result["vertebra_mask_shape_ok"] = vertebra_mask.shape == EXPECTED_MASK_SHAPE
    result["region_mask_shape_ok"] = region_mask.shape == EXPECTED_MASK_SHAPE
    result["vertebra_mask_nonzero"] = bool(vertebra_mask.any())
    result["region_values_ok"] = set(np.unique(region_mask)).issubset(
        REGION_MASK_ALLOWED_VALUES
    )
    # Region classes present in this bag (transverse foramina are absent on
    # some planes and occasionally in the whole bag; recorded, not enforced).
    for region_value in (1, 2, 3, 4):
        result[f"region_{region_value}_present"] = bool(
            (region_mask == region_value).any()
        )

    for file_name in BAG_FILES:
        key = file_name.removesuffix(".npy")
        result[f"{key}_sha256"] = sha256_of_file(bag_dir / file_name)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    region_df = load_region_labels(REGION_CSV)
    vertebra_df = load_vertebra_labels(TRAIN_CSV)

    print("== inventory scan ==")
    inventory = inventory_dataset()
    inventory.to_csv(OUTPUT_DIR / "bag_inventory.csv", index=False)
    complete = inventory[
        (inventory[[f"{f.removesuffix('.npy')}_bytes" for f in BAG_FILES]] > 0).all(
            axis=1
        )
    ]
    print(f"bags found: {len(inventory)} (complete: {len(complete)})")
    print(f"studies with data: {inventory['study_id'].nunique()}")

    print("== annotated bag full read check (268 bags) ==")
    manifest_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for row in region_df.itertuples(index=False):
        bag_dir = DATASET_DIR / row.study_id / row.level
        entry: dict[str, object] = {
            "study_id": row.study_id,
            "level": row.level,
            "n_runs": row.n_runs,
            "region_1": row.region_1,
            "region_2": row.region_2,
            "region_3": row.region_3,
            "region_4": row.region_4,
        }
        if not bag_dir.is_dir():
            failures.append(f"missing bag dir: {bag_dir}")
            manifest_rows.append(entry)
            continue
        checks = check_annotated_bag(bag_dir)
        entry.update(checks)
        failed_checks = [k for k, v in checks.items() if k.endswith("_ok") and not v]
        if failed_checks:
            failures.append(f"{row.study_id}/{row.level}: {failed_checks}")
        if not checks["vertebra_mask_nonzero"]:
            failures.append(f"{row.study_id}/{row.level}: empty vertebra mask")
        manifest_rows.append(entry)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "annotated_bag_manifest.csv", index=False)

    print("== cross-checks against train.csv ==")
    studies_with_data = set(inventory["study_id"])
    studies_in_train = set(vertebra_df["study_id"])
    missing_studies = sorted(studies_in_train - studies_with_data)
    extra_studies = sorted(studies_with_data - studies_in_train)

    bag_keys = set(zip(inventory["study_id"], inventory["level"], strict=True))
    available = vertebra_df[
        [
            key in bag_keys
            for key in zip(vertebra_df["study_id"], vertebra_df["level"], strict=True)
        ]
    ]
    annotated_in_train = region_df.merge(
        vertebra_df, on=["study_id", "level"], how="left"
    )
    annotated_not_fractured = annotated_in_train[annotated_in_train["fractured"] != 1]
    if len(annotated_not_fractured):
        failures.append(
            "annotated bags without positive vertebra label: "
            f"{len(annotated_not_fractured)}"
        )

    # Transverse-foramen mask availability vs labels: a bag labeled positive
    # for R2/R3 whose region mask lacks that class cannot be learned from.
    for region_value, column in ((2, "region_2"), (3, "region_3")):
        labeled = manifest[manifest[column] == 1]
        mask_absent = labeled[~labeled[f"region_{region_value}_present"].fillna(False)]
        if len(mask_absent):
            failures.append(
                f"{column} positive but mask class absent: "
                f"{len(mask_absent)} bags: "
                + ", ".join(f"{r.study_id}/{r.level}" for r in mask_absent.itertuples())
            )

    report = {
        "bags_total": int(len(inventory)),
        "bags_complete": int(len(complete)),
        "studies_with_data": int(inventory["study_id"].nunique()),
        "studies_in_train_csv": int(len(studies_in_train)),
        "studies_missing_data": missing_studies,
        "studies_not_in_train_csv": extra_studies,
        "vertebra_labels_available": int(len(available)),
        "vertebra_positives_available": int(available["fractured"].sum()),
        "annotated_bags_checked": int(len(manifest)),
        "failures": failures,
    }
    (OUTPUT_DIR / "check_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
