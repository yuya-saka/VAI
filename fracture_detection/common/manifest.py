"""凍結foldとラベルから全アーム共通manifestを構築する。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from fracture_detection.common.constants import (
    BAG_FILE_COLUMNS,
    EXCLUDED_LEVELS_CSV,
    EXCLUDED_STUDIES_CSV,
    FOLDS_CSV,
    INVENTORY_CSV,
    MANIFEST_COLUMNS,
    REGION_COLUMNS,
    REGION_CSV,
    REGION_TARGET_VALID_COLUMNS,
    SUPERVISED_MANIFEST_COLUMNS,
    TRAIN_CSV,
)
from fracture_detection.common.region_validity import (
    attach_region_target_validity,
    load_annotation_coverage,
)
from fracture_detection.folds.load_labels import (
    load_region_labels,
    load_vertebra_labels,
)


def _assert_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{name}に重複キーがあります: {columns}")


def assemble_manifest(
    inventory: pd.DataFrame,
    vertebra_labels: pd.DataFrame,
    region_labels: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """読み込み済み表を結合し、1行1椎体の共通manifestを返す。"""
    required_inventory = {"study_id", "level", *BAG_FILE_COLUMNS}
    if not required_inventory.issubset(inventory.columns):
        missing = sorted(required_inventory - set(inventory.columns))
        raise ValueError(f"棚卸し表に必要な列がありません: {missing}")

    complete = inventory[(inventory[list(BAG_FILE_COLUMNS)] > 0).all(axis=1)][
        ["study_id", "level"]
    ].copy()
    complete["study_id"] = complete["study_id"].astype(str)
    complete["level"] = complete["level"].astype(str)

    vertebra_labels = vertebra_labels.copy()
    region_labels = region_labels.copy()
    folds = folds.copy()
    for frame in (vertebra_labels, region_labels, folds):
        frame["study_id"] = frame["study_id"].astype(str)

    _assert_unique(complete, ["study_id", "level"], "完備bag一覧")
    _assert_unique(vertebra_labels, ["study_id", "level"], "椎体ラベル")
    _assert_unique(region_labels, ["study_id", "level"], "領域ラベル")
    _assert_unique(folds, ["study_id"], "fold一覧")

    manifest = complete.merge(
        vertebra_labels,
        on=["study_id", "level"],
        how="left",
        validate="one_to_one",
    )
    if manifest["fractured"].isna().any():
        raise ValueError("椎体骨折ラベルのない完備bagがあります")

    manifest = manifest.merge(
        folds[["study_id", "fold"]],
        on="study_id",
        how="left",
        validate="many_to_one",
    )
    if manifest["fold"].isna().any():
        raise ValueError("fold未割り当ての完備bagがあります")

    manifest = manifest.merge(
        region_labels[["study_id", "level", *REGION_COLUMNS]],
        on=["study_id", "level"],
        how="left",
        validate="one_to_one",
        indicator="region_merge",
    )
    manifest["has_region_target"] = manifest["region_merge"].eq("both")
    manifest = manifest.drop(columns="region_merge")
    manifest[list(REGION_COLUMNS)] = manifest[list(REGION_COLUMNS)].fillna(0)
    manifest = manifest.rename(columns={"fractured": "vertebra_target"})

    integer_columns = ["fold", "vertebra_target", *REGION_COLUMNS]
    manifest[integer_columns] = manifest[integer_columns].astype(int)
    manifest = manifest[list(MANIFEST_COLUMNS)]
    return manifest.sort_values(["study_id", "level"]).reset_index(drop=True)


def apply_quality_exclusions(
    manifest: pd.DataFrame,
    excluded_studies: pd.DataFrame,
    excluded_levels: pd.DataFrame,
) -> pd.DataFrame:
    """Stage1と同じ品質除外リストをmanifestへ適用する。"""
    required_study_columns = {"study_uid"}
    required_level_columns = {"study_uid", "vertebra"}
    if not required_study_columns.issubset(excluded_studies.columns):
        raise ValueError("excluded_studies.csvにstudy_uid列がありません")
    if not required_level_columns.issubset(excluded_levels.columns):
        raise ValueError("excluded_levels.csvにstudy_uid/vertebra列がありません")

    study_ids = set(excluded_studies["study_uid"].astype(str))
    level_keys = set(
        zip(
            excluded_levels["study_uid"].astype(str),
            excluded_levels["vertebra"].astype(str),
            strict=True,
        )
    )
    study_mask = manifest["study_id"].astype(str).isin(study_ids)
    level_mask = pd.Series(
        [
            (str(study_id), str(level)) in level_keys
            for study_id, level in zip(
                manifest["study_id"], manifest["level"], strict=True
            )
        ],
        index=manifest.index,
    )
    removed = study_mask | level_mask
    filtered = manifest.loc[~removed].reset_index(drop=True)
    filtered.attrs["quality_exclusions"] = {
        "rows_before": int(len(manifest)),
        "rows_removed": int(removed.sum()),
        "rows_after": int(len(filtered)),
        "listed_studies": int(len(study_ids)),
        "listed_levels": int(len(level_keys)),
    }
    return filtered


def build_manifest(
    inventory_csv: Path = INVENTORY_CSV,
    train_csv: Path = TRAIN_CSV,
    region_csv: Path = REGION_CSV,
    folds_csv: Path = FOLDS_CSV,
    excluded_studies_csv: Path = EXCLUDED_STUDIES_CSV,
    excluded_levels_csv: Path = EXCLUDED_LEVELS_CSV,
    annotation_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """確定済み入力ファイルから共通manifestを構築する。"""
    inventory = pd.read_csv(inventory_csv, dtype={"study_id": str, "level": str})
    vertebra_labels = load_vertebra_labels(train_csv)
    region_labels = load_region_labels(region_csv)
    folds = pd.read_csv(folds_csv, dtype={"study_id": str})
    excluded_studies = pd.read_csv(excluded_studies_csv, dtype={"study_uid": str})
    excluded_levels = pd.read_csv(
        excluded_levels_csv, dtype={"study_uid": str, "vertebra": str}
    )
    manifest = assemble_manifest(inventory, vertebra_labels, region_labels, folds)
    filtered = apply_quality_exclusions(manifest, excluded_studies, excluded_levels)
    coverage = (
        load_annotation_coverage()
        if annotation_coverage is None
        else annotation_coverage
    )
    supervised = attach_region_target_validity(filtered, coverage)
    return supervised[list(SUPERVISED_MANIFEST_COLUMNS)]


def write_manifest(manifest: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """manifestとSHA256を含むメタデータを安全に書き出す。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.csv"
    temporary_path = output_dir / "input_manifest.csv.tmp"
    manifest.to_csv(temporary_path, index=False)
    temporary_path.replace(manifest_path)

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "rows": int(len(manifest)),
        "studies": int(manifest["study_id"].nunique()),
        "region_annotated_rows": int(manifest["has_region_target"].sum()),
        "region_complete_rows": int(manifest["annotation_complete"].sum()),
        "region_valid_cells": {
            column: int(manifest[column].sum())
            for column in REGION_TARGET_VALID_COLUMNS
        },
        "sha256": digest,
        "quality_exclusions": manifest.attrs.get("quality_exclusions"),
        "excluded_studies_sha256": hashlib.sha256(
            EXCLUDED_STUDIES_CSV.read_bytes()
        ).hexdigest(),
        "excluded_levels_sha256": hashlib.sha256(
            EXCLUDED_LEVELS_CSV.read_bytes()
        ).hexdigest(),
    }
    metadata_path = output_dir / "input_manifest_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path, metadata_path


def main() -> None:
    """既定パスから共通manifestを生成する。"""
    output_dir = Path(__file__).resolve().parent / "outputs"
    manifest = build_manifest()
    manifest_path, metadata_path = write_manifest(manifest, output_dir)
    print(f"共通manifestを書き込みました: {manifest_path}")
    print(f"メタデータを書き込みました: {metadata_path}")


if __name__ == "__main__":
    main()
