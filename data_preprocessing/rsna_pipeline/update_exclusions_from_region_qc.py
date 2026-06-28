"""Update RSNA vertebra-level exclusions from reference-based region QC."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
RSNA_DATA_DIR: Final = PROJECT_ROOT / "data" / "rsna_data"
DEFAULT_QC_SCORES: Final = (
    PROJECT_ROOT / "Unet" / "outputs" / "sdf_qc_reference_based" / "level_scores.csv"
)
DEFAULT_BBOX_CSV: Final = RSNA_DATA_DIR / "train_bounding_boxes.csv"
DEFAULT_EXCLUDED_STUDIES_CSV: Final = RSNA_DATA_DIR / "excluded_studies.csv"
DEFAULT_EXCLUDED_LEVELS_CSV: Final = RSNA_DATA_DIR / "excluded_levels.csv"
EXCLUSION_REASON: Final = "sdf_region_qc_outlier_no_bbox"
STUDY_FIELDNAMES: Final = ("study_uid", "reason", "detail")
LEVEL_FIELDNAMES: Final = (
    "study_uid",
    "vertebra",
    "component_count",
    "volume_mm3",
    "volume_z_score",
    "tilt_magnitude_deg",
    "reason",
)


def load_bbox_study_ids(path: Path) -> set[str]:
    """Return study IDs with at least one bounding-box annotation."""
    with path.open(encoding="utf-8", newline="") as file:
        return {
            row["StudyInstanceUID"]
            for row in csv.DictReader(file)
            if row.get("StudyInstanceUID")
        }


def load_failed_levels(path: Path) -> set[tuple[str, str]]:
    """Return failed (study ID, vertebra) pairs."""
    with path.open(encoding="utf-8", newline="") as file:
        return {
            (row["study_id"], row["vertebra"])
            for row in csv.DictReader(file)
            if row.get("passed") == "0"
            and row.get("study_id")
            and row.get("vertebra")
        }


def load_rows(path: Path, fieldnames: tuple[str, ...]) -> list[dict[str, str]]:
    """Load and validate an exclusion CSV."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != fieldnames:
            raise ValueError(f"Unexpected exclusion CSV fields: {reader.fieldnames}")
        return [
            {field: row.get(field, "") for field in fieldnames}
            for row in reader
            if row.get("study_uid")
        ]


def remove_qc_study_exclusions(
    existing_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Remove the superseded study-level QC exclusions."""
    retained_rows = [
        row for row in existing_rows if row["reason"] != EXCLUSION_REASON
    ]
    return retained_rows, len(existing_rows) - len(retained_rows)


def update_level_exclusions(
    existing_rows: list[dict[str, str]],
    failed_levels: set[tuple[str, str]],
    bbox_study_ids: set[str],
) -> tuple[list[dict[str, str]], int]:
    """Append failed bbox-negative vertebra levels, preserving existing rows."""
    existing_keys = {
        (row["study_uid"], row["vertebra"])
        for row in existing_rows
    }
    new_keys = sorted(
        (study_id, vertebra)
        for study_id, vertebra in failed_levels
        if study_id not in bbox_study_ids
        and (study_id, vertebra) not in existing_keys
    )
    new_rows = [
        {
            "study_uid": study_id,
            "vertebra": vertebra,
            "component_count": "",
            "volume_mm3": "",
            "volume_z_score": "",
            "tilt_magnitude_deg": "",
            "reason": EXCLUSION_REASON,
        }
        for study_id, vertebra in new_keys
    ]
    return [*existing_rows, *new_rows], len(new_rows)


def write_rows_atomic(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: tuple[str, ...],
) -> None:
    """Write the exclusion CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exclude bbox-negative reference-QC failures by vertebra level."
    )
    parser.add_argument("--qc-scores", type=Path, default=DEFAULT_QC_SCORES)
    parser.add_argument("--bbox-csv", type=Path, default=DEFAULT_BBOX_CSV)
    parser.add_argument(
        "--excluded-studies-csv",
        type=Path,
        default=DEFAULT_EXCLUDED_STUDIES_CSV,
    )
    parser.add_argument(
        "--excluded-levels-csv",
        type=Path,
        default=DEFAULT_EXCLUDED_LEVELS_CSV,
    )
    arguments = parser.parse_args()

    study_rows = load_rows(arguments.excluded_studies_csv, STUDY_FIELDNAMES)
    level_rows = load_rows(arguments.excluded_levels_csv, LEVEL_FIELDNAMES)
    retained_study_rows, removed_study_count = remove_qc_study_exclusions(study_rows)
    failed_levels = load_failed_levels(arguments.qc_scores)
    bbox_study_ids = load_bbox_study_ids(arguments.bbox_csv)
    updated_level_rows, added_level_count = update_level_exclusions(
        level_rows,
        failed_levels,
        bbox_study_ids,
    )
    write_rows_atomic(
        arguments.excluded_studies_csv,
        retained_study_rows,
        STUDY_FIELDNAMES,
    )
    write_rows_atomic(
        arguments.excluded_levels_csv,
        updated_level_rows,
        LEVEL_FIELDNAMES,
    )
    print(
        f"[DONE] removed_studies={removed_study_count} "
        f"added_levels={added_level_count} "
        f"study_total={len(retained_study_rows)} "
        f"level_total={len(updated_level_rows)}"
    )


if __name__ == "__main__":
    main()
