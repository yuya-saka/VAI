"""Label loading utilities for the fracture_detection study.

Two label sources:
- data/rsna_data/train.csv: vertebra-level fracture labels (wide, C1-C7 columns).
- data/rsna_data/fracture_region_labels_dicom.csv: 4-region labels, one row per
  annotation run.

A `run` is NOT a repeated annotation of the same thing. The annotation tool
(Unet/dicom_bbox_annotation_tool) groups each vertebra's fracture bboxes into
runs of contiguous DICOM slices, so two runs on one vertebra are two disjoint
fracture sites, each shown and judged separately (gaps between them are 5-50
slices). The annotator confirmed (2026-08-07) that every run's labels are
correct as recorded.

Therefore the per-vertebra region label is the OR across that vertebra's runs:
a vertebra has a fracture in region r if any of its fracture sites reaches r.
Taking only one run would silently discard a real fracture site.

This yields 268 bags / 160 studies, R1=78 R2=59 R3=72 R4=158. Earlier documents
recorded R1=77 R3=71 R4=155 from a keep-latest-run rule that dropped sites;
those counts are superseded.
"""

from pathlib import Path

import pandas as pd

REGION_COLUMNS = ["region_1", "region_2", "region_3", "region_4"]
LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

EXPECTED_ANNOTATED_BAGS = 268
EXPECTED_ANNOTATED_STUDIES = 160
EXPECTED_REGION_POSITIVES = {
    "region_1": 78,  # vertebral body
    "region_2": 59,  # right transverse foramen
    "region_3": 72,  # left transverse foramen
    "region_4": 158,  # posterior elements
}


def load_region_labels(csv_path: Path) -> pd.DataFrame:
    """Load region labels as one row per vertebra, OR-aggregated over its runs.

    Returns columns: study_id, level, n_runs, region_1..region_4.
    Raises ValueError if the counts differ from the confirmed values, which
    would mean the CSV changed and every downstream number must be re-derived.
    """
    raw = pd.read_csv(csv_path)
    df = raw.groupby(["study_id", "level"], as_index=False).agg(
        n_runs=("run_id", "nunique"),
        region_1=("region_1", "max"),
        region_2=("region_2", "max"),
        region_3=("region_3", "max"),
        region_4=("region_4", "max"),
    )
    df = df.sort_values(["study_id", "level"]).reset_index(drop=True)

    if len(df) != EXPECTED_ANNOTATED_BAGS:
        raise ValueError(f"expected {EXPECTED_ANNOTATED_BAGS} bags, got {len(df)}")
    if df["study_id"].nunique() != EXPECTED_ANNOTATED_STUDIES:
        raise ValueError(
            f"expected {EXPECTED_ANNOTATED_STUDIES} studies, "
            f"got {df['study_id'].nunique()}"
        )
    for column, expected in EXPECTED_REGION_POSITIVES.items():
        actual = int(df[column].sum())
        if actual != expected:
            raise ValueError(f"{column}: expected {expected} positives, got {actual}")
    return df


def load_vertebra_labels(csv_path: Path) -> pd.DataFrame:
    """Load train.csv into long form: study_id, level, fractured (0/1)."""
    wide = pd.read_csv(csv_path)
    long = wide.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=LEVELS,
        var_name="level",
        value_name="fractured",
    )
    long = long.rename(columns={"StudyInstanceUID": "study_id"})
    return long.sort_values(["study_id", "level"]).reset_index(drop=True)
