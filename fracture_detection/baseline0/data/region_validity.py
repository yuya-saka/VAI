"""Partial-annotation coverage and per-region target validity."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from fracture_detection.baseline0.data.constants import (
    ANNOTATION_COMPLETE_COLUMN,
    REGION_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)


def load_annotation_coverage() -> pd.DataFrame:
    """Derive bag-level run completion from the annotation source inventory."""
    annotation_tool = importlib.import_module("Unet.dicom_bbox_annotation_tool.server")
    targets = annotation_tool.augment_with_missing_fractured_levels(
        annotation_tool.build_targets(
            annotation_tool.DEFAULT_BBOX_CSV,
            annotation_tool.DEFAULT_METADATA_DIR,
        ),
        annotation_tool.DEFAULT_BBOX_CSV,
        annotation_tool.DEFAULT_METADATA_DIR,
        annotation_tool.DEFAULT_TRAIN_CSV,
        annotation_tool.DEFAULT_TRAIN_IMAGES_DIR,
        annotation_tool.DEFAULT_SEGMENTATION_DIR,
        annotation_tool.DEFAULT_AUGMENT_CACHE,
    )
    label_keys = set(
        annotation_tool.LabelStore(annotation_tool.DEFAULT_LABEL_CSV).read()
    )
    target_keys = {target.label_key for target in targets}
    unmatched = label_keys - target_keys
    if unmatched:
        raise ValueError(
            f"Region labels do not match annotation targets: {len(unmatched)}"
        )

    annotated_bags = {(study_id, level) for study_id, level, _ in label_keys}
    rows: list[dict[str, Any]] = []
    for study_id, level in sorted(annotated_bags):
        bag_targets = [
            target
            for target in targets
            if target.study_id == study_id and target.level == level
        ]
        annotatable = [target for target in bag_targets if target.rows]
        annotated_runs = sum(target.label_key in label_keys for target in annotatable)
        bbox_missing_runs = sum(not target.rows for target in bag_targets)
        unannotated_runs = len(annotatable) - annotated_runs
        rows.append(
            {
                "study_id": study_id,
                "level": level,
                "expected_annotation_runs": len(annotatable),
                "annotated_runs": annotated_runs,
                "unannotated_runs": unannotated_runs,
                "bbox_missing_runs": bbox_missing_runs,
                ANNOTATION_COMPLETE_COLUMN: (
                    unannotated_runs == 0 and bbox_missing_runs == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_region_target_validity(
    frame: pd.DataFrame,
    annotation_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Mark observed positives and fully reviewed zeros as valid per region."""
    required = {"study_id", "level", "has_region_target", *REGION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"region validity requires columns: {sorted(missing)}")
    coverage_required = {"study_id", "level", ANNOTATION_COMPLETE_COLUMN}
    coverage_missing = coverage_required - set(annotation_coverage.columns)
    if coverage_missing:
        raise ValueError(
            f"annotation coverage requires columns: {sorted(coverage_missing)}"
        )
    if annotation_coverage.duplicated(["study_id", "level"]).any():
        raise ValueError("annotation coverage contains duplicate bags")

    attrs = dict(frame.attrs)
    stale_columns = [
        column
        for column in (ANNOTATION_COMPLETE_COLUMN, *REGION_TARGET_VALID_COLUMNS)
        if column in frame.columns
    ]
    base = frame.drop(columns=stale_columns)
    merged = base.merge(
        annotation_coverage[["study_id", "level", ANNOTATION_COMPLETE_COLUMN]],
        on=["study_id", "level"],
        how="left",
        validate="one_to_one",
    )
    annotated = merged["has_region_target"].astype(bool)
    if merged.loc[annotated, ANNOTATION_COMPLETE_COLUMN].isna().any():
        raise ValueError("Annotation coverage is missing for an annotated bag")
    merged[ANNOTATION_COMPLETE_COLUMN] = (
        merged[ANNOTATION_COMPLETE_COLUMN].astype("boolean").fillna(False).astype(bool)
    )
    for region, validity_column in zip(
        REGION_COLUMNS, REGION_TARGET_VALID_COLUMNS, strict=True
    ):
        merged[validity_column] = (
            merged[region].eq(1) | merged[ANNOTATION_COMPLETE_COLUMN]
        ) & annotated
    merged.attrs = attrs
    return merged
