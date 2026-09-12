"""N/A/U bag group resolution and inventory validation.

Group definitions come from fracture_detection/REGION_MIL_DESIGN.md section 4:

- N (negative): ``vertebra_target == 0``. All four regions are negative by
  the logical definition of the whole label; no annotation column is used.
- A (annotated positive): ``vertebra_target == 1`` and
  ``has_region_target == True``. All four ``region_1..region_4`` cells are
  valid targets and 0 always means "no fracture" -- never "unknown". The
  legacy ``region_*_target_valid`` / ``annotation_complete`` columns handled
  by ``baseline0/data/region_validity.py`` describe a different (superseded)
  contract and are intentionally not read here.
- U (weak positive): ``vertebra_target == 1`` and ``has_region_target`` is
  not True. Only the bag-level "at least one region is positive" fact is
  used; no per-region target is attached.

An all-zero A bag or a negative bag carrying ``has_region_target=True`` both
contradict the label contract above and are treated as fatal input errors
rather than silently coerced, per the design doc's "input contradiction"
requirement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from fracture_detection.baseline0.data.constants import REGION_COLUMNS

NEGATIVE_GROUP = "negative"
ANNOTATED_GROUP = "annotated_positive"
WEAK_GROUP = "weak_positive"
BAG_GROUPS = (NEGATIVE_GROUP, ANNOTATED_GROUP, WEAK_GROUP)

# Integer encoding stored in the dataset's `group_id` tensor, in the same
# order as BAG_GROUPS.
GROUP_IDS = {NEGATIVE_GROUP: 0, ANNOTATED_GROUP: 1, WEAK_GROUP: 2}
_GROUP_NAMES_BY_ID = {value: key for key, value in GROUP_IDS.items()}


def group_name_by_id(group_id: int) -> str:
    """Reverse lookup from the integer group_id tensor value to its name."""
    if group_id not in _GROUP_NAMES_BY_ID:
        raise ValueError(f"未知のgroup_idです: {group_id}")
    return _GROUP_NAMES_BY_ID[group_id]


GROUP_COLUMN = "bag_group"

_REQUIRED_COLUMNS = {"study_id", "level", "vertebra_target", "has_region_target"}


def resolve_bag_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``bag_group`` column and validate the N/A/U label contract.

    Raises ``ValueError`` if a negative bag carries ``has_region_target=True``
    or an annotated positive bag has all four region cells at 0 (both
    contradict the whole label).
    """
    missing = (_REQUIRED_COLUMNS | set(REGION_COLUMNS)) - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")

    resolved = manifest.reset_index(drop=True).copy()
    is_positive = resolved["vertebra_target"].astype(float) == 1.0
    has_region_target = resolved["has_region_target"].astype(bool)

    negative_with_region_target = (~is_positive) & has_region_target
    if negative_with_region_target.any():
        offending = resolved.loc[
            negative_with_region_target, ["study_id", "level"]
        ].to_dict("records")
        raise ValueError(
            f"whole-negative bags must not carry has_region_target=True: {offending}"
        )

    region_cells = resolved[list(REGION_COLUMNS)].astype(float)
    all_zero_annotated = (
        is_positive & has_region_target & (region_cells.sum(axis=1) == 0)
    )
    if all_zero_annotated.any():
        offending = resolved.loc[all_zero_annotated, ["study_id", "level"]].to_dict(
            "records"
        )
        raise ValueError(
            "annotated positive bags with all-zero region cells contradict "
            f"the whole-positive label: {offending}"
        )

    group = pd.Series(WEAK_GROUP, index=resolved.index, dtype=object)
    group[~is_positive] = NEGATIVE_GROUP
    group[is_positive & has_region_target] = ANNOTATED_GROUP
    resolved[GROUP_COLUMN] = group
    return resolved


@dataclass(frozen=True)
class GroupInventory:
    """Per-manifest-slice N/A/U bag and cell counts, saved before training."""

    n_negative: int
    n_annotated: int
    n_weak: int
    n_annotated_cells: int
    n_annotated_positive_cells: int

    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


def build_group_inventory(manifest: pd.DataFrame) -> GroupInventory:
    """Summarize N/A/U counts for a manifest slice (typically train-only)."""
    resolved = resolve_bag_groups(manifest)
    counts = resolved[GROUP_COLUMN].value_counts()
    annotated = resolved[resolved[GROUP_COLUMN] == ANNOTATED_GROUP]
    annotated_cells = annotated[list(REGION_COLUMNS)].astype(float).to_numpy()
    return GroupInventory(
        n_negative=int(counts.get(NEGATIVE_GROUP, 0)),
        n_annotated=int(counts.get(ANNOTATED_GROUP, 0)),
        n_weak=int(counts.get(WEAK_GROUP, 0)),
        n_annotated_cells=int(annotated_cells.size),
        n_annotated_positive_cells=int(annotated_cells.sum()),
    )


def group_row_indices(resolved_manifest: pd.DataFrame) -> dict[str, list[int]]:
    """Map each bag group to the 0-based row positions of a resolved manifest.

    ``resolved_manifest`` must already have a ``bag_group`` column (from
    ``resolve_bag_groups``) and be reset to a contiguous 0..N-1 index -- these
    positions are exactly the indices a matching ``Dataset.__getitem__``
    accepts.
    """
    if GROUP_COLUMN not in resolved_manifest.columns:
        raise ValueError(f"manifest is missing the {GROUP_COLUMN!r} column")
    return {
        group: resolved_manifest.index[
            resolved_manifest[GROUP_COLUMN] == group
        ].tolist()
        for group in BAG_GROUPS
    }
