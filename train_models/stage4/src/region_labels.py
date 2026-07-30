"""Load and classify Stage4 region supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RegionLabel = NDArray[np.int8]
RegionLabelMap = dict[tuple[str, str], RegionLabel]
RegionSupervision = Literal["strong", "weak", "negative"]

REGION_COLUMNS = ("region_1", "region_2", "region_3", "region_4")


def load_region_labels(csv_path: Path) -> RegionLabelMap:
    """Return OR-aggregated R1-R4 labels keyed by study and vertebra level."""
    frame = pd.read_csv(csv_path, dtype={"study_id": str, "level": str})
    required = {"study_id", "level", *REGION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"region label CSV is missing columns: {sorted(missing)}")
    values = frame.loc[:, REGION_COLUMNS].to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("region labels must be binary")

    labels: RegionLabelMap = {}
    for (study_id, level), group in frame.groupby(
        ["study_id", "level"], sort=False, dropna=False
    ):
        key = (str(study_id), str(level))
        labels[key] = group.loc[:, REGION_COLUMNS].max(axis=0).to_numpy(dtype=np.int8)
    return labels


def region_supervision_of(
    label: int,
    key: tuple[str, str],
    region_labels: RegionLabelMap,
) -> RegionSupervision:
    """Classify a bag as negative, strong positive, or weak positive."""
    if label == 0:
        return "negative"
    if label != 1:
        raise ValueError(f"vertebra label must be binary, got {label}")
    return "strong" if key in region_labels else "weak"
