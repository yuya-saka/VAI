"""Bag dataset for the weak-label region-MIL model.

Returns CT + vertebra mask + the 4-class anatomical region label map, the
whole-bag N/A/U group, and a region target that is only meaningful for the
annotated-positive (A) group. No pseudo-label columns and no legacy
``region_*_target_valid`` / ``annotation_complete`` validity are read (see
``fracture_detection/weak/data/groups.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fracture_detection.baseline0.data.constants import REGION_COLUMNS
from fracture_detection.weak.data_pipeline.augmentation import (
    apply_bag_transform_with_regions,
)
from fracture_detection.weak.data_pipeline.constants import (
    DATASET_DIR,
    EXPECTED_CT_DTYPE,
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
    EXPECTED_REGION_MASK_SHAPE,
    N_REGIONS,
    REGION_MASK_FILENAME,
    REGION_MASK_VALUES,
)
from fracture_detection.weak.data_pipeline.groups import (
    ANNOTATED_GROUP,
    GROUP_COLUMN,
    GROUP_IDS,
    NEGATIVE_GROUP,
    resolve_bag_groups,
)

_REQUIRED_COLUMNS = (
    "study_id",
    "level",
    "fold",
    "vertebra_target",
    "has_region_target",
    *REGION_COLUMNS,
)


class WeakBagDataset(Dataset[dict[str, Any]]):
    """CT, vertebra mask, region label map, and N/A/U group per bag."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        transform: A.ReplayCompose | None = None,
    ) -> None:
        missing = set(_REQUIRED_COLUMNS) - set(manifest.columns)
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        self.manifest = resolve_bag_groups(manifest)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        study_id = str(row["study_id"])
        level = str(row["level"])
        bag_dir = self.dataset_dir / study_id / level
        ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
        whole_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
        region_mask = np.load(bag_dir / REGION_MASK_FILENAME, allow_pickle=False)
        _validate_bag_arrays(ct, whole_mask, region_mask)

        if self.transform is not None:
            ct, whole_mask, region_mask = apply_bag_transform_with_regions(
                ct, whole_mask, region_mask, self.transform
            )

        mask_channel = np.rint(whole_mask * 255.0).astype(np.uint8)[:, None]
        inputs = torch.from_numpy(
            np.ascontiguousarray(np.concatenate([ct, mask_channel], axis=1))
        )
        if inputs.shape[1] != 6:
            raise ValueError(f"weak/入力ch数が不正です: {inputs.shape}")

        bag_group = str(row[GROUP_COLUMN])
        region_target = _resolve_region_target(row, bag_group)

        return {
            "inputs": inputs,
            "region_mask": torch.from_numpy(np.ascontiguousarray(region_mask)),
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "region_target": region_target,
            "group_id": torch.tensor(GROUP_IDS[bag_group], dtype=torch.int64),
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }


def _resolve_region_target(row: pd.Series, bag_group: str) -> Tensor:
    """Return the 4-cell region target: real 0/1 for N/A, unused zeros for U."""
    if bag_group == NEGATIVE_GROUP:
        return torch.zeros(N_REGIONS, dtype=torch.float32)
    if bag_group == ANNOTATED_GROUP:
        values = [float(row[column]) for column in REGION_COLUMNS]
        return torch.tensor(values, dtype=torch.float32)
    # weak_positive: no per-region target exists; losses.py never reads this
    # tensor for the weak-positive group (see compute_weak_losses).
    return torch.zeros(N_REGIONS, dtype=torch.float32)


def _validate_bag_arrays(
    ct: np.ndarray, whole_mask: np.ndarray, region_mask: np.ndarray
) -> None:
    """Validate the fixed CT / vertebra-mask / region-mask bag contract."""
    if ct.shape != EXPECTED_CT_SHAPE or str(ct.dtype) != EXPECTED_CT_DTYPE:
        raise ValueError(f"CT配列が不正です: shape={ct.shape}, dtype={ct.dtype}")
    if whole_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"椎体mask形状が不正です: {whole_mask.shape}")
    if not np.issubdtype(whole_mask.dtype, np.integer) or not np.any(whole_mask):
        raise ValueError("椎体maskは非空の整数配列である必要があります")
    if region_mask.shape != EXPECTED_REGION_MASK_SHAPE:
        raise ValueError(f"4領域mask形状が不正です: {region_mask.shape}")
    region_values = set(np.unique(region_mask).tolist())
    if not region_values.issubset(set(REGION_MASK_VALUES)):
        raise ValueError(f"4領域maskの値が不正です: {sorted(region_values)}")
