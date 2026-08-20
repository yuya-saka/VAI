"""全アーム共通の10ch canonical dataset。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fracture_detection.common.augmentation import (
    CanonicalAugmentation,
    apply_canonical_augmentation,
    build_uint8_inputs,
)
from fracture_detection.common.constants import (
    DATASET_DIR,
    MANIFEST_COLUMNS,
    N_REGIONS,
    REGION_COLUMNS,
)
from fracture_detection.common.dataset import validate_arrays
from fracture_detection.common.sampling import SampleIndex
from fracture_detection.core.rng import sample_seed


class CanonicalFractureDataset(Dataset[dict[str, Tensor | str]]):
    """CT5 + whole + R1..R4を常に返すcanonical dataset。"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        augmentation: CanonicalAugmentation | None = None,
        *,
        base_seed: int = 20260807,
        outer_fold: int = 0,
        stream: str = "natural",
    ) -> None:
        missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
        if missing:
            raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_dir = dataset_dir
        self.augmentation = augmentation
        self.base_seed = base_seed
        self.outer_fold = outer_fold
        self.stream = stream

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, requested: int | SampleIndex) -> dict[str, Tensor | str]:
        if isinstance(requested, SampleIndex):
            index = requested.index
            epoch = requested.epoch
            ordinal = requested.ordinal
        else:
            index = requested
            epoch = 0
            ordinal = requested
        row = self.manifest.iloc[index]
        study_id = str(row["study_id"])
        level = str(row["level"])
        bag_dir = self.dataset_dir / study_id / level
        ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
        whole_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
        region_mask = np.load(bag_dir / "region_4class.npy", allow_pickle=False)
        validate_arrays(ct, whole_mask, region_mask)
        region_targets = row[list(REGION_COLUMNS)].to_numpy(dtype=np.float32)
        if self.augmentation is not None:
            seed = sample_seed(
                self.base_seed,
                self.outer_fold,
                epoch,
                self.stream,
                ordinal,
            )
            ct, whole_mask, region_mask, region_targets = apply_canonical_augmentation(
                ct,
                whole_mask,
                region_mask,
                region_targets,
                self.augmentation,
                seed,
            )
        inputs, region_masks = build_uint8_inputs(ct, whole_mask, region_mask)
        has_region_target = bool(row["has_region_target"])
        return {
            "inputs": torch.from_numpy(inputs),
            "region_masks": torch.from_numpy(region_masks),
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "region_targets": torch.from_numpy(np.ascontiguousarray(region_targets)),
            "region_target_valid": torch.full(
                (N_REGIONS,), has_region_target, dtype=torch.bool
            ),
            "has_region_target": torch.tensor(has_region_target),
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }
