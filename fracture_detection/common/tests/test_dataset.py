from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracture_detection.common.constants import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
)
from fracture_detection.common.dataset import FractureDataset


def _write_sample(dataset_dir: Path) -> pd.DataFrame:
    bag_dir = dataset_dir / "study-a" / "C1"
    bag_dir.mkdir(parents=True)
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    vertebra_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)

    ct[0, 0, 7, 13] = 255
    vertebra_mask[:, 5:20, 10:30] = 1
    region_mask[0, 7, 13] = 1
    region_mask[0, 8, 14] = 2
    region_mask[0, 9, 15] = 3
    region_mask[0, 10, 16] = 4
    np.save(bag_dir / "ct.npy", ct)
    np.save(bag_dir / "vertebra_mask.npy", vertebra_mask)
    np.save(bag_dir / "region_4class.npy", region_mask)

    return pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "fold": 0,
                "vertebra_target": 1,
                "has_region_target": True,
                "region_1": 1,
                "region_2": 1,
                "region_3": 0,
                "region_4": 0,
            }
        ]
    )


def test_dataset_preserves_orientation_and_builds_ten_channels(
    tmp_path: Path,
) -> None:
    manifest = _write_sample(tmp_path)
    dataset = FractureDataset(manifest, dataset_dir=tmp_path)

    sample = dataset[0]

    assert sample["ct"].shape == (15, 5, 224, 224)
    assert sample["masks"].shape == (15, 5, 224, 224)
    assert sample["ct"][0, 0, 7, 13] == 1.0
    assert sample["ct"][0, 0, 13, 7] == 0.0
    assert sample["masks"][0, 1, 7, 13] == 1.0
    assert sample["masks"][0, 2, 8, 14] == 1.0
    assert sample["masks"][0, 3, 9, 15] == 1.0
    assert sample["masks"][0, 4, 10, 16] == 1.0
    assert torch.equal(sample["region_target_valid"], torch.ones(4, dtype=torch.bool))
