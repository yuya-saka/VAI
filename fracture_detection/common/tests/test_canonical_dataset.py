from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracture_detection.common.augmentation import build_canonical_augmentation
from fracture_detection.common.canonical_dataset import CanonicalFractureDataset
from fracture_detection.common.constants import (
    REGION_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)
from fracture_detection.common.sampling import SampleIndex


def _augmentation(horizontal_flip_probability: float) -> dict[str, float | int]:
    return {
        "horizontal_flip_probability": horizontal_flip_probability,
        "affine_probability": 0.0,
        "shift_limit": 0.3,
        "scale_lower": 0.7,
        "scale_upper": 1.3,
        "rotate_limit": 45.0,
        "border_mode": 4,
        "brightness_limit": 0.1,
        "contrast_limit": 0.0,
        "intensity_probability": 0.0,
        "blur_noise_probability": 0.0,
        "noise_variance_lower": 3.0,
        "noise_variance_upper": 9.0,
        "distortion_probability": 0.0,
        "cutout_probability": 0.0,
        "cutout_ratio": 0.5,
    }


def _write_bag(root: Path) -> pd.DataFrame:
    bag_dir = root / "study" / "C1"
    bag_dir.mkdir(parents=True)
    ct = np.zeros((15, 5, 224, 224), dtype=np.uint8)
    ct[..., 20:40] = 200
    whole = np.ones((15, 224, 224), dtype=np.uint8)
    region = np.zeros((15, 224, 224), dtype=np.uint8)
    region[..., 10:20] = 2
    region[..., -20:-10] = 3
    np.save(bag_dir / "ct.npy", ct)
    np.save(bag_dir / "vertebra_mask.npy", whole)
    np.save(bag_dir / "region_4class.npy", region)
    row: dict[str, object] = {
        "study_id": "study",
        "level": "C1",
        "fold": 0,
        "vertebra_target": 1,
        "has_region_target": True,
        "annotation_complete": False,
    }
    row.update(dict(zip(REGION_COLUMNS, [0, 1, 0, 1], strict=True)))
    row.update(
        dict(zip(REGION_TARGET_VALID_COLUMNS, [False, True, False, True], strict=True))
    )
    return pd.DataFrame([row])


def test_canonical_dataset_returns_uint8_ten_channel_sample(tmp_path: Path) -> None:
    dataset = CanonicalFractureDataset(_write_bag(tmp_path), dataset_dir=tmp_path)

    sample = dataset[0]

    assert isinstance(sample["inputs"], torch.Tensor)
    assert sample["inputs"].shape == (15, 10, 224, 224)
    assert sample["inputs"].dtype == torch.uint8
    assert sample["region_masks"].shape == (15, 4, 224, 224)


def test_canonical_horizontal_flip_swaps_r2_r3_targets_and_channels(
    tmp_path: Path,
) -> None:
    dataset = CanonicalFractureDataset(
        _write_bag(tmp_path),
        dataset_dir=tmp_path,
        augmentation=build_canonical_augmentation(_augmentation(1.0)),
    )

    sample = dataset[SampleIndex(index=0, epoch=2, ordinal=3)]

    assert torch.equal(sample["region_targets"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.equal(
        sample["region_target_valid"],
        torch.tensor([False, False, True, True]),
    )
    inputs = sample["inputs"]
    assert torch.all(inputs[:, 7, :, 10:20] == 255)
    assert torch.all(inputs[:, 8, :, -20:-10] == 255)


def test_canonical_augmentation_is_repeatable_for_same_sample_index(
    tmp_path: Path,
) -> None:
    dataset = CanonicalFractureDataset(
        _write_bag(tmp_path),
        dataset_dir=tmp_path,
        augmentation=build_canonical_augmentation(_augmentation(0.5)),
    )
    requested = SampleIndex(index=0, epoch=4, ordinal=12)

    first = dataset[requested]
    second = dataset[requested]

    assert torch.equal(first["inputs"], second["inputs"])
    assert torch.equal(first["region_targets"], second["region_targets"])
