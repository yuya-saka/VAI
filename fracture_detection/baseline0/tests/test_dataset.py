from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracture_detection.baseline0.data.dataset import (
    Baseline0Dataset,
    apply_bag_transform,
    augment_from_config,
    default_augmentation,
)
from fracture_detection.common.constants import EXPECTED_CT_SHAPE, EXPECTED_MASK_SHAPE


def _write_bag(dataset_dir: Path) -> pd.DataFrame:
    bag_dir = dataset_dir / "study-a" / "C1"
    bag_dir.mkdir(parents=True)
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    vertebra_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    ct[:, :, 20, 40] = 255
    vertebra_mask[:, 10:50, 30:70] = 1
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
                "has_region_target": False,
                "region_1": 0,
                "region_2": 0,
                "region_3": 0,
                "region_4": 0,
            }
        ]
    )


def test_baseline0_dataset_returns_ct_and_whole_mask_only(tmp_path: Path) -> None:
    manifest = _write_bag(tmp_path)
    dataset = Baseline0Dataset(manifest, dataset_dir=tmp_path)

    sample = dataset[0]

    assert sample["inputs"].shape == (15, 6, 224, 224)
    assert sample["inputs"].dtype == torch.uint8
    assert sample["inputs"][0, 0, 20, 40] == 255
    assert sample["inputs"][0, 5, 20, 40] == 255
    assert sample["inputs"][0, 5, 40, 20] == 0
    assert sample["vertebra_target"] == torch.tensor(1.0)


def test_apply_bag_transform_keeps_mask_binary_and_geometry_shared() -> None:
    ct = np.zeros((15, 5, 32, 32), dtype=np.float32)
    whole_mask = np.zeros((15, 32, 32), dtype=np.float32)
    ct[:, :, 8, 12] = 1.0
    whole_mask[:, 8, 12] = 1.0
    transform = augment_from_config(
        {
            "affine_probability": 1.0,
            "shift_limit": 0.0,
            "scale_lower": 1.0,
            "scale_upper": 1.0,
            "rotate_limit": 0.0,
            "brightness_limit": 0.0,
            "contrast_limit": 0.0,
            "intensity_probability": 0.0,
            "blur_noise_probability": 0.0,
            "distortion_probability": 0.0,
            "cutout_probability": 0.0,
        },
    )

    augmented_ct, augmented_mask = apply_bag_transform(ct, whole_mask, transform)

    assert np.array_equal(augmented_ct[:, 0], augmented_ct[:, 4])
    assert np.array_equal(augmented_mask, whole_mask)
    assert set(np.unique(augmented_mask)) == {0.0, 1.0}


def test_apply_bag_transform_calls_augmentation_once_for_whole_bag() -> None:
    class RecordingTransform:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def __call__(
            self, *, image: np.ndarray, mask: np.ndarray
        ) -> dict[str, np.ndarray]:
            self.calls.append((image.shape, mask.shape))
            return {"image": image, "mask": mask}

    ct = np.zeros((15, 5, 32, 32), dtype=np.float32)
    whole_mask = np.zeros((15, 32, 32), dtype=np.float32)
    transform = RecordingTransform()

    augmented_ct, augmented_mask = apply_bag_transform(  # type: ignore[arg-type]
        ct, whole_mask, transform
    )

    assert transform.calls == [((32, 32, 75), (32, 32, 15))]
    assert augmented_ct.shape == ct.shape
    assert augmented_mask.shape == whole_mask.shape


def test_apply_bag_transform_preserves_uint8_range() -> None:
    ct = np.full((15, 5, 32, 32), 128, dtype=np.uint8)
    whole_mask = np.ones((15, 32, 32), dtype=np.uint8)
    transform = augment_from_config(
        {
            "affine_probability": 0.0,
            "intensity_probability": 0.0,
            "blur_noise_probability": 0.0,
            "distortion_probability": 0.0,
            "cutout_probability": 0.0,
        }
    )

    augmented_ct, augmented_mask = apply_bag_transform(ct, whole_mask, transform)

    assert augmented_ct.dtype == np.uint8
    assert int(augmented_ct.min()) == 128
    assert int(augmented_ct.max()) == 128
    assert set(np.unique(augmented_mask)) == {1.0}


def test_default_augmentation_matches_stage1_except_orientation_changes() -> None:
    values = default_augmentation()
    transform = augment_from_config()
    transform_names = [item.__class__.__name__ for item in transform.transforms]

    assert values == {
        "affine_probability": 0.7,
        "shift_limit": 0.3,
        "scale_lower": 0.7,
        "scale_upper": 1.3,
        "rotate_limit": 45.0,
        "border_mode": 4.0,
        "brightness_limit": 0.1,
        "contrast_limit": 0.0,
        "intensity_probability": 0.7,
        "blur_noise_probability": 0.5,
        "noise_variance_lower": 3.0,
        "noise_variance_upper": 9.0,
        "distortion_probability": 0.5,
        "cutout_probability": 0.05,
        "cutout_ratio": 0.5,
    }
    assert transform_names == [
        "RandomBrightnessContrast",
        "Affine",
        "OneOf",
        "OneOf",
        "CoarseDropout",
    ]
    assert not {"HorizontalFlip", "VerticalFlip", "Transpose"} & set(transform_names)
