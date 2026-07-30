from __future__ import annotations

from pathlib import Path

import numpy as np

from train_models.stage2.src.dataset import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
    get_train_transforms,
    remap_regions_after_horizontal_flip,
)
from train_models.stage4.src.data_utils import collect_items
from train_models.stage4.src.dataset import (
    RSNAStage4Dataset,
    swap_region_label_after_horizontal_flip,
)

ROOT = Path(__file__).resolve().parents[3]


def _flip_only_transform() -> object:
    return get_train_transforms(
        {
            "horizontal_flip_p": 1.0,
            "vertical_flip_p": 0.0,
            "transpose_p": 0.0,
            "brightness_p": 0.0,
            "ssr_p": 0.0,
            "blur_noise_p": 0.0,
            "distortion_p": 0.0,
            "cutout_p": 0.0,
        }
    )


def test_double_flip_restores_image_mask_and_region_label() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    mask = np.asarray([[1, 2, 3, 4]], dtype=np.uint8)
    label = np.asarray([1, 1, 0, 1], dtype=np.int8)

    restored_image = np.flip(np.flip(image, axis=-1), axis=-1)
    restored_mask = remap_regions_after_horizontal_flip(
        np.flip(
            remap_regions_after_horizontal_flip(np.flip(mask, axis=-1)),
            axis=-1,
        )
    )
    restored_label = swap_region_label_after_horizontal_flip(
        swap_region_label_after_horizontal_flip(label)
    )

    np.testing.assert_array_equal(restored_image, image)
    np.testing.assert_array_equal(restored_mask, mask)
    np.testing.assert_array_equal(restored_label, label)


def test_r2_only_label_becomes_r3_only_after_flip(tmp_path: Path) -> None:
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    vertebra_mask = np.ones(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask[:, :, :112] = 2
    region_mask[:, :, 112:] = 3
    ct_path = tmp_path / "ct.npy"
    mask_path = tmp_path / "mask.npy"
    region_path = tmp_path / "region.npy"
    np.save(ct_path, ct)
    np.save(mask_path, vertebra_mask)
    np.save(region_path, region_mask)
    item = {
        "study_uid": "study",
        "vertebra": "C3",
        "label": 1,
        "ct_path": ct_path,
        "mask_path": mask_path,
        "region_mask_path": region_path,
        "region_label": np.asarray([0, 1, 0, 0], dtype=np.int8),
        "region_supervision": "strong",
    }
    dataset = RSNAStage4Dataset(
        [item],
        mode="train",
        transform=_flip_only_transform(),
        p_rand_order=0.0,
    )

    _, regions, _, region_label, supervised = dataset[0]

    np.testing.assert_array_equal(region_label.numpy(), [0, 0, 1, 0])
    assert supervised.item() is True
    assert np.all(regions.numpy()[:, :, :112] == 2)
    assert np.all(regions.numpy()[:, :, 112:] == 3)


def test_first_100_strong_samples_keep_mask_label_swap_consistent() -> None:
    items = collect_items(
        ROOT / "data/rsna_data/fracture_dataset_blind",
        ROOT / "data/rsna_data/train.csv",
        ROOT / "data/rsna_data/fracture_region_labels_dicom.csv",
        ROOT / "data/rsna_data/excluded_studies.csv",
        ROOT / "data/rsna_data/excluded_levels.csv",
    )
    strong_items = [item for item in items if item["region_supervision"] == "strong"][
        :100
    ]

    assert len(strong_items) == 100
    for item in strong_items:
        region_mask = np.load(item["region_mask_path"], allow_pickle=False)
        before_counts = np.asarray(
            [(region_mask == region_id).sum() for region_id in range(1, 5)]
        )
        flipped_mask = remap_regions_after_horizontal_flip(
            np.flip(region_mask, axis=-1)
        )
        after_counts = np.asarray(
            [(flipped_mask == region_id).sum() for region_id in range(1, 5)]
        )
        flipped_label = swap_region_label_after_horizontal_flip(item["region_label"])

        np.testing.assert_array_equal(after_counts, before_counts[[0, 2, 1, 3]])
        np.testing.assert_array_equal(
            flipped_label,
            np.asarray(item["region_label"])[[0, 2, 1, 3]],
        )
