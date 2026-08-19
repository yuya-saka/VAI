from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracture_detection.common.constants import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
)
from fracture_detection.common.dataset import (
    FractureDataset,
    build_mask_channels,
    flip_horizontal,
)


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


def _region_bag() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """R1〜R4を別々の位置に置いた1椎体分の配列を作る。"""
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    vertebra_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)

    ct[0, 0, 30, 60] = 255
    vertebra_mask[0, 30, 60] = 1
    region_mask[0, 40, 100] = 1  # R1 椎体（正中）
    region_mask[0, 50, 20] = 2  # R2 右横突孔
    region_mask[0, 50, 200] = 3  # R3 左横突孔
    region_mask[0, 60, 100] = 4  # R4 後方要素（正中）
    region_targets = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return ct, vertebra_mask, region_mask, region_targets


def test_flip_horizontal_swaps_only_the_two_transverse_foramina() -> None:
    ct, vertebra_mask, region_mask, region_targets = _region_bag()

    flipped_ct, flipped_vertebra, flipped_region, flipped_targets = flip_horizontal(
        ct, vertebra_mask, region_mask, region_targets
    )

    width = EXPECTED_MASK_SHAPE[-1]
    # 画素は鏡像位置へ移る。
    assert flipped_ct[0, 0, 30, width - 1 - 60] == 255
    assert flipped_vertebra[0, 30, width - 1 - 60] == 1
    # R2があった位置は鏡像先でR3になる（値の入れ替え）。
    assert flipped_region[0, 50, width - 1 - 20] == 3
    assert flipped_region[0, 50, width - 1 - 200] == 2
    # 正中のR1/R4は値が変わらない。
    assert flipped_region[0, 40, width - 1 - 100] == 1
    assert flipped_region[0, 60, width - 1 - 100] == 4
    # ラベルもR2/R3だけ入れ替わる。
    assert flipped_targets.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert flipped_region.dtype == region_mask.dtype


def test_flip_horizontal_twice_is_identity() -> None:
    ct, vertebra_mask, region_mask, region_targets = _region_bag()

    once = flip_horizontal(ct, vertebra_mask, region_mask, region_targets)
    twice = flip_horizontal(*once)

    assert np.array_equal(twice[0], ct)
    assert np.array_equal(twice[1], vertebra_mask)
    assert np.array_equal(twice[2], region_mask)
    assert np.array_equal(twice[3], region_targets)


def test_flip_horizontal_keeps_region_channels_consistent_with_labels() -> None:
    """マスクチャンネルとラベルが同じ入れ替えを受けることを確認する。"""
    ct, vertebra_mask, region_mask, region_targets = _region_bag()
    _, flipped_vertebra, flipped_region, _ = flip_horizontal(
        ct, vertebra_mask, region_mask, region_targets
    )

    original = build_mask_channels(vertebra_mask, region_mask)
    flipped = build_mask_channels(flipped_vertebra, flipped_region)

    # R2チャンネル(index 2)の反転はR3チャンネル(index 3)と一致する。
    assert np.array_equal(flipped[:, 2], original[:, 3][..., ::-1])
    assert np.array_equal(flipped[:, 3], original[:, 2][..., ::-1])
    # R1/R4チャンネルは自分自身の反転と一致する。
    assert np.array_equal(flipped[:, 1], original[:, 1][..., ::-1])
    assert np.array_equal(flipped[:, 4], original[:, 4][..., ::-1])
