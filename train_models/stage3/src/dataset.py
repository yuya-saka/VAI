"""Stage3 dataset with deterministic spatial-scramble control."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from train_models.stage2.src.dataset import RSNARegionDataset


def scramble_region_mask(
    region_mask: np.ndarray, sample_key: str, seed: int
) -> np.ndarray:
    """Circularly shift each plane while preserving every region's area exactly."""
    height, width = region_mask.shape[-2:]
    shifted = np.empty_like(region_mask)
    min_y, max_y = max(1, height // 4), max(2, height // 2)
    min_x, max_x = max(1, width // 4), max(2, width // 2)
    for plane_index, plane in enumerate(region_mask):
        digest = hashlib.sha256(f"{seed}:{sample_key}:{plane_index}".encode()).digest()
        offset_y = min_y + int.from_bytes(digest[:4], "little") % (max_y - min_y + 1)
        offset_x = min_x + int.from_bytes(digest[4:8], "little") % (max_x - min_x + 1)
        shifted[plane_index] = np.roll(plane, shift=(offset_y, offset_x), axis=(0, 1))
    return shifted


class RSNAStage3Dataset(RSNARegionDataset):
    """Stage2-compatible dataset that optionally destroys mask-image correspondence."""

    def __init__(
        self,
        *args: Any,
        region_mode: str = "masked",
        scramble_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.region_mode = region_mode
        self.scramble_seed = scramble_seed

    def __getitem__(self, index: int) -> Any:
        output = super().__getitem__(index)
        if self.region_mode != "scramble":
            return output
        images, regions, *metadata = output
        item = self.items[index]
        key = f"{item['study_uid']}:{item['vertebra']}"
        scrambled = scramble_region_mask(regions.numpy(), key, self.scramble_seed)
        return (images, regions.new_tensor(scrambled), *metadata)
