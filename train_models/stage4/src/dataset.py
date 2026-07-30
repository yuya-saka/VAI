"""Stage4 dataset with flip-aware human region labels."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from train_models.stage2.src.dataset import validate_arrays
from train_models.stage3.src.dataset import RSNAStage3Dataset, scramble_region_mask

REGION_LABEL_HORIZONTAL_ORDER = np.asarray([0, 2, 1, 3])


def swap_region_label_after_horizontal_flip(
    region_label: np.ndarray,
) -> np.ndarray:
    """Swap only right/left foramen labels after a horizontal reflection."""
    label = np.asarray(region_label, dtype=np.int8)
    if label.shape != (4,):
        raise ValueError(f"region_label must have shape (4,), got {label.shape}")
    return label[REGION_LABEL_HORIZONTAL_ORDER]


class RSNAStage4Dataset(RSNAStage3Dataset):
    """Return Stage3 inputs plus region targets and supervision masks."""

    def __getitem__(self, index: int | tuple[int, bool]) -> Any:
        supervision_override: bool | None = None
        if isinstance(index, tuple):
            index, supervision_override = index
        item = self.items[index]
        ct = np.load(item["ct_path"], allow_pickle=False)
        vertebra_mask = np.load(item["mask_path"], allow_pickle=False)
        region_mask = np.load(item["region_mask_path"], allow_pickle=False)
        validate_arrays(ct, vertebra_mask, region_mask)

        flip_applied = False
        if self.transform is None:
            images = np.concatenate([ct, vertebra_mask[:, None]], axis=1)
            regions = region_mask.astype(np.uint8, copy=False)
        else:
            images, regions, flip_applied = self._augment_volume(
                ct, vertebra_mask, region_mask
            )

        if self.mode == "train" and random.random() < self.p_rand_order:
            indices = np.random.permutation(images.shape[0])
            images = images[indices]
            regions = regions[indices]

        if self.region_mode == "scramble":
            key = f"{item['study_uid']}:{item['vertebra']}"
            regions = scramble_region_mask(regions, key, self.scramble_seed)

        region_label = np.asarray(item["region_label"], dtype=np.int8)
        if flip_applied:
            region_label = swap_region_label_after_horizontal_flip(region_label)
        supervised = item["region_supervision"] == "strong"
        if supervision_override is not None:
            supervised = supervision_override
        output: tuple[Tensor, ...] = (
            torch.from_numpy(np.ascontiguousarray(images)),
            torch.from_numpy(np.ascontiguousarray(regions)),
            torch.tensor(float(item["label"]), dtype=torch.float32),
            torch.from_numpy(np.ascontiguousarray(region_label)).float(),
            torch.tensor(supervised, dtype=torch.bool),
        )
        if not self.include_metadata:
            return output
        return (*output, str(item["study_uid"]), str(item["vertebra"]))
