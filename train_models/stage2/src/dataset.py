"""Dataset and synchronized augmentations for Stage2 region MIL."""

from __future__ import annotations

import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

EXPECTED_CT_SHAPE = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE = (15, 224, 224)
REGION_REMAP_HORIZONTAL = np.array([0, 1, 3, 2, 4], dtype=np.uint8)


def get_train_transforms(aug_cfg: dict[str, Any]) -> A.ReplayCompose:
    """Build transforms whose spatial parameters can be replayed for all planes."""
    image_size = 224
    cutout_size = int(image_size * float(aug_cfg.get("cutout_ratio", 0.5)))
    border_mode = int(aug_cfg.get("ssr_border_mode", cv2.BORDER_REFLECT_101))
    shift_limit = float(aug_cfg.get("shift_limit", 0.3))
    scale_limit = float(aug_cfg.get("scale_limit", 0.3))
    rotate_limit = float(aug_cfg.get("rotate_limit", 45))

    return A.ReplayCompose(
        [
            A.HorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
            A.VerticalFlip(p=float(aug_cfg.get("vertical_flip_p", 0.0))),
            A.Transpose(p=float(aug_cfg.get("transpose_p", 0.0))),
            A.RandomBrightnessContrast(
                brightness_limit=float(aug_cfg.get("brightness_limit", 0.1)),
                contrast_limit=0.0,
                p=float(aug_cfg.get("brightness_p", 0.7)),
            ),
            A.Affine(
                translate_percent=(-shift_limit, shift_limit),
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                rotate=(-rotate_limit, rotate_limit),
                border_mode=border_mode,
                p=float(aug_cfg.get("ssr_p", 0.7)),
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(
                        std_range=(np.sqrt(3.0) / 255.0, np.sqrt(9.0) / 255.0)
                    ),
                ],
                p=float(aug_cfg.get("blur_noise_p", 0.5)),
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                ],
                p=float(aug_cfg.get("distortion_p", 0.5)),
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(cutout_size, cutout_size),
                hole_width_range=(cutout_size, cutout_size),
                p=float(aug_cfg.get("cutout_p", 0.05)),
            ),
        ],
        additional_targets={"region_mask": "mask"},
    )


def replay_applied_horizontal_flip(replay: dict[str, Any]) -> bool:
    """Return whether HorizontalFlip was applied in an Albumentations replay."""
    return any(
        transform.get("__class_fullname__", "").endswith("HorizontalFlip")
        and bool(transform.get("applied", False))
        for transform in replay.get("transforms", [])
    )


def remap_regions_after_horizontal_flip(region_mask: np.ndarray) -> np.ndarray:
    """Restore anatomical left/right region identifiers after image reflection."""
    return REGION_REMAP_HORIZONTAL[region_mask]


def validate_arrays(
    ct: np.ndarray,
    vertebra_mask: np.ndarray,
    region_mask: np.ndarray,
) -> None:
    """Validate one Stage2 sample before augmentation."""
    if ct.shape != EXPECTED_CT_SHAPE:
        raise ValueError(f"invalid ct shape: {ct.shape}")
    if vertebra_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"invalid vertebra mask shape: {vertebra_mask.shape}")
    if region_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"invalid region mask shape: {region_mask.shape}")
    if not np.issubdtype(ct.dtype, np.integer):
        raise ValueError(f"invalid ct dtype: {ct.dtype}")
    if region_mask.min() < 0 or region_mask.max() > 4:
        raise ValueError("region mask values must be in [0, 4]")
    if not np.any(region_mask > 0):
        raise ValueError("region mask contains no valid region")


class RSNARegionDataset(Dataset):
    """Return full-plane inputs and anatomical region masks for one vertebra."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        mode: str = "train",
        transform: A.ReplayCompose | None = None,
        p_rand_order: float = 0.2,
        include_metadata: bool = False,
    ) -> None:
        self.items = items
        self.mode = mode
        self.transform = transform
        self.p_rand_order = p_rand_order
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, str, str]:
        item = self.items[index]
        ct = np.load(item["ct_path"], allow_pickle=False)
        vertebra_mask = np.load(item["mask_path"], allow_pickle=False)
        region_mask = np.load(item["region_mask_path"], allow_pickle=False)
        validate_arrays(ct, vertebra_mask, region_mask)

        if self.transform is None:
            images = np.concatenate([ct, vertebra_mask[:, None]], axis=1)
            regions = region_mask.astype(np.uint8, copy=False)
        else:
            images, regions, _ = self._augment_volume(ct, vertebra_mask, region_mask)

        if self.mode == "train" and random.random() < self.p_rand_order:
            indices = np.random.permutation(images.shape[0])
            images = images[indices]
            regions = regions[indices]

        output = (
            torch.from_numpy(np.ascontiguousarray(images)),
            torch.from_numpy(np.ascontiguousarray(regions)),
            torch.tensor(float(item["label"]), dtype=torch.float32),
        )
        if not self.include_metadata:
            return output
        return (*output, str(item["study_uid"]), str(item["vertebra"]))

    def _augment_volume(
        self,
        ct: np.ndarray,
        vertebra_mask: np.ndarray,
        region_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply one shared spatial transform to all planes at once.

        All 15 planes are stacked into the channel axis so Albumentations
        samples and applies each transform's parameters exactly once instead
        of once per plane via replay; every plane still receives the same
        geometry. ``(2, 3, 0, 1)`` is its own inverse, so the same transpose
        both builds the stack and restores the original axis order.
        """
        n_planes, n_ct_channels, height, width = ct.shape
        image_stack = ct.transpose(2, 3, 0, 1).reshape(
            height, width, n_planes * n_ct_channels
        )
        mask_stack = vertebra_mask.transpose(1, 2, 0)
        region_stack = region_mask.transpose(1, 2, 0)

        augmented = self.transform(
            image=image_stack, mask=mask_stack, region_mask=region_stack
        )

        images = (
            augmented["image"]
            .reshape(height, width, n_planes, n_ct_channels)
            .transpose(2, 3, 0, 1)
        )
        vertebra_masks = augmented["mask"].transpose(2, 0, 1)
        regions = augmented["region_mask"].transpose(2, 0, 1)
        combined = np.concatenate([images, vertebra_masks[:, None]], axis=1)

        flip_applied = replay_applied_horizontal_flip(augmented["replay"])
        if flip_applied:
            regions = remap_regions_after_horizontal_flip(regions)
        return combined, regions, flip_applied
