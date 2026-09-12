"""Synchronized CT + vertebra-mask + region-label-map augmentation.

Reuses baseline0's frozen augmentation recipe (``default_augmentation`` /
``build_train_transform`` / ``augment_from_config``) so this package's
augmentation strength matches Baseline 0 exactly, per the user's instruction.
Only the synchronization plumbing that also carries the 4-class region label
map through the same ``albumentations.ReplayCompose`` call is new here (the
same technique baseline0 uses for the whole-vertebra mask, extended to a
second mask channel so all three arrays receive one identical transform).

The region label map is never blended: mask interpolation is nearest-neighbor
(see ``build_train_transform``'s ``mask_interpolation=cv2.INTER_NEAREST``),
and the value is re-quantized after any resampling.
"""

from __future__ import annotations

import numpy as np
from albumentations import ReplayCompose

from fracture_detection.baseline0.data.dataset import (
    augment_from_config as augment_from_config,
)
from fracture_detection.baseline0.data.dataset import (
    build_train_transform as build_train_transform,
)
from fracture_detection.baseline0.data.dataset import (
    default_augmentation as default_augmentation,
)
from fracture_detection.weak.data_pipeline.constants import N_REGIONS


def apply_bag_transform_with_regions(
    ct: np.ndarray,
    whole_mask: np.ndarray,
    region_mask: np.ndarray,
    transform: ReplayCompose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one synchronized transform to the 15-plane CT and both masks.

    All 15 planes of the CT are stacked into the image channel axis and the
    whole-vertebra mask and 4-class region mask are stacked into one mask
    target, so a single ``ReplayCompose`` call applies the identical geometry
    to all three arrays (mirrors ``baseline0.data.dataset.apply_bag_transform``,
    extended with a second mask channel block).
    """
    if ct.ndim != 4 or whole_mask.ndim != 3 or region_mask.ndim != 3:
        raise ValueError("CT must be 4D and both masks must be 3D")
    if (
        ct.shape[0] != whole_mask.shape[0]
        or ct.shape[0] != region_mask.shape[0]
        or ct.shape[2:] != whole_mask.shape[1:]
        or ct.shape[2:] != region_mask.shape[1:]
    ):
        raise ValueError("CT, whole mask, and region mask shapes do not match")

    plane_count, channel_count, height, width = ct.shape
    image_stack = ct.transpose(2, 3, 0, 1).reshape(
        height, width, plane_count * channel_count
    )
    combined_mask_stack = np.concatenate(
        [whole_mask.transpose(1, 2, 0), region_mask.transpose(1, 2, 0)], axis=-1
    )
    augmented = transform(image=image_stack, mask=combined_mask_stack)
    augmented_ct = (
        augmented["image"]
        .reshape(height, width, plane_count, channel_count)
        .transpose(2, 3, 0, 1)
    )
    augmented_combined = augmented["mask"].transpose(2, 0, 1)
    augmented_whole = augmented_combined[:plane_count]
    augmented_region = augmented_combined[plane_count:]

    if np.issubdtype(ct.dtype, np.integer):
        maximum = np.iinfo(ct.dtype).max
        augmented_ct = np.clip(augmented_ct, 0, maximum).astype(ct.dtype)
    else:
        augmented_ct = np.clip(augmented_ct, 0.0, 1.0).astype(np.float32)
    augmented_whole = (augmented_whole > 0.5).astype(np.float32)
    augmented_region = np.clip(np.rint(augmented_region), 0, N_REGIONS).astype(np.uint8)
    return augmented_ct, augmented_whole, augmented_region
