"""全アーム共通のlaterality-safe bag augmentation。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import albumentations as A
import cv2
import numpy as np

from fracture_detection.common.dataset import (
    LR_SWAPPED_REGION_ORDER,
    build_mask_channels,
    flip_horizontal,
)


@dataclass(frozen=True)
class CanonicalAugmentation:
    """手動水平反転とAlbumentations変換をまとめる。"""

    horizontal_flip_probability: float
    transform: A.ReplayCompose


def build_canonical_augmentation(
    augmentation: dict[str, Any],
) -> CanonicalAugmentation:
    """凍結設定からorientation-safeな共通変換を作る。"""
    probability = float(augmentation["horizontal_flip_probability"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("horizontal flip probabilityは0から1が必要です")
    cutout_size = int(224 * float(augmentation["cutout_ratio"]))
    noise_std_lower = np.sqrt(float(augmentation["noise_variance_lower"])) / 255.0
    noise_std_upper = np.sqrt(float(augmentation["noise_variance_upper"])) / 255.0
    transform = A.ReplayCompose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=float(augmentation["brightness_limit"]),
                contrast_limit=float(augmentation["contrast_limit"]),
                p=float(augmentation["intensity_probability"]),
            ),
            A.Affine(
                translate_percent=(
                    -float(augmentation["shift_limit"]),
                    float(augmentation["shift_limit"]),
                ),
                scale=(
                    float(augmentation["scale_lower"]),
                    float(augmentation["scale_upper"]),
                ),
                rotate=(
                    -float(augmentation["rotate_limit"]),
                    float(augmentation["rotate_limit"]),
                ),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=int(augmentation["border_mode"]),
                p=float(augmentation["affine_probability"]),
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(std_range=(noise_std_lower, noise_std_upper)),
                ],
                p=float(augmentation["blur_noise_probability"]),
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                ],
                p=float(augmentation["distortion_probability"]),
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(cutout_size, cutout_size),
                hole_width_range=(cutout_size, cutout_size),
                fill_mask=None,
                p=float(augmentation["cutout_probability"]),
            ),
        ]
    )
    return CanonicalAugmentation(probability, transform)


def apply_canonical_augmentation(
    ct: np.ndarray,
    vertebra_mask: np.ndarray,
    region_mask: np.ndarray,
    region_targets: np.ndarray,
    region_target_valid: np.ndarray,
    augmentation: CanonicalAugmentation,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """75ch imageと30ch maskを1回で変換する。"""
    if region_target_valid.shape != region_targets.shape:
        raise ValueError("region targetとvalidityの形状が一致しません")
    if random.Random(seed).random() < augmentation.horizontal_flip_probability:
        ct, vertebra_mask, region_mask, region_targets = flip_horizontal(
            ct, vertebra_mask, region_mask, region_targets
        )
        region_target_valid = np.ascontiguousarray(
            region_target_valid[list(LR_SWAPPED_REGION_ORDER)]
        )
    plane_count, channel_count, height, width = ct.shape
    image_stack = ct.transpose(2, 3, 0, 1).reshape(
        height, width, plane_count * channel_count
    )
    mask_stack = (
        np.concatenate([vertebra_mask, region_mask], axis=0)
        .reshape(2, plane_count, height, width)
        .transpose(2, 3, 0, 1)
        .reshape(height, width, 2 * plane_count)
    )
    augmentation.transform.set_random_seed(seed)
    transformed = augmentation.transform(image=image_stack, mask=mask_stack)
    transformed_ct = (
        transformed["image"]
        .reshape(height, width, plane_count, channel_count)
        .transpose(2, 3, 0, 1)
    )
    transformed_masks = (
        transformed["mask"].reshape(height, width, 2, plane_count).transpose(2, 3, 0, 1)
    )
    transformed_ct = np.clip(transformed_ct, 0, 255).astype(np.uint8)
    transformed_whole = (transformed_masks[0] > 0.5).astype(np.uint8)
    transformed_region = np.rint(transformed_masks[1]).astype(np.uint8)
    values = set(np.unique(transformed_region).tolist())
    if not values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"augmentation後のregion mask値が不正です: {values}")
    return (
        transformed_ct,
        transformed_whole,
        transformed_region,
        region_targets,
        region_target_valid,
    )


def build_uint8_inputs(
    ct: np.ndarray, vertebra_mask: np.ndarray, region_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """canonical 10ch入力と4領域attention targetをuint8で返す。"""
    mask_channels = build_mask_channels(vertebra_mask, region_mask)
    masks_uint8 = np.rint(mask_channels * 255.0).astype(np.uint8)
    inputs = np.concatenate([ct, masks_uint8], axis=1)
    return np.ascontiguousarray(inputs), np.ascontiguousarray(masks_uint8[:, 1:])
