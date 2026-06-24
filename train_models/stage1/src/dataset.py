"""
RSNA頸椎骨折分類用Dataset。

各アイテムは1椎体 (1 study × 1 vertebra)。
ct.npy (15, 5, 224, 224) + vertebra_mask.npy (15, 224, 224)
を結合して (15, 6, 224, 224) の入力を構築する。
"""

from __future__ import annotations

import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def get_train_transforms(aug_cfg: dict) -> A.Compose:
    """
    RSNA 2022 1位解法スタイルの訓練時augmentationを構築する。

    CT (5ch) のみに適用される intensity 変換と、
    CT + mask 両方に適用される spatial 変換を含む。
    albumentations の image= に CT (HWC)、mask= に vertebra mask を渡す前提。
    """
    image_size = 224
    cutout_size = int(image_size * float(aug_cfg.get("cutout_ratio", 0.5)))
    border_mode = int(aug_cfg.get("ssr_border_mode", cv2.BORDER_REFLECT_101))
    shift_limit = float(aug_cfg.get("shift_limit", 0.3))
    scale_limit = float(aug_cfg.get("scale_limit", 0.3))
    rotate_limit = float(aug_cfg.get("rotate_limit", 45))

    return A.Compose(
        [
            A.HorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
            A.VerticalFlip(p=float(aug_cfg.get("vertical_flip_p", 0.5))),
            A.Transpose(p=float(aug_cfg.get("transpose_p", 0.5))),
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
                p=float(aug_cfg.get("cutout_p", 0.5)),
            ),
        ]
    )


def get_valid_transforms() -> A.Compose:
    """検証時のtransform（データは既に224×224のためno-op）。"""
    return A.Compose([])


class RSNAFractureDataset(Dataset):
    """
    RSNA頸椎骨折分類Dataset（1アイテム = 1椎体）。

    ct.npy (15, 5, 224, 224) uint8 と
    vertebra_mask.npy (15, 224, 224) uint8 を読み込み、
    per-sliceでaugmentationを適用して (15, 6, 224, 224) uint8 を返す。
    float32変換と0-1正規化はGPU転送後にtrainer側で行う。

    Args:
        items: 各要素が
            {"study_uid", "vertebra", "label", "ct_path", "mask_path"}
            のdictのリスト
        mode: "train" | "valid"
        transform: albumentations Compose（CT(image) + mask同期用）
        p_rand_order: スライス順序ランダム化確率（train時のみ有効）
    """

    def __init__(
        self,
        items: list[dict],
        mode: str = "train",
        transform: A.Compose | None = None,
        p_rand_order: float = 0.2,
    ) -> None:
        self.items = items
        self.mode = mode
        self.transform = transform
        self.p_rand_order = p_rand_order

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.items[idx]
        label = int(item["label"])

        ct = np.load(item["ct_path"], allow_pickle=False)
        mask = np.load(item["mask_path"], allow_pickle=False)

        n_slices = ct.shape[0]
        if self.transform is None:
            images = np.concatenate([ct, mask[:, np.newaxis]], axis=1)
        else:
            images = np.empty((n_slices, 6, 224, 224), dtype=np.uint8)

            for s in range(n_slices):
                ct_slice = ct[s].transpose(1, 2, 0)
                mask_slice = mask[s]

                augmented = self.transform(image=ct_slice, mask=mask_slice)
                ct_aug = augmented["image"]
                mask_aug = augmented["mask"]

                combined = np.concatenate([ct_aug, mask_aug[:, :, np.newaxis]], axis=2)
                images[s] = combined.transpose(2, 0, 1)

        # スライス順序ランダム化（train時のみ）
        if self.mode == "train" and random.random() < self.p_rand_order:
            indices = np.random.permutation(n_slices)
            images = images[indices]

        images_t = torch.from_numpy(images)
        labels_t = torch.full((n_slices,), label, dtype=torch.float32)  # (15,) 全同値

        return images_t, labels_t
