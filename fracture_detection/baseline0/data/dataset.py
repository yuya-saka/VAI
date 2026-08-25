"""Baseline 0の6チャネル入力とflipなしのデータ拡張。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fracture_detection.baseline0.data.constants import (
    DATASET_DIR,
    EXPECTED_CT_DTYPE,
    EXPECTED_CT_SHAPE,
    EXPECTED_MANIFEST_ROWS,
    EXPECTED_MASK_SHAPE,
    INPUT_MANIFEST_CSV,
    MANIFEST_COLUMNS,
)


def build_train_transform(augmentation: dict[str, float]) -> A.ReplayCompose:
    """CTのみの強度変換とCT・マスク同期の空間変換を構築する。"""
    horizontal_flip_probability = float(augmentation["horizontal_flip_probability"])
    affine_probability = float(augmentation["affine_probability"])
    shift_limit = float(augmentation["shift_limit"])
    scale_lower = float(augmentation["scale_lower"])
    scale_upper = float(augmentation["scale_upper"])
    rotate_limit = float(augmentation["rotate_limit"])
    brightness_limit = float(augmentation["brightness_limit"])
    contrast_limit = float(augmentation["contrast_limit"])
    intensity_probability = float(augmentation["intensity_probability"])
    blur_noise_probability = float(augmentation["blur_noise_probability"])
    border_mode = int(augmentation["border_mode"])
    noise_std_lower = np.sqrt(float(augmentation["noise_variance_lower"])) / 255.0
    noise_std_upper = np.sqrt(float(augmentation["noise_variance_upper"])) / 255.0
    distortion_probability = float(augmentation["distortion_probability"])
    cutout_probability = float(augmentation["cutout_probability"])
    cutout_size = int(224 * float(augmentation["cutout_ratio"]))

    return A.ReplayCompose(
        [
            # Baseline 0はwholeラベルのみで左右依存がないためそのまま適用できる。
            # 領域ラベルを持つアームはcommon.dataset.flip_horizontalを使い、
            # R2/R3のラベルとマスク値を同時に入れ替えること。
            A.HorizontalFlip(p=horizontal_flip_probability),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=intensity_probability,
            ),
            A.Affine(
                translate_percent=(-shift_limit, shift_limit),
                scale=(scale_lower, scale_upper),
                rotate=(-rotate_limit, rotate_limit),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=border_mode,
                p=affine_probability,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(std_range=(noise_std_lower, noise_std_upper)),
                ],
                p=blur_noise_probability,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                ],
                p=distortion_probability,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(cutout_size, cutout_size),
                hole_width_range=(cutout_size, cutout_size),
                p=cutout_probability,
            ),
        ]
    )


def apply_bag_transform(
    ct: np.ndarray,
    whole_mask: np.ndarray,
    transform: A.ReplayCompose,
) -> tuple[np.ndarray, np.ndarray]:
    """全15面をchannelへ積み、Stage1と同じ1回の変換で同期拡張する。"""
    if ct.ndim != 4 or whole_mask.ndim != 3:
        raise ValueError("CTは4次元、whole maskは3次元である必要があります")
    if ct.shape[0] != whole_mask.shape[0] or ct.shape[2:] != whole_mask.shape[1:]:
        raise ValueError("CTとwhole maskの形状が一致しません")

    plane_count, channel_count, height, width = ct.shape
    image_stack = ct.transpose(2, 3, 0, 1).reshape(
        height, width, plane_count * channel_count
    )
    mask_stack = whole_mask.transpose(1, 2, 0)
    augmented = transform(image=image_stack, mask=mask_stack)
    augmented_ct = (
        augmented["image"]
        .reshape(height, width, plane_count, channel_count)
        .transpose(2, 3, 0, 1)
    )
    augmented_mask = augmented["mask"].transpose(2, 0, 1)

    if np.issubdtype(ct.dtype, np.integer):
        maximum = np.iinfo(ct.dtype).max
        augmented_ct = np.clip(augmented_ct, 0, maximum).astype(ct.dtype)
    else:
        augmented_ct = np.clip(augmented_ct, 0.0, 1.0).astype(np.float32)
    return augmented_ct, (augmented_mask > 0.5).astype(np.float32)


def load_manifest(manifest_path: Path = INPUT_MANIFEST_CSV) -> pd.DataFrame:
    """凍結済みfullマニフェストを読み込む。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifestがありません: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "level": str})
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    if manifest.duplicated(["study_id", "level"]).any():
        raise ValueError("manifestに重複したstudy_id・levelがあります")
    if len(manifest) != EXPECTED_MANIFEST_ROWS:
        raise ValueError(
            f"品質除外済みfull manifestは{EXPECTED_MANIFEST_ROWS:,} bagが必要です: "
            f"{len(manifest)}"
        )
    return manifest.reset_index(drop=True)


class Baseline0Dataset(Dataset[dict[str, Tensor | str]]):
    """共通DatasetをCT 5チャネルと椎体全体マスク1チャネルへ変換する。"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        transform: A.ReplayCompose | None = None,
    ) -> None:
        missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"manifestに必要な列がありません: {sorted(missing_columns)}"
            )
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.manifest.iloc[index]
        study_id = str(row["study_id"])
        level = str(row["level"])
        bag_dir = self.dataset_dir / study_id / level
        ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
        whole_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
        _validate_baseline_arrays(ct, whole_mask)

        if self.transform is not None:
            augmented_ct, augmented_mask = apply_bag_transform(
                ct, whole_mask, self.transform
            )
            ct = augmented_ct
            whole_mask = augmented_mask

        mask_channel = np.rint(whole_mask * 255.0).astype(np.uint8)[:, None]
        inputs = torch.from_numpy(
            np.ascontiguousarray(np.concatenate([ct, mask_channel], axis=1))
        )
        if inputs.shape[1] != 6:
            raise ValueError(f"Baseline 0入力ch数が不正です: {inputs.shape}")
        return {
            "inputs": inputs,
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }


def _validate_baseline_arrays(ct: np.ndarray, whole_mask: np.ndarray) -> None:
    """Baseline 0が読むCTと椎体maskの固定契約を検証する。"""
    if ct.shape != EXPECTED_CT_SHAPE or str(ct.dtype) != EXPECTED_CT_DTYPE:
        raise ValueError(f"CT配列が不正です: shape={ct.shape}, dtype={ct.dtype}")
    if whole_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"椎体mask形状が不正です: {whole_mask.shape}")
    if not np.issubdtype(whole_mask.dtype, np.integer) or not np.any(whole_mask):
        raise ValueError("椎体maskは非空の整数配列である必要があります")


def default_augmentation() -> dict[str, float]:
    """Baseline 0の凍結augmentation設定を返す。"""
    return {
        "horizontal_flip_probability": 0.50,
        "affine_probability": 0.70,
        "shift_limit": 0.30,
        "scale_lower": 0.70,
        "scale_upper": 1.30,
        "rotate_limit": 45.0,
        "border_mode": 4.0,
        "brightness_limit": 0.10,
        "contrast_limit": 0.0,
        "intensity_probability": 0.70,
        "blur_noise_probability": 0.50,
        "noise_variance_lower": 3.0,
        "noise_variance_upper": 9.0,
        "distortion_probability": 0.50,
        "cutout_probability": 0.05,
        "cutout_ratio": 0.50,
    }


def augment_from_config(
    override: dict[str, Any] | None = None,
) -> A.ReplayCompose:
    """凍結既定値にYAML設定を重ねた学習用変換を返す。"""
    values = default_augmentation()
    if override is not None:
        unknown = set(override) - set(values)
        if unknown:
            raise ValueError(f"augmentationに不明な設定があります: {sorted(unknown)}")
        values = {**values, **{key: float(value) for key, value in override.items()}}
    return build_train_transform(values)
