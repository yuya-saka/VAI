"""Baseline 1の6チャネル入力と反射なしのデータ拡張。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fracture_detection.cohorts.constants import (
    INPUT_MANIFEST_CSV,
    MATCHED_COHORT_CSV,
)
from fracture_detection.common.constants import DATASET_DIR, MANIFEST_COLUMNS
from fracture_detection.common.dataset import FractureDataset

DataMode = Literal["matched", "full"]


def build_train_transform(augmentation: dict[str, float]) -> A.ReplayCompose:
    """CTのみの強度変換とCT・マスク同期の空間変換を構築する。"""
    affine_probability = float(augmentation["affine_probability"])
    shift_limit = float(augmentation["shift_limit"])
    scale_lower = float(augmentation["scale_lower"])
    scale_upper = float(augmentation["scale_upper"])
    rotate_limit = float(augmentation["rotate_limit"])
    brightness_limit = float(augmentation["brightness_limit"])
    contrast_limit = float(augmentation["contrast_limit"])
    intensity_probability = float(augmentation["intensity_probability"])
    blur_noise_probability = float(augmentation["blur_noise_probability"])
    noise_std = float(augmentation["noise_std"])

    return A.ReplayCompose(
        [
            A.Affine(
                translate_percent=(-shift_limit, shift_limit),
                scale=(scale_lower, scale_upper),
                rotate=(-rotate_limit, rotate_limit),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                fill_mask=0.0,
                p=affine_probability,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=intensity_probability,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0)),
                    A.GaussNoise(std_range=(noise_std / 2, noise_std)),
                ],
                p=blur_noise_probability,
            ),
        ]
    )


def apply_bag_transform(
    ct: np.ndarray,
    whole_mask: np.ndarray,
    transform: A.ReplayCompose,
) -> tuple[np.ndarray, np.ndarray]:
    """同じ再生パラメータで全CT面・チャネルと椎体全体マスクを変換する。"""
    if ct.ndim != 4 or whole_mask.ndim != 3:
        raise ValueError("CTは4次元、whole maskは3次元である必要があります")
    if ct.shape[0] != whole_mask.shape[0] or ct.shape[2:] != whole_mask.shape[1:]:
        raise ValueError("CTとwhole maskの形状が一致しません")

    reference = transform(image=ct[0, 0], mask=whole_mask[0])
    replay = reference["replay"]
    augmented_ct = np.empty_like(ct, dtype=np.float32)
    augmented_mask = np.empty_like(whole_mask, dtype=np.float32)

    for plane_index in range(ct.shape[0]):
        for channel_index in range(ct.shape[1]):
            augmented = A.ReplayCompose.replay(
                replay,
                image=ct[plane_index, channel_index],
                mask=whole_mask[plane_index],
            )
            augmented_ct[plane_index, channel_index] = augmented["image"]
            if channel_index == 0:
                augmented_mask[plane_index] = augmented["mask"]

    return np.clip(augmented_ct, 0.0, 1.0), (augmented_mask > 0.5).astype(np.float32)


def load_mode_manifest(
    mode: DataMode,
    input_manifest_path: Path = INPUT_MANIFEST_CSV,
    matched_cohort_path: Path = MATCHED_COHORT_CSV,
) -> pd.DataFrame:
    """設定に対応する固定マニフェストを読み込む。"""
    if mode == "matched":
        manifest_path = matched_cohort_path
    elif mode == "full":
        manifest_path = input_manifest_path
    else:
        raise ValueError(f"不正なdata.modeです: {mode}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifestがありません: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "level": str})
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    if manifest.duplicated(["study_id", "level"]).any():
        raise ValueError("manifestに重複したstudy_id・levelがあります")
    return manifest.reset_index(drop=True)


def split_fold_manifest(
    manifest: pd.DataFrame, validation_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """固定fold列だけを使って学習用・検証用へ分割する。"""
    if validation_fold not in range(5):
        raise ValueError(f"validation foldが不正です: {validation_fold}")
    if "fold" not in manifest:
        raise ValueError("manifestにfold列がありません")
    train = manifest[manifest["fold"].ne(validation_fold)].reset_index(drop=True)
    validation = manifest[manifest["fold"].eq(validation_fold)].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("trainまたはvalidationが空です")
    return train, validation


class Baseline1Dataset(Dataset[dict[str, Tensor | str]]):
    """共通DatasetをCT 5チャネルと椎体全体マスク1チャネルへ変換する。"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        transform: A.ReplayCompose | None = None,
    ) -> None:
        self.base_dataset = FractureDataset(manifest, dataset_dir=dataset_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.base_dataset[index]
        ct = sample["ct"]
        masks = sample["masks"]
        if not isinstance(ct, Tensor) or not isinstance(masks, Tensor):
            raise TypeError("common datasetのCTまたはmaskがTensorではありません")
        whole_mask = masks[:, 0]

        if self.transform is not None:
            augmented_ct, augmented_mask = apply_bag_transform(
                ct.numpy(), whole_mask.numpy(), self.transform
            )
            ct = torch.from_numpy(np.ascontiguousarray(augmented_ct))
            whole_mask = torch.from_numpy(np.ascontiguousarray(augmented_mask))

        inputs = torch.cat([ct, whole_mask.unsqueeze(1)], dim=1)
        if inputs.shape[1] != 6:
            raise ValueError(f"Baseline 1入力ch数が不正です: {inputs.shape}")
        return {
            "inputs": inputs,
            "vertebra_target": sample["vertebra_target"],
            "fold": sample["fold"],
            "study_id": sample["study_id"],
            "level": sample["level"],
        }


def default_augmentation(mode: DataMode) -> dict[str, float]:
    """確定仕様に対応する設定別のデータ拡張設定を返す。"""
    if mode == "matched":
        return {
            "affine_probability": 0.70,
            "shift_limit": 0.07,
            "scale_lower": 0.88,
            "scale_upper": 1.12,
            "rotate_limit": 40.0,
            "brightness_limit": 0.12,
            "contrast_limit": 0.15,
            "intensity_probability": 0.40,
            "blur_noise_probability": 0.20,
            "noise_std": 0.03,
        }
    if mode == "full":
        return {
            "affine_probability": 0.50,
            "shift_limit": 0.05,
            "scale_lower": 0.90,
            "scale_upper": 1.10,
            "rotate_limit": 40.0,
            "brightness_limit": 0.10,
            "contrast_limit": 0.10,
            "intensity_probability": 0.30,
            "blur_noise_probability": 0.15,
            "noise_std": 0.02,
        }
    raise ValueError(f"不正なdata.modeです: {mode}")


def augment_from_config(
    mode: DataMode, override: dict[str, Any] | None = None
) -> A.ReplayCompose:
    """設定別の既定値にYAML設定を重ねた学習用変換を返す。"""
    values = default_augmentation(mode)
    if override is not None:
        unknown = set(override) - set(values)
        if unknown:
            raise ValueError(f"augmentationに不明な設定があります: {sorted(unknown)}")
        values = {**values, **{key: float(value) for key, value in override.items()}}
    return build_train_transform(values)
