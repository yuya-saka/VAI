"""region_branchの6ch入力・4領域mask・教師信号を返すDataset。

horizontal flipはCT・椎体mask・4領域maskを同じ`ReplayCompose`呼び出しで同期変換する。
`region_4class.npy`はラベル値そのもの（1..4=REGION_COLUMNS、0=背景）なので、
flipで空間位置が変わっても値は入れ替えない。head r は常に解剖学的領域 r を予測する
（`baseline0/data/dataset.py`のflip時ラベル入れ替えコメントとは異なる規約。
統合4領域モデルと単一領域モデルの両方でこの規約が一致する）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fracture_detection.baseline0.data.dataset import (
    augment_from_config as augment_from_config,
)
from fracture_detection.baseline0.data.dataset import (
    build_train_transform as build_train_transform,
)
from fracture_detection.baseline0.data.dataset import (
    default_augmentation as default_augmentation,
)
from fracture_detection.baseline0.data.dataset import load_manifest as load_manifest
from fracture_detection.region_branch.data_pipeline.constants import (
    DATASET_DIR,
    EXPECTED_REGION_MASK_SHAPE,
    N_REGIONS,
    REGION_COLUMNS,
    REGION_MASK_FILENAME,
    REGION_SCORE_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)

EXPECTED_CT_SHAPE = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE = (15, 224, 224)
EXPECTED_CT_DTYPE = "uint8"

SUPERVISED_COLUMNS = (
    "study_id",
    "level",
    "fold",
    "vertebra_target",
    "has_region_target",
    *REGION_COLUMNS,
    *REGION_TARGET_VALID_COLUMNS,
)


def apply_bag_transform_with_regions(
    ct: np.ndarray,
    whole_mask: np.ndarray,
    region_mask: np.ndarray,
    transform: A.ReplayCompose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """15面のCT・椎体mask・4領域maskを1回の変換で同期拡張する。"""
    if ct.ndim != 4 or whole_mask.ndim != 3 or region_mask.ndim != 3:
        raise ValueError("CTは4次元、mask群は3次元である必要があります")
    if (
        ct.shape[0] != whole_mask.shape[0]
        or ct.shape[0] != region_mask.shape[0]
        or ct.shape[2:] != whole_mask.shape[1:]
        or ct.shape[2:] != region_mask.shape[1:]
    ):
        raise ValueError("CT・椎体mask・4領域maskの形状が一致しません")

    plane_count, channel_count, height, width = ct.shape
    image_stack = ct.transpose(2, 3, 0, 1).reshape(
        height, width, plane_count * channel_count
    )
    # whole maskとregion maskを1つのmask targetへ連結し、単一のreplayで同期させる。
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


class RegionBranchDataset(Dataset[dict[str, Any]]):
    """CT 5ch + 椎体全体mask 1ch + 4領域mask + 領域教師信号を返す。"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        transform: A.ReplayCompose | None = None,
    ) -> None:
        missing_columns = set(SUPERVISED_COLUMNS) - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"manifestに必要な列がありません: {sorted(missing_columns)}"
            )
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.has_teacher_scores = set(REGION_SCORE_COLUMNS).issubset(manifest.columns)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        study_id = str(row["study_id"])
        level = str(row["level"])
        bag_dir = self.dataset_dir / study_id / level
        ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
        whole_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
        region_mask = np.load(bag_dir / REGION_MASK_FILENAME, allow_pickle=False)
        _validate_bag_arrays(ct, whole_mask, region_mask)

        if self.transform is not None:
            ct, whole_mask, region_mask = apply_bag_transform_with_regions(
                ct, whole_mask, region_mask, self.transform
            )

        mask_channel = np.rint(whole_mask * 255.0).astype(np.uint8)[:, None]
        inputs = torch.from_numpy(
            np.ascontiguousarray(np.concatenate([ct, mask_channel], axis=1))
        )
        if inputs.shape[1] != 6:
            raise ValueError(f"region_branch入力ch数が不正です: {inputs.shape}")

        region_targets = torch.tensor(
            [float(row[column]) for column in REGION_COLUMNS], dtype=torch.float32
        )
        region_target_valid = torch.tensor(
            [bool(row[column]) for column in REGION_TARGET_VALID_COLUMNS],
            dtype=torch.bool,
        )
        if self.has_teacher_scores:
            region_scores = torch.tensor(
                [float(row[column]) for column in REGION_SCORE_COLUMNS],
                dtype=torch.float32,
            )
        else:
            region_scores = torch.full((N_REGIONS,), float("nan"), dtype=torch.float32)

        return {
            "inputs": inputs,
            "region_mask": torch.from_numpy(np.ascontiguousarray(region_mask)),
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "region_targets": region_targets,
            "region_target_valid": region_target_valid,
            "region_scores": region_scores,
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }


def _validate_bag_arrays(
    ct: np.ndarray, whole_mask: np.ndarray, region_mask: np.ndarray
) -> None:
    """region_branchが読むCT・椎体mask・4領域maskの固定契約を検証する。"""
    if ct.shape != EXPECTED_CT_SHAPE or str(ct.dtype) != EXPECTED_CT_DTYPE:
        raise ValueError(f"CT配列が不正です: shape={ct.shape}, dtype={ct.dtype}")
    if whole_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"椎体mask形状が不正です: {whole_mask.shape}")
    if not np.issubdtype(whole_mask.dtype, np.integer) or not np.any(whole_mask):
        raise ValueError("椎体maskは非空の整数配列である必要があります")
    if region_mask.shape != EXPECTED_REGION_MASK_SHAPE:
        raise ValueError(f"4領域mask形状が不正です: {region_mask.shape}")
    region_values = set(np.unique(region_mask).tolist())
    if not region_values.issubset(set(range(N_REGIONS + 1))):
        raise ValueError(f"4領域maskの値が不正です: {sorted(region_values)}")
