"""region_branchの6ch入力・4領域mask・教師信号を返すDataset。

全幾何変換はCT・椎体mask・4領域maskを同じ`ReplayCompose`呼び出しで同期変換する。
`region_4class.npy`はラベル値そのもの（1..4=REGION_COLUMNS、0=背景）なので、
変換で空間位置が変わっても値は入れ替えない。head r は常に解剖学的領域 r を予測する
（`baseline0/data/dataset.py`のflip時ラベル入れ替えコメントとは異なる規約。
統合4領域モデルと単一領域モデルの両方でこの規約が一致する）。

セル単位の統一教師信号契約（`.claude/docs/REGION_MODEL_DESIGN_JA.md` §5.1）:

    if vertebra_target == 0:
        target = 0, target_valid = True, hard_valid = True
    elif human_valid:
        target = human 0/1, target_valid = True, hard_valid = True
    else:
        target = q, target_valid = True, pseudo_valid = True (artifactがあれば)

hard GTは常にCAM soft targetより優先する。GTとpseudoは同じ`target` tensorへ
解決し、source別target tensorやsource別lossを作らない。
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
    REGION_PSEUDO_TARGET_COLUMNS,
    REGION_SHARE_COLUMNS,
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
    """CT入力、mask、GT/pseudo統一region targetを返す。"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
        transform: A.ReplayCompose | None = None,
        require_pseudo_targets: bool = False,
    ) -> None:
        missing_columns = set(SUPERVISED_COLUMNS) - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"manifestに必要な列がありません: {sorted(missing_columns)}"
            )
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.require_pseudo_targets = require_pseudo_targets
        self.has_pseudo_columns = set(REGION_SHARE_COLUMNS) | set(
            REGION_PSEUDO_TARGET_COLUMNS
        ) <= set(manifest.columns)
        if require_pseudo_targets and not self.has_pseudo_columns:
            raise ValueError(
                "require_pseudo_targets=Trueですが、manifestにCAM share/pseudo target"
                "列がありません。build_outer_fold_loadersでattach_pseudo_targetsを"
                "呼んでいるか確認してください"
            )

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

        target, target_valid, hard_valid, pseudo_valid = self._resolve_region_targets(
            row, study_id, level
        )

        return {
            "inputs": inputs,
            "region_mask": torch.from_numpy(np.ascontiguousarray(region_mask)),
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "region_target": target,
            "region_target_valid": target_valid,
            "region_hard_valid": hard_valid,
            "region_pseudo_valid": pseudo_valid,
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }

    def _resolve_region_targets(
        self, row: pd.Series, study_id: str, level: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """1 bag分の4領域をGT優先の単一target tensorへ解決する。"""
        vertebra_negative = float(row["vertebra_target"]) == 0.0
        target = [0.0] * N_REGIONS
        target_valid = [False] * N_REGIONS
        hard_valid = [False] * N_REGIONS
        pseudo_valid = [False] * N_REGIONS

        for region_index, (region_column, valid_column) in enumerate(
            zip(REGION_COLUMNS, REGION_TARGET_VALID_COLUMNS, strict=True)
        ):
            if vertebra_negative:
                target[region_index] = 0.0
                target_valid[region_index] = True
                hard_valid[region_index] = True
                continue

            human_valid = bool(row[valid_column])
            if human_valid:
                target[region_index] = float(row[region_column])
                target_valid[region_index] = True
                hard_valid[region_index] = True
                continue

            # whole-positive, human-unknown cell
            value = self._pseudo_value(row, region_index)
            if value is not None:
                target[region_index] = value
                target_valid[region_index] = True
                pseudo_valid[region_index] = True
            elif self.require_pseudo_targets:
                share_column = REGION_SHARE_COLUMNS[region_index]
                target_column = REGION_PSEUDO_TARGET_COLUMNS[region_index]
                raise ValueError(
                    f"study_id={study_id}, level={level}, region={region_column}: "
                    f"whole-positiveかつhuman-unknownのcellにpseudo target"
                    f"({target_column}/{share_column})がありません"
                )

        return (
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(target_valid, dtype=torch.bool),
            torch.tensor(hard_valid, dtype=torch.bool),
            torch.tensor(pseudo_valid, dtype=torch.bool),
        )

    def _pseudo_value(self, row: pd.Series, region_index: int) -> float | None:
        """pseudo target列が存在し有限であればfloatを、そうでなければNoneを返す。"""
        if not self.has_pseudo_columns:
            return None
        column = REGION_PSEUDO_TARGET_COLUMNS[region_index]
        value = float(row[column])
        if not np.isfinite(value):
            return None
        return value


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
