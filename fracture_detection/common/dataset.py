"""反転拡張を行わない共通2.5D dataset。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fracture_detection.common.constants import (
    DATASET_DIR,
    EXPECTED_CT_DTYPE,
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
    MANIFEST_COLUMNS,
    N_REGIONS,
    REGION_COLUMNS,
)


def validate_arrays(
    ct: np.ndarray,
    vertebra_mask: np.ndarray,
    region_mask: np.ndarray,
) -> None:
    """1椎体分の配列が固定データ契約を満たすか検証する。"""
    if ct.shape != EXPECTED_CT_SHAPE:
        raise ValueError(f"CT形状が不正です: {ct.shape}")
    if str(ct.dtype) != EXPECTED_CT_DTYPE:
        raise ValueError(f"CTデータ型が不正です: {ct.dtype}")
    if vertebra_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"椎体マスク形状が不正です: {vertebra_mask.shape}")
    if region_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"領域マスク形状が不正です: {region_mask.shape}")
    if not np.issubdtype(vertebra_mask.dtype, np.integer):
        raise ValueError(f"椎体マスクのデータ型が不正です: {vertebra_mask.dtype}")
    if not np.issubdtype(region_mask.dtype, np.integer):
        raise ValueError(f"領域マスクのデータ型が不正です: {region_mask.dtype}")
    if not np.any(vertebra_mask):
        raise ValueError("椎体マスクが空です")
    region_values = set(np.unique(region_mask).tolist())
    if not region_values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"領域マスクに不正な値があります: {sorted(region_values)}")


def build_mask_channels(
    vertebra_mask: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    """全体マスクとR1〜R4を5チャンネルへ変換する。"""
    whole = (vertebra_mask > 0)[:, None]
    regions = np.stack(
        [(region_mask == value) for value in range(1, N_REGIONS + 1)], axis=1
    )
    return np.concatenate([whole, regions], axis=1).astype(np.float32)


class FractureDataset(Dataset[dict[str, Tensor | str]]):
    """1要素につき1椎体のCT・5マスク・ラベルを返す。

    CTは0〜1のfloat32へ正規化する。マスクは全体、R1、R2、R3、R4の順で、
    CTとは別のテンソルとして返す。反転・転置・左右入れ替えは一切行わない。
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_dir: Path = DATASET_DIR,
    ) -> None:
        missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"manifestに必要な列がありません: {sorted(missing_columns)}"
            )
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.manifest.iloc[index]
        study_id = str(row["study_id"])
        level = str(row["level"])
        bag_dir = self.dataset_dir / study_id / level

        ct = np.load(bag_dir / "ct.npy", allow_pickle=False)
        vertebra_mask = np.load(bag_dir / "vertebra_mask.npy", allow_pickle=False)
        region_mask = np.load(bag_dir / "region_4class.npy", allow_pickle=False)
        validate_arrays(ct, vertebra_mask, region_mask)

        mask_channels = build_mask_channels(vertebra_mask, region_mask)
        plane_region_valid = mask_channels[:, 1:].any(axis=(2, 3))
        region_targets = row[list(REGION_COLUMNS)].to_numpy(dtype=np.float32)
        has_region_target = bool(row["has_region_target"])

        return {
            "ct": torch.from_numpy(np.ascontiguousarray(ct)).float().div_(255.0),
            "masks": torch.from_numpy(np.ascontiguousarray(mask_channels)),
            "vertebra_target": torch.tensor(
                float(row["vertebra_target"]), dtype=torch.float32
            ),
            "region_targets": torch.from_numpy(region_targets),
            "region_target_valid": torch.full(
                (N_REGIONS,), has_region_target, dtype=torch.bool
            ),
            "plane_region_valid": torch.from_numpy(
                np.ascontiguousarray(plane_region_valid)
            ),
            "fold": torch.tensor(int(row["fold"]), dtype=torch.int64),
            "study_id": study_id,
            "level": level,
        }
