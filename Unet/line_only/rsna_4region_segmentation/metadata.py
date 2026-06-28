"""processing_metadata からの情報読み込み。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import PROCESSING_METADATA_DIR, VERTEBRA_LEVELS


def load_metadata(
    study_id: str,
    metadata_dir: Path | None = None,
) -> dict | None:
    """study の processing_metadata JSON を読み込む。"""
    if metadata_dir is None:
        metadata_dir = PROCESSING_METADATA_DIR
    meta_path = metadata_dir / f"{study_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def find_max_area_plane_index(metadata: dict, vertebra: str) -> int | None:
    """メタデータから max_area_forced プレーンの sequence_index を返す。"""
    planes = metadata["vertebrae"][vertebra]["classifier_planes"]["planes"]
    for plane in planes:
        if plane.get("max_area_forced", False):
            return int(plane["sequence_index"])
    return None


def load_max_area_indices(
    study_id: str,
    metadata_dir: Path | None = None,
) -> dict[str, int]:
    """study メタデータから椎体別 max-area プレーン番号を返す。"""
    if metadata_dir is None:
        metadata_dir = PROCESSING_METADATA_DIR
    metadata = json.loads(
        (metadata_dir / f"{study_id}.json").read_text(encoding="utf-8")
    )
    return {
        vertebra: next(
            i
            for i, plane in enumerate(
                metadata["vertebrae"][vertebra]["classifier_planes"]["planes"]
            )
            if plane.get("max_area_forced")
        )
        for vertebra in VERTEBRA_LEVELS
    }


def load_classifier_plane_z_offsets(
    study_id: str,
    vertebra: str,
    metadata_dir: Path | None = None,
) -> tuple[list[float], float, int]:
    """15 プレーンの z オフセット (mm) と max_area インデックスを返す。

    戻り値:
        (z_offsets_mm, slice_spacing_mm, max_area_idx)
    """
    if metadata_dir is None:
        metadata_dir = PROCESSING_METADATA_DIR
    meta = json.loads((metadata_dir / f"{study_id}.json").read_text(encoding="utf-8"))
    dz: float = meta["dicom_geometry"]["median_slice_spacing_mm"]
    planes = meta["vertebrae"][vertebra]["classifier_planes"]["planes"]

    max_area_idx = next(i for i, p in enumerate(planes) if p.get("max_area_forced"))
    max_center = np.asarray(planes[max_area_idx]["center_lps_mm"], dtype=np.float64)
    normal = np.asarray(planes[max_area_idx]["normal_lps"], dtype=np.float64)

    z_offsets = [
        float(
            np.dot(
                np.asarray(p["center_lps_mm"], dtype=np.float64) - max_center,
                normal,
            )
        )
        for p in planes
    ]
    return z_offsets, dz, max_area_idx
