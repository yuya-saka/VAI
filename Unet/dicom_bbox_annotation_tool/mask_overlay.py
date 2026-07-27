"""椎体NIfTI maskを元DICOMスライスへ投影する処理。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import nibabel as nib
import numpy as np
import numpy.typing as npt
from pydicom.dataset import Dataset
from scipy import ndimage  # type: ignore[import-untyped]

Uint8Array = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

LPS_TO_RAS: Final = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
DEFAULT_MASK_CACHE_ITEMS: Final = 2


class MaskOverlayError(ValueError):
    """椎体maskを安全にDICOMへ投影できない場合に送出する。"""


@dataclass(frozen=True)
class VertebraMaskVolume:
    """処理済み椎体maskとRAS物理座標からの逆変換。"""

    mask: Uint8Array
    inverse_affine_ras: FloatArray


class VertebraMaskCache:
    """処理済み3D椎体maskを少数だけメモリへ保持する。"""

    def __init__(self, max_items: int = DEFAULT_MASK_CACHE_ITEMS) -> None:
        if max_items <= 0:
            raise ValueError("max_itemsは正数である必要があります")
        self._max_items = max_items
        self._items: dict[tuple[str, str], VertebraMaskVolume] = {}
        self._lock = threading.Lock()

    def get(
        self,
        segmentation_dir: Path,
        study_id: str,
        level: str,
    ) -> VertebraMaskVolume:
        """study・椎体に対応する処理済みmaskを返す。"""
        key = study_id, level
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                return cached
            loaded = load_mask_volume(
                segmentation_dir / study_id / f"vertebrae_{level}.nii.gz"
            )
            if len(self._items) >= self._max_items:
                self._items.pop(next(iter(self._items)))
            self._items[key] = loaded
            return loaded


def load_mask_volume(path: Path) -> VertebraMaskVolume:
    """前処理と同じ最大連結成分maskを読み込む。"""
    try:
        image: Any = nib.load(path)
        data = np.asanyarray(image.dataobj)
        affine_ras = np.asarray(image.affine, dtype=np.float64)
        mask = _largest_component(data)
        inverse_affine = np.linalg.inv(affine_ras)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        raise MaskOverlayError(f"椎体maskを読み込めません: {path}") from error
    return VertebraMaskVolume(
        mask=mask,
        inverse_affine_ras=np.asarray(inverse_affine, dtype=np.float64),
    )


def _largest_component(data: npt.ArrayLike) -> Uint8Array:
    """前処理と同じ26近傍で最大連結成分だけを残す。"""
    values = np.asarray(data)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.number):
        raise MaskOverlayError("椎体maskは数値の3次元配列である必要があります")
    if not np.all(np.isfinite(values)):
        raise MaskOverlayError("椎体maskに非有限値があります")
    binary = values > 0
    occupied = np.argwhere(binary)
    if not len(occupied):
        raise MaskOverlayError("椎体maskが空です")
    minimum = occupied.min(axis=0)
    maximum = occupied.max(axis=0) + 1
    crop = tuple(
        slice(int(low), int(high)) for low, high in zip(minimum, maximum, strict=True)
    )
    labeled, component_count = ndimage.label(
        binary[crop],
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(sizes)) + 1
    local_indices = np.argwhere(labeled == largest_label)
    largest_indices = local_indices + minimum
    mask = np.zeros(binary.shape, dtype=np.uint8)
    mask[tuple(largest_indices.T)] = 1
    if component_count <= 0:
        raise MaskOverlayError("椎体maskの連結成分がありません")
    return mask


def sample_mask_on_dicom(
    dataset: Dataset,
    volume: VertebraMaskVolume,
) -> Uint8Array:
    """元DICOMの全ピクセル中心で3D maskを最近傍サンプリングする。"""
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        origin_lps = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        spacing = np.asarray(dataset.PixelSpacing, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise MaskOverlayError("DICOMのmask投影用ジオメトリが不正です") from error
    if rows <= 0 or columns <= 0:
        raise MaskOverlayError("DICOM画像サイズが不正です")
    if origin_lps.shape != (3,) or orientation.shape != (6,) or spacing.shape != (2,):
        raise MaskOverlayError("DICOMジオメトリのshapeが不正です")
    if np.any(spacing <= 0.0):
        raise MaskOverlayError("DICOM PixelSpacingは正数である必要があります")

    inverse_linear = volume.inverse_affine_ras[:3, :3]
    origin_ras = LPS_TO_RAS @ origin_lps
    column_step_ras = LPS_TO_RAS @ (orientation[:3] * spacing[1])
    row_step_ras = LPS_TO_RAS @ (orientation[3:] * spacing[0])
    origin_voxel = inverse_linear @ origin_ras + volume.inverse_affine_ras[:3, 3]
    column_step_voxel = inverse_linear @ column_step_ras
    row_step_voxel = inverse_linear @ row_step_ras

    row_coordinates = np.arange(rows, dtype=np.float64)[:, None]
    column_coordinates = np.arange(columns, dtype=np.float64)[None, :]
    voxel_coordinates = (
        origin_voxel[:, None, None]
        + row_step_voxel[:, None, None] * row_coordinates[None, :, :]
        + column_step_voxel[:, None, None] * column_coordinates[None, :, :]
    )
    indices = np.rint(voxel_coordinates).astype(np.int64)
    shape = np.asarray(volume.mask.shape, dtype=np.int64)
    valid = np.all((indices >= 0) & (indices < shape[:, None, None]), axis=0)
    sampled = np.zeros((rows, columns), dtype=np.uint8)
    sampled[valid] = volume.mask[
        indices[0, valid],
        indices[1, valid],
        indices[2, valid],
    ]
    return sampled
