"""Sample CT and mask planes around the maximum-area vertebra cross-section."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from data_preprocessing.rsna_pipeline.classifier_plane_sampling import (
    WINDOW_LOW_HU,
    FloatVolume,
    Uint8Volume,
    apply_bone_window,
)
from data_preprocessing.rsna_pipeline.dicom_geometry import DicomSeriesGeometry
from data_preprocessing.rsna_pipeline.mask_processing import ProcessedVertebraMask
from data_preprocessing.rsna_pipeline.orientation import OrientationSearchResult
from data_preprocessing.rsna_pipeline.plane_sampling import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PIXEL_SPACING_MM,
    PhysicalPlane,
    sample_nifti_physical_planes,
    sample_physical_planes,
)

SEG_PLANE_COUNT: Final = 5
SEG_SLICE_INDEX_OFFSETS: Final = (-2, -1, 0, 1, 2)


class SegmentationPlaneSamplingError(ValueError):
    """Raised when segmentation-plane sampling fails validation."""


@dataclass(frozen=True)
class SampledSegmentationPlanes:
    """Windowed CT and binary masks for planes around the max-area cross-section."""

    ct: Uint8Volume
    vertebra_mask: Uint8Volume
    channel_offsets_mm: tuple[float, ...]


def sample_segmentation_planes(
    hu_volume: FloatVolume,
    geometry: DicomSeriesGeometry,
    processed_mask: ProcessedVertebraMask,
    orientation: OrientationSearchResult,
) -> SampledSegmentationPlanes:
    """Sample 5 single-channel planes centred on the max-area cross-section."""
    center = np.asarray(orientation.max_area_center_lps_mm, dtype=np.float64)
    normal = np.asarray(orientation.normal_lps, dtype=np.float64)
    offsets_mm = tuple(
        float(index_offset) * geometry.slice_spacing_mm
        for index_offset in SEG_SLICE_INDEX_OFFSETS
    )
    planes = tuple(
        PhysicalPlane(
            center=_vector3(center + offset_mm * normal),
            row_basis=orientation.row_basis_lps,
            column_basis=orientation.column_basis_lps,
        )
        for offset_mm in offsets_mm
    )
    sampled_hu = sample_physical_planes(
        hu_volume,
        geometry,
        planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=1,
        cval=WINDOW_LOW_HU,
    )
    sampled_mask = sample_nifti_physical_planes(
        processed_mask.mask,
        processed_mask.affine_ras,
        planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=0,
        cval=0.0,
    )
    expected_shape = (SEG_PLANE_COUNT, *DEFAULT_OUTPUT_SIZE)
    if sampled_hu.shape != expected_shape:
        raise SegmentationPlaneSamplingError(
            f"HU shape {sampled_hu.shape} does not match expected {expected_shape}"
        )
    ct = apply_bone_window(sampled_hu)
    vertebra_mask = np.asarray(sampled_mask > 0, dtype=np.uint8)
    return SampledSegmentationPlanes(
        ct=ct,
        vertebra_mask=vertebra_mask,
        channel_offsets_mm=offsets_mm,
    )


def write_study_segmentation_outputs_atomic(
    output_directory: Path,
    sampled_vertebrae: dict[str, SampledSegmentationPlanes],
) -> None:
    """Write seg_ct.npy and seg_vertebra_mask.npy into each level directory."""
    for level, sampled in sampled_vertebrae.items():
        level_directory = output_directory / level
        level_directory.mkdir(parents=True, exist_ok=True)
        write_npy_atomic(level_directory / "seg_ct.npy", sampled.ct)
        write_npy_atomic(
            level_directory / "seg_vertebra_mask.npy",
            sampled.vertebra_mask,
        )


def segmentation_sampling_metadata(
    sampled: SampledSegmentationPlanes,
    output_directory: Path,
) -> dict[str, object]:
    """Serialize segmentation sampling settings and output paths."""
    return {
        "seg_ct_path": str(output_directory / "seg_ct.npy"),
        "seg_vertebra_mask_path": str(output_directory / "seg_vertebra_mask.npy"),
        "output_size_row_column": list(DEFAULT_OUTPUT_SIZE),
        "pixel_spacing_row_column_mm": list(DEFAULT_PIXEL_SPACING_MM),
        "slice_index_offsets": list(SEG_SLICE_INDEX_OFFSETS),
        "channel_offsets_mm": list(sampled.channel_offsets_mm),
        "plane_count": SEG_PLANE_COUNT,
        "ct_shape_plane_row_column": list(sampled.ct.shape),
    }


def write_npy_atomic(path: Path, array: npt.NDArray) -> None:
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as file:
            np.save(file, array)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _vector3(values: npt.ArrayLike) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    return (float(v[0]), float(v[1]), float(v[2]))
