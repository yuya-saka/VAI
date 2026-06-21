"""Sample and persist fixed classifier CT and vertebra-mask planes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pydicom

from data_preprocessing.rsna_pipeline.classifier_planes import (
    ClassifierPlane,
    ClassifierPlanePlan,
)
from data_preprocessing.rsna_pipeline.dicom_geometry import DicomSeriesGeometry
from data_preprocessing.rsna_pipeline.mask_processing import ProcessedVertebraMask
from data_preprocessing.rsna_pipeline.plane_sampling import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PIXEL_SPACING_MM,
    PhysicalPlane,
    sample_nifti_physical_planes,
    sample_physical_planes,
)

WINDOW_LEVEL_HU: Final = 400.0
WINDOW_WIDTH_HU: Final = 2000.0
WINDOW_LOW_HU: Final = WINDOW_LEVEL_HU - WINDOW_WIDTH_HU / 2.0
WINDOW_HIGH_HU: Final = WINDOW_LEVEL_HU + WINDOW_WIDTH_HU / 2.0
EXPECTED_PLANE_COUNT: Final = 15
CHANNEL_INDEX_OFFSETS: Final = (-2, -1, 0, 1, 2)
EXPECTED_CHANNEL_COUNT: Final = len(CHANNEL_INDEX_OFFSETS)

FloatVolume = npt.NDArray[np.float32]
Uint8Volume = npt.NDArray[np.uint8]


class ClassifierPlaneSamplingError(ValueError):
    """Raised when classifier-plane sampling fails validation."""


@dataclass(frozen=True)
class ClassifierPlaneSamplingQc:
    """QC measurements for one vertebra's sampled planes."""

    ct_shape_plane_channel_row_column: tuple[int, int, int, int]
    shape_plane_row_column: tuple[int, int, int]
    mask_pixel_counts: tuple[int, ...]
    mask_boundary_pixel_counts: tuple[int, ...]
    mask_touches_fov_boundary: tuple[bool, ...]
    empty_mask_plane_indices: tuple[int, ...]
    bbox_forced_empty_mask_plane_indices: tuple[int, ...]
    all_ct_finite: bool
    all_masks_non_empty: bool
    all_masks_inside_fov: bool


@dataclass(frozen=True)
class SampledClassifierPlanes:
    """Windowed CT, binary masks, and sampling QC."""

    ct: Uint8Volume
    vertebra_mask: Uint8Volume
    channel_offsets_mm: tuple[float, ...]
    qc: ClassifierPlaneSamplingQc


def load_hu_volume(geometry: DicomSeriesGeometry) -> FloatVolume:
    """Read ordered DICOM pixel data and convert every slice to HU."""
    slices: list[FloatVolume] = []
    for slice_metadata in geometry.slices:
        dataset = pydicom.dcmread(slice_metadata.path)
        try:
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
            slope = float(dataset.get("RescaleSlope", 1.0))
            intercept = float(dataset.get("RescaleIntercept", 0.0))
        except (AttributeError, TypeError, ValueError) as error:
            raise ClassifierPlaneSamplingError(
                f"Invalid DICOM pixel data: {slice_metadata.path}"
            ) from error
        if pixels.shape != (geometry.rows, geometry.columns):
            raise ClassifierPlaneSamplingError(
                f"DICOM pixel shape mismatch: {slice_metadata.path}"
            )
        hu_slice = pixels * slope + intercept
        if not np.all(np.isfinite(hu_slice)):
            raise ClassifierPlaneSamplingError(
                f"DICOM HU values are not finite: {slice_metadata.path}"
            )
        slices.append(np.asarray(hu_slice, dtype=np.float32))
    volume = np.stack(slices, axis=0)
    if volume.shape != geometry.shape:
        raise ClassifierPlaneSamplingError("HU volume shape does not match geometry")
    return np.asarray(volume, dtype=np.float32)


def apply_bone_window(
    hu_values: npt.ArrayLike,
    *,
    level_hu: float = WINDOW_LEVEL_HU,
    width_hu: float = WINDOW_WIDTH_HU,
) -> Uint8Volume:
    """Apply the established self-site bone window and convert to uint8."""
    if width_hu <= 0.0:
        raise ClassifierPlaneSamplingError("Window width must be positive")
    values = np.asarray(hu_values, dtype=np.float32)
    lower = level_hu - width_hu / 2.0
    upper = level_hu + width_hu / 2.0
    normalized = (np.clip(values, lower, upper) - lower) / (upper - lower)
    return np.asarray(normalized * 255.0, dtype=np.uint8)


def sample_classifier_planes(
    hu_volume: FloatVolume,
    geometry: DicomSeriesGeometry,
    processed_mask: ProcessedVertebraMask,
    plane_plan: ClassifierPlanePlan,
) -> SampledClassifierPlanes:
    """Sample one vertebra's CT and mask planes and enforce output QC."""
    if len(plane_plan.planes) != EXPECTED_PLANE_COUNT:
        raise ClassifierPlaneSamplingError(
            f"Expected {EXPECTED_PLANE_COUNT} classifier planes"
        )
    center_planes = tuple(plane.physical_plane() for plane in plane_plan.planes)
    channel_offsets_mm = tuple(
        float(offset) * geometry.slice_spacing_mm for offset in CHANNEL_INDEX_OFFSETS
    )
    channel_planes = tuple(
        shifted_plane
        for plane in plane_plan.planes
        for shifted_plane in _channel_physical_planes(
            plane,
            channel_offsets_mm,
        )
    )
    sampled_hu = sample_physical_planes(
        hu_volume,
        geometry,
        channel_planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=1,
        cval=WINDOW_LOW_HU,
    )
    sampled_mask = sample_nifti_physical_planes(
        processed_mask.mask,
        processed_mask.affine_ras,
        center_planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=0,
        cval=0.0,
    )
    ct = apply_bone_window(sampled_hu).reshape(
        EXPECTED_PLANE_COUNT,
        EXPECTED_CHANNEL_COUNT,
        *DEFAULT_OUTPUT_SIZE,
    )
    binary_mask = np.asarray(
        np.asarray(sampled_mask, dtype=np.uint8) > 0,
        dtype=np.uint8,
    )
    qc = _sampling_qc(ct, binary_mask)
    if not qc.all_ct_finite:
        raise ClassifierPlaneSamplingError("Sampled CT contains non-finite values")
    bbox_forced_empty_indices = tuple(
        index
        for index in qc.empty_mask_plane_indices
        if plane_plan.planes[index].bbox_forced
    )
    invalid_empty_indices = set(qc.empty_mask_plane_indices) - set(
        bbox_forced_empty_indices
    )
    if invalid_empty_indices:
        raise ClassifierPlaneSamplingError("A classifier plane has an empty mask")
    qc = replace(
        qc,
        bbox_forced_empty_mask_plane_indices=bbox_forced_empty_indices,
    )
    return SampledClassifierPlanes(
        ct=ct,
        vertebra_mask=binary_mask,
        channel_offsets_mm=channel_offsets_mm,
        qc=qc,
    )


def write_study_classifier_outputs_atomic(
    output_directory: Path,
    sampled_vertebrae: dict[str, SampledClassifierPlanes],
) -> None:
    """Atomically replace one study directory containing all vertebra outputs."""
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            suffix=".tmp",
            dir=output_directory.parent,
        )
    )
    backup_directory = output_directory.with_name(f".{output_directory.name}.backup")
    try:
        for level, sampled in sampled_vertebrae.items():
            level_directory = staging_directory / level
            level_directory.mkdir()
            np.save(level_directory / "ct.npy", sampled.ct)
            np.save(level_directory / "vertebra_mask.npy", sampled.vertebra_mask)
            (level_directory / "sampling_qc.json").write_text(
                json.dumps(asdict(sampled.qc), indent=2) + "\n",
                encoding="utf-8",
            )
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
        if output_directory.exists():
            os.replace(output_directory, backup_directory)
        try:
            os.replace(staging_directory, output_directory)
        except Exception:
            if backup_directory.exists():
                os.replace(backup_directory, output_directory)
            raise
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise


def sampling_metadata(
    sampled: SampledClassifierPlanes,
    output_directory: Path,
) -> dict[str, object]:
    """Serialize sampling settings, output paths, and QC."""
    return {
        "output_directory": str(output_directory),
        "ct_path": str(output_directory / "ct.npy"),
        "vertebra_mask_path": str(output_directory / "vertebra_mask.npy"),
        "output_size_row_column": list(DEFAULT_OUTPUT_SIZE),
        "pixel_spacing_row_column_mm": list(DEFAULT_PIXEL_SPACING_MM),
        "channel_offsets_mm": list(sampled.channel_offsets_mm),
        "channel_order": "center_minus_2_to_center_plus_2",
        "window_level_hu": WINDOW_LEVEL_HU,
        "window_width_hu": WINDOW_WIDTH_HU,
        "hu_clip": [WINDOW_LOW_HU, WINDOW_HIGH_HU],
        "ct_interpolation": "linear",
        "mask_interpolation": "nearest",
        "qc": asdict(sampled.qc),
    }


def _sampling_qc(
    ct: Uint8Volume,
    mask: Uint8Volume,
) -> ClassifierPlaneSamplingQc:
    expected_ct_shape = (
        EXPECTED_PLANE_COUNT,
        EXPECTED_CHANNEL_COUNT,
        *DEFAULT_OUTPUT_SIZE,
    )
    expected_mask_shape = (EXPECTED_PLANE_COUNT, *DEFAULT_OUTPUT_SIZE)
    if ct.shape != expected_ct_shape or mask.shape != expected_mask_shape:
        raise ClassifierPlaneSamplingError(
            f"Sampled CT must have shape {expected_ct_shape} and mask "
            f"must have shape {expected_mask_shape}"
        )
    mask_pixel_counts = tuple(int(value) for value in mask.sum(axis=(1, 2)))
    boundary_pixels = np.concatenate(
        (
            mask[:, 0, :],
            mask[:, -1, :],
            mask[:, 1:-1, 0],
            mask[:, 1:-1, -1],
        ),
        axis=1,
    )
    boundary_pixel_counts = tuple(
        int(value)
        for value in np.asarray(
            boundary_pixels.sum(axis=1),
            dtype=np.int64,
        ).tolist()
    )
    touches_boundary = tuple(
        bool(value)
        for value in np.asarray(
            boundary_pixels.any(axis=1),
            dtype=np.bool_,
        ).tolist()
    )
    return ClassifierPlaneSamplingQc(
        ct_shape_plane_channel_row_column=expected_ct_shape,
        shape_plane_row_column=expected_mask_shape,
        mask_pixel_counts=mask_pixel_counts,
        mask_boundary_pixel_counts=boundary_pixel_counts,
        mask_touches_fov_boundary=touches_boundary,
        empty_mask_plane_indices=tuple(
            index for index, value in enumerate(mask_pixel_counts) if value == 0
        ),
        bbox_forced_empty_mask_plane_indices=(),
        all_ct_finite=bool(np.all(np.isfinite(ct))),
        all_masks_non_empty=all(value > 0 for value in mask_pixel_counts),
        all_masks_inside_fov=not any(touches_boundary),
    )


def _channel_physical_planes(
    plane: ClassifierPlane,
    channel_offsets_mm: tuple[float, ...],
) -> tuple[PhysicalPlane, ...]:
    center = np.asarray(plane.center_lps_mm, dtype=np.float64)
    normal = np.asarray(plane.normal_lps, dtype=np.float64)
    shifted_planes: list[PhysicalPlane] = []
    for offset_mm in channel_offsets_mm:
        shifted_center = center + offset_mm * normal
        shifted_planes.append(
            PhysicalPlane(
                center=(
                    float(shifted_center[0]),
                    float(shifted_center[1]),
                    float(shifted_center[2]),
                ),
                row_basis=plane.row_basis_lps,
                column_basis=plane.column_basis_lps,
            )
        )
    return tuple(shifted_planes)
