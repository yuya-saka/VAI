"""Cleaning and physical geometry extraction for vertebra masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]

from data_preprocessing.rsna_pipeline.dicom_geometry import DicomSeriesGeometry

FloatArray = npt.NDArray[np.float64]
BinaryMask = npt.NDArray[np.uint8]

RAS_TO_LPS: Final = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
DEFAULT_ALIGNMENT_TOLERANCE_VOXELS: Final = 1.5
DEFAULT_ALIGNMENT_SAMPLE_COUNT: Final = 4096


class MaskProcessingError(ValueError):
    """Raised when a vertebra mask cannot be processed safely."""


@dataclass(frozen=True)
class MaskDicomAlignment:
    """Mask extent expressed in DICOM volume coordinates."""

    minimum_voxel: tuple[float, float, float]
    maximum_voxel: tuple[float, float, float]
    outside_fraction: float
    sample_count: int


@dataclass(frozen=True)
class ProcessedVertebraMask:
    """Cleaned mask and physical measurements in LPS millimeters."""

    mask: BinaryMask
    affine_ras: FloatArray
    voxel_count: int
    component_count: int
    retained_fraction: float
    volume_mm3: float
    centroid_lps_mm: tuple[float, float, float]
    bbox_min_lps_mm: tuple[float, float, float]
    bbox_max_lps_mm: tuple[float, float, float]
    superior_inferior_range_mm: tuple[float, float]
    dicom_alignment: MaskDicomAlignment | None


def load_and_process_vertebra_mask(
    mask_path: Path,
    *,
    dicom_geometry: DicomSeriesGeometry | None = None,
    superior_inferior_axis_lps: npt.ArrayLike = (0.0, 0.0, 1.0),
    alignment_tolerance_voxels: float = DEFAULT_ALIGNMENT_TOLERANCE_VOXELS,
) -> ProcessedVertebraMask:
    """Load one NIfTI mask and return its cleaned physical geometry."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Vertebra mask not found: {mask_path}")

    image: Any = nib.load(mask_path)
    mask_data = np.asanyarray(image.dataobj)
    return process_vertebra_mask(
        mask_data,
        np.asarray(image.affine, dtype=np.float64),
        dicom_geometry=dicom_geometry,
        superior_inferior_axis_lps=superior_inferior_axis_lps,
        alignment_tolerance_voxels=alignment_tolerance_voxels,
    )


def process_vertebra_mask(
    mask_data: npt.ArrayLike,
    affine_ras: npt.ArrayLike,
    *,
    dicom_geometry: DicomSeriesGeometry | None = None,
    superior_inferior_axis_lps: npt.ArrayLike = (0.0, 0.0, 1.0),
    alignment_tolerance_voxels: float = DEFAULT_ALIGNMENT_TOLERANCE_VOXELS,
) -> ProcessedVertebraMask:
    """Clean a 3D mask and calculate physical ROI measurements."""
    data = np.asarray(mask_data)
    affine = _validated_affine(affine_ras)
    axis = _unit_vector(superior_inferior_axis_lps)
    _validate_mask_data(data)
    if alignment_tolerance_voxels < 0.0:
        raise MaskProcessingError("Alignment tolerance must be non-negative")

    binary_mask = data > 0
    (
        cleaned_mask,
        occupied_indices,
        component_count,
        retained_fraction,
    ) = _largest_component(binary_mask)
    physical_centers_lps = _indices_to_lps(occupied_indices, affine)
    linear_lps = RAS_TO_LPS @ affine[:3, :3]

    centroid = physical_centers_lps.mean(axis=0)
    half_extent_lps = 0.5 * np.sum(np.abs(linear_lps), axis=1)
    bbox_min = physical_centers_lps.min(axis=0) - half_extent_lps
    bbox_max = physical_centers_lps.max(axis=0) + half_extent_lps

    projections = physical_centers_lps @ axis
    projection_half_extent = 0.5 * float(np.sum(np.abs(axis @ linear_lps)))
    superior_inferior_range = (
        float(projections.min() - projection_half_extent),
        float(projections.max() + projection_half_extent),
    )

    alignment = None
    if dicom_geometry is not None:
        alignment_indices = _sample_alignment_indices(occupied_indices)
        alignment = validate_mask_dicom_alignment(
            _indices_to_lps(alignment_indices, affine),
            dicom_geometry,
            tolerance_voxels=alignment_tolerance_voxels,
        )

    voxel_count = int(occupied_indices.shape[0])
    voxel_volume_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return ProcessedVertebraMask(
        mask=cleaned_mask,
        affine_ras=affine,
        voxel_count=voxel_count,
        component_count=component_count,
        retained_fraction=retained_fraction,
        volume_mm3=voxel_count * voxel_volume_mm3,
        centroid_lps_mm=_vector3_tuple(centroid),
        bbox_min_lps_mm=_vector3_tuple(bbox_min),
        bbox_max_lps_mm=_vector3_tuple(bbox_max),
        superior_inferior_range_mm=superior_inferior_range,
        dicom_alignment=alignment,
    )


def validate_mask_dicom_alignment(
    physical_centers_lps: npt.ArrayLike,
    dicom_geometry: DicomSeriesGeometry,
    *,
    tolerance_voxels: float = DEFAULT_ALIGNMENT_TOLERANCE_VOXELS,
) -> MaskDicomAlignment:
    """Require cleaned mask voxel centers to lie inside the DICOM volume."""
    coordinates = np.asarray(physical_centers_lps, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise MaskProcessingError("Physical mask coordinates must have shape (N, 3)")
    if coordinates.shape[0] == 0:
        raise MaskProcessingError("Physical mask coordinates must not be empty")
    if tolerance_voxels < 0.0:
        raise MaskProcessingError("Alignment tolerance must be non-negative")

    voxel_coordinates = dicom_geometry.patient_to_voxel(coordinates)
    minimum_voxel = voxel_coordinates.min(axis=0)
    maximum_voxel = voxel_coordinates.max(axis=0)
    shape_limit = np.asarray(dicom_geometry.shape, dtype=np.float64) - 1.0
    outside = np.any(
        (voxel_coordinates < -tolerance_voxels)
        | (voxel_coordinates > shape_limit + tolerance_voxels),
        axis=1,
    )
    outside_fraction = float(outside.mean())
    if outside_fraction > 0.0:
        raise MaskProcessingError(
            "Vertebra mask extends outside the DICOM patient-coordinate volume"
        )

    return MaskDicomAlignment(
        minimum_voxel=_vector3_tuple(minimum_voxel),
        maximum_voxel=_vector3_tuple(maximum_voxel),
        outside_fraction=outside_fraction,
        sample_count=coordinates.shape[0],
    )


def _largest_component(
    binary_mask: npt.NDArray[np.bool_],
) -> tuple[BinaryMask, npt.NDArray[np.intp], int, float]:
    occupied_indices = np.argwhere(binary_mask)
    if occupied_indices.shape[0] == 0:
        raise MaskProcessingError("Vertebra mask is empty")

    minimum_indices = occupied_indices.min(axis=0)
    maximum_indices = occupied_indices.max(axis=0) + 1
    crop_slices = tuple(
        slice(int(minimum), int(maximum))
        for minimum, maximum in zip(
            minimum_indices,
            maximum_indices,
            strict=True,
        )
    )
    cropped_mask = binary_mask[crop_slices]
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, component_count = ndimage.label(cropped_mask, structure=structure)

    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(component_sizes)) + 1
    largest_size = int(component_sizes[largest_label - 1])
    total_size = int(component_sizes.sum())
    local_indices = np.argwhere(labeled == largest_label)
    largest_indices = local_indices + minimum_indices
    cleaned_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    cleaned_mask[tuple(largest_indices.T)] = 1
    return (
        cleaned_mask,
        largest_indices,
        component_count,
        largest_size / total_size,
    )


def _sample_alignment_indices(
    occupied_indices: npt.NDArray[np.intp],
    *,
    maximum_samples: int = DEFAULT_ALIGNMENT_SAMPLE_COUNT,
) -> npt.NDArray[np.intp]:
    if occupied_indices.shape[0] <= maximum_samples:
        return occupied_indices

    extrema_indices = {
        int(np.argmin(occupied_indices[:, axis]))
        for axis in range(occupied_indices.shape[1])
    }
    extrema_indices.update(
        int(np.argmax(occupied_indices[:, axis]))
        for axis in range(occupied_indices.shape[1])
    )
    remaining_count = maximum_samples - len(extrema_indices)
    evenly_spaced = np.linspace(
        0,
        occupied_indices.shape[0] - 1,
        remaining_count,
        dtype=np.intp,
    )
    selected_indices = np.asarray(
        sorted(extrema_indices | set(int(value) for value in evenly_spaced)),
        dtype=np.intp,
    )
    return occupied_indices[selected_indices]


def _indices_to_lps(
    indices: npt.NDArray[np.intp],
    affine_ras: FloatArray,
) -> FloatArray:
    homogeneous = np.column_stack(
        (indices.astype(np.float64), np.ones(indices.shape[0]))
    )
    physical_ras = (homogeneous @ affine_ras.T)[:, :3]
    return physical_ras @ RAS_TO_LPS.T


def _validated_affine(affine_ras: npt.ArrayLike) -> FloatArray:
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4):
        raise MaskProcessingError("NIfTI affine must have shape (4, 4)")
    if not np.all(np.isfinite(affine)):
        raise MaskProcessingError("NIfTI affine must contain finite values")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0)):
        raise MaskProcessingError("NIfTI affine has an invalid homogeneous row")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise MaskProcessingError("NIfTI affine is singular")
    return affine


def _validate_mask_data(mask_data: npt.NDArray[np.generic]) -> None:
    if mask_data.ndim != 3:
        raise MaskProcessingError("Vertebra mask must be three-dimensional")
    if not np.issubdtype(mask_data.dtype, np.number):
        raise MaskProcessingError("Vertebra mask must be numeric")
    if not np.all(np.isfinite(mask_data)):
        raise MaskProcessingError("Vertebra mask must contain finite values")


def _unit_vector(values: npt.ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise MaskProcessingError("Superior-inferior axis must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise MaskProcessingError("Superior-inferior axis must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise MaskProcessingError("Superior-inferior axis must have non-zero length")
    return vector / norm


def _vector3_tuple(values: npt.ArrayLike) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return (float(vector[0]), float(vector[1]), float(vector[2]))
