"""Direct physical-plane sampling from a native DICOM volume."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]

from data_preprocessing.rsna_pipeline.dicom_geometry import DicomSeriesGeometry
from data_preprocessing.rsna_pipeline.mask_processing import RAS_TO_LPS

DEFAULT_OUTPUT_SIZE: Final = (224, 224)
DEFAULT_PIXEL_SPACING_MM: Final = (0.4, 0.4)
DEFAULT_CHUNK_SIZE: Final = 8


@dataclass(frozen=True)
class PhysicalPlane:
    """One output plane described in LPS patient coordinates."""

    center: tuple[float, float, float]
    row_basis: tuple[float, float, float]
    column_basis: tuple[float, float, float]


def sample_physical_planes(
    volume: npt.NDArray[np.generic],
    geometry: DicomSeriesGeometry,
    planes: Sequence[PhysicalPlane],
    *,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    pixel_spacing_mm: tuple[float, float] = DEFAULT_PIXEL_SPACING_MM,
    interpolation_order: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cval: float = 0.0,
) -> npt.NDArray[np.generic]:
    """Sample physical planes without rotating or resampling the full volume."""
    _validate_sampling_inputs(
        volume=volume,
        geometry=geometry,
        planes=planes,
        output_size=output_size,
        pixel_spacing_mm=pixel_spacing_mm,
        interpolation_order=interpolation_order,
        chunk_size=chunk_size,
    )
    if not planes:
        return np.empty((0, *output_size), dtype=volume.dtype)

    row_offsets, column_offsets = _pixel_offsets(
        output_size,
        pixel_spacing_mm,
    )
    sampled_chunks: list[npt.NDArray[np.generic]] = []
    for start in range(0, len(planes), chunk_size):
        plane_chunk = planes[start : start + chunk_size]
        patient_coordinates = _patient_coordinates(
            plane_chunk,
            row_offsets,
            column_offsets,
        )
        voxel_coordinates = geometry.patient_to_voxel(patient_coordinates)
        sampled_chunks.append(
            _sample_coordinate_chunk(
                volume,
                voxel_coordinates,
                plane_count=len(plane_chunk),
                output_size=output_size,
                interpolation_order=interpolation_order,
                cval=cval,
            )
        )
    return np.concatenate(sampled_chunks, axis=0)


def sample_nifti_physical_planes(
    volume: npt.NDArray[np.generic],
    affine_ras: npt.ArrayLike,
    planes: Sequence[PhysicalPlane],
    *,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    pixel_spacing_mm: tuple[float, float] = DEFAULT_PIXEL_SPACING_MM,
    interpolation_order: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cval: float = 0.0,
) -> npt.NDArray[np.generic]:
    """Sample LPS physical planes from a volume described by a NIfTI RAS affine."""
    affine = _validated_nifti_affine(affine_ras)
    _validate_common_inputs(
        volume=volume,
        planes=planes,
        output_size=output_size,
        pixel_spacing_mm=pixel_spacing_mm,
        interpolation_order=interpolation_order,
        chunk_size=chunk_size,
    )
    if not planes:
        return np.empty((0, *output_size), dtype=volume.dtype)

    inverse_affine = np.linalg.inv(affine)
    row_offsets, column_offsets = _pixel_offsets(
        output_size,
        pixel_spacing_mm,
    )
    sampled_chunks: list[npt.NDArray[np.generic]] = []
    for start in range(0, len(planes), chunk_size):
        plane_chunk = planes[start : start + chunk_size]
        patient_lps = _patient_coordinates(
            plane_chunk,
            row_offsets,
            column_offsets,
        )
        patient_ras = patient_lps @ RAS_TO_LPS
        homogeneous = np.concatenate(
            (
                patient_ras,
                np.ones((*patient_ras.shape[:-1], 1), dtype=np.float64),
            ),
            axis=-1,
        )
        voxel_coordinates = (homogeneous @ inverse_affine.T)[..., :3]
        sampled_chunks.append(
            _sample_coordinate_chunk(
                volume,
                voxel_coordinates,
                plane_count=len(plane_chunk),
                output_size=output_size,
                interpolation_order=interpolation_order,
                cval=cval,
            )
        )
    return np.concatenate(sampled_chunks, axis=0)


def _sample_coordinate_chunk(
    volume: npt.NDArray[np.generic],
    voxel_coordinates: npt.NDArray[np.float64],
    *,
    plane_count: int,
    output_size: tuple[int, int],
    interpolation_order: int,
    cval: float,
) -> npt.NDArray[np.generic]:
    coordinate_matrix = np.moveaxis(voxel_coordinates, -1, 0).reshape(3, -1)
    sampled = ndimage.map_coordinates(
        volume,
        coordinate_matrix,
        order=interpolation_order,
        mode="constant",
        cval=cval,
        prefilter=interpolation_order > 1,
    )
    return cast(
        npt.NDArray[np.generic],
        sampled.reshape(plane_count, output_size[0], output_size[1]),
    )


def _patient_coordinates(
    planes: Sequence[PhysicalPlane],
    row_offsets: npt.NDArray[np.float64],
    column_offsets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    centers = np.asarray([plane.center for plane in planes], dtype=np.float64)
    row_bases = np.asarray(
        [_unit_vector(plane.row_basis) for plane in planes],
        dtype=np.float64,
    )
    column_bases = np.asarray(
        [_unit_vector(plane.column_basis) for plane in planes],
        dtype=np.float64,
    )
    dot_products = np.sum(row_bases * column_bases, axis=1)
    if not np.allclose(dot_products, 0.0, atol=1e-6, rtol=0.0):
        raise ValueError("Plane row and column bases must be orthogonal")

    return (
        centers[:, None, None, :]
        + column_offsets[None, :, :, None] * row_bases[:, None, None, :]
        + row_offsets[None, :, :, None] * column_bases[:, None, None, :]
    )


def _pixel_offsets(
    output_size: tuple[int, int],
    pixel_spacing_mm: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    height, width = output_size
    row_spacing, column_spacing = pixel_spacing_mm
    row_values = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * (
        row_spacing
    )
    column_values = (
        np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    ) * column_spacing
    return np.meshgrid(row_values, column_values, indexing="ij")


def _unit_vector(values: tuple[float, float, float]) -> npt.NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Plane basis vector must have non-zero length")
    return vector / norm


def _validate_sampling_inputs(
    *,
    volume: npt.NDArray[np.generic],
    geometry: DicomSeriesGeometry,
    planes: Sequence[PhysicalPlane],
    output_size: tuple[int, int],
    pixel_spacing_mm: tuple[float, float],
    interpolation_order: int,
    chunk_size: int,
) -> None:
    if volume.shape != geometry.shape:
        raise ValueError(
            f"Volume shape {volume.shape} does not match geometry {geometry.shape}"
        )
    _validate_common_inputs(
        volume=volume,
        planes=planes,
        output_size=output_size,
        pixel_spacing_mm=pixel_spacing_mm,
        interpolation_order=interpolation_order,
        chunk_size=chunk_size,
    )


def _validate_common_inputs(
    *,
    volume: npt.NDArray[np.generic],
    planes: Sequence[PhysicalPlane],
    output_size: tuple[int, int],
    pixel_spacing_mm: tuple[float, float],
    interpolation_order: int,
    chunk_size: int,
) -> None:
    if volume.ndim != 3:
        raise ValueError("Volume must be three-dimensional")
    if any(value <= 0 for value in output_size):
        raise ValueError("Output dimensions must be positive")
    if any(value <= 0.0 for value in pixel_spacing_mm):
        raise ValueError("Pixel spacing must be positive")
    if interpolation_order not in range(6):
        raise ValueError("Interpolation order must be between 0 and 5")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if not isinstance(planes, Sequence):
        raise TypeError("Planes must be a sequence")


def _validated_nifti_affine(affine_ras: npt.ArrayLike) -> npt.NDArray[np.float64]:
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("NIfTI affine must have shape (4, 4)")
    if not np.all(np.isfinite(affine)):
        raise ValueError("NIfTI affine must be finite")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise ValueError("NIfTI affine must be invertible")
    return affine
