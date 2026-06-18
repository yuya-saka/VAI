"""DICOM series geometry in patient-coordinate millimeters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pydicom
from pydicom.errors import InvalidDicomError

FloatArray = npt.NDArray[np.float64]

ORIENTATION_TOLERANCE: Final = 1e-4
POSITION_TOLERANCE_MM: Final = 1e-3
SPACING_TOLERANCE: Final = 1e-4

_HEADER_TAGS: Final = (
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
)


class DicomGeometryError(ValueError):
    """Base error for invalid DICOM series geometry."""


class MultipleDicomSeriesError(DicomGeometryError):
    """Raised when a study directory contains multiple DICOM series."""


class GeometryValidationError(DicomGeometryError):
    """Raised when required geometry is missing or inconsistent."""


@dataclass(frozen=True)
class DicomSliceMetadata:
    """Geometry metadata for one DICOM image."""

    path: Path
    sop_instance_uid: str
    series_instance_uid: str
    instance_number: int | None
    image_position: tuple[float, float, float]
    slice_position_mm: float


@dataclass(frozen=True)
class DicomSeriesGeometry:
    """Ordered DICOM series geometry in LPS patient coordinates."""

    series_instance_uid: str
    slices: tuple[DicomSliceMetadata, ...]
    row_direction: tuple[float, float, float]
    column_direction: tuple[float, float, float]
    slice_normal: tuple[float, float, float]
    pixel_spacing: tuple[float, float]
    slice_spacing_mm: float
    rows: int
    columns: int
    spacing_deviations_mm: tuple[float, ...]

    @property
    def slice_positions_mm(self) -> tuple[float, ...]:
        """Return ordered slice positions along the series normal."""
        return tuple(item.slice_position_mm for item in self.slices)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return volume shape as slice, row, column."""
        return (len(self.slices), self.rows, self.columns)

    def has_irregular_slice_spacing(
        self,
        *,
        absolute_tolerance_mm: float = 0.1,
        relative_tolerance: float = 0.05,
    ) -> bool:
        """Return whether any adjacent spacing deviates beyond tolerance."""
        tolerance = max(
            absolute_tolerance_mm,
            relative_tolerance * self.slice_spacing_mm,
        )
        return any(abs(value) > tolerance for value in self.spacing_deviations_mm)

    def voxel_to_patient(
        self,
        voxel_coordinates: npt.ArrayLike,
    ) -> FloatArray:
        """Convert ``(..., slice, row, column)`` coordinates to LPS mm."""
        coordinates = _coordinate_array(voxel_coordinates)
        slice_indices = coordinates[..., 0]
        row_indices = coordinates[..., 1]
        column_indices = coordinates[..., 2]
        origins = self._origins_at_indices(slice_indices)

        row_direction = np.asarray(self.row_direction, dtype=np.float64)
        column_direction = np.asarray(self.column_direction, dtype=np.float64)
        row_spacing, column_spacing = self.pixel_spacing
        return (
            origins
            + column_indices[..., None] * column_spacing * row_direction
            + row_indices[..., None] * row_spacing * column_direction
        )

    def patient_to_voxel(
        self,
        patient_coordinates: npt.ArrayLike,
    ) -> FloatArray:
        """Convert ``(..., x, y, z)`` LPS-mm coordinates to volume indices."""
        coordinates = _coordinate_array(patient_coordinates)
        normal = np.asarray(self.slice_normal, dtype=np.float64)
        projected_positions = np.sum(coordinates * normal, axis=-1)
        slice_indices = _interpolate_with_extrapolation(
            projected_positions,
            np.asarray(self.slice_positions_mm, dtype=np.float64),
            np.arange(len(self.slices), dtype=np.float64),
        )
        origins = self._origins_at_indices(slice_indices)
        offsets = coordinates - origins

        row_direction = np.asarray(self.row_direction, dtype=np.float64)
        column_direction = np.asarray(self.column_direction, dtype=np.float64)
        row_spacing, column_spacing = self.pixel_spacing
        column_indices = np.sum(offsets * row_direction, axis=-1) / column_spacing
        row_indices = np.sum(offsets * column_direction, axis=-1) / row_spacing
        return np.stack((slice_indices, row_indices, column_indices), axis=-1)

    def _origins_at_indices(self, slice_indices: npt.ArrayLike) -> FloatArray:
        indices = np.asarray(slice_indices, dtype=np.float64)
        source_indices = np.arange(len(self.slices), dtype=np.float64)
        origins = np.asarray(
            [item.image_position for item in self.slices],
            dtype=np.float64,
        )
        components = [
            _interpolate_with_extrapolation(
                indices,
                source_indices,
                origins[:, axis],
            )
            for axis in range(3)
        ]
        return np.stack(components, axis=-1)


@dataclass(frozen=True)
class _HeaderRecord:
    path: Path
    sop_instance_uid: str
    series_instance_uid: str
    instance_number: int | None
    image_position: tuple[float, float, float]
    image_orientation: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]
    rows: int
    columns: int


def load_dicom_series(
    study_directory: Path,
    *,
    orientation_tolerance: float = ORIENTATION_TOLERANCE,
    spacing_tolerance: float = SPACING_TOLERANCE,
) -> DicomSeriesGeometry:
    """Read header-only metadata and build one ordered DICOM series."""
    if not study_directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {study_directory}")

    paths = sorted(path for path in study_directory.iterdir() if path.is_file())
    if not paths:
        raise GeometryValidationError("DICOM directory contains no files")

    records = tuple(_read_header(path) for path in paths)
    series_uids = {record.series_instance_uid for record in records}
    if len(series_uids) != 1:
        raise MultipleDicomSeriesError(
            f"Expected one DICOM series, found {len(series_uids)}"
        )

    first = records[0]
    row_direction, column_direction, normal = _validated_orientation(
        first.image_orientation
    )
    for record in records[1:]:
        current_row, current_column, _ = _validated_orientation(
            record.image_orientation
        )
        if not np.allclose(
            current_row,
            row_direction,
            atol=orientation_tolerance,
            rtol=0.0,
        ) or not np.allclose(
            current_column,
            column_direction,
            atol=orientation_tolerance,
            rtol=0.0,
        ):
            raise GeometryValidationError(
                f"Inconsistent image orientation: {record.path}"
            )
        if not np.allclose(
            record.pixel_spacing,
            first.pixel_spacing,
            atol=spacing_tolerance,
            rtol=0.0,
        ):
            raise GeometryValidationError(
                f"Inconsistent pixel spacing: {record.path}"
            )
        if (record.rows, record.columns) != (first.rows, first.columns):
            raise GeometryValidationError(
                f"Inconsistent image dimensions: {record.path}"
            )

    ordered_records = sorted(
        records,
        key=lambda record: float(np.dot(record.image_position, normal)),
    )
    projected_positions = np.asarray(
        [
            float(np.dot(record.image_position, normal))
            for record in ordered_records
        ],
        dtype=np.float64,
    )
    slice_spacing, spacing_deviations = _slice_spacing(projected_positions)

    slices = tuple(
        DicomSliceMetadata(
            path=record.path,
            sop_instance_uid=record.sop_instance_uid,
            series_instance_uid=record.series_instance_uid,
            instance_number=record.instance_number,
            image_position=record.image_position,
            slice_position_mm=float(position),
        )
        for record, position in zip(
            ordered_records,
            projected_positions,
            strict=True,
        )
    )
    return DicomSeriesGeometry(
        series_instance_uid=first.series_instance_uid,
        slices=slices,
        row_direction=_vector3_tuple(row_direction),
        column_direction=_vector3_tuple(column_direction),
        slice_normal=_vector3_tuple(normal),
        pixel_spacing=first.pixel_spacing,
        slice_spacing_mm=slice_spacing,
        rows=first.rows,
        columns=first.columns,
        spacing_deviations_mm=spacing_deviations,
    )


def _read_header(path: Path) -> _HeaderRecord:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=list(_HEADER_TAGS),
        )
    except InvalidDicomError as error:
        raise GeometryValidationError(f"Invalid DICOM file: {path}") from error

    try:
        image_position = cast(
            tuple[float, float, float],
            _float_tuple(dataset.ImagePositionPatient, 3),
        )
        image_orientation = cast(
            tuple[float, float, float, float, float, float],
            _float_tuple(dataset.ImageOrientationPatient, 6),
        )
        pixel_spacing = cast(
            tuple[float, float],
            _float_tuple(dataset.PixelSpacing, 2),
        )
        instance_number = (
            int(dataset.InstanceNumber)
            if "InstanceNumber" in dataset
            else None
        )
        return _HeaderRecord(
            path=path,
            sop_instance_uid=str(dataset.SOPInstanceUID),
            series_instance_uid=str(dataset.SeriesInstanceUID),
            instance_number=instance_number,
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            rows=int(dataset.Rows),
            columns=int(dataset.Columns),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GeometryValidationError(
            f"Missing or invalid DICOM geometry tag: {path}"
        ) from error


def _validated_orientation(
    orientation: tuple[float, float, float, float, float, float],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    row_direction = _normalized_vector(np.asarray(orientation[:3]))
    column_direction = _normalized_vector(np.asarray(orientation[3:]))
    if not np.isclose(
        np.dot(row_direction, column_direction),
        0.0,
        atol=ORIENTATION_TOLERANCE,
    ):
        raise GeometryValidationError("DICOM orientation vectors are not orthogonal")
    normal = _normalized_vector(np.cross(row_direction, column_direction))
    return row_direction, column_direction, normal


def _normalized_vector(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise GeometryValidationError("DICOM orientation vector has zero length")
    return np.asarray(vector, dtype=np.float64) / norm


def _slice_spacing(
    projected_positions: FloatArray,
) -> tuple[float, tuple[float, ...]]:
    if len(projected_positions) < 2:
        raise GeometryValidationError("DICOM series requires at least two slices")
    spacings = np.diff(projected_positions)
    if np.any(spacings <= POSITION_TOLERANCE_MM):
        raise GeometryValidationError("Duplicate or non-monotonic slice positions")
    median_spacing = float(np.median(spacings))
    deviations = tuple(float(value - median_spacing) for value in spacings)
    return median_spacing, deviations


def _coordinate_array(coordinates: npt.ArrayLike) -> FloatArray:
    array = np.asarray(coordinates, dtype=np.float64)
    if array.shape == () or array.shape[-1] != 3:
        raise ValueError("Coordinates must have shape (..., 3)")
    return array


def _float_tuple(values: object, expected_length: int) -> tuple[float, ...]:
    if not isinstance(values, Iterable):
        raise TypeError("DICOM geometry value must be iterable")
    converted = tuple(float(value) for value in values)
    if len(converted) != expected_length:
        raise ValueError(f"Expected {expected_length} values")
    return converted


def _vector3_tuple(values: npt.ArrayLike) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("Expected a three-dimensional vector")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _interpolate_with_extrapolation(
    values: npt.ArrayLike,
    source: FloatArray,
    target: FloatArray,
) -> FloatArray:
    values_array = np.asarray(values, dtype=np.float64)
    if len(source) < 2:
        raise GeometryValidationError("At least two slices are required")

    flat_values = values_array.reshape(-1)
    indices = np.searchsorted(source, flat_values, side="right") - 1
    indices = np.clip(indices, 0, len(source) - 2)
    source_low = source[indices]
    source_high = source[indices + 1]
    fractions = (flat_values - source_low) / (source_high - source_low)
    interpolated = target[indices] + fractions * (
        target[indices + 1] - target[indices]
    )
    return interpolated.reshape(values_array.shape)
