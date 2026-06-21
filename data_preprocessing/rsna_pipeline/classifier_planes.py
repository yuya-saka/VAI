"""Build fixed classifier planes with mandatory bbox coverage."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pydicom
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from data_preprocessing.rsna_pipeline.classifier_plane_selection import (
    select_classifier_plane_positions,
)
from data_preprocessing.rsna_pipeline.dicom_geometry import DicomSeriesGeometry
from data_preprocessing.rsna_pipeline.mask_processing import (
    RAS_TO_LPS,
    ProcessedVertebraMask,
)
from data_preprocessing.rsna_pipeline.orientation import OrientationSearchResult
from data_preprocessing.rsna_pipeline.plane_sampling import PhysicalPlane

FloatArray = npt.NDArray[np.float64]

LOW_VOLUME_QUANTILE: Final = 0.01
HIGH_VOLUME_QUANTILE: Final = 0.99


class ClassifierPlaneError(ValueError):
    """Raised when classifier planes cannot be constructed safely."""


@dataclass(frozen=True)
class BoundingBoxCenter:
    """Representative center of one contiguous bbox slice interval."""

    slice_number: int
    patient_lps_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ClassifierPlane:
    """One classifier plane and its selection provenance."""

    position_mm: float
    center_lps_mm: tuple[float, float, float]
    row_basis_lps: tuple[float, float, float]
    column_basis_lps: tuple[float, float, float]
    normal_lps: tuple[float, float, float]
    bbox_forced: bool
    max_area_forced: bool
    bbox_slice_numbers: tuple[int, ...]

    def physical_plane(self) -> PhysicalPlane:
        """Return the sampling-plane representation."""
        return PhysicalPlane(
            center=self.center_lps_mm,
            row_basis=self.row_basis_lps,
            column_basis=self.column_basis_lps,
        )


@dataclass(frozen=True)
class ClassifierPlanePlan:
    """Fixed-count classifier planes for one vertebra."""

    robust_range_mm: tuple[float, float]
    full_range_mm: tuple[float, float]
    planes: tuple[ClassifierPlane, ...]


def load_study_bbox_centers(
    bbox_csv_path: Path,
    study_id: str,
    geometry: DicomSeriesGeometry,
) -> tuple[BoundingBoxCenter, ...]:
    """Load one representative physical bbox center per contiguous interval."""
    if not bbox_csv_path.is_file():
        raise FileNotFoundError(f"Bounding-box CSV not found: {bbox_csv_path}")

    rows = _study_bbox_rows(bbox_csv_path, study_id)
    if not rows:
        return ()
    slice_groups = _contiguous_slice_groups(row.slice_number for row in rows)
    slice_index_by_number = _slice_index_by_number(geometry)
    centers: list[BoundingBoxCenter] = []
    for slice_group in slice_groups:
        representative_slice = slice_group[(len(slice_group) - 1) // 2]
        slice_rows = [row for row in rows if row.slice_number == representative_slice]
        row_center = float(np.mean([row.y + row.height / 2.0 for row in slice_rows]))
        column_center = float(np.mean([row.x + row.width / 2.0 for row in slice_rows]))
        try:
            slice_index = slice_index_by_number[representative_slice]
        except KeyError as error:
            raise ClassifierPlaneError(
                f"BBox slice {representative_slice} has no matching DICOM image"
            ) from error
        dicom_path = geometry.slices[slice_index].path
        patient_point = (
            _dicom_pixel_to_patient(
                dicom_path,
                row_center,
                column_center,
            )
            if dicom_path.is_file()
            else geometry.voxel_to_patient(
                (float(slice_index), row_center, column_center)
            )
        )
        centers.append(
            BoundingBoxCenter(
                slice_number=representative_slice,
                patient_lps_mm=_vector3_tuple(patient_point),
            )
        )
    return tuple(centers)


def assign_bbox_centers_to_vertebrae(
    bbox_centers: tuple[BoundingBoxCenter, ...],
    processed_masks: dict[str, ProcessedVertebraMask],
    orientations: dict[str, OrientationSearchResult],
    *,
    unique_levels: tuple[str, ...] = (),
) -> dict[str, tuple[BoundingBoxCenter, ...]]:
    """Assign every bbox interval to exactly one nearest vertebra."""
    assignments: dict[str, list[BoundingBoxCenter]] = {
        level: [] for level in processed_masks
    }
    ranges = {
        level: _projection_range(processed_masks[level], orientations[level])
        for level in processed_masks
    }
    centroids = {
        level: float(
            np.dot(
                processed_masks[level].centroid_lps_mm,
                orientations[level].normal_lps,
            )
        )
        for level in processed_masks
    }
    if len(bbox_centers) == len(unique_levels) and unique_levels:
        return _assign_bbox_centers_uniquely(
            bbox_centers,
            unique_levels,
            assignments,
            ranges,
            centroids,
            orientations,
        )
    for bbox_center in bbox_centers:
        point = np.asarray(bbox_center.patient_lps_mm, dtype=np.float64)
        level = min(
            processed_masks,
            key=lambda candidate: _assignment_score(
                point,
                ranges[candidate],
                centroids[candidate],
                orientations[candidate].normal_lps,
                candidate,
            ),
        )
        assignments[level].append(bbox_center)
    return {level: tuple(centers) for level, centers in assignments.items()}


def load_study_fracture_levels(
    label_csv_path: Path,
    study_id: str,
) -> tuple[str, ...]:
    """Load positive C1-C7 labels for one study."""
    if not label_csv_path.is_file():
        return ()
    with label_csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["StudyInstanceUID"] != study_id:
                continue
            return tuple(
                level
                for level in (f"C{index}" for index in range(1, 8))
                if int(float(row[level])) == 1
            )
    return ()


def build_classifier_plane_plan(
    processed_mask: ProcessedVertebraMask,
    orientation: OrientationSearchResult,
    *,
    bbox_centers: tuple[BoundingBoxCenter, ...] = (),
    plane_count: int = 15,
) -> ClassifierPlanePlan:
    """Build one fixed-count plane plan in the corrected physical frame."""
    normal = np.asarray(orientation.normal_lps, dtype=np.float64)
    centroid = np.asarray(processed_mask.centroid_lps_mm, dtype=np.float64)
    projections = _mask_projections(processed_mask, normal)
    robust_low, robust_high = np.quantile(
        projections,
        (LOW_VOLUME_QUANTILE, HIGH_VOLUME_QUANTILE),
    )
    bbox_positions_by_slice = {
        center.slice_number: float(np.dot(center.patient_lps_mm, normal))
        for center in bbox_centers
    }
    selection = select_classifier_plane_positions(
        float(robust_low),
        float(robust_high),
        bbox_positions_mm=tuple(bbox_positions_by_slice.values()),
        required_positions_mm=(orientation.max_area_position_mm,),
        plane_count=plane_count,
    )

    centroid_position = float(np.dot(centroid, normal))
    unsorted_planes = tuple(
        _classifier_plane(
            position,
            centroid,
            centroid_position,
            orientation,
            bbox_positions_by_slice,
            orientation.max_area_position_mm,
        )
        for position in selection.positions_mm
    )
    planes = tuple(
        sorted(
            unsorted_planes,
            key=lambda plane: plane.center_lps_mm[2],
            reverse=True,
        )
    )
    return ClassifierPlanePlan(
        robust_range_mm=(float(robust_low), float(robust_high)),
        full_range_mm=(
            float(projections.min()),
            float(projections.max()),
        ),
        planes=planes,
    )


@dataclass(frozen=True)
class _BoundingBoxRow:
    x: float
    y: float
    width: float
    height: float
    slice_number: int


def _study_bbox_rows(
    bbox_csv_path: Path,
    study_id: str,
) -> tuple[_BoundingBoxRow, ...]:
    rows: list[_BoundingBoxRow] = []
    with bbox_csv_path.open(newline="", encoding="utf-8") as file:
        for raw_row in csv.DictReader(file):
            if raw_row["StudyInstanceUID"] != study_id:
                continue
            rows.append(
                _BoundingBoxRow(
                    x=float(raw_row["x"]),
                    y=float(raw_row["y"]),
                    width=float(raw_row["width"]),
                    height=float(raw_row["height"]),
                    slice_number=int(raw_row["slice_number"]),
                )
            )
    return tuple(rows)


def _contiguous_slice_groups(
    slice_numbers: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    unique_numbers = sorted(set(int(value) for value in slice_numbers))
    if not unique_numbers:
        return ()
    groups: list[list[int]] = [[unique_numbers[0]]]
    for number in unique_numbers[1:]:
        if number == groups[-1][-1] + 1:
            groups[-1].append(number)
            continue
        groups.append([number])
    return tuple(tuple(group) for group in groups)


def _slice_index_by_number(
    geometry: DicomSeriesGeometry,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, slice_metadata in enumerate(geometry.slices):
        file_number = _integer_stem(slice_metadata.path)
        if file_number is not None:
            result[file_number] = index
        if slice_metadata.instance_number is not None:
            result.setdefault(slice_metadata.instance_number, index)
    return result


def _integer_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _dicom_pixel_to_patient(
    dicom_path: Path,
    row: float,
    column: float,
) -> FloatArray:
    dataset = pydicom.dcmread(
        dicom_path,
        stop_before_pixels=True,
        specific_tags=[
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
        ],
    )
    origin = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
    spacing = np.asarray(dataset.PixelSpacing, dtype=np.float64)
    return np.asarray(
        origin
        + column * spacing[1] * orientation[:3]
        + row * spacing[0] * orientation[3:],
        dtype=np.float64,
    )


def _projection_range(
    processed_mask: ProcessedVertebraMask,
    orientation: OrientationSearchResult,
) -> tuple[float, float]:
    projections = _mask_projections(
        processed_mask,
        np.asarray(orientation.normal_lps, dtype=np.float64),
    )
    return (float(projections.min()), float(projections.max()))


def _mask_projections(
    processed_mask: ProcessedVertebraMask,
    normal_lps: FloatArray,
) -> FloatArray:
    occupied_indices = np.argwhere(processed_mask.mask > 0)
    homogeneous = np.column_stack(
        (occupied_indices.astype(np.float64), np.ones(occupied_indices.shape[0]))
    )
    physical_ras = (homogeneous @ processed_mask.affine_ras.T)[:, :3]
    physical_lps = physical_ras @ RAS_TO_LPS.T
    return np.asarray(physical_lps @ normal_lps, dtype=np.float64)


def _assignment_score(
    point_lps: FloatArray,
    projection_range: tuple[float, float],
    centroid_position: float,
    normal_lps: tuple[float, float, float],
    level: str,
) -> tuple[float, float, str]:
    position = float(np.dot(point_lps, normal_lps))
    low, high = projection_range
    range_distance = max(low - position, 0.0, position - high)
    centroid_distance = abs(position - centroid_position)
    return (range_distance, centroid_distance, level)


def _assign_bbox_centers_uniquely(
    bbox_centers: tuple[BoundingBoxCenter, ...],
    unique_levels: tuple[str, ...],
    assignments: dict[str, list[BoundingBoxCenter]],
    ranges: dict[str, tuple[float, float]],
    centroids: dict[str, float],
    orientations: dict[str, OrientationSearchResult],
) -> dict[str, tuple[BoundingBoxCenter, ...]]:
    if any(level not in assignments for level in unique_levels):
        raise ClassifierPlaneError("Fracture label contains an unknown vertebra level")
    costs = np.empty((len(bbox_centers), len(unique_levels)), dtype=np.float64)
    for row_index, bbox_center in enumerate(bbox_centers):
        point = np.asarray(bbox_center.patient_lps_mm, dtype=np.float64)
        for column_index, level in enumerate(unique_levels):
            range_distance, centroid_distance, _ = _assignment_score(
                point,
                ranges[level],
                centroids[level],
                orientations[level].normal_lps,
                level,
            )
            costs[row_index, column_index] = (
                range_distance * 1_000_000.0 + centroid_distance
            )
    row_indices, column_indices = linear_sum_assignment(costs)
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        assignments[unique_levels[int(column_index)]].append(
            bbox_centers[int(row_index)]
        )
    return {level: tuple(centers) for level, centers in assignments.items()}


def _classifier_plane(
    position_mm: float,
    centroid_lps: FloatArray,
    centroid_position_mm: float,
    orientation: OrientationSearchResult,
    bbox_positions_by_slice: dict[int, float],
    max_area_position_mm: float,
) -> ClassifierPlane:
    normal = np.asarray(orientation.normal_lps, dtype=np.float64)
    center = centroid_lps + (position_mm - centroid_position_mm) * normal
    bbox_slice_numbers = tuple(
        sorted(
            slice_number
            for slice_number, bbox_position in bbox_positions_by_slice.items()
            if np.isclose(
                bbox_position,
                position_mm,
                atol=1e-6,
                rtol=0.0,
            )
        )
    )
    return ClassifierPlane(
        position_mm=position_mm,
        center_lps_mm=_vector3_tuple(center),
        row_basis_lps=orientation.row_basis_lps,
        column_basis_lps=orientation.column_basis_lps,
        normal_lps=orientation.normal_lps,
        bbox_forced=bool(bbox_slice_numbers),
        max_area_forced=bool(
            np.isclose(
                max_area_position_mm,
                position_mm,
                atol=1e-6,
                rtol=0.0,
            )
        ),
        bbox_slice_numbers=bbox_slice_numbers,
    )


def _vector3_tuple(values: npt.ArrayLike) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return (float(vector[0]), float(vector[1]), float(vector[2]))
