"""Canonical 3D geometry for RSNA per-slice fracture bounding boxes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import pydicom

from data_preprocessing.rsna_pipeline.plane_sampling import PhysicalPlane

FloatArray = npt.NDArray[np.float64]
Uint8Array = npt.NDArray[np.uint8]

OUTPUT_SIZE: Final = (224, 224)
PIXEL_SPACING_MM: Final = (0.4, 0.4)
PLANE_COUNT: Final = 15
CENTER_PLANE_INDEX: Final = 7
GEOMETRY_TOLERANCE_MM: Final = 1e-6
DEFAULT_DENSE_STEP_MM: Final = 0.4
DEFAULT_CONTEXT_MARGIN_MM: Final = 2.0

CELL_EDGES: Final = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


class BboxGeometryError(ValueError):
    """Raised when bbox geometry cannot be constructed safely."""


class UnsupportedBboxGeometryModeError(BboxGeometryError):
    """Raised when exact bbox geometry is unavailable for a study."""


@dataclass(frozen=True)
class SliceBbox:
    """One native DICOM-slice bbox represented by four LPS corners."""

    slice_number: int
    series_index: int
    plane_origin_lps_mm: tuple[float, float, float]
    plane_normal_lps: tuple[float, float, float]
    corners_lps_mm: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BboxRun:
    """One contiguous run of per-slice bbox rectangles."""

    rectangles: tuple[SliceBbox, ...]
    median_slice_spacing_mm: float

    @property
    def slice_numbers(self) -> tuple[int, ...]:
        return tuple(rectangle.slice_number for rectangle in self.rectangles)


@dataclass(frozen=True)
class BboxCell:
    """One eight-vertex cell in the piecewise 3D bbox envelope."""

    vertices_lps_mm: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BboxCenteredPlaneSelection:
    """One bbox-run-centred sequence of corrected physical planes."""

    planes: tuple[PhysicalPlane, ...]
    positions_mm: tuple[float, ...]
    bbox_support_range_mm: tuple[float, float]
    bbox_center_position_mm: float
    plane_spacing_mm: float


def assigned_slice_to_level(metadata: dict) -> dict[int, str]:
    """Return the canonical bbox slice-to-vertebra assignment."""
    mapping: dict[int, str] = {}
    for level, vertebra in metadata.get("vertebrae", {}).items():
        classifier = vertebra.get("classifier_planes", {})
        for raw_slice_number in classifier.get("assigned_bbox_slice_numbers", []):
            slice_number = int(raw_slice_number)
            previous = mapping.get(slice_number)
            if previous is not None:
                raise BboxGeometryError(
                    f"BBox slice {slice_number} is assigned to both {previous} and {level}"
                )
            mapping[slice_number] = str(level)
    return mapping


def load_study_bbox_runs(
    study_id: str,
    bbox_csv_path: Path,
    study_directory: Path,
    metadata: dict,
) -> dict[str, tuple[BboxRun, ...]]:
    """Load one study's raw bbox rows as canonical level-specific runs."""
    if not bbox_csv_path.is_file():
        raise FileNotFoundError(f"Bounding-box CSV not found: {bbox_csv_path}")
    slice_to_level = assigned_slice_to_level(metadata)
    slice_lookup = _metadata_slice_lookup(metadata)
    rectangles_by_level: dict[str, list[SliceBbox]] = {}
    with bbox_csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["StudyInstanceUID"] != study_id:
                continue
            slice_number = int(row["slice_number"])
            level = slice_to_level.get(slice_number)
            if level is None:
                raise BboxGeometryError(
                    f"{study_id}: bbox slice {slice_number} has no canonical level"
                )
            slice_info = slice_lookup.get(slice_number)
            if slice_info is None:
                raise BboxGeometryError(
                    f"{study_id}: bbox slice {slice_number} has no DICOM metadata"
                )
            series_index, source_file = slice_info
            rectangle = load_slice_bbox(
                study_directory / source_file,
                slice_number=slice_number,
                series_index=series_index,
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["width"]),
                height=float(row["height"]),
            )
            rectangles_by_level.setdefault(level, []).append(rectangle)

    spacing = float(metadata["dicom_geometry"]["median_slice_spacing_mm"])
    return {
        level: split_contiguous_bbox_runs(rectangles, spacing)
        for level, rectangles in rectangles_by_level.items()
    }


def load_slice_bbox(
    dicom_path: Path,
    *,
    slice_number: int,
    series_index: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> SliceBbox:
    """Convert one raw DICOM bbox rectangle to subpixel LPS corners."""
    if width <= 0.0 or height <= 0.0:
        raise BboxGeometryError("BBox width and height must be positive")
    dataset = pydicom.dcmread(
        dicom_path,
        stop_before_pixels=True,
        specific_tags=[
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
        ],
    )
    try:
        origin = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        spacing = np.asarray(dataset.PixelSpacing, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise BboxGeometryError(f"Invalid DICOM bbox geometry: {dicom_path}") from error
    if origin.shape != (3,) or orientation.shape != (6,) or spacing.shape != (2,):
        raise BboxGeometryError(f"Invalid DICOM bbox geometry shape: {dicom_path}")
    row_direction = _unit_vector(orientation[:3])
    column_direction = _unit_vector(orientation[3:])
    normal = _unit_vector(np.cross(row_direction, column_direction))
    row_spacing, column_spacing = float(spacing[0]), float(spacing[1])
    corners = tuple(
        _vector3_tuple(
            origin
            + column * column_spacing * row_direction
            + row * row_spacing * column_direction
        )
        for column, row in (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        )
    )
    return SliceBbox(
        slice_number=slice_number,
        series_index=series_index,
        plane_origin_lps_mm=_vector3_tuple(origin),
        plane_normal_lps=_vector3_tuple(normal),
        corners_lps_mm=corners,
    )


def split_contiguous_bbox_runs(
    rectangles: list[SliceBbox] | tuple[SliceBbox, ...],
    median_slice_spacing_mm: float,
) -> tuple[BboxRun, ...]:
    """Split rectangles by adjacency in the true DICOM series index."""
    if median_slice_spacing_mm <= 0.0:
        raise BboxGeometryError("Median slice spacing must be positive")
    ordered = sorted(rectangles, key=lambda rectangle: rectangle.series_index)
    if not ordered:
        return ()
    if len({rectangle.series_index for rectangle in ordered}) != len(ordered):
        raise BboxGeometryError("Multiple bbox rows share one DICOM series index")
    grouped: list[list[SliceBbox]] = [[ordered[0]]]
    for rectangle in ordered[1:]:
        if rectangle.series_index == grouped[-1][-1].series_index + 1:
            grouped[-1].append(rectangle)
        else:
            grouped.append([rectangle])
    return tuple(BboxRun(tuple(group), median_slice_spacing_mm) for group in grouped)


def build_bbox_cells(run: BboxRun) -> tuple[BboxCell, ...]:
    """Build a capped piecewise 3D envelope for one contiguous bbox run."""
    if not run.rectangles:
        raise BboxGeometryError("BBox run must contain at least one rectangle")
    rectangles = run.rectangles
    if len(rectangles) == 1:
        rectangle = rectangles[0]
        normal = np.asarray(rectangle.plane_normal_lps, dtype=np.float64)
        offset = normal * (run.median_slice_spacing_mm * 0.5)
        return (
            _cell(
                _shift_corners(rectangle.corners_lps_mm, -offset),
                _shift_corners(rectangle.corners_lps_mm, offset),
            ),
        )

    cells: list[BboxCell] = []
    first = rectangles[0]
    second = rectangles[1]
    start_delta = 0.5 * (
        np.asarray(second.plane_origin_lps_mm, dtype=np.float64)
        - np.asarray(first.plane_origin_lps_mm, dtype=np.float64)
    )
    cells.append(
        _cell(
            _shift_corners(first.corners_lps_mm, -start_delta),
            first.corners_lps_mm,
        )
    )
    for lower, upper in zip(rectangles, rectangles[1:], strict=False):
        cells.append(_cell(lower.corners_lps_mm, upper.corners_lps_mm))
    previous = rectangles[-2]
    last = rectangles[-1]
    end_delta = 0.5 * (
        np.asarray(last.plane_origin_lps_mm, dtype=np.float64)
        - np.asarray(previous.plane_origin_lps_mm, dtype=np.float64)
    )
    cells.append(
        _cell(
            last.corners_lps_mm,
            _shift_corners(last.corners_lps_mm, end_delta),
        )
    )
    return tuple(cells)


def intersect_bbox_cell_with_plane(
    cell: BboxCell,
    plane: PhysicalPlane,
    *,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    pixel_spacing_mm: tuple[float, float] = PIXEL_SPACING_MM,
) -> FloatArray:
    """Return one cell-plane intersection polygon as ``(row, column)`` pixels."""
    vertices = np.asarray(cell.vertices_lps_mm, dtype=np.float64)
    center, row_basis, column_basis, normal = _plane_arrays(plane)
    distances = (vertices - center) @ normal
    points: list[FloatArray] = []
    for index, distance in enumerate(distances):
        if abs(float(distance)) <= GEOMETRY_TOLERANCE_MM:
            points.append(vertices[index])
    for first_index, second_index in CELL_EDGES:
        first_distance = float(distances[first_index])
        second_distance = float(distances[second_index])
        if first_distance * second_distance >= 0.0:
            continue
        interpolation = first_distance / (first_distance - second_distance)
        points.append(
            vertices[first_index]
            + interpolation * (vertices[second_index] - vertices[first_index])
        )
    unique = _unique_points(points)
    if len(unique) < 3:
        return np.empty((0, 2), dtype=np.float64)
    height, width = output_size
    row_spacing, column_spacing = pixel_spacing_mm
    half_row = (height - 1) / 2.0
    half_column = (width - 1) / 2.0
    projected = np.asarray(
        [
            (
                float(np.dot(point - center, column_basis)) / row_spacing + half_row,
                float(np.dot(point - center, row_basis)) / column_spacing + half_column,
            )
            for point in unique
        ],
        dtype=np.float64,
    )
    hull = cv2.convexHull(
        projected[:, ::-1].astype(np.float32),
        returnPoints=True,
    ).reshape(-1, 2)
    return np.asarray(hull[:, ::-1], dtype=np.float64)


def bbox_plane_polygons(
    run: BboxRun,
    plane: PhysicalPlane,
    *,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    pixel_spacing_mm: tuple[float, float] = PIXEL_SPACING_MM,
) -> tuple[FloatArray, ...]:
    """Return all non-empty cell intersections for one run and plane."""
    polygons = tuple(
        intersect_bbox_cell_with_plane(
            cell,
            plane,
            output_size=output_size,
            pixel_spacing_mm=pixel_spacing_mm,
        )
        for cell in build_bbox_cells(run)
    )
    return tuple(polygon for polygon in polygons if len(polygon) >= 3)


def rasterize_bbox_polygons(
    polygons: tuple[FloatArray, ...],
    *,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    supersample: int = 4,
) -> Uint8Array:
    """Rasterize polygons as uint8 partial-pixel occupancy in ``[0, 255]``."""
    occupancy, _ = _rasterize_polygon_union(
        polygons,
        output_size=output_size,
        supersample=supersample,
    )
    return occupancy


def _rasterize_polygon_union(
    polygons: tuple[FloatArray, ...],
    *,
    output_size: tuple[int, int],
    supersample: int,
) -> tuple[Uint8Array, tuple[FloatArray, ...]]:
    if supersample <= 0:
        raise BboxGeometryError("Supersample factor must be positive")
    height, width = output_size
    canvas = np.zeros((height * supersample, width * supersample), dtype=np.uint8)
    for polygon in polygons:
        points = np.rint(polygon[:, ::-1] * supersample).astype(np.int32)
        cv2.fillConvexPoly(canvas, points, 255, lineType=cv2.LINE_8)
    raw_contours, _ = cv2.findContours(
        canvas,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = tuple(
        np.asarray(
            contour.reshape(-1, 2)[:, ::-1] / float(supersample),
            dtype=np.float64,
        )
        for contour in raw_contours
        if len(contour) >= 3
    )
    if supersample == 1:
        return canvas, contours
    return (
        np.asarray(
            cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        ),
        contours,
    )


def render_bbox_run(
    run: BboxRun,
    planes: tuple[PhysicalPlane, ...],
    *,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    pixel_spacing_mm: tuple[float, float] = PIXEL_SPACING_MM,
    supersample: int = 4,
) -> tuple[Uint8Array, tuple[tuple[FloatArray, ...], ...]]:
    """Render one run on corrected planes and retain exact polygon vertices."""
    cell_polygons = tuple(
        bbox_plane_polygons(
            run,
            plane,
            output_size=output_size,
            pixel_spacing_mm=pixel_spacing_mm,
        )
        for plane in planes
    )
    rendered = tuple(
        _rasterize_polygon_union(
            polygons,
            output_size=output_size,
            supersample=supersample,
        )
        for polygons in cell_polygons
    )
    occupancy = np.stack(tuple(item[0] for item in rendered), axis=0)
    union_contours = tuple(item[1] for item in rendered)
    return occupancy, union_contours


def select_bbox_centered_planes(
    run: BboxRun,
    reference_plane: PhysicalPlane,
    reference_position_mm: float,
    robust_range_mm: tuple[float, float],
    *,
    dense_step_mm: float = DEFAULT_DENSE_STEP_MM,
    context_margin_mm: float = DEFAULT_CONTEXT_MARGIN_MM,
    plane_count: int = PLANE_COUNT,
) -> BboxCenteredPlaneSelection:
    """Select symmetric corrected planes with occupied bbox geometry at the centre."""
    if plane_count <= 0 or plane_count % 2 == 0:
        raise BboxGeometryError("Plane count must be a positive odd number")
    if dense_step_mm <= 0.0 or context_margin_mm < 0.0:
        raise BboxGeometryError("Invalid bbox-centered sampling parameters")
    _, _, _, normal = _plane_arrays(reference_plane)
    support_low, support_high = bbox_support_range_mm(run, normal)
    candidate_positions = np.arange(
        support_low,
        support_high + dense_step_mm * 0.5,
        dense_step_mm,
        dtype=np.float64,
    )
    if candidate_positions.size == 0:
        candidate_positions = np.asarray([(support_low + support_high) * 0.5])
    candidate_planes = tuple(
        _shifted_plane(
            reference_plane,
            normal,
            float(position - reference_position_mm),
        )
        for position in candidate_positions
    )
    areas = np.asarray(
        [
            _visible_cross_section_area(
                bbox_plane_polygons(run, plane),
                output_size=OUTPUT_SIZE,
            )
            for plane in candidate_planes
        ],
        dtype=np.float64,
    )
    occupied = np.flatnonzero(areas > 0.0)
    if occupied.size == 0:
        raise BboxGeometryError("BBox run does not intersect the corrected output FOV")
    cumulative = np.cumsum(areas)
    median_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    center_index = min(
        (int(index) for index in occupied),
        key=lambda index: (abs(index - median_index), -areas[index]),
    )
    center_position = float(candidate_positions[center_index])
    robust_low, robust_high = robust_range_mm
    if robust_low >= robust_high:
        raise BboxGeometryError("Robust vertebra range must be increasing")
    bbox_half_extent = max(
        center_position - support_low,
        support_high - center_position,
    )
    half_extent = max(
        bbox_half_extent + context_margin_mm,
        (robust_high - robust_low) * 0.5,
    )
    center_plane_index = plane_count // 2
    spacing = half_extent / center_plane_index
    positions = tuple(
        center_position + (center_plane_index - index) * spacing
        for index in range(plane_count)
    )
    planes = tuple(
        _shifted_plane(
            reference_plane,
            normal,
            position - reference_position_mm,
        )
        for position in positions
    )
    return BboxCenteredPlaneSelection(
        planes=planes,
        positions_mm=positions,
        bbox_support_range_mm=(support_low, support_high),
        bbox_center_position_mm=center_position,
        plane_spacing_mm=spacing,
    )


def bbox_support_range_mm(
    run: BboxRun,
    normal_lps: npt.ArrayLike,
) -> tuple[float, float]:
    """Return the complete capped bbox envelope range along one normal."""
    normal = _unit_vector(normal_lps)
    vertices = np.asarray(
        [vertex for cell in build_bbox_cells(run) for vertex in cell.vertices_lps_mm],
        dtype=np.float64,
    )
    positions = vertices @ normal
    return float(positions.min()), float(positions.max())


def contours_to_json(
    contours: tuple[tuple[FloatArray, ...], ...],
) -> list[dict[str, object]]:
    """Serialize plane polygons without losing floating-point vertices."""
    return [
        {
            "plane_index": plane_index,
            "components": [polygon.tolist() for polygon in polygons],
        }
        for plane_index, polygons in enumerate(contours)
    ]


def load_metadata(path: Path) -> dict:
    """Load one processing metadata JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_slice_lookup(metadata: dict) -> dict[int, tuple[int, str]]:
    lookup: dict[int, tuple[int, str]] = {}
    for index, item in enumerate(metadata["dicom_geometry"]["slices"]):
        source_file = str(item["source_file"])
        try:
            lookup[int(Path(source_file).stem)] = (index, source_file)
        except ValueError:
            pass
        instance_number = item.get("instance_number")
        if instance_number is not None:
            lookup.setdefault(int(instance_number), (index, source_file))
    return lookup


def _cell(
    lower_corners: tuple[tuple[float, float, float], ...],
    upper_corners: tuple[tuple[float, float, float], ...],
) -> BboxCell:
    if len(lower_corners) != 4 or len(upper_corners) != 4:
        raise BboxGeometryError("BBox rectangles must have four corners")
    return BboxCell(tuple((*lower_corners, *upper_corners)))


def _shift_corners(
    corners: tuple[tuple[float, float, float], ...],
    offset: npt.ArrayLike,
) -> tuple[tuple[float, float, float], ...]:
    shift = np.asarray(offset, dtype=np.float64)
    return tuple(
        _vector3_tuple(np.asarray(corner, dtype=np.float64) + shift)
        for corner in corners
    )


def _plane_arrays(
    plane: PhysicalPlane,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    center = np.asarray(plane.center, dtype=np.float64)
    row_basis = _unit_vector(plane.row_basis)
    column_basis = _unit_vector(plane.column_basis)
    normal = _unit_vector(np.cross(row_basis, column_basis))
    return center, row_basis, column_basis, normal


def _shifted_plane(
    reference_plane: PhysicalPlane,
    normal: FloatArray,
    offset_mm: float,
) -> PhysicalPlane:
    center = np.asarray(reference_plane.center, dtype=np.float64) + offset_mm * normal
    return PhysicalPlane(
        center=_vector3_tuple(center),
        row_basis=reference_plane.row_basis,
        column_basis=reference_plane.column_basis,
    )


def _unique_points(points: list[FloatArray]) -> list[FloatArray]:
    unique: list[FloatArray] = []
    for point in points:
        if any(
            np.linalg.norm(point - existing) <= GEOMETRY_TOLERANCE_MM
            for existing in unique
        ):
            continue
        unique.append(np.asarray(point, dtype=np.float64))
    return unique


def _visible_cross_section_area(
    polygons: tuple[FloatArray, ...],
    *,
    output_size: tuple[int, int],
) -> float:
    mask = rasterize_bbox_polygons(
        polygons,
        output_size=output_size,
        supersample=1,
    )
    return float(np.count_nonzero(mask))


def _unit_vector(values: npt.ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise BboxGeometryError("Geometry vector must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise BboxGeometryError("Geometry vector must have non-zero length")
    return np.asarray(vector / norm, dtype=np.float64)


def _vector3_tuple(values: npt.ArrayLike) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return float(vector[0]), float(vector[1]), float(vector[2])
