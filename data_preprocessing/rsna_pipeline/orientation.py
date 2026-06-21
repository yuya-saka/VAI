"""Physical-coordinate orientation search for vertebra masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from data_preprocessing.rsna_pipeline.mask_processing import RAS_TO_LPS

FloatArray = npt.NDArray[np.float64]

DEFAULT_COARSE_RANGE_DEG: Final = 30.0
DEFAULT_COARSE_STEP_DEG: Final = 5.0
DEFAULT_FINE_STEP_DEG: Final = 1.0
DEFAULT_ISOTROPIC_SPACING_MM: Final = 0.4
DEFAULT_MARGIN_MM: Final = 2.0


class OrientationSearchError(ValueError):
    """Raised when physical orientation search inputs are invalid."""


@dataclass(frozen=True)
class OrientationSearchResult:
    """Best correction and corrected physical plane basis."""

    rx_deg: float
    ry_deg: float
    max_area_mm2: float
    max_area_position_mm: float
    max_area_center_lps_mm: tuple[float, float, float]
    row_basis_lps: tuple[float, float, float]
    column_basis_lps: tuple[float, float, float]
    normal_lps: tuple[float, float, float]
    at_search_boundary: bool


@dataclass(frozen=True)
class _IsotropicMaskGrid:
    """Transient isotropic mask and its local physical coordinate mapping."""

    mask: npt.NDArray[np.uint8]
    local_minimum_mm: FloatArray
    spacing_mm: float


def find_best_physical_orientation(
    mask: npt.ArrayLike,
    affine_ras: npt.ArrayLike,
    *,
    base_row_basis_lps: npt.ArrayLike = (1.0, 0.0, 0.0),
    base_column_basis_lps: npt.ArrayLike = (0.0, 1.0, 0.0),
    coarse_range_deg: float = DEFAULT_COARSE_RANGE_DEG,
    coarse_step_deg: float = DEFAULT_COARSE_STEP_DEG,
    fine_step_deg: float = DEFAULT_FINE_STEP_DEG,
    isotropic_spacing_mm: float = DEFAULT_ISOTROPIC_SPACING_MM,
    margin_mm: float = DEFAULT_MARGIN_MM,
) -> OrientationSearchResult:
    """Find the correction maximizing physical mask cross-sectional area."""
    binary_mask = np.asarray(mask) > 0
    affine = _validated_affine(affine_ras)
    row_basis, column_basis, normal_basis = _validated_basis(
        base_row_basis_lps,
        base_column_basis_lps,
    )
    _validate_search_parameters(
        binary_mask=binary_mask,
        coarse_range_deg=coarse_range_deg,
        coarse_step_deg=coarse_step_deg,
        fine_step_deg=fine_step_deg,
        isotropic_spacing_mm=isotropic_spacing_mm,
        margin_mm=margin_mm,
    )

    base_basis = np.column_stack((row_basis, column_basis, normal_basis))
    isotropic_grid = _resample_isotropic_local_mask(
        binary_mask,
        affine,
        base_basis,
        spacing_mm=isotropic_spacing_mm,
        margin_mm=margin_mm,
    )
    best_rx, best_ry, best_pixel_area = _find_best_tilt(
        isotropic_grid.mask,
        coarse_range_deg=coarse_range_deg,
        coarse_step_deg=coarse_step_deg,
        fine_step_deg=fine_step_deg,
    )
    corrected_basis = _corrected_basis(base_basis, best_rx, best_ry)
    max_area_center_lps = _maximum_area_plane_center_lps(
        isotropic_grid,
        base_basis,
        best_rx,
        best_ry,
    )
    max_area_position_mm = float(np.dot(max_area_center_lps, corrected_basis[:, 2]))
    return OrientationSearchResult(
        rx_deg=best_rx,
        ry_deg=best_ry,
        max_area_mm2=best_pixel_area * isotropic_spacing_mm**2,
        max_area_position_mm=max_area_position_mm,
        max_area_center_lps_mm=_vector3_tuple(max_area_center_lps),
        row_basis_lps=_vector3_tuple(corrected_basis[:, 0]),
        column_basis_lps=_vector3_tuple(corrected_basis[:, 1]),
        normal_lps=_vector3_tuple(corrected_basis[:, 2]),
        at_search_boundary=(
            abs(best_rx) >= coarse_range_deg or abs(best_ry) >= coarse_range_deg
        ),
    )


def _resample_isotropic_local_mask(
    binary_mask: npt.NDArray[np.bool_],
    affine_ras: FloatArray,
    base_basis_lps: FloatArray,
    *,
    spacing_mm: float,
    margin_mm: float,
) -> _IsotropicMaskGrid:
    occupied_indices = np.argwhere(binary_mask)
    points_lps = _indices_to_lps(occupied_indices, affine_ras)
    local_points = points_lps @ base_basis_lps
    local_minimum = local_points.min(axis=0) - margin_mm
    local_maximum = local_points.max(axis=0) + margin_mm
    shape = np.maximum(
        2,
        np.ceil((local_maximum - local_minimum) / spacing_mm).astype(int) + 1,
    )

    output_indices = np.indices(tuple(int(value) for value in shape))
    local_coordinates = (
        np.moveaxis(output_indices, 0, -1).astype(np.float64) * spacing_mm
        + local_minimum
    )
    patient_lps = local_coordinates @ base_basis_lps.T
    source_indices = _lps_to_indices(patient_lps, affine_ras)
    coordinate_matrix = np.moveaxis(source_indices, -1, 0).reshape(3, -1)
    sampled = ndimage.map_coordinates(
        binary_mask.astype(np.uint8),
        coordinate_matrix,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    result = np.asarray(
        sampled.reshape(tuple(int(value) for value in shape)),
        dtype=np.uint8,
    )
    if not np.any(result):
        raise OrientationSearchError("Isotropic mask resampling produced no foreground")
    return _IsotropicMaskGrid(
        mask=result,
        local_minimum_mm=np.asarray(local_minimum, dtype=np.float64),
        spacing_mm=spacing_mm,
    )


def _find_best_tilt(
    isotropic_mask: npt.NDArray[np.uint8],
    *,
    coarse_range_deg: float,
    coarse_step_deg: float,
    fine_step_deg: float,
) -> tuple[float, float, float]:
    coarse_mask = ndimage.zoom(
        isotropic_mask.astype(np.float32),
        0.5,
        order=0,
        prefilter=False,
    )
    coarse_angles = _angle_values(
        -coarse_range_deg,
        coarse_range_deg,
        coarse_step_deg,
    )
    coarse_rx, coarse_ry, _ = _search_rotations(
        coarse_mask,
        coarse_angles,
        coarse_angles,
    )

    fine_x = _angle_values(
        coarse_rx - coarse_step_deg,
        coarse_rx + coarse_step_deg,
        fine_step_deg,
    )
    fine_y = _angle_values(
        coarse_ry - coarse_step_deg,
        coarse_ry + coarse_step_deg,
        fine_step_deg,
    )
    return _search_rotations(isotropic_mask, fine_x, fine_y)


def _search_rotations(
    mask: npt.NDArray[np.generic],
    rx_values: FloatArray,
    ry_values: FloatArray,
) -> tuple[float, float, float]:
    best_rx = 0.0
    best_ry = 0.0
    best_area = -1.0
    for rx_deg in rx_values:
        for ry_deg in ry_values:
            area = _max_axial_mask_area(
                mask,
                float(rx_deg),
                float(ry_deg),
            )
            candidate = (float(rx_deg), float(ry_deg))
            if _is_better_candidate(
                area,
                candidate,
                best_area,
                (best_rx, best_ry),
            ):
                best_rx, best_ry, best_area = candidate[0], candidate[1], area
    return best_rx, best_ry, best_area


def _max_axial_mask_area(
    mask: npt.NDArray[np.generic],
    rx_deg: float,
    ry_deg: float,
) -> float:
    rotated = _rotated_mask(mask, rx_deg, ry_deg)
    return float(rotated.sum(axis=(0, 1)).max())


def _rotated_mask(
    mask: npt.NDArray[np.generic],
    rx_deg: float,
    ry_deg: float,
) -> npt.NDArray[np.generic]:
    center = np.asarray(mask.shape, dtype=np.float64) / 2.0
    rotation = Rotation.from_euler(
        "XY",
        [rx_deg, ry_deg],
        degrees=True,
    ).as_matrix()
    offset = center - rotation.T @ center
    return cast(
        npt.NDArray[np.generic],
        ndimage.affine_transform(
            mask,
            rotation.T,
            offset=offset,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ),
    )


def _maximum_area_plane_center_lps(
    isotropic_grid: _IsotropicMaskGrid,
    base_basis_lps: FloatArray,
    rx_deg: float,
    ry_deg: float,
) -> FloatArray:
    rotated = _rotated_mask(isotropic_grid.mask, rx_deg, ry_deg)
    areas = np.asarray(rotated.sum(axis=(0, 1)), dtype=np.float64)
    maximum_index = int(np.argmax(areas))

    center = np.asarray(isotropic_grid.mask.shape, dtype=np.float64) / 2.0
    destination_index = np.asarray(
        (center[0], center[1], float(maximum_index)),
        dtype=np.float64,
    )
    rotation = Rotation.from_euler(
        "XY",
        [rx_deg, ry_deg],
        degrees=True,
    ).as_matrix()
    source_index = rotation.T @ (destination_index - center) + center
    source_local_mm = (
        isotropic_grid.local_minimum_mm + source_index * isotropic_grid.spacing_mm
    )
    return np.asarray(source_local_mm @ base_basis_lps.T, dtype=np.float64)


def _corrected_basis(
    base_basis: FloatArray,
    rx_deg: float,
    ry_deg: float,
) -> FloatArray:
    local_rotation = Rotation.from_euler(
        "XY",
        [rx_deg, ry_deg],
        degrees=True,
    ).as_matrix()
    return np.asarray(base_basis @ local_rotation.T, dtype=np.float64)


def _is_better_candidate(
    candidate_area: float,
    candidate_angles: tuple[float, float],
    best_area: float,
    best_angles: tuple[float, float],
) -> bool:
    if candidate_area > best_area:
        return True
    if candidate_area != best_area:
        return False
    candidate_norm = candidate_angles[0] ** 2 + candidate_angles[1] ** 2
    best_norm = best_angles[0] ** 2 + best_angles[1] ** 2
    if candidate_norm != best_norm:
        return candidate_norm < best_norm
    return candidate_angles < best_angles


def _validated_basis(
    row_basis_lps: npt.ArrayLike,
    column_basis_lps: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    row_basis = _unit_vector(row_basis_lps)
    column_basis = _unit_vector(column_basis_lps)
    if not np.isclose(
        np.dot(row_basis, column_basis),
        0.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise OrientationSearchError("Base plane basis must be orthogonal")
    normal_basis = _unit_vector(np.cross(row_basis, column_basis))
    return row_basis, column_basis, normal_basis


def _validated_affine(affine_ras: npt.ArrayLike) -> FloatArray:
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4):
        raise OrientationSearchError("NIfTI affine must have shape (4, 4)")
    if not np.all(np.isfinite(affine)):
        raise OrientationSearchError("NIfTI affine must be finite")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise OrientationSearchError("NIfTI affine is singular")
    return affine


def _validate_search_parameters(
    *,
    binary_mask: npt.NDArray[np.bool_],
    coarse_range_deg: float,
    coarse_step_deg: float,
    fine_step_deg: float,
    isotropic_spacing_mm: float,
    margin_mm: float,
) -> None:
    if binary_mask.ndim != 3:
        raise OrientationSearchError("Mask must be three-dimensional")
    if not np.any(binary_mask):
        raise OrientationSearchError("Mask must not be empty")
    positive_values = (
        coarse_range_deg,
        coarse_step_deg,
        fine_step_deg,
        isotropic_spacing_mm,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values):
        raise OrientationSearchError(
            "Search range, steps, and spacing must be positive"
        )
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise OrientationSearchError("Margin must be non-negative and finite")


def _indices_to_lps(
    indices: npt.NDArray[np.intp],
    affine_ras: FloatArray,
) -> FloatArray:
    homogeneous = np.column_stack(
        (indices.astype(np.float64), np.ones(indices.shape[0]))
    )
    physical_ras = (homogeneous @ affine_ras.T)[:, :3]
    return physical_ras @ RAS_TO_LPS.T


def _lps_to_indices(
    patient_lps: FloatArray,
    affine_ras: FloatArray,
) -> FloatArray:
    patient_ras = patient_lps @ RAS_TO_LPS.T
    inverse_affine = np.linalg.inv(affine_ras)
    homogeneous = np.concatenate(
        (
            patient_ras,
            np.ones((*patient_ras.shape[:-1], 1), dtype=np.float64),
        ),
        axis=-1,
    )
    return np.asarray(homogeneous @ inverse_affine.T, dtype=np.float64)[..., :3]


def _angle_values(
    minimum: float,
    maximum: float,
    step: float,
) -> FloatArray:
    return np.arange(minimum, maximum + step / 2.0, step, dtype=np.float64)


def _unit_vector(values: npt.ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise OrientationSearchError("Basis vector must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise OrientationSearchError("Basis vector must have non-zero length")
    return vector / norm


def _vector3_tuple(values: npt.ArrayLike) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return (float(vector[0]), float(vector[1]), float(vector[2]))
