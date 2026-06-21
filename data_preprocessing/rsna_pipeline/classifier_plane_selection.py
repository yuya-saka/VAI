"""Classifier-plane selection with mandatory bounding-box coverage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np

DEFAULT_PLANE_COUNT: Final = 15
POSITION_TOLERANCE_MM: Final = 1e-6


@dataclass(frozen=True)
class ClassifierPlaneSelection:
    """Selected physical positions and their mandatory subsets."""

    positions_mm: tuple[float, ...]
    bbox_positions_mm: tuple[float, ...]
    required_positions_mm: tuple[float, ...]


def representative_bbox_slice_numbers(
    slice_numbers: Iterable[int],
) -> tuple[int, ...]:
    """Choose one actual slice from each contiguous bbox slice interval."""
    unique_numbers = sorted(set(slice_numbers))
    if not unique_numbers:
        return ()
    if any(number < 0 for number in unique_numbers):
        raise ValueError("BBox slice numbers must be non-negative")

    groups: list[list[int]] = [[unique_numbers[0]]]
    for number in unique_numbers[1:]:
        if number == groups[-1][-1] + 1:
            groups[-1].append(number)
            continue
        groups.append([number])
    return tuple(group[(len(group) - 1) // 2] for group in groups)


def select_classifier_plane_positions(
    t_low_mm: float,
    t_high_mm: float,
    *,
    bbox_positions_mm: Sequence[float] = (),
    required_positions_mm: Sequence[float] = (),
    plane_count: int = DEFAULT_PLANE_COUNT,
) -> ClassifierPlaneSelection:
    """Select fixed-count planes while preserving all mandatory positions."""
    _validate_selection_inputs(
        t_low_mm=t_low_mm,
        t_high_mm=t_high_mm,
        bbox_positions_mm=bbox_positions_mm,
        required_positions_mm=required_positions_mm,
        plane_count=plane_count,
    )
    bbox_positions = _unique_positions(bbox_positions_mm)
    required_positions = _unique_positions((*bbox_positions, *required_positions_mm))
    if len(required_positions) > plane_count:
        raise ValueError("Required positions exceed the classifier plane count")

    range_low = min((t_low_mm, *required_positions))
    range_high = max((t_high_mm, *required_positions))
    uniform_positions = np.linspace(
        range_low,
        range_high,
        plane_count,
        dtype=np.float64,
    )
    if not required_positions:
        return ClassifierPlaneSelection(
            positions_mm=tuple(float(value) for value in uniform_positions),
            bbox_positions_mm=(),
            required_positions_mm=(),
        )

    selected_positions = list(required_positions)
    candidates = [
        float(value)
        for value in uniform_positions
        if not _contains_close(selected_positions, float(value))
    ]
    while len(selected_positions) < plane_count:
        next_position = max(
            candidates,
            key=lambda candidate: (
                _distance_to_nearest(candidate, selected_positions),
                -candidate,
            ),
        )
        selected_positions.append(next_position)
        candidates.remove(next_position)

    return ClassifierPlaneSelection(
        positions_mm=tuple(sorted(selected_positions)),
        bbox_positions_mm=bbox_positions,
        required_positions_mm=required_positions,
    )


def _validate_selection_inputs(
    *,
    t_low_mm: float,
    t_high_mm: float,
    bbox_positions_mm: Sequence[float],
    required_positions_mm: Sequence[float],
    plane_count: int,
) -> None:
    if not isfinite(t_low_mm) or not isfinite(t_high_mm):
        raise ValueError("Plane range must be finite")
    if t_low_mm >= t_high_mm:
        raise ValueError("Plane range must satisfy t_low_mm < t_high_mm")
    if plane_count <= 0:
        raise ValueError("Plane count must be positive")
    if any(not isfinite(position) for position in bbox_positions_mm):
        raise ValueError("BBox positions must be finite")
    if any(not isfinite(position) for position in required_positions_mm):
        raise ValueError("Required positions must be finite")


def _unique_positions(positions: Sequence[float]) -> tuple[float, ...]:
    unique_positions: list[float] = []
    for position in sorted(float(value) for value in positions):
        if _contains_close(unique_positions, position):
            continue
        unique_positions.append(position)
    return tuple(unique_positions)


def _contains_close(positions: Sequence[float], target: float) -> bool:
    return any(
        abs(position - target) <= POSITION_TOLERANCE_MM for position in positions
    )


def _distance_to_nearest(
    position: float,
    selected_positions: Sequence[float],
) -> float:
    return min(abs(position - selected) for selected in selected_positions)
