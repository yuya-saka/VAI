"""Generation-stage audit primitives for Baseline 0 Grad-CAM region scores.

The pseudo-label plan uses the region-wise Grad-CAM density of an already
trained vertebra-level model as the teacher signal. Before spending training
runs on it, two properties have to hold:

1. the CAM localizes because the model found the lesion, not because it
   memorized the vertebra it was trained on (teacher memorization), and
2. the region score survives plausible errors in the four-region segmentation
   masks used to aggregate it (mask boundary sensitivity).

This module holds the pure, testable pieces used by both checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from fracture_detection.common.constants import EXPECTED_MASK_SHAPE, N_REGIONS

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]

PerturbationKind = Literal["identity", "erode", "dilate", "shift"]
ScalarT = TypeVar("ScalarT", bound=np.generic)

# One 7x7 Grad-CAM cell covers 32 input pixels. At the frozen 0.4 mm/px input
# geometry, 4 px is 1.6 mm and 8 px is 3.2 mm, which brackets a plausible
# segmentation error. Larger offsets are reported descriptively only.
GATE_PERTURBATION_PIXELS = 4


@dataclass(frozen=True)
class MaskPerturbation:
    """One deterministic perturbation of the four-region mask."""

    name: str
    kind: PerturbationKind
    amount_pixels: int = 0
    dy: int = 0
    dx: int = 0

    def __post_init__(self) -> None:
        if self.kind == "identity" and (self.amount_pixels or self.dy or self.dx):
            raise ValueError("identity perturbation must not carry an amount")
        if self.kind in {"erode", "dilate"} and self.amount_pixels < 1:
            raise ValueError(f"{self.kind} needs a positive radius")
        if self.kind == "shift" and self.dy == 0 and self.dx == 0:
            raise ValueError("shift needs a non-zero offset")


def default_perturbations() -> tuple[MaskPerturbation, ...]:
    """Return the frozen perturbation grid used by the audit."""
    perturbations: list[MaskPerturbation] = [MaskPerturbation("identity", "identity")]
    for radius in (2, 4, 8):
        perturbations.append(
            MaskPerturbation(f"erode_{radius}", "erode", amount_pixels=radius)
        )
        perturbations.append(
            MaskPerturbation(f"dilate_{radius}", "dilate", amount_pixels=radius)
        )
    for offset in (4, 8):
        perturbations.append(MaskPerturbation(f"shift_x_p{offset}", "shift", dx=offset))
        perturbations.append(
            MaskPerturbation(f"shift_x_m{offset}", "shift", dx=-offset)
        )
        perturbations.append(MaskPerturbation(f"shift_y_p{offset}", "shift", dy=offset))
        perturbations.append(
            MaskPerturbation(f"shift_y_m{offset}", "shift", dy=-offset)
        )
    return tuple(perturbations)


def _structuring_element(radius: int) -> UInt8Array:
    size = 2 * radius + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return element.astype(np.uint8)


def _shift_planes(mask: BoolArray, dy: int, dx: int) -> BoolArray:
    """Shift every plane by (dy, dx) with zero fill (no wrap-around)."""
    shifted = np.zeros_like(mask)
    height, width = mask.shape[-2:]
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    shifted[..., dst_y, dst_x] = mask[..., src_y, src_x]
    return shifted


def perturb_region(
    region: BoolArray,
    whole: BoolArray,
    perturbation: MaskPerturbation,
) -> BoolArray:
    """Apply one perturbation to a single region mask, plane by plane.

    Dilated and shifted masks are intersected with the unperturbed whole-vertebra
    mask: an error in how the vertebra is subdivided does not move tissue outside
    the vertebra, and the whole mask is the reference denominator of the score.
    """
    if region.shape != whole.shape:
        raise ValueError("region and whole masks must have the same shape")
    if perturbation.kind == "identity":
        return region
    if perturbation.kind == "shift":
        return _shift_planes(region, perturbation.dy, perturbation.dx) & whole

    element = _structuring_element(perturbation.amount_pixels)
    operation = cv2.erode if perturbation.kind == "erode" else cv2.dilate
    planes = [
        operation(plane.astype(np.uint8), element).astype(bool) for plane in region
    ]
    changed = np.stack(planes, axis=0)
    if perturbation.kind == "dilate":
        return changed & whole
    return changed


def region_density_enrichment(
    cams: FloatArray,
    whole_mask: UInt8Array,
    region_mask: UInt8Array,
    perturbation: MaskPerturbation,
) -> FloatArray:
    """Area-corrected CAM density of each region, relative to the whole vertebra.

    The score is ``(CAM mass in region / region area) / (CAM mass in vertebra /
    vertebra area)``. Masses and areas are summed over all 15 planes before the
    ratio, so a single noisy plane cannot dominate. Regions that vanish under a
    perturbation get NaN.
    """
    if cams.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Invalid CAM shape: {cams.shape}")
    if whole_mask.shape != cams.shape or region_mask.shape != cams.shape:
        raise ValueError("CAM and mask shapes must match")
    if not np.isfinite(cams).all() or np.any(cams < 0):
        raise ValueError("CAM values must be finite and non-negative")

    whole = whole_mask > 0
    whole_area = int(whole.sum())
    whole_mass = float(cams[whole].sum(dtype=np.float64))
    if whole_area == 0 or whole_mass <= 0.0:
        return np.full(N_REGIONS, np.nan, dtype=np.float32)
    whole_density = whole_mass / whole_area

    scores = np.full(N_REGIONS, np.nan, dtype=np.float32)
    for index in range(N_REGIONS):
        region = (region_mask == index + 1) & whole
        perturbed = perturb_region(region, whole, perturbation)
        area = int(perturbed.sum())
        if area == 0:
            continue
        mass = float(cams[perturbed].sum(dtype=np.float64))
        scores[index] = np.float32((mass / area) / whole_density)
    return scores


def teacher_role(bag_fold: int, teacher_fold: int, n_folds: int = 5) -> str:
    """Role of one bag for the Baseline 0 run whose outer fold is ``teacher_fold``.

    The frozen protocol is outer=k, inner=(k+1)%n, train=the rest
    (``common.splits.resolve_nested_folds``). ``train`` bags are in-sample for
    the teacher, ``inner`` bags were only used for checkpoint selection, and
    ``outer`` bags were never seen at all.
    """
    if n_folds != 5:
        raise ValueError("The frozen research contract requires n_folds=5")
    if bag_fold not in range(n_folds) or teacher_fold not in range(n_folds):
        raise ValueError(f"Invalid fold: bag={bag_fold}, teacher={teacher_fold}")
    if bag_fold == teacher_fold:
        return "outer"
    if bag_fold == (teacher_fold + 1) % n_folds:
        return "inner"
    return "train"


def flip_planes_horizontally(values: NDArray[ScalarT]) -> NDArray[ScalarT]:
    """Mirror the last axis of a plane stack.

    Baseline 0 only consumes CT and the whole-vertebra mask, so the four-region
    mask never enters the forward pass. Flipping the CAM back into the original
    frame therefore needs no R2/R3 swap; the original region mask stays valid.
    """
    return cast(NDArray[ScalarT], np.ascontiguousarray(values[..., ::-1]))


def gate_perturbation_names(
    perturbations: tuple[MaskPerturbation, ...] | None = None,
) -> tuple[str, ...]:
    """Names of the perturbations the kill criterion is applied to.

    Only the plausible-segmentation-error magnitude gates the decision. Larger
    offsets stay in the report as a descriptive sensitivity curve.
    """
    grid = default_perturbations() if perturbations is None else perturbations
    return tuple(
        item.name
        for item in grid
        if max(abs(item.amount_pixels), abs(item.dy), abs(item.dx))
        == GATE_PERTURBATION_PIXELS
    )
