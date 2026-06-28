"""3D boundary surface interpolation from sparse 2D boundary line predictions.

Mathematical basis:
    L_k(z) = S_k ∩ Π(z)

where:
    L_k(z): k-th boundary line at slice z
    S_k:    k-th 3D boundary surface (zero-level set of the signed-distance field φ_k)
    Π(z):   the slice plane at position z

    φ_k(x, y, z) = cos(φ_k(z))·x + sin(φ_k(z))·y - ρ_k(z)·D

z-direction interpolation is performed on (cos φ, sin φ, ρ) jointly.
Since each individual SDF is linear in (x, y), the interpolated SDF is also
linear, so the zero-level set at any z is always a straight line.

Sign consistency:
    phi ∈ [0, π) normalization is insufficient — nearly-horizontal lines can
    have normals pointing in opposite x-directions across slices.
    Solution: for each line, evaluate the SDF at the image centre for all
    anchor slices, then flip any slice whose sign differs from the centre
    anchor.  Flipping is done by negating ρ (equivalent to negating the SDF).

Extrapolation:
    Outside the anchor range, numpy.interp clips to the nearest boundary
    value (nearest-neighbour extrapolation).  The user accepted that the
    outer slices do not need strict accuracy.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

LINE_KEYS: Final = ("line_1", "line_2", "line_3", "line_4")
DEFAULT_IMAGE_SIZE: Final = 224
DEFAULT_LINE_LENGTH_PX: Final = 80.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sdf_at_image_centre(
    phi: float,
    rho_norm: float,
    image_size: int,
) -> float:
    """Return the signed-distance-field value at the image centre pixel.

    The image centre (cx, cy) is used as the reference point for sign
    unification.  Converting to math coordinates (y-up, origin at centre):
        x_math = 0, y_math = 0  (the centre maps to the origin)
    so SDF = cos(phi)*0 + sin(phi)*0 - rho_norm*D = -rho_norm * D.
    """
    D = math.sqrt(2.0) * image_size
    return -rho_norm * D


def _unify_signs(
    phi_list: list[float | None],
    rho_list: list[float | None],
    centre_idx: int,
    image_size: int,
) -> tuple[list[float | None], list[float | None]]:
    """Flip (phi, rho) pairs so the SDF sign at the image centre is consistent.

    The centre anchor (centre_idx) is treated as the reference.
    Any other anchor whose SDF sign at the image centre differs from the
    reference is flipped by adding π to phi and negating rho.

    Args:
        phi_list:   phi_rad per anchor (None if prediction failed).
        rho_list:   rho_normalized per anchor (None if prediction failed).
        centre_idx: index of the centre anchor (typically 2 for 5 anchors).
        image_size: square image side length in pixels.

    Returns:
        (phi_unified, rho_unified) with the same length as the inputs.
    """
    phi_out: list[float | None] = list(phi_list)
    rho_out: list[float | None] = list(rho_list)

    ref_phi = phi_list[centre_idx]
    ref_rho = rho_list[centre_idx]
    if ref_phi is None or ref_rho is None:
        return phi_out, rho_out

    sign_ref = math.copysign(1.0, _sdf_at_image_centre(ref_phi, ref_rho, image_size))

    for i, (phi, rho) in enumerate(zip(phi_list, rho_list, strict=True)):
        if i == centre_idx or phi is None or rho is None:
            continue
        sdf_val = _sdf_at_image_centre(phi, rho, image_size)
        if sdf_val == 0.0:
            continue
        if math.copysign(1.0, sdf_val) != sign_ref:
            phi_out[i] = phi + math.pi  # may go outside [0, π); that is intentional
            rho_out[i] = -rho

    return phi_out, rho_out


def _phi_rho_to_endpoints(
    phi: float,
    rho_norm: float,
    length_px: float,
    image_size: int,
    centroid_image: tuple[float, float] | None = None,
) -> list[list[float]]:
    """Convert a (phi, rho_normalized) line to a two-point list in image coords.

    Coordinate conventions (matches moments_to_phi_rho / detect_line_moments):
        Math coords: y-up, origin at image centre.
        Image coords: y-down, origin at top-left corner.

    The line direction vector in math coords is the 90°-CCW rotation of the
    normal: direction = (-sin(phi), cos(phi)).

    Args:
        phi:       Normal angle in radians (may be outside [0, π) after sign flip).
        rho_norm:  Signed distance from image centre, normalised by D = √2·image_size.
        length_px: Full length of the endpoint segment in pixels.
        image_size: Square image side length.
        centroid_image: Preferred segment centre in image coordinates. It is
                        projected onto the interpolated SDF line before endpoint
                        construction. If omitted, the closest point to the image
                        centre is used.

    Returns:
        [[x1, y1], [x2, y2]] in image coordinates.
    """
    D = math.sqrt(2.0) * image_size
    centre = image_size / 2.0

    nx = math.cos(phi)
    ny = math.sin(phi)

    if centroid_image is None:
        xbar = nx * rho_norm * D
        ybar = ny * rho_norm * D
    else:
        centroid_x, centroid_y = centroid_image
        x_candidate = float(centroid_x) - centre
        y_candidate = -(float(centroid_y) - centre)
        signed_offset = nx * x_candidate + ny * y_candidate - rho_norm * D
        xbar = x_candidate - signed_offset * nx
        ybar = y_candidate - signed_offset * ny

    # Line direction (90° CCW from normal, in math coords)
    vx = -ny
    vy = nx

    half = length_px * 0.5
    x1_math = xbar - half * vx
    y1_math = ybar - half * vy
    x2_math = xbar + half * vx
    y2_math = ybar + half * vy

    # Math → image coords
    x1 = x1_math + centre
    y1 = -y1_math + centre
    x2 = x2_math + centre
    y2 = -y2_math + centre

    return [[x1, y1], [x2, y2]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SDFBoundaryInterpolator:
    """Interpolates / extrapolates 4 boundary lines to arbitrary z positions.

    Usage example:
        phi_rho = {
            "line_1": [(phi0, rho0), (phi1, rho1), ..., (phi4, rho4)],
            ...
        }
        interp = SDFBoundaryInterpolator(phi_rho, z_offsets=[-2,-1,0,1,2])
        lines = interp.get_lines(z=3.5)  # extrapolated
        # → {"line_1": [[x1,y1],[x2,y2]], "line_2": ..., ...}

    The z_offsets can be in any consistent unit (slice indices, mm, etc.).
    Within the anchor range, linear interpolation is used; outside, the
    nearest boundary value is extrapolated.
    """

    def __init__(
        self,
        phi_rho_anchors: dict[str, list[tuple[float | None, float | None]]],
        z_offsets: list[float],
        centre_idx: int = 2,
        image_size: int = DEFAULT_IMAGE_SIZE,
        line_length_px: float = DEFAULT_LINE_LENGTH_PX,
        centroid_anchors: dict[
            str,
            list[tuple[float, float] | None],
        ] | None = None,
        line_lengths_px: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            phi_rho_anchors: {line_key → [(phi_rad, rho_norm), ...]}
                             Use (None, None) for failed predictions.
            z_offsets:       z position for each anchor, same length as each list.
            centre_idx:      Index of the centre anchor; used for sign reference.
            image_size:      Square image side length in pixels.
            line_length_px:  Backward-compatible default endpoint segment length.
            centroid_anchors: Heatmap centroids in image coordinates for each line.
            line_lengths_px: Per-line training-set average segment lengths.
        """
        if len(z_offsets) < 2:
            raise ValueError("Need at least 2 anchor slices for interpolation.")

        self._image_size = image_size
        self._line_length_px = line_length_px
        self._line_lengths_px = line_lengths_px or {}
        self._z_arr = np.asarray(z_offsets, dtype=np.float64)

        # For each line, keep SDF geometry and the heatmap centroid trajectory.
        self._line_data: dict[
            str,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        for line_key in LINE_KEYS:
            raw = phi_rho_anchors.get(line_key, [])
            if len(raw) != len(z_offsets):
                raise ValueError(
                    f"{line_key} anchor count must match z_offsets."
                )
            phi_raw = [p for p, _ in raw]
            rho_raw = [r for _, r in raw]
            raw_centroids = (
                centroid_anchors.get(line_key, [])
                if centroid_anchors is not None
                else []
            )
            if raw_centroids and len(raw_centroids) != len(z_offsets):
                raise ValueError(
                    f"{line_key} centroid count must match z_offsets."
                )

            phi_u, rho_u = _unify_signs(phi_raw, rho_raw, centre_idx, image_size)

            z_valid, cos_vals, sin_vals, rho_vals = [], [], [], []
            centroid_x_vals, centroid_y_vals = [], []
            for i, (phi, rho) in enumerate(zip(phi_u, rho_u, strict=True)):
                if phi is None or rho is None:
                    continue
                z_valid.append(z_offsets[i])
                cos_vals.append(math.cos(phi))
                sin_vals.append(math.sin(phi))
                rho_vals.append(rho)
                centroid = raw_centroids[i] if raw_centroids else None
                if centroid is None:
                    point = _phi_rho_to_endpoints(
                        phi,
                        rho,
                        0.0,
                        image_size,
                    )[0]
                    centroid_x_vals.append(point[0])
                    centroid_y_vals.append(point[1])
                else:
                    centroid_x_vals.append(float(centroid[0]))
                    centroid_y_vals.append(float(centroid[1]))

            if len(z_valid) < 2:
                continue  # insufficient anchors; this line will be skipped

            self._line_data[line_key] = (
                np.asarray(z_valid, dtype=np.float64),
                np.asarray(cos_vals, dtype=np.float64),
                np.asarray(sin_vals, dtype=np.float64),
                np.asarray(rho_vals, dtype=np.float64),
                np.asarray(centroid_x_vals, dtype=np.float64),
                np.asarray(centroid_y_vals, dtype=np.float64),
            )

    @property
    def available_lines(self) -> list[str]:
        """Line keys with sufficient valid anchors for interpolation."""
        return list(self._line_data.keys())

    def get_lines(
        self,
        z: float,
    ) -> dict[str, list[list[float]]] | None:
        """Return interpolated / extrapolated endpoint pairs at position z.

        Returns None if fewer than 4 lines have sufficient anchor data.

        The (cos φ, sin φ, ρ) components are linearly interpolated in z.
        Outside the anchor range, numpy.interp clips to the boundary value,
        giving nearest-neighbour extrapolation.  The interpolated normal
        (a, b) = (cos_interp, sin_interp) is then re-normalised to unit
        length before converting back to (phi, rho).

        Returns:
            {line_key: [[x1, y1], [x2, y2]]} or None.
        """
        if len(self._line_data) < 4:
            return None

        result: dict[str, list[list[float]]] = {}
        for line_key, (
            z_v,
            cos_v,
            sin_v,
            rho_v,
            centroid_x_v,
            centroid_y_v,
        ) in self._line_data.items():
            a = float(np.interp(z, z_v, cos_v))
            b = float(np.interp(z, z_v, sin_v))
            rho_interp = float(np.interp(z, z_v, rho_v))
            centroid_image = (
                float(np.interp(z, z_v, centroid_x_v)),
                float(np.interp(z, z_v, centroid_y_v)),
            )

            norm = math.sqrt(a * a + b * b)
            if norm < 1e-8:
                return None  # degenerate; skip entire slice

            # Re-normalise: the interpolated (a, b) may not be unit length
            phi_interp = math.atan2(b / norm, a / norm)
            rho_renorm = rho_interp / norm  # scale to match the unit normal

            result[line_key] = _phi_rho_to_endpoints(
                phi_interp,
                rho_renorm,
                self._line_lengths_px.get(line_key, self._line_length_px),
                self._image_size,
                centroid_image=centroid_image,
            )

        return result

    def get_lines_batch(
        self,
        z_targets: list[float],
    ) -> dict[float, dict[str, list[list[float]]] | None]:
        """Return interpolated lines for a list of target z positions.

        Returns:
            {z: lines_dict or None}
        """
        return {z: self.get_lines(z) for z in z_targets}
