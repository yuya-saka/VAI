"""単一3D平面のfitとz傾斜検証。"""

from __future__ import annotations

import numpy as np

from Unet.line_surface_3d.analysis.plane_feasibility import (
    LineObservation,
    SurfaceRecord,
    evaluate_record,
    fit_surface,
)


def _make_plane_record(slope: float) -> SurfaceRecord:
    """既知のz傾斜を持つ平面の交線列を作る。"""
    normal = np.array([0.6, 0.8], dtype=np.float64)
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    observations = []
    for z_index in range(7):
        rho = 4.0 + slope * z_index
        centroid = rho * normal
        positions = np.linspace(-20.0, 20.0, 9)
        points = centroid[None, :] + positions[:, None] * tangent[None, :]
        observations.append(
            LineObservation(
                z_index=z_index,
                points_xy=points,
                centroid_xy=centroid,
                normal_xy=normal,
            )
        )
    return SurfaceRecord(
        sample="sample_test",
        vertebra="C3",
        line_key="line_1",
        observations=tuple(observations),
    )


def test_fit_surface_recovers_signed_z_tilt() -> None:
    """平面fitがz傾斜の大きさと向きを復元する。"""
    record = _make_plane_record(slope=-0.35)

    result = fit_surface(record, spacing_mm=0.4)

    assert np.isclose(result.slope_px_per_slice, -0.35)
    assert result.signed_tilt_deg < 0
    assert np.isclose(result.point_residual_rms_px, 0.0, atol=1e-7)
    assert result.loo_sign_agreement == 1.0


def test_plane_extrapolation_beats_no_tilt_on_exact_plane() -> None:
    """中央観測から両端を予測するとz傾斜あり平面が正解する。"""
    record = _make_plane_record(slope=0.5)

    results = [
        result
        for result in evaluate_record(record)
        if result.protocol == "central_to_edges"
    ]
    plane_errors = [
        result.point_error_rms_px for result in results if result.model == "plane"
    ]
    no_tilt_errors = [
        result.point_error_rms_px for result in results if result.model == "no_tilt"
    ]

    assert max(plane_errors) < 1e-7
    assert min(no_tilt_errors) > 0.0


def test_half_to_half_extrapolation_preserves_tilt_direction() -> None:
    """観測の片半分だけでも反対側へ正しく平面を延長できる。"""
    record = _make_plane_record(slope=-0.4)

    results = [
        result
        for result in evaluate_record(record)
        if result.protocol == "half_to_half_extrapolation" and result.model == "plane"
    ]

    assert results
    assert max(result.point_error_rms_px for result in results) < 1e-7
