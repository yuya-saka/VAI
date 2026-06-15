from __future__ import annotations

import numpy as np

from data_preprocessing.learning_dataset.propagate_lines_z import (
    MaskGeometry,
    align_polyline_direction,
    collect_vertebra_pairs,
    confidence_from_provenance,
    constrain_extrapolation,
    denormalize_points,
    load_nrrd_as_nifti,
    normalize_points,
    propagate_normalized_line,
    resample_polyline,
    stabilize_polyline,
)


def test_resample_polyline_returns_uniform_fixed_count() -> None:
    points = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]

    result = resample_polyline(points, point_count=5)

    assert result.shape == (5, 2)
    np.testing.assert_allclose(result[0], [0.0, 0.0])
    np.testing.assert_allclose(result[-1], [2.0, 2.0])


def test_normalize_and_denormalize_points_round_trip() -> None:
    geometry = MaskGeometry(
        centroid=np.array([100.0, 80.0]),
        scale=np.array([50.0, 40.0]),
    )
    points = np.array([[75.0, 60.0], [125.0, 100.0]])

    normalized = normalize_points(points, geometry)
    restored = denormalize_points(normalized, geometry)

    np.testing.assert_allclose(restored, points)


def test_align_polyline_direction_reverses_mismatched_order() -> None:
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    reversed_points = reference[::-1].copy()

    aligned = align_polyline_direction(reversed_points, reference)

    np.testing.assert_allclose(aligned, reference)


def test_propagate_normalized_line_interpolates_between_anchors() -> None:
    anchors = {
        10: np.array([[0.0, 0.0], [1.0, 0.0]]),
        20: np.array([[0.0, 1.0], [1.0, 1.0]]),
    }

    result, provenance, distance = propagate_normalized_line(anchors, 15)

    np.testing.assert_allclose(result, [[0.0, 0.5], [1.0, 0.5]])
    assert provenance == "interpolated"
    assert distance == 5


def test_propagate_normalized_line_caps_far_extrapolation() -> None:
    anchors = {
        10: np.array([[0.0, 0.0], [1.0, 0.0]]),
        11: np.array([[0.1, 0.0], [1.1, 0.0]]),
        12: np.array([[0.2, 0.0], [1.2, 0.0]]),
    }

    result, provenance, distance = propagate_normalized_line(
        anchors,
        target_slice_idx=30,
        max_extrapolation=0.35,
    )

    displacement = result - anchors[12]
    assert np.all(np.linalg.norm(displacement, axis=1) <= 0.350001)
    assert provenance == "extrapolated"
    assert distance == 18


def test_constrain_extrapolation_preserves_small_displacement() -> None:
    displacement = np.array([[0.1, 0.2], [0.0, 0.1]])

    result = constrain_extrapolation(displacement, 0.35)

    np.testing.assert_allclose(result, displacement)


def test_confidence_decays_only_for_extrapolation() -> None:
    assert confidence_from_provenance("manual", 20, 0.4, 4.0) == 1.0
    assert confidence_from_provenance("interpolated", 20, 0.4, 4.0) == 0.9
    assert confidence_from_provenance("extrapolated", 10, 0.4, 4.0) < 0.5


def test_collect_vertebra_pairs_requires_source_and_lines(tmp_path) -> None:
    source_root = tmp_path / "source"
    anchor_root = tmp_path / "anchor"
    (source_root / "sample1" / "C3").mkdir(parents=True)
    (anchor_root / "sample1" / "C3").mkdir(parents=True)
    (anchor_root / "sample1" / "C4").mkdir(parents=True)
    (anchor_root / "sample1" / "C3" / "lines.json").write_text("{}")
    (anchor_root / "sample1" / "C4" / "lines.json").write_text("{}")

    result = collect_vertebra_pairs(source_root, anchor_root)

    assert result == [("sample1", "C3")]


def test_stabilize_polyline_expands_degenerate_points() -> None:
    geometry = MaskGeometry(
        centroid=np.array([112.0, 112.0]),
        scale=np.array([40.0, 30.0]),
    )
    points = np.repeat([[0.0, 0.0]], repeats=8, axis=0)

    result = stabilize_polyline(points, geometry)

    assert np.linalg.norm(result[-1] - result[0]) >= 4.0


def test_load_nrrd_as_nifti_reads_gzip_float_volume(tmp_path) -> None:
    import gzip

    data = np.arange(24, dtype=np.float32).reshape((2, 3, 4), order="F")
    header = "\n".join(
        [
            "NRRD0004",
            "type: float",
            "dimension: 3",
            "sizes: 2 3 4",
            "space directions: (1,0,0) (0,1,0) (0,0,1)",
            "endian: little",
            "encoding: gzip",
            "space origin: (0,0,0)",
            "",
            "",
        ]
    ).encode("ascii")
    path = tmp_path / "volume.nrrd"
    path.write_bytes(header + gzip.compress(data.tobytes(order="F")))

    image = load_nrrd_as_nifti(path)

    np.testing.assert_allclose(image.get_fdata(), data)
