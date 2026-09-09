from __future__ import annotations

import numpy as np
import pytest

from fracture_detection.baseline0.pseudo_labeling.calibration import (
    N_REGIONS,
    SharedLogitShareCalibration,
    apply_shared_logit_share_calibration,
    average_view_shares,
    cross_validated_calibration_quality,
    enrichment_to_share,
    fit_shared_logit_share_calibration,
    project_sum_at_least_one,
    shares_to_logit_features,
    summarize_probability_distribution,
)


def test_enrichment_to_share_normalizes_rows_to_one() -> None:
    enrichment = np.array([[1.0, 1.0, 2.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    shares = enrichment_to_share(enrichment)
    np.testing.assert_allclose(shares.sum(axis=-1), 1.0)
    np.testing.assert_allclose(shares[0], [0.25, 0.25, 0.5, 0.0])


def test_enrichment_to_share_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        enrichment_to_share(np.array([[1.0, -0.1, 0.0, 0.0]]))


def test_enrichment_to_share_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        enrichment_to_share(np.array([[1.0, np.nan, 0.0, 0.0]]))


def test_enrichment_to_share_rejects_zero_mass_row() -> None:
    with pytest.raises(ValueError, match="positive"):
        enrichment_to_share(np.array([[0.0, 0.0, 0.0, 0.0]]))


def test_average_view_shares_averages_after_sharing_not_before() -> None:
    # Two views with very different magnitudes but the same underlying pattern.
    view_a = enrichment_to_share(np.array([[3.0, 1.0, 0.0, 0.0]]))
    view_b = enrichment_to_share(np.array([[30.0, 40.0, 0.0, 0.0]]))
    stacked = np.stack([view_a, view_b], axis=0)

    share_then_average = average_view_shares(stacked)
    average_then_share = enrichment_to_share(
        np.stack(
            [np.array([[3.0, 1.0, 0.0, 0.0]]), np.array([[30.0, 40.0, 0.0, 0.0]])],
            axis=0,
        ).mean(axis=0)
    )

    assert not np.allclose(share_then_average, average_then_share)
    np.testing.assert_allclose(share_then_average.sum(axis=-1), 1.0)
    # mean(share(view)) manually: view_a=[0.75,0.25,0,0], view_b=[3/7,4/7,0,0]
    expected = (
        np.array([0.75, 0.25, 0.0, 0.0]) + np.array([3 / 7, 4 / 7, 0.0, 0.0])
    ) / 2
    np.testing.assert_allclose(share_then_average[0], expected)


def test_average_view_shares_rejects_a_view_that_is_not_normalized() -> None:
    not_shared = np.array([[[0.5, 0.5, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]]])
    with pytest.raises(ValueError, match="must already be a region share"):
        average_view_shares(not_shared)


def test_average_view_shares_requires_a_leading_view_axis() -> None:
    with pytest.raises(ValueError, match="leading view axis"):
        average_view_shares(np.array([0.25, 0.25, 0.25, 0.25]))


def test_shares_to_logit_features_clips_before_the_logit() -> None:
    shares = np.array([[0.0, 1.0, 0.5, 0.005]])
    features = shares_to_logit_features(shares, share_floor=0.01)
    expected_extreme = np.log(0.01 / 0.99)
    assert features[0, 0] == pytest.approx(expected_extreme)
    assert features[0, 1] == pytest.approx(-expected_extreme)
    assert features[0, 2] == pytest.approx(0.0, abs=1e-9)


def _synthetic_fit_population(
    rng: np.random.Generator, n_bags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bags whose positive region has a high share, matching a positive slope."""
    positive_region = rng.integers(0, N_REGIONS, size=n_bags)
    shares = rng.uniform(0.05, 0.15, size=(n_bags, N_REGIONS))
    shares[np.arange(n_bags), positive_region] = rng.uniform(0.5, 0.9, size=n_bags)
    shares = shares / shares.sum(axis=1, keepdims=True)
    targets = np.zeros((n_bags, N_REGIONS), dtype=np.int64)
    targets[np.arange(n_bags), positive_region] = 1
    study_ids = np.array([f"study_{index}" for index in range(n_bags)])
    return shares, targets, study_ids


def test_fit_shared_logit_share_calibration_recovers_a_positive_slope() -> None:
    rng = np.random.default_rng(0)
    shares, targets, study_ids = _synthetic_fit_population(rng, 200)

    calibration = fit_shared_logit_share_calibration(shares, targets, study_ids, 0)

    assert calibration.slope > 0
    assert calibration.n_fit_bags == 200
    assert calibration.n_fit_studies == 200


def test_fit_shared_logit_share_calibration_rejects_a_nonpositive_slope() -> None:
    rng = np.random.default_rng(1)
    shares, targets, study_ids = _synthetic_fit_population(rng, 200)
    # Invert the label so a high share predicts a negative label instead.
    inverted_targets = 1 - targets

    with pytest.raises(ValueError, match="slope must be positive"):
        fit_shared_logit_share_calibration(shares, inverted_targets, study_ids, 0)


def test_fit_shared_logit_share_calibration_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        fit_shared_logit_share_calibration(
            np.zeros((3, 4)), np.zeros((3, 3)), ["a", "b", "c"], 0
        )


def test_fit_shared_logit_share_calibration_rejects_soft_targets() -> None:
    with pytest.raises(ValueError, match="hard 0/1"):
        fit_shared_logit_share_calibration(
            np.full((3, 4), 0.25), np.full((3, 4), 0.5), ["a", "b", "c"], 0
        )


def test_apply_shared_logit_share_calibration_is_region_agnostic() -> None:
    """The same (slope, intercept) map applies uniformly regardless of region."""
    calibration = SharedLogitShareCalibration(
        student_outer_fold=0,
        slope=1.5,
        intercept=-0.2,
        n_fit_bags=10,
        n_fit_studies=10,
    )
    # Region 1's share in bag A equals region 3's share in bag B.
    shares = np.array([[0.4, 0.2, 0.2, 0.2], [0.2, 0.2, 0.4, 0.2]])

    probabilities = apply_shared_logit_share_calibration(shares, calibration)

    assert probabilities[0, 0] == pytest.approx(probabilities[1, 2])
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_project_sum_at_least_one_only_rescales_incoherent_rows() -> None:
    probabilities = np.array([[0.1, 0.1, 0.1, 0.1], [0.6, 0.3, 0.1, 0.05]])

    projected = project_sum_at_least_one(probabilities)

    np.testing.assert_allclose(projected[0].sum(), 1.0)
    np.testing.assert_allclose(projected[1], probabilities[1])


def test_project_sum_at_least_one_preserves_within_bag_rank() -> None:
    probabilities = np.array([[0.05, 0.2, 0.02, 0.01]])

    projected = project_sum_at_least_one(probabilities)

    assert np.argsort(projected[0]).tolist() == np.argsort(probabilities[0]).tolist()
    assert projected.sum() >= 1.0 - 1e-9
    assert (projected <= 1.0).all()


def test_project_sum_at_least_one_rejects_out_of_range_probabilities() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        project_sum_at_least_one(np.array([[1.5, 0.0, 0.0, 0.0]]))


def test_cross_validated_calibration_quality_reports_oof_metrics() -> None:
    rng = np.random.default_rng(2)
    shares, targets, study_ids = _synthetic_fit_population(rng, 100)

    quality = cross_validated_calibration_quality(shares, targets, study_ids, 0)

    assert quality["oof_available"] == 1.0
    assert 0.0 <= quality["oof_ap"] <= 1.0
    assert quality["oof_brier"] >= 0.0


def test_cross_validated_calibration_quality_handles_too_few_studies() -> None:
    shares = np.full((3, 4), 0.25)
    targets = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    quality = cross_validated_calibration_quality(
        shares, targets, ["only_one_study"] * 3, 0
    )
    assert quality["oof_available"] == 0.0


def test_summarize_probability_distribution_reports_quantiles_and_projection_rate() -> (
    None
):
    probabilities = np.array([[0.5, 0.4, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1]])

    summary = summarize_probability_distribution(probabilities)

    assert summary["n_bags"] == 2.0
    assert summary["n_cells"] == 8.0
    assert summary["projected_fraction"] == pytest.approx(0.5)
    assert summary["sum_p50"] > 0.0


def test_full_pipeline_produces_bounded_finite_projected_targets() -> None:
    rng = np.random.default_rng(3)
    fit_shares, fit_targets, fit_studies = _synthetic_fit_population(rng, 150)
    calibration = fit_shared_logit_share_calibration(
        fit_shares, fit_targets, fit_studies, 0
    )

    pool_enrichment = rng.uniform(0.1, 5.0, size=(50, N_REGIONS))
    pool_shares = enrichment_to_share(pool_enrichment)
    raw_q = apply_shared_logit_share_calibration(pool_shares, calibration)
    projected = project_sum_at_least_one(raw_q)

    assert np.isfinite(projected).all()
    assert (projected >= 0).all() and (projected <= 1).all()
    assert (projected.sum(axis=-1) >= 1.0 - 1e-9).all()
