from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from fracture_detection.baseline0.analysis.pseudo_label import (
    TEMPERATURE_FLOOR,
    RegionPairBatch,
    build_region_pair_batch,
    log_score,
    pairwise_confidence,
    pairwise_ranking_loss,
    region_balanced_pairwise_ranking_loss,
    region_temperature,
)
from fracture_detection.baseline0.cli.generate_pseudo_labels import (
    _compute_temperatures,
)
from fracture_detection.common.constants import REGION_COLUMNS


def test_log_score_floors_zero_density_instead_of_diverging() -> None:
    density = np.array([0.0, 1.0, np.e], dtype=np.float64)
    logs = log_score(density)
    assert np.isfinite(logs).all()
    assert logs[0] < logs[1] < logs[2]
    assert logs[2] == pytest.approx(1.0, abs=1e-6)


def test_log_score_matches_between_numpy_and_torch() -> None:
    density = np.array([0.5, 2.0], dtype=np.float64)
    numpy_logs = log_score(density)
    torch_logs = log_score(torch.tensor(density))
    assert torch.allclose(torch.tensor(numpy_logs), torch_logs)


def test_region_temperature_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    scores = rng.lognormal(size=500)
    first = region_temperature(scores, n_pairs=1000, seed=7)
    second = region_temperature(scores, n_pairs=1000, seed=7)
    assert first == second


def test_region_temperature_differs_across_seeds_but_stays_close() -> None:
    rng = np.random.default_rng(2)
    scores = rng.lognormal(size=2000)
    values = [region_temperature(scores, n_pairs=5000, seed=s) for s in range(5)]
    assert len(set(values)) > 1
    assert max(values) / min(values) < 1.5


def test_region_temperature_scales_with_the_spread_of_the_population() -> None:
    rng = np.random.default_rng(3)
    tight = np.exp(rng.normal(0, 0.1, size=2000))
    wide = np.exp(rng.normal(0, 1.0, size=2000))
    assert region_temperature(tight, seed=11) < region_temperature(wide, seed=11)


def test_region_temperature_ignores_nonfinite_and_nonpositive_scores() -> None:
    rng = np.random.default_rng(4)
    clean = rng.lognormal(size=1000)
    contaminated = np.concatenate([clean, np.array([np.nan, 0.0, -1.0, np.inf] * 50)])
    rng.shuffle(contaminated)
    clean_value = region_temperature(clean, seed=5)
    contaminated_value = region_temperature(contaminated, seed=5)
    assert contaminated_value == pytest.approx(clean_value, rel=0.05)


def test_region_temperature_rejects_a_degenerate_population() -> None:
    with pytest.raises(ValueError):
        region_temperature(np.array([1.0]))
    with pytest.raises(ValueError):
        region_temperature(np.array([np.nan, -1.0, 0.0]))


def test_region_temperature_never_goes_below_the_floor() -> None:
    identical = np.full(200, 3.0)
    assert region_temperature(identical, seed=1) == pytest.approx(TEMPERATURE_FLOOR)


def test_region_temperature_handles_a_two_bag_population_without_collapsing() -> None:
    scores = np.array([1.0, 4.0])
    temperature = region_temperature(scores, n_pairs=1000, seed=1)
    assert temperature > TEMPERATURE_FLOOR


def test_region_temperature_rejects_nonpositive_pair_count() -> None:
    with pytest.raises(ValueError, match="n_pairs must be positive"):
        region_temperature(np.array([1.0, 2.0]), n_pairs=0)


def test_compute_temperatures_uses_only_fracture_positive_bags() -> None:
    positive_scores = np.array([1.0, 2.0, 4.0, 8.0])
    frame = pd.DataFrame(
        {
            "teacher_outer_fold": [0] * 6,
            "vertebra_target": [1, 1, 1, 1, 0, 0],
            **{
                f"{region}_score": [*positive_scores, 1e-9, 1e9]
                for region in REGION_COLUMNS
            },
        }
    )

    temperatures = _compute_temperatures(frame, n_pairs=100, seed=9)
    expected = region_temperature(positive_scores, n_pairs=100, seed=9)

    assert temperatures["population"].eq("fracture_positive").all()
    assert temperatures["n_bags"].eq(4).all()
    assert temperatures["n_defined"].eq(4).all()
    assert temperatures["temperature"].to_numpy() == pytest.approx(expected)


def test_pairwise_confidence_is_near_half_for_a_vanishing_gap() -> None:
    equal = torch.zeros(1)
    confidence = pairwise_confidence(equal, equal, temperature=1.0)
    assert confidence.item() == pytest.approx(0.5, abs=1e-6)


def test_pairwise_confidence_saturates_for_a_gap_much_larger_than_temperature() -> None:
    large_gap = torch.tensor([10.0])
    zero = torch.zeros(1)
    confidence = pairwise_confidence(large_gap, zero, temperature=0.5)
    assert confidence.item() > 0.999
    reversed_confidence = pairwise_confidence(zero, large_gap, temperature=0.5)
    assert reversed_confidence.item() < 0.001


def test_pairwise_confidence_is_antisymmetric() -> None:
    a = torch.tensor([1.3, -0.4])
    b = torch.tensor([0.2, 0.9])
    forward = pairwise_confidence(a, b, temperature=0.7)
    backward = pairwise_confidence(b, a, temperature=0.7)
    assert torch.allclose(forward, 1.0 - backward, atol=1e-6)


def test_pairwise_confidence_rejects_a_nonpositive_temperature() -> None:
    zero = torch.zeros(1)
    with pytest.raises(ValueError):
        pairwise_confidence(zero, zero, temperature=0.0)
    with pytest.raises(ValueError):
        pairwise_confidence(zero, zero, temperature=-1.0)


def test_ranking_loss_is_small_when_the_student_agrees_with_a_confident_teacher() -> (
    None
):
    student_i = torch.tensor([4.0])
    student_j = torch.tensor([-4.0])
    teacher_i = torch.tensor([3.0])
    teacher_j = torch.tensor([-3.0])
    loss = pairwise_ranking_loss(student_i, student_j, teacher_i, teacher_j, 1.0)
    assert loss.item() < 0.05


def test_ranking_loss_is_large_when_the_student_disagrees_with_a_confident_teacher() -> (
    None
):
    student_i = torch.tensor([-4.0])
    student_j = torch.tensor([4.0])
    teacher_i = torch.tensor([3.0])
    teacher_j = torch.tensor([-3.0])
    loss = pairwise_ranking_loss(student_i, student_j, teacher_i, teacher_j, 1.0)
    assert loss.item() > 3.0


def test_ranking_loss_weighs_a_confident_pair_more_than_a_near_tie() -> None:
    # Same student disagreement, but the teacher's evidence gap differs.
    student_i = torch.tensor([-1.0])
    student_j = torch.tensor([1.0])
    confident_teacher_i = torch.tensor([5.0])
    confident_teacher_j = torch.tensor([-5.0])
    near_tie_teacher_i = torch.tensor([0.05])
    near_tie_teacher_j = torch.tensor([-0.05])
    confident_loss = pairwise_ranking_loss(
        student_i, student_j, confident_teacher_i, confident_teacher_j, 1.0
    )
    near_tie_loss = pairwise_ranking_loss(
        student_i, student_j, near_tie_teacher_i, near_tie_teacher_j, 1.0
    )
    assert confident_loss.item() > near_tie_loss.item()


def test_ranking_loss_does_not_backpropagate_into_the_teacher_scores() -> None:
    student_i = torch.tensor([0.3], requires_grad=True)
    student_j = torch.tensor([-0.1], requires_grad=True)
    teacher_i = torch.tensor([1.2], requires_grad=True)
    teacher_j = torch.tensor([0.4], requires_grad=True)
    loss = pairwise_ranking_loss(student_i, student_j, teacher_i, teacher_j, 1.0)
    loss.backward()
    assert student_i.grad is not None
    assert teacher_i.grad is None


def test_ranking_loss_rejects_mismatched_shapes() -> None:
    a = torch.zeros(2)
    b = torch.zeros(3)
    with pytest.raises(ValueError):
        pairwise_ranking_loss(a, a, b, b, 1.0)
    with pytest.raises(ValueError):
        pairwise_ranking_loss(a, a, a, torch.zeros(2, 1), 1.0)


def test_build_region_pairs_filters_negatives_human_targets_and_undefined_scores() -> (
    None
):
    teacher_scores = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [0.0, 4.0],
            [5.0, float("nan")],
        ]
    )
    vertebra_targets = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    human_target_valid = torch.tensor(
        [
            [False, False],
            [True, False],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    generator = torch.Generator().manual_seed(17)

    pairs = build_region_pair_batch(
        teacher_scores,
        vertebra_targets,
        torch.zeros(5, dtype=torch.int64),
        generator,
        human_target_valid,
    )

    assert pairs.pair_counts_by_region.tolist() == [2, 3]
    assert pairs.n_pairs == 5
    for region_index, expected in ((0, {0, 4}), (1, {0, 1, 3})):
        region_mask = pairs.region_indices.eq(region_index)
        assert set(pairs.left_bag_indices[region_mask].tolist()) == expected
        assert set(pairs.right_bag_indices[region_mask].tolist()) == expected
    assert not torch.any(pairs.left_bag_indices.eq(pairs.right_bag_indices))


def test_build_region_pairs_pseudo_only_does_not_exclude_annotated_cells() -> None:
    teacher_scores = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.ones(3)
    teacher_folds = torch.zeros(3, dtype=torch.int64)

    pairs = build_region_pair_batch(
        teacher_scores,
        targets,
        teacher_folds,
        torch.Generator().manual_seed(1),
    )

    assert pairs.pair_counts_by_region.tolist() == [3]
    assert set(pairs.left_bag_indices.tolist()) == {0, 1, 2}


def test_build_region_pairs_keeps_exact_ties_as_soft_pairs() -> None:
    pairs = build_region_pair_batch(
        torch.ones(4, 1),
        torch.ones(4),
        torch.zeros(4, dtype=torch.int64),
        torch.Generator().manual_seed(3),
    )
    targets = pairwise_confidence(
        pairs.teacher_log_score_left,
        pairs.teacher_log_score_right,
        temperature=1.0,
    )
    assert pairs.n_pairs == 4
    assert torch.allclose(targets, torch.full((4,), 0.5))


def test_build_region_pairs_is_reproducible_for_generator_state() -> None:
    scores = torch.arange(1, 9, dtype=torch.float32).unsqueeze(1)
    targets = torch.ones(8)
    teacher_folds = torch.full((8,), 2, dtype=torch.int64)

    first = build_region_pair_batch(
        scores,
        targets,
        teacher_folds,
        torch.Generator().manual_seed(11),
    )
    second = build_region_pair_batch(
        scores,
        targets,
        teacher_folds,
        torch.Generator().manual_seed(11),
    )
    third = build_region_pair_batch(
        scores,
        targets,
        teacher_folds,
        torch.Generator().manual_seed(12),
    )

    assert torch.equal(first.left_bag_indices, second.left_bag_indices)
    assert torch.equal(first.right_bag_indices, second.right_bag_indices)
    assert not torch.equal(first.left_bag_indices, third.left_bag_indices)


def test_build_region_pairs_rejects_mixed_teachers() -> None:
    with pytest.raises(ValueError, match="exactly one teacher"):
        build_region_pair_batch(
            torch.ones(2, 1),
            torch.ones(2),
            torch.tensor([0, 1]),
            torch.Generator().manual_seed(1),
        )


def test_region_balanced_pairwise_loss_matches_manual_region_means() -> None:
    teacher_scores = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    pairs = build_region_pair_batch(
        teacher_scores,
        torch.ones(4),
        torch.zeros(4, dtype=torch.int64),
        torch.Generator().manual_seed(5),
    )
    student_logits = torch.tensor(
        [[0.1, 0.2], [0.4, -0.1], [0.7, 0.5], [-0.3, 0.9]],
        requires_grad=True,
    )
    temperatures = torch.tensor([0.7, 1.3])

    loss = region_balanced_pairwise_ranking_loss(student_logits, pairs, temperatures)
    pair_targets = torch.sigmoid(
        (pairs.teacher_log_score_left - pairs.teacher_log_score_right)
        / temperatures[pairs.region_indices]
    )
    pair_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits[pairs.left_bag_indices, pairs.region_indices]
        - student_logits[pairs.right_bag_indices, pairs.region_indices],
        pair_targets,
        reduction="none",
    )
    expected = torch.stack(
        [
            pair_losses[pairs.region_indices.eq(region_index)].mean()
            for region_index in range(2)
        ]
    ).mean()

    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert student_logits.grad is not None


def test_region_pair_builder_detaches_teacher_scores() -> None:
    teacher_scores = torch.tensor([[1.0], [2.0]], requires_grad=True)
    pairs = build_region_pair_batch(
        teacher_scores,
        torch.ones(2),
        torch.zeros(2, dtype=torch.int64),
        torch.Generator().manual_seed(2),
    )
    student_logits = torch.tensor([[0.1], [0.2]], requires_grad=True)

    loss = region_balanced_pairwise_ranking_loss(student_logits, pairs, torch.ones(1))
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_scores.grad is None


def test_region_balanced_pairwise_loss_returns_connected_zero_without_pairs() -> None:
    pairs = build_region_pair_batch(
        torch.tensor([[1.0], [0.0]]),
        torch.ones(2),
        torch.zeros(2, dtype=torch.int64),
        torch.Generator().manual_seed(4),
    )
    student_logits = torch.tensor([[0.2], [-0.3]], requires_grad=True)

    loss = region_balanced_pairwise_ranking_loss(student_logits, pairs, torch.ones(1))
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(student_logits.grad, torch.zeros_like(student_logits))


def test_region_balanced_pairwise_loss_rejects_inconsistent_region_counts() -> None:
    pairs = RegionPairBatch(
        left_bag_indices=torch.tensor([0, 1]),
        right_bag_indices=torch.tensor([1, 0]),
        region_indices=torch.tensor([1, 1]),
        teacher_log_score_left=torch.tensor([0.0, 1.0]),
        teacher_log_score_right=torch.tensor([1.0, 0.0]),
        pair_counts_by_region=torch.tensor([2, 0]),
    )
    with pytest.raises(ValueError, match="counts do not match region indices"):
        region_balanced_pairwise_ranking_loss(torch.zeros(2, 2), pairs, torch.ones(2))
