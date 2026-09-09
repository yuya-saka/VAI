from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch
from torch import Tensor, nn

from fracture_detection.baseline0.cli import generate_pseudo_labels
from fracture_detection.baseline0.data.constants import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
    N_PLANES,
    REGION_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)
from fracture_detection.baseline0.modeling.model import Baseline0Model
from fracture_detection.baseline0.pseudo_labeling.scoring import (
    TEMPERATURE_FLOOR,
    RegionPairBatch,
    build_region_pair_batch,
    log_score,
    pairwise_confidence,
    pairwise_ranking_loss,
    region_balanced_pairwise_ranking_loss,
    region_temperature,
)

# ``scoring.py`` (cross-case pairwise ranking) is retired from the active
# pseudo-label pipeline (see ``.claude/docs/REGION_MODEL_DESIGN_JA.md``
# Section 5), but the module itself is left in place and its pure functions
# stay covered below so the historical implementation does not silently rot.


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


# --- generate_pseudo_labels.py (current CAM-soft-target pipeline) ---------


def test_guard_output_blocks_existing_new_artifacts_without_overwrite(
    tmp_path: Path,
) -> None:
    (tmp_path / generate_pseudo_labels.REGION_TARGETS_CSV).write_text("x")

    with pytest.raises(FileExistsError):
        generate_pseudo_labels._guard_output(tmp_path, overwrite=False)

    generate_pseudo_labels._guard_output(tmp_path, overwrite=True)


def test_guard_output_does_not_see_the_retired_pairwise_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "pseudo_label_scores.csv").write_text("x")
    (tmp_path / "pseudo_label_temperatures.csv").write_text("x")

    generate_pseudo_labels._guard_output(tmp_path, overwrite=False)


def test_atomic_write_csv_leaves_no_temp_file_and_is_readable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "out.csv"

    generate_pseudo_labels._atomic_write_csv(frame, path)

    assert path.exists()
    assert not path.with_suffix(".csv.tmp").exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)


def test_atomic_write_text_leaves_no_temp_file_and_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "out.json"

    generate_pseudo_labels._atomic_write_text('{"a": 1}\n', path)

    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'
    assert not path.with_suffix(".json.tmp").exists()


def _complete_frame(n_bags: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive_region = rng.integers(0, 4, size=n_bags)
    shares = rng.uniform(0.05, 0.15, size=(n_bags, 4))
    shares[np.arange(n_bags), positive_region] = rng.uniform(0.5, 0.9, size=n_bags)
    shares = shares / shares.sum(axis=1, keepdims=True)
    targets = np.zeros((n_bags, 4), dtype=np.int64)
    targets[np.arange(n_bags), positive_region] = 1
    frame = pd.DataFrame(
        {
            "study_id": [f"study_{index}" for index in range(n_bags)],
            "has_region_target": True,
        }
    )
    for region_index, region_column in enumerate(REGION_COLUMNS):
        frame[region_column] = targets[:, region_index]
    for valid_column in REGION_TARGET_VALID_COLUMNS:
        frame[valid_column] = True
    return frame, shares


def test_calibrate_fold_fits_and_produces_bounded_projected_targets() -> None:
    frame, shares = _complete_frame(n_bags=200, seed=0)
    namespace = _fake_args()

    q, calibration_row = generate_pseudo_labels._calibrate_fold(
        0, frame, shares, namespace
    )

    assert calibration_row["oof_available"] == 1.0
    assert calibration_row["slope"] > 0
    assert np.isfinite(q).all()
    assert (q >= 0).all() and (q <= 1).all()
    assert (q.sum(axis=-1) >= 1.0 - 1e-9).all()


def test_calibrate_fold_reports_no_targets_below_the_minimum_fit_population() -> None:
    frame, shares = _complete_frame(n_bags=1, seed=1)
    namespace = _fake_args()

    q, calibration_row = generate_pseudo_labels._calibrate_fold(
        0, frame, shares, namespace
    )

    assert calibration_row["oof_available"] == 0.0
    assert calibration_row["n_fit_bags"] == 1
    assert np.isnan(q).all()


def _fake_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "share_floor": generate_pseudo_labels.SHARE_FLOOR,
        "regularization_c": generate_pseudo_labels.REGULARIZATION_C,
    }
    return argparse.Namespace(**{**defaults, **overrides})


class _DeterministicEncoder(nn.Module):
    """A 1x1 conv + ReLU whose positive constant weights guarantee a CAM

    that is non-negative and strictly positive anywhere the input (CT or the
    whole-vertebra mask channel) is non-zero, independent of random init.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(6, 4, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 0.05)
        self.bn2 = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bn2(self.conv(inputs)).mean(dim=(-2, -1))


class _DeterministicBaseline0(Baseline0Model):
    """A tiny stand-in for Baseline0Model with a guaranteed-positive CAM.

    ``load_baseline0_checkpoint`` builds a real EfficientNetV2 backbone,
    which is unaffordable in a fast unit test; this model swaps that out
    while keeping the exact hook target (``encoder.bn2``) and forward
    contract ``compute_gradcam`` relies on.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.n_planes = N_PLANES
        self.encoder = _DeterministicEncoder()
        self.head = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.head.weight, 1.0)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, plane_count, channels, height, width = inputs.shape
        features = self.encoder(
            inputs.reshape(batch_size * plane_count, channels, height, width)
        )
        return self.head(features).reshape(batch_size, plane_count)


def _write_smoke_bag(dataset_dir: Path, study_id: str, level: str) -> None:
    bag_dir = dataset_dir / study_id / level
    bag_dir.mkdir(parents=True)
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole_mask[:, 40:184, 40:184] = 1
    region_mask[:, 40:112, 40:112] = 1
    region_mask[:, 40:112, 112:184] = 2
    region_mask[:, 112:184, 40:112] = 3
    region_mask[:, 112:184, 112:184] = 4
    np.save(bag_dir / "ct.npy", ct)
    np.save(bag_dir / "vertebra_mask.npy", whole_mask)
    np.save(bag_dir / "region_4class.npy", region_mask)


def _write_smoke_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub-checkpoint")


def test_run_generation_end_to_end_smoke_writes_the_three_new_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    experiment_dir = tmp_path / "experiment"
    output_dir = tmp_path / "pseudo_labels"

    rows: list[dict[str, object]] = []
    for fold in range(5):
        study_id = f"study_{fold}"
        _write_smoke_bag(dataset_dir, study_id, "C1")
        _write_smoke_checkpoint(experiment_dir / f"outer{fold}" / "best_model.pt")
        row = {
            "study_id": study_id,
            "level": "C1",
            "fold": fold,
            "vertebra_target": 1,
            "has_region_target": False,
        }
        for region_column in REGION_COLUMNS:
            row[region_column] = 0
        for valid_column in REGION_TARGET_VALID_COLUMNS:
            row[valid_column] = False
        rows.append(row)
    manifest = pd.DataFrame(rows)

    monkeypatch.setattr(generate_pseudo_labels, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        generate_pseudo_labels,
        "load_baseline0_checkpoint",
        lambda checkpoint_path, device: _DeterministicBaseline0().to(device).eval(),
    )

    namespace = _fake_args(
        experiment_dir=experiment_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        checkpoint_name="best_model.pt",
        device="cpu",
        batch_size=4,
        limit_bags=1,
        overwrite=False,
    )

    result_dir = generate_pseudo_labels.run_generation(namespace)

    assert result_dir == output_dir
    targets_path = output_dir / generate_pseudo_labels.REGION_TARGETS_CSV
    calibration_path = output_dir / generate_pseudo_labels.CALIBRATION_CSV
    metadata_path = output_dir / generate_pseudo_labels.METADATA_JSON
    assert targets_path.exists()
    assert calibration_path.exists()
    assert metadata_path.exists()

    targets = pd.read_csv(targets_path, dtype={"study_id": str, "level": str})
    assert len(targets) == 10
    assert set(targets["student_outer_fold"]) == set(range(5))
    assert set(targets["teacher_outer_fold"]) == set(range(5))
    for student_outer_fold, subset in targets.groupby("student_outer_fold"):
        assert set(subset["teacher_outer_fold"]) == {
            student_outer_fold,
            (student_outer_fold + 1) % 5,
        }
    for region_column in REGION_COLUMNS:
        share_values = targets[f"{region_column}_cam_share"].to_numpy()
        assert np.isfinite(share_values).all()
        assert (share_values >= 0).all() and (share_values <= 1).all()

    calibration = pd.read_csv(calibration_path)
    assert len(calibration) == 5
    # --limit-bags=1 leaves a single-bag train split, below the minimum fit
    # population, so every fold falls back to the "no calibration" branch.
    assert calibration["oof_available"].eq(0.0).all()

    metadata = generate_pseudo_labels.json.loads(metadata_path.read_text())
    assert metadata["smoke_only"] is True
    assert metadata["views"] == [
        view.name for view in generate_pseudo_labels.DEFAULT_TTA_VIEWS
    ]


def test_run_generation_respects_the_overwrite_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "pseudo_labels"
    output_dir.mkdir()
    (output_dir / generate_pseudo_labels.REGION_TARGETS_CSV).write_text("existing")
    monkeypatch.setattr(generate_pseudo_labels, "load_manifest", lambda: pd.DataFrame())
    namespace = _fake_args(
        experiment_dir=tmp_path / "experiment",
        dataset_dir=tmp_path / "dataset",
        output_dir=output_dir,
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        generate_pseudo_labels.run_generation(namespace)
