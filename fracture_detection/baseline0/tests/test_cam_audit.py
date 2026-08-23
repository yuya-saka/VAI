from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fracture_detection.baseline0.analysis.cam_audit import (
    GATE_PERTURBATION_PIXELS,
    MaskPerturbation,
    default_perturbations,
    flip_planes_horizontally,
    gate_perturbation_names,
    perturb_region,
    region_density_enrichment,
    teacher_role,
)
from fracture_detection.baseline0.analysis.cam_audit_report import (
    HFLIP_TTA,
    NO_TTA,
    audit_verdict,
    laterality_summary,
    localization_table,
    memorization_table,
    perturbation_table,
    select_role_frame,
    tta_table,
)
from fracture_detection.common.constants import (
    EXPECTED_MASK_SHAPE,
    N_REGIONS,
    REGION_COLUMNS,
)

PLANES, HEIGHT, WIDTH = EXPECTED_MASK_SHAPE


def _square_masks() -> tuple[np.ndarray, np.ndarray]:
    """Whole mask covering the centre, split into four horizontal region bands."""
    whole = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole[:, 48:176, 48:176] = 1
    region = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    for index in range(N_REGIONS):
        top = 48 + index * 32
        region[:, top : top + 32, 48:176] = index + 1
    return whole, region


def test_teacher_role_matches_the_frozen_nested_protocol() -> None:
    for teacher in range(5):
        assert teacher_role(teacher, teacher) == "outer"
        assert teacher_role((teacher + 1) % 5, teacher) == "inner"
        train_folds = [
            fold for fold in range(5) if fold not in {teacher, (teacher + 1) % 5}
        ]
        assert len(train_folds) == 3
        for fold in train_folds:
            assert teacher_role(fold, teacher) == "train"


def test_teacher_role_rejects_out_of_range_folds() -> None:
    with pytest.raises(ValueError):
        teacher_role(5, 0)
    with pytest.raises(ValueError):
        teacher_role(0, 0, n_folds=4)


def test_each_bag_is_in_sample_for_three_teachers_and_held_out_once() -> None:
    for fold in range(5):
        roles = [teacher_role(fold, teacher) for teacher in range(5)]
        assert roles.count("train") == 3
        assert roles.count("inner") == 1
        assert roles.count("outer") == 1


def test_identity_perturbation_returns_the_region_unchanged() -> None:
    whole, region_mask = _square_masks()
    region = (region_mask == 1) & (whole > 0)
    result = perturb_region(region, whole > 0, MaskPerturbation("identity", "identity"))
    assert np.array_equal(result, region)


def test_erosion_shrinks_and_dilation_grows_inside_the_vertebra() -> None:
    whole, region_mask = _square_masks()
    whole_bool = whole > 0
    region = (region_mask == 2) & whole_bool
    eroded = perturb_region(
        region, whole_bool, MaskPerturbation("erode_4", "erode", amount_pixels=4)
    )
    dilated = perturb_region(
        region, whole_bool, MaskPerturbation("dilate_4", "dilate", amount_pixels=4)
    )
    assert eroded.sum() < region.sum() < dilated.sum()
    assert np.all(dilated <= whole_bool)


def test_shift_moves_the_region_without_wrapping_around() -> None:
    whole = np.ones(EXPECTED_MASK_SHAPE, dtype=bool)
    region = np.zeros(EXPECTED_MASK_SHAPE, dtype=bool)
    region[:, 10:20, 10:20] = True
    shifted = perturb_region(region, whole, MaskPerturbation("s", "shift", dx=5))
    assert shifted[:, 10:20, 15:25].all()
    assert not shifted[:, 10:20, 10:15].any()
    assert shifted.sum() == region.sum()

    to_edge = perturb_region(region, whole, MaskPerturbation("s", "shift", dx=-15))
    assert to_edge.sum() < region.sum()


def test_density_enrichment_recovers_a_known_ratio() -> None:
    whole, region_mask = _square_masks()
    cams = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.float32)
    # All CAM mass lands inside region 1, which is a quarter of the vertebra.
    cams[region_mask == 1] = 1.0
    scores = region_density_enrichment(
        cams, whole, region_mask, MaskPerturbation("identity", "identity")
    )
    assert scores[0] == pytest.approx(4.0, rel=1e-5)
    assert scores[1:] == pytest.approx(np.zeros(3), abs=1e-6)


def test_density_enrichment_is_nan_when_the_cam_is_empty() -> None:
    whole, region_mask = _square_masks()
    cams = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.float32)
    scores = region_density_enrichment(
        cams, whole, region_mask, MaskPerturbation("identity", "identity")
    )
    assert np.isnan(scores).all()


def test_density_enrichment_is_nan_for_a_region_erased_by_erosion() -> None:
    whole = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole[:, 40:80, 40:80] = 1
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask[:, 40:80, 40:80] = 1
    region_mask[:, 50:53, 50:53] = 2
    cams = np.ones(EXPECTED_MASK_SHAPE, dtype=np.float32)
    scores = region_density_enrichment(
        cams,
        whole,
        region_mask,
        MaskPerturbation("erode_8", "erode", amount_pixels=8),
    )
    assert np.isnan(scores[1])
    assert np.isfinite(scores[0])


def test_density_enrichment_rejects_negative_cams() -> None:
    whole, region_mask = _square_masks()
    cams = np.full(EXPECTED_MASK_SHAPE, -1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        region_density_enrichment(
            cams, whole, region_mask, MaskPerturbation("identity", "identity")
        )


def test_horizontal_flip_is_an_involution() -> None:
    values = np.arange(PLANES * HEIGHT * WIDTH, dtype=np.float32).reshape(
        EXPECTED_MASK_SHAPE
    )
    assert np.array_equal(
        flip_planes_horizontally(flip_planes_horizontally(values)), values
    )


def test_gate_perturbations_are_the_plausible_error_magnitude() -> None:
    names = gate_perturbation_names()
    grid = {item.name: item for item in default_perturbations()}
    assert names
    for name in names:
        item = grid[name]
        magnitude = max(abs(item.amount_pixels), abs(item.dy), abs(item.dx))
        assert magnitude == GATE_PERTURBATION_PIXELS
    assert "identity" not in names
    assert "erode_8" not in names


def test_perturbation_grid_rejects_degenerate_definitions() -> None:
    with pytest.raises(ValueError):
        MaskPerturbation("bad", "erode", amount_pixels=0)
    with pytest.raises(ValueError):
        MaskPerturbation("bad", "shift")
    with pytest.raises(ValueError):
        MaskPerturbation("bad", "identity", amount_pixels=2)


def _synthetic_scores(
    *,
    train_shift: float = 0.0,
    n_bags: int = 40,
) -> pd.DataFrame:
    """Build an audit score table with a controllable in-sample advantage."""
    rng = np.random.default_rng(7)
    noise = 1.4
    rows: list[dict[str, object]] = []
    perturbations = [item.name for item in default_perturbations()]
    for bag in range(n_bags):
        fold = bag % 5
        labels = {
            column: int(bag % (index + 2) == 0)
            for index, column in enumerate(REGION_COLUMNS)
        }
        base = {
            column: float(labels[column]) + rng.normal(0.0, noise)
            for column in REGION_COLUMNS
        }
        for teacher in range(5):
            role = teacher_role(fold, teacher)
            for tta in (NO_TTA, HFLIP_TTA):
                for perturbation in perturbations:
                    row: dict[str, object] = {
                        "study_id": f"S{bag:03d}",
                        "level": "C4",
                        "fold": fold,
                        "teacher": teacher,
                        "role": role,
                        "tta": tta,
                        "perturbation": perturbation,
                        "bag_probability": 0.5,
                        "cam_total": 1.0,
                    }
                    for column in REGION_COLUMNS:
                        score = base[column] + 3.0
                        if role == "train":
                            score += train_shift * (2.0 * labels[column] - 1.0)
                        row[f"{column}_score"] = score
                        row[column] = labels[column]
                        row[f"{column}_target_valid"] = True
                    rows.append(row)
    return pd.DataFrame(rows)


def test_select_role_frame_returns_one_row_per_bag_from_the_lowest_teacher() -> None:
    scores = _synthetic_scores()
    frame = select_role_frame(scores, "train")
    assert len(frame) == scores["study_id"].nunique()
    assert not frame.duplicated(["study_id", "level"]).any()
    for _, row in frame.iterrows():
        expected = min(
            teacher
            for teacher in range(5)
            if teacher_role(int(row["fold"]), teacher) == "train"
        )
        assert int(row["teacher"]) == expected


def test_memorization_gate_passes_when_in_sample_matches_held_out() -> None:
    table = memorization_table(_synthetic_scores(train_shift=0.0), n_bootstrap=50)
    assert len(table) == N_REGIONS
    assert table["auroc_difference"].abs().max() < 1e-9
    assert not table["auroc_gate_failed"].any()
    assert not table["smd_gate_failed"].any()


def test_memorization_gate_fires_on_a_large_in_sample_advantage() -> None:
    table = memorization_table(_synthetic_scores(train_shift=2.5), n_bootstrap=50)
    assert (table["auroc_difference"] > 0).all()
    assert int(table["auroc_gate_failed"].sum()) >= 2
    verdict = audit_verdict(table, perturbation_table(_synthetic_scores()))
    assert verdict["memorization_gate_failed"]
    assert not verdict["proceed_to_pseudo_label_generation"]


def test_perturbation_and_tta_tables_report_every_region() -> None:
    scores = _synthetic_scores()
    perturbation = perturbation_table(scores)
    variants = set(perturbation["variant"])
    assert "identity" not in variants
    assert len(perturbation) == len(variants) * N_REGIONS
    assert not perturbation["spearman_gate_failed"].any()
    flip = tta_table(scores)
    assert len(flip) == N_REGIONS
    assert flip["spearman_vs_identity"].min() == pytest.approx(1.0)


def test_verdict_passes_when_both_gates_hold() -> None:
    scores = _synthetic_scores()
    verdict = audit_verdict(
        memorization_table(scores, n_bootstrap=50), perturbation_table(scores)
    )
    assert verdict["proceed_to_pseudo_label_generation"]


def test_localization_table_reports_every_region_with_an_interval() -> None:
    table = localization_table(_synthetic_scores(), n_bootstrap=50)
    assert len(table) == N_REGIONS
    assert table["auroc"].notna().all()
    assert (table["auroc_ci_low"] <= table["auroc"]).all()
    assert (table["auroc"] <= table["auroc_ci_high"]).all()
    assert (table["n_undefined"] == 0).all()


def test_laterality_summary_scores_the_side_that_carries_the_label() -> None:
    scores = _synthetic_scores()
    identity = (
        scores["perturbation"].eq("identity")
        & scores["tta"].eq(NO_TTA)
        & scores["role"].eq("outer")
    )
    exclusive = scores["region_2"].ne(scores["region_3"])
    # Make the labelled side win everywhere, so the rate must be exactly one.
    right_wins = identity & exclusive & scores["region_2"].eq(1)
    left_wins = identity & exclusive & scores["region_3"].eq(1)
    scores.loc[right_wins, "region_2_score"] = 9.0
    scores.loc[right_wins, "region_3_score"] = 1.0
    scores.loc[left_wins, "region_3_score"] = 9.0
    scores.loc[left_wins, "region_2_score"] = 1.0
    summary = laterality_summary(scores, n_bootstrap=50)
    assert summary["n_bags"] > 0
    assert summary["win_rate"] == pytest.approx(1.0)

    flipped = scores.copy()
    flipped.loc[right_wins, ["region_2_score", "region_3_score"]] = [1.0, 9.0]
    flipped.loc[left_wins, ["region_2_score", "region_3_score"]] = [9.0, 1.0]
    assert laterality_summary(flipped, n_bootstrap=50)["win_rate"] == pytest.approx(0.0)


def test_laterality_summary_skips_bags_with_an_undefined_side() -> None:
    scores = _synthetic_scores()
    full = laterality_summary(scores, n_bootstrap=10)
    scores.loc[scores["region_2_score"].notna(), "region_2_score"] = np.nan
    empty = laterality_summary(scores, n_bootstrap=10)
    assert full["n_bags"] > 0
    assert empty["n_bags"] == 0
