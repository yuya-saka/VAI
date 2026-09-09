from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from fracture_detection.baseline0.cli.attention import (
    summarize_annotated_localization,
    summarize_annotated_targets,
)
from fracture_detection.baseline0.data.constants import (
    EXPECTED_CT_SHAPE,
    EXPECTED_MASK_SHAPE,
)
from fracture_detection.baseline0.modeling.model import Baseline0Model
from fracture_detection.baseline0.pseudo_labeling.cam_audit import (
    MaskPerturbation,
    region_density_enrichment,
)
from fracture_detection.baseline0.pseudo_labeling.gradcam import (
    DEFAULT_TTA_VIEWS,
    TTAView,
    _rotation_matrix,  # testing the private warp helper directly
    _warp_planes,
    anatomical_attention_metrics,
    apply_tta_view_to_inputs,
    compute_gradcam,
    invert_tta_view_on_cam,
    prepare_inputs,
    select_stratified_high_scores,
)

IDENTITY = MaskPerturbation("identity", "identity")


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(6, 4, kernel_size=1, bias=False)
        self.bn2 = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.bn2(self.conv(inputs))
        return features.mean(dim=(-2, -1))


class TinyBaseline0(Baseline0Model):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.n_planes = 15
        self.encoder = TinyEncoder()
        self.head = nn.Linear(4, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, plane_count, channels, height, width = inputs.shape
        features = self.encoder(
            inputs.reshape(batch_size * plane_count, channels, height, width)
        )
        return self.head(features).reshape(batch_size, plane_count)


def test_prepare_inputs_adds_binary_mask_channel() -> None:
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole_mask[:, 10:20, 30:40] = 1

    inputs = prepare_inputs(ct, whole_mask)

    assert inputs.shape == (15, 6, 224, 224)
    assert torch.equal(inputs[:, 5], torch.from_numpy(whole_mask).float())


def test_compute_gradcam_supports_independent_batched_bags() -> None:
    torch.manual_seed(7)
    model = TinyBaseline0().eval()
    batched_inputs = torch.rand(2, 15, 6, 8, 8)

    batched = compute_gradcam(model, batched_inputs, torch.device("cpu"))
    first = compute_gradcam(model, batched_inputs[:1], torch.device("cpu"))

    assert batched.cams.shape == (2, 15, 8, 8)
    assert batched.plane_probabilities.shape == (2, 15)
    assert batched.bag_probabilities.shape == (2,)
    np.testing.assert_allclose(batched.cams[0], first.cams[0], atol=1e-6)
    np.testing.assert_allclose(
        batched.bag_probabilities[0], first.bag_probabilities[0], atol=1e-6
    )


def test_anatomical_attention_metrics_separates_area_and_density() -> None:
    cams = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.float32)
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole_mask[:, :2, :4] = 1
    region_mask[:, :2, :2] = 1
    region_mask[:, :2, 2:4] = 2
    cams[:, :2, :2] = 2.0
    cams[:, :2, 2:4] = 1.0

    metrics = anatomical_attention_metrics(cams, whole_mask, region_mask)

    assert metrics["in_vertebra_mass_fraction"] == 1.0
    assert metrics["vertebra_density_enrichment"] > 1.0
    assert metrics["region_1_area_fraction"] == 0.5
    assert metrics["region_1_mass_fraction"] == 2.0 / 3.0
    assert metrics["region_1_density_enrichment"] == 4.0 / 3.0
    assert metrics["region_2_density_enrichment"] == 2.0 / 3.0


def test_select_stratified_high_scores_keeps_each_fold_level_category() -> None:
    predictions = pd.DataFrame(
        [
            {
                "study_id": "a",
                "level": "C1",
                "fold": 0,
                "category": "TP",
                "vertebra_score": 0.8,
            },
            {
                "study_id": "b",
                "level": "C1",
                "fold": 0,
                "category": "TP",
                "vertebra_score": 0.9,
            },
            {
                "study_id": "c",
                "level": "C2",
                "fold": 0,
                "category": "TP",
                "vertebra_score": 0.7,
            },
            {
                "study_id": "d",
                "level": "C1",
                "fold": 1,
                "category": "FP",
                "vertebra_score": 0.6,
            },
        ]
    )

    selected = select_stratified_high_scores(predictions, ("TP", "FP"), 1)

    assert selected["study_id"].tolist() == ["b", "c", "d"]


def test_summarize_annotated_targets_compares_positive_and_negative_bags() -> None:
    metrics = pd.DataFrame(
        {
            "has_region_target": [True, True],
            "annotation_complete": [True, True],
            "region_1": [1, 0],
            "region_2": [0, 1],
            "region_3": [0, 0],
            "region_4": [1, 1],
            "region_1_mass_fraction": [0.6, 0.2],
            "region_2_mass_fraction": [0.1, 0.3],
            "region_3_mass_fraction": [0.1, 0.1],
            "region_4_mass_fraction": [0.2, 0.4],
            "region_1_density_enrichment": [1.5, 0.5],
            "region_2_density_enrichment": [0.5, 1.5],
            "region_3_density_enrichment": [1.0, 1.0],
            "region_4_density_enrichment": [0.8, 1.2],
        }
    )

    summary = summarize_annotated_targets(metrics).set_index("target_region")

    assert summary.loc["region_1", "n_positive"] == 1
    assert summary.loc["region_1", "n_negative"] == 1
    assert summary.loc["region_1", "n_unknown"] == 0
    assert summary.loc["region_1", "density_enrichment_mean_difference"] == 1.0


def test_summarize_annotated_localization_finds_density_signal() -> None:
    metrics = pd.DataFrame(
        {
            "study_id": ["a", "b", "c", "d"],
            "level": ["C1", "C1", "C2", "C2"],
            "has_region_target": [True] * 4,
            "annotation_complete": [True] * 4,
            "region_1": [0, 1, 0, 1],
            "region_2": [1, 0, 1, 0],
            "region_3": [0, 1, 0, 1],
            "region_4": [1, 0, 1, 0],
            "region_1_density_enrichment": [0.5, 1.5, 0.7, 1.7],
            "region_2_density_enrichment": [1.5, 0.5, 1.7, 0.7],
            "region_3_density_enrichment": [0.5, 1.5, 0.7, 1.7],
            "region_4_density_enrichment": [1.5, 0.5, 1.7, 0.7],
        }
    )

    summary = summarize_annotated_localization(
        metrics, bootstrap_samples=20, seed=7
    ).set_index("region")

    assert summary.loc["region_1", "density_auroc"] == 1.0
    assert summary.loc["region_1", "within_level_rank_auroc"] == 1.0
    assert summary.loc["region_1", "density_difference_ci_low"] > 0


def test_region_summaries_exclude_unreviewed_zero_but_keep_positive() -> None:
    metrics = pd.DataFrame(
        {
            "study_id": ["complete", "unknown", "positive"],
            "level": ["C1", "C1", "C1"],
            "has_region_target": [True, True, True],
            "annotation_complete": [True, False, False],
            "region_1": [0, 0, 1],
            "region_2": [1, 1, 1],
            "region_3": [1, 1, 1],
            "region_4": [1, 1, 1],
            "region_1_mass_fraction": [0.1, 0.9, 0.8],
            "region_2_mass_fraction": [0.1, 0.1, 0.1],
            "region_3_mass_fraction": [0.1, 0.1, 0.1],
            "region_4_mass_fraction": [0.1, 0.1, 0.1],
            "region_1_density_enrichment": [0.1, 0.9, 0.8],
            "region_2_density_enrichment": [1.0, 1.0, 1.0],
            "region_3_density_enrichment": [1.0, 1.0, 1.0],
            "region_4_density_enrichment": [1.0, 1.0, 1.0],
        }
    )

    target_summary = summarize_annotated_targets(metrics).set_index("target_region")
    localization = summarize_annotated_localization(
        metrics, bootstrap_samples=20, seed=7
    ).set_index("region")

    assert target_summary.loc["region_1", "n_positive"] == 1
    assert target_summary.loc["region_1", "n_negative"] == 1
    assert target_summary.loc["region_1", "n_unknown"] == 1
    assert localization.loc["region_1", "density_auroc"] == 1.0


def test_default_tta_views_are_the_frozen_four_view_set() -> None:
    assert [view.name for view in DEFAULT_TTA_VIEWS] == [
        "identity",
        "horizontal_flip",
        "rotation_plus10",
        "rotation_minus10",
    ]
    assert [view.kind for view in DEFAULT_TTA_VIEWS] == [
        "identity",
        "horizontal_flip",
        "rotation",
        "rotation",
    ]


def test_tta_view_rejects_a_rotation_without_an_angle() -> None:
    with pytest.raises(ValueError, match="non-zero angle"):
        TTAView("bad", "rotation")


def test_tta_view_rejects_an_angle_on_a_non_rotation_view() -> None:
    with pytest.raises(ValueError, match="must not carry an angle"):
        TTAView("bad", "identity", degrees=5.0)


def test_apply_tta_view_to_inputs_identity_returns_the_same_arrays() -> None:
    ct = np.arange(np.prod(EXPECTED_CT_SHAPE), dtype=np.uint8).reshape(
        EXPECTED_CT_SHAPE
    )
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    view = TTAView("identity", "identity")

    warped_ct, warped_mask = apply_tta_view_to_inputs(ct, whole_mask, view)

    assert warped_ct is ct
    assert warped_mask is whole_mask


def test_apply_tta_view_to_inputs_preserves_shape_and_dtype_for_every_view() -> None:
    ct = np.zeros(EXPECTED_CT_SHAPE, dtype=np.uint8)
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole_mask[:, 50:150, 50:150] = 1

    for view in DEFAULT_TTA_VIEWS:
        warped_ct, warped_mask = apply_tta_view_to_inputs(ct, whole_mask, view)
        assert warped_ct.shape == EXPECTED_CT_SHAPE
        assert warped_ct.dtype == np.uint8
        assert warped_mask.shape == EXPECTED_MASK_SHAPE
        assert warped_mask.dtype == np.uint8


def test_horizontal_flip_view_round_trips_exactly_on_a_synthetic_cam() -> None:
    rng = np.random.default_rng(0)
    native_cam = rng.uniform(0.0, 1.0, size=EXPECTED_MASK_SHAPE).astype(np.float32)
    view = TTAView("horizontal_flip", "horizontal_flip")

    transformed = native_cam[..., ::-1]
    recovered = invert_tta_view_on_cam(transformed, view)

    np.testing.assert_allclose(recovered, native_cam)


def _blob_masks() -> tuple[np.ndarray, np.ndarray]:
    """A whole mask split into four large, spatially separated region blocks."""
    whole_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    region_mask = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.uint8)
    whole_mask[:, 40:184, 40:184] = 1
    region_mask[:, 40:112, 40:112] = 1
    region_mask[:, 40:112, 112:184] = 2
    region_mask[:, 112:184, 40:112] = 3
    region_mask[:, 112:184, 112:184] = 4
    return whole_mask, region_mask


def test_rotation_view_round_trip_recovers_region_enrichment_on_a_smooth_cam() -> None:
    whole_mask, region_mask = _blob_masks()
    native_cam = np.zeros(EXPECTED_MASK_SHAPE, dtype=np.float32)
    native_cam[:, 40:112, 40:112] = 3.0
    native_cam[:, 40:112, 112:184] = 1.0
    native_cam[:, 112:184, 40:112] = 0.5
    native_cam[:, 112:184, 112:184] = 0.2

    expected = region_density_enrichment(native_cam, whole_mask, region_mask, IDENTITY)

    for view in (
        TTAView("rotation_plus10", "rotation", degrees=10.0),
        TTAView("rotation_minus10", "rotation", degrees=-10.0),
    ):
        forward_matrix = _rotation_matrix(view.degrees, native_cam.shape[-1])
        transformed = _warp_planes(native_cam, forward_matrix, nearest=False)
        recovered_cam = invert_tta_view_on_cam(transformed, view)

        assert np.isfinite(recovered_cam).all()
        assert (recovered_cam >= 0).all()

        recovered = region_density_enrichment(
            recovered_cam, whole_mask, region_mask, IDENTITY
        )
        np.testing.assert_allclose(recovered, expected, rtol=0.1, atol=0.05)
