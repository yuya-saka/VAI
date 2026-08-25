from __future__ import annotations

import numpy as np
import pandas as pd
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
from fracture_detection.baseline0.pseudo_labeling.gradcam import (
    anatomical_attention_metrics,
    compute_gradcam,
    prepare_inputs,
    select_stratified_high_scores,
)


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
