"""重複窓集約とendpoints再構成のテスト。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from line_surface_3d.src.inference import (
    OnlineRibbonAggregate,
    predict_loader,
)
from line_surface_3d.utils.detection import line_from_ribbon
from line_surface_3d.utils.region_eval import evaluate_prediction_tree
from PIL import Image


def test_identical_windows_have_zero_disagreement() -> None:
    """一致する重複窓の分散は0になる。"""
    aggregate = OnlineRibbonAggregate()
    for _ in range(3):
        aggregate.add(2.0, -1.0, 1.0, 0.0, 40.0, 0.8)
    result = aggregate.finalize()
    assert result["centroid_x"] == 2.0
    assert result["centroid_y"] == -1.0
    assert result["centroid_disagreement_px"] == 0.0
    assert result["angle_disagreement_deg"] == 0.0
    assert result["overlap_count"] == 3.0


def test_disagreement_increases_for_conflicting_windows() -> None:
    """位置・角度の不一致が指標へ反映される。"""
    aggregate = OnlineRibbonAggregate()
    aggregate.add(0.0, 0.0, 1.0, 0.0, 40.0, 0.8)
    aggregate.add(4.0, 0.0, 0.0, 1.0, 40.0, 0.8)
    result = aggregate.finalize()
    assert math.isclose(result["centroid_disagreement_px"], 2.0)
    assert result["angle_disagreement_deg"] > 0.0


def test_line_from_ribbon_reconstructs_horizontal_line() -> None:
    """垂直法線から水平線を再構成する。"""
    line = line_from_ribbon(
        centroid_x=0.0,
        centroid_y=0.0,
        doubled_cosine=-1.0,
        doubled_sine=0.0,
        length=40.0,
        image_size=64,
    )
    first, second = line["endpoints"]
    assert math.isclose(first[1], 32.0, abs_tol=1e-5)
    assert math.isclose(second[1], 32.0, abs_tol=1e-5)
    assert math.isclose(abs(second[0] - first[0]), 40.0, abs_tol=1e-5)


class _ZeroModel(nn.Module):
    """一定logitを返す推論統合テスト用モデル。"""

    def forward(
        self,
        inputs: torch.Tensor,
        vertebra_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del vertebra_indices
        return torch.zeros(
            inputs.shape[0],
            12,
            inputs.shape[-2],
            inputs.shape[-1],
            device=inputs.device,
        )


def test_predict_loader_writes_full_slice_tree(tmp_path: Path) -> None:
    """1窓から全z・4線のJSONを生成する。"""
    batch = {
        "image": torch.zeros(1, 6, 16, 16),
        "heatmaps": torch.zeros(1, 3, 4, 16, 16),
        "label_mask": torch.zeros(1, 3, 4, dtype=torch.bool),
        "slice_indices": torch.tensor([[5, 6, 7]]),
        "sample": ["sample1"],
        "vertebra": ["C1"],
    }
    summary = predict_loader(
        _ZeroModel(),
        [batch],
        torch.device("cpu"),
        {
            "data": {"slab_size": 3, "image_size": 16, "sigma": 1.5},
            "evaluation": {"line_extend_ratio": 1.0},
        },
        tmp_path,
    )
    lines = json.loads(
        (tmp_path / "sample1" / "C1" / "lines.json").read_text(encoding="utf-8")
    )
    assert summary["window_count"] == 1
    assert sorted(lines) == ["5", "6", "7"]
    assert all(len(slice_lines) == 4 for slice_lines in lines.values())


def _region_lines() -> dict[str, list[list[float]]]:
    """4領域を形成できる合流線を返す。"""
    return {
        "line_1": [[11.0, 8.0], [14.0, 3.0]],
        "line_2": [[11.0, 8.0], [15.0, 10.0]],
        "line_3": [[5.0, 8.0], [2.0, 3.0]],
        "line_4": [[5.0, 8.0], [1.0, 10.0]],
    }


def test_region_evaluation_counts_formed_regions(tmp_path: Path) -> None:
    """予測treeから領域欠損率とreformatを生成する。"""
    prediction_root = tmp_path / "predictions"
    dense_root = tmp_path / "dense"
    annotation_root = tmp_path / "annotation"
    prediction_dir = prediction_root / "sample1" / "C1"
    dense_dir = dense_root / "sample1" / "C1"
    annotation_dir = annotation_root / "sample1" / "C1"
    prediction_dir.mkdir(parents=True)
    (dense_dir / "images").mkdir(parents=True)
    (dense_dir / "masks").mkdir()
    annotation_dir.mkdir(parents=True)
    lines = {str(index): _region_lines() for index in range(3)}
    surface = {
        str(index): {
            f"line_{line_index}": {
                "centroid_math": [0.0, 0.0],
                "normal_angle_deg": 0.0,
            }
            for line_index in range(1, 5)
        }
        for index in range(3)
    }
    (prediction_dir / "lines.json").write_text(
        json.dumps(lines),
        encoding="utf-8",
    )
    (prediction_dir / "surface.json").write_text(
        json.dumps(surface),
        encoding="utf-8",
    )
    (annotation_dir / "lines.json").write_text(
        json.dumps({"1": _region_lines()}),
        encoding="utf-8",
    )
    image = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1:15, 1:15] = 255
    for slice_index in range(3):
        Image.fromarray(image).save(
            dense_dir / "images" / f"slice_{slice_index:03d}.png"
        )
        Image.fromarray(mask).save(dense_dir / "masks" / f"slice_{slice_index:03d}.png")
    summary = evaluate_prediction_tree(
        prediction_root,
        dense_root,
        annotation_root,
        tmp_path / "evaluation",
        image_size=16,
        spacing_mm=0.4,
        bin_width_mm=3.2,
    )
    assert summary["scopes"]["all"]["slice_count"] == 3
    assert summary["scopes"]["all"]["any_missing_count"] == 0
    assert (tmp_path / "evaluation" / "reformats" / "sample1_C1.png").exists()
