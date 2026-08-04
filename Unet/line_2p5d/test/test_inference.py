"""学習後の画像・予測JSON保存テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Unet.line_2p5d.src.inference import predict_lines_and_save, save_examples


class FixedHeatmapModel(nn.Module):
    """固定heatmapを全5スライスへ返すテストモデル。"""

    def __init__(self, image_size: int = 16) -> None:
        super().__init__()
        y_grid, _ = np.indices((image_size, image_size), dtype=np.float32)
        heatmaps = np.stack(
            [
                np.exp(-0.5 * ((y_grid - y_value) / 1.0) ** 2)
                for y_value in (3, 6, 9, 12)
            ]
        )
        probabilities = torch.from_numpy(heatmaps).clamp(1e-4, 1 - 1e-4)
        self.register_buffer("logits", torch.logit(probabilities))

    def forward(
        self,
        images: torch.Tensor,
        vertebra_index: torch.Tensor,
    ) -> torch.Tensor:
        """batch・スライス方向へ固定logitを展開する。"""
        del vertebra_index
        batch_size, slice_count = images.shape[:2]
        return self.logits.view(1, 1, 4, 16, 16).expand(
            batch_size,
            slice_count,
            -1,
            -1,
            -1,
        )


def _batch() -> dict[str, object]:
    """可視化用の単一batchを返す。"""
    lines = {
        f"line_{index + 1}": [[2.0, float(y_value)], [13.0, float(y_value)]]
        for index, y_value in enumerate((3, 6, 9, 12))
    }
    return {
        "image": torch.full((1, 5, 2, 16, 16), 0.5),
        "heatmaps": torch.zeros((1, 4, 16, 16)),
        "line_params_gt": torch.tensor(
            [[[0.0, -5.0], [0.0, -2.0], [0.0, 1.0], [0.0, 4.0]]]
        ),
        "sample": ["sample1"],
        "vertebra": ["C1"],
        "slice_idx": torch.tensor([3]),
        "lines_json": [json.dumps(lines)],
    }


def test_inference_saves_line_only_style_outputs(tmp_path: Path) -> None:
    """比較画像・3列画像・予測JSONをテスト全件へ保存する。"""
    model = FixedHeatmapModel()
    output_dir = tmp_path / "test_lines"
    config = {
        "data": {"image_size": 16},
        "evaluation": {
            "heatmap_threshold": {"mode": "adaptive", "min": 0.1, "peak_ratio": 0.4},
            "min_line_confidence": 0.05,
            "line_extend_ratio": 1.25,
        },
    }

    summary = predict_lines_and_save(
        config,
        model,
        [_batch()],
        torch.device("cpu"),
        output_dir,
    )

    prefix = "sample1_C1_slice003"
    assert summary["n_samples"] == 1
    assert (output_dir / f"{prefix}_comparison.png").exists()
    assert (output_dir / f"{prefix}_heatmap_lines.png").exists()
    prediction_path = output_dir / f"{prefix}_PRED_lines.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["line_extend_ratio"] == 1.25
    assert prediction["pred_lines"]["line_1"]["length"] == 13.75


def test_save_examples_writes_four_images_per_sample(tmp_path: Path) -> None:
    """GT・予測のgridとoverlayを各1枚保存する。"""
    save_examples(
        FixedHeatmapModel(),
        [_batch()],
        torch.device("cpu"),
        tmp_path,
        n_save=1,
        tag="TEST",
    )
    assert len(list(tmp_path.glob("*.png"))) == 4
