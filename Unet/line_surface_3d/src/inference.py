"""全高sliding-window推論と重複窓集約。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.detection import estimate_line_length, line_from_ribbon
from ..utils.ribbon import compute_heatmap_moments, fit_ribbon
from .evaluation import vertebra_indices
from .experiment import save_json
from .model import reshape_slab_heatmaps


@dataclass
class OnlineRibbonAggregate:
    """同一global z・同一線への重複窓予測を逐次集約する。"""

    count: int = 0
    centroid_x_sum: float = 0.0
    centroid_y_sum: float = 0.0
    centroid_square_sum: float = 0.0
    cosine_sum: float = 0.0
    sine_sum: float = 0.0
    length_sum: float = 0.0
    confidence_sum: float = 0.0

    def add(
        self,
        centroid_x: float,
        centroid_y: float,
        cosine: float,
        sine: float,
        length: float,
        confidence: float,
    ) -> None:
        """1窓分の予測を追加する。"""
        self.count += 1
        self.centroid_x_sum += centroid_x
        self.centroid_y_sum += centroid_y
        self.centroid_square_sum += centroid_x**2 + centroid_y**2
        self.cosine_sum += cosine
        self.sine_sum += sine
        self.length_sum += length
        self.confidence_sum += confidence

    def finalize(self) -> dict[str, float]:
        """平均面と重複不一致を返す。"""
        if self.count <= 0:
            raise ValueError("空のaggregateは確定できません")
        centroid_x = self.centroid_x_sum / self.count
        centroid_y = self.centroid_y_sum / self.count
        mean_cosine = self.cosine_sum / self.count
        mean_sine = self.sine_sum / self.count
        resultant = math.hypot(mean_cosine, mean_sine)
        if resultant <= 1e-8:
            normalized_cosine, normalized_sine = 1.0, 0.0
        else:
            normalized_cosine = mean_cosine / resultant
            normalized_sine = mean_sine / resultant
        centroid_variance = max(
            0.0,
            self.centroid_square_sum / self.count - centroid_x**2 - centroid_y**2,
        )
        angle_disagreement = math.degrees(
            0.5 * math.acos(min(1.0, max(0.0, resultant)))
        )
        return {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "doubled_cosine": normalized_cosine,
            "doubled_sine": normalized_sine,
            "length": self.length_sum / self.count,
            "confidence": self.confidence_sum / self.count,
            "overlap_count": float(self.count),
            "centroid_disagreement_px": math.sqrt(centroid_variance),
            "angle_disagreement_deg": angle_disagreement,
            "orientation_resultant": min(1.0, resultant),
        }


def aggregate_prediction(
    aggregates: dict[tuple[str, str, int, int], OnlineRibbonAggregate],
    sample: str,
    vertebra: str,
    slice_index: int,
    line_index: int,
    values: dict[str, float],
) -> None:
    """global keyへ1つの窓予測を追加する。"""
    key = (sample, vertebra, slice_index, line_index)
    aggregate = aggregates.setdefault(key, OnlineRibbonAggregate())
    aggregate.add(
        centroid_x=values["centroid_x"],
        centroid_y=values["centroid_y"],
        cosine=values["doubled_cosine"],
        sine=values["doubled_sine"],
        length=values["length"],
        confidence=values["confidence"],
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """全推論窓を集約し、椎体ごとの全高4線JSONを保存する。"""
    model.eval()
    slab_size = int(config["data"]["slab_size"])
    image_size = int(config["data"]["image_size"])
    sigma = float(config["data"]["sigma"])
    extend_ratio = float(config.get("evaluation", {}).get("line_extend_ratio", 1.0))
    aggregates: dict[
        tuple[str, str, int, int],
        OnlineRibbonAggregate,
    ] = {}
    window_count = 0
    for batch in loader:
        images = batch["image"].to(device).float()
        logits = model(images, vertebra_indices(batch, device))
        heatmaps = torch.sigmoid(reshape_slab_heatmaps(logits, slab_size))
        ribbon_fit = fit_ribbon(compute_heatmap_moments(heatmaps))
        fitted_values = ribbon_fit.fitted_values.cpu().numpy()
        major_variances = ribbon_fit.major_variance.cpu().numpy()
        confidences = ribbon_fit.confidence.cpu().numpy()
        slice_indices = batch["slice_indices"].cpu().numpy()
        batch_size = heatmaps.shape[0]
        for batch_index in range(batch_size):
            sample = str(batch["sample"][batch_index])
            vertebra = str(batch["vertebra"][batch_index])
            for slab_index in range(slab_size):
                slice_index = int(slice_indices[batch_index, slab_index])
                for line_index in range(4):
                    fitted = fitted_values[batch_index, slab_index, line_index]
                    major_variance = float(
                        major_variances[batch_index, slab_index, line_index]
                    )
                    aggregate_prediction(
                        aggregates,
                        sample,
                        vertebra,
                        slice_index,
                        line_index,
                        {
                            "centroid_x": float(fitted[0]),
                            "centroid_y": float(fitted[1]),
                            "doubled_cosine": float(fitted[2]),
                            "doubled_sine": float(fitted[3]),
                            "length": estimate_line_length(
                                major_variance,
                                sigma,
                                image_size,
                            ),
                            "confidence": float(
                                confidences[batch_index, slab_index, line_index]
                            ),
                        },
                    )
            window_count += 1

    grouped: dict[tuple[str, str], dict[int, dict[int, dict[str, float]]]] = {}
    for (sample, vertebra, slice_index, line_index), aggregate in aggregates.items():
        grouped.setdefault((sample, vertebra), {}).setdefault(slice_index, {})[
            line_index
        ] = aggregate.finalize()

    vertebra_summaries: list[dict[str, Any]] = []
    for (sample, vertebra), slices in sorted(grouped.items()):
        lines_output: dict[str, dict[str, list[list[float]]]] = {}
        surface_output: dict[str, dict[str, Any]] = {}
        disagreements: list[float] = []
        for slice_index, line_values in sorted(slices.items()):
            lines_output[str(slice_index)] = {}
            surface_output[str(slice_index)] = {}
            for line_index in range(4):
                values = line_values[line_index]
                line_name = f"line_{line_index + 1}"
                line = line_from_ribbon(
                    centroid_x=values["centroid_x"],
                    centroid_y=values["centroid_y"],
                    doubled_cosine=values["doubled_cosine"],
                    doubled_sine=values["doubled_sine"],
                    length=values["length"],
                    image_size=image_size,
                    extend_ratio=extend_ratio,
                )
                lines_output[str(slice_index)][line_name] = line["endpoints"]
                surface_output[str(slice_index)][line_name] = {
                    **values,
                    **line,
                }
                disagreements.append(values["centroid_disagreement_px"])
        vertebra_dir = output_root / sample / vertebra
        save_json(vertebra_dir / "lines.json", lines_output)
        save_json(vertebra_dir / "surface.json", surface_output)
        vertebra_summaries.append(
            {
                "sample": sample,
                "vertebra": vertebra,
                "slice_count": len(slices),
                "mean_centroid_disagreement_px": (
                    sum(disagreements) / len(disagreements)
                    if disagreements
                    else float("nan")
                ),
            }
        )
    summary = {
        "window_count": window_count,
        "vertebra_count": len(vertebra_summaries),
        "vertebrae": vertebra_summaries,
    }
    save_json(output_root / "inference_summary.json", summary)
    return summary
