"""Stage4 gradient-conflict and pooling-collapse diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

GRADIENT_CONFLICT_THRESHOLD = -0.20
GRADIENT_RATIO_THRESHOLD = 3.0
WINNER_SHARE_THRESHOLD = 0.75
POOL_CONCENTRATION_THRESHOLD = 0.80


def gradient_alignment(
    vertebra_loss: Tensor,
    region_loss_value: Tensor,
    parameters: Iterable[torch.nn.Parameter],
    lambda_region: float,
) -> dict[str, float]:
    """Measure gradient cosine and weighted norm ratio on shared parameters."""
    shared = [parameter for parameter in parameters if parameter.requires_grad]
    vertebra_gradients = torch.autograd.grad(
        vertebra_loss,
        shared,
        retain_graph=True,
        allow_unused=True,
    )
    region_gradients = torch.autograd.grad(
        region_loss_value,
        shared,
        retain_graph=True,
        allow_unused=True,
    )
    dot = torch.zeros((), device=vertebra_loss.device, dtype=torch.float64)
    vertebra_square = torch.zeros_like(dot)
    region_square = torch.zeros_like(dot)
    for vertebra_gradient, region_gradient in zip(
        vertebra_gradients, region_gradients, strict=True
    ):
        if vertebra_gradient is None or region_gradient is None:
            continue
        vertebra_values = vertebra_gradient.detach().double()
        region_values = region_gradient.detach().double()
        dot += (vertebra_values * region_values).sum()
        vertebra_square += vertebra_values.square().sum()
        region_square += region_values.square().sum()
    vertebra_norm = vertebra_square.sqrt()
    region_norm = region_square.sqrt()
    denominator = vertebra_norm * region_norm
    cosine = dot / denominator if denominator > 0 else torch.zeros_like(dot)
    ratio = (
        lambda_region * region_norm / vertebra_norm
        if vertebra_norm > 0
        else torch.full_like(dot, float("inf"))
    )
    return {
        "gradient_cosine": float(cosine.cpu()),
        "weighted_gradient_norm_ratio": float(ratio.cpu()),
    }


def pooling_diagnostics(
    region_logits: Tensor,
    region_pool_weights: Tensor,
    targets: Tensor,
    region_valid: Tensor,
) -> dict[str, float]:
    """Summarize winning regions and single-region pooling concentration."""
    positive = (targets > 0) & region_valid.any(dim=1)
    if not positive.any():
        return {
            **{f"winner_share_r{index + 1}": 0.0 for index in range(4)},
            "max_winner_share": 0.0,
            "pool_weight_above_095_fraction": 0.0,
        }
    valid_logits = torch.where(
        region_valid[positive],
        region_logits[positive].float(),
        torch.full_like(region_logits[positive].float(), -torch.inf),
    )
    winners = valid_logits.argmax(dim=1)
    winner_shares = torch.bincount(winners, minlength=4).float() / winners.numel()
    concentrated = region_pool_weights[positive].float().max(dim=1).values > 0.95
    result = {
        f"winner_share_r{index + 1}": float(winner_shares[index].cpu())
        for index in range(4)
    }
    return {
        **result,
        "max_winner_share": float(winner_shares.max().cpu()),
        "pool_weight_above_095_fraction": float(concentrated.float().mean().cpu()),
    }


@dataclass
class DiagnosticHistory:
    """Aggregate step diagnostics and apply consecutive-epoch warning rules."""

    gradient_conflict_epochs: int = 0
    gradient_ratio_epochs: int = 0
    step_records: list[dict[str, float]] = field(default_factory=list)

    def add_step(self, record: dict[str, float]) -> None:
        self.step_records.append(dict(record))

    def summarize_epoch(self) -> dict[str, float | bool]:
        if not self.step_records:
            raise ValueError("no diagnostic step records were added")
        cosine = float(
            np.median([record["gradient_cosine"] for record in self.step_records])
        )
        ratio = float(
            np.median(
                [record["weighted_gradient_norm_ratio"] for record in self.step_records]
            )
        )
        max_winner_share = max(
            record["max_winner_share"] for record in self.step_records
        )
        concentration = max(
            record["pool_weight_above_095_fraction"] for record in self.step_records
        )
        self.gradient_conflict_epochs = (
            self.gradient_conflict_epochs + 1
            if cosine < GRADIENT_CONFLICT_THRESHOLD
            else 0
        )
        self.gradient_ratio_epochs = (
            self.gradient_ratio_epochs + 1 if ratio > GRADIENT_RATIO_THRESHOLD else 0
        )
        summary: dict[str, float | bool] = {
            "gradient_cosine_median": cosine,
            "weighted_gradient_norm_ratio_median": ratio,
            "max_winner_share": max_winner_share,
            "pool_weight_above_095_fraction": concentration,
            "warn_gradient_conflict": self.gradient_conflict_epochs >= 2,
            "warn_gradient_ratio": self.gradient_ratio_epochs >= 2,
            "warn_winner_collapse": max_winner_share > WINNER_SHARE_THRESHOLD,
            "warn_pool_concentration": concentration > POOL_CONCENTRATION_THRESHOLD,
        }
        self.step_records.clear()
        return summary
