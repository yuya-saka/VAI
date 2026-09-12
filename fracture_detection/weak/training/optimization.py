"""AdamW parameter groups and per-group cosine LR schedule for fine-tuning.

Two groups -- "transferred" (the Baseline 0 CNN trunk) and "new" (FPN, region
BiLSTM, region head) -- each with its own learning rate and minimum learning
rate. ``baseline0.training.optimization.create_cosine_scheduler`` cannot be
reused because it requires both groups to share one minimum LR
(``backbone_min_learning_rate == head_min_learning_rate``); this package
needs different minimums per group, so a ``LambdaLR``-based scheduler is
written here instead (the same technique ``region_branch`` uses for its own
two-group fine-tuning schedule).
"""

from __future__ import annotations

import math
from typing import Protocol, cast

import torch
from torch import nn

TRANSFERRED_CATEGORY = "transferred"
NEW_CATEGORY = "new"


class WeakFineTuningModel(Protocol):
    """Model contract separating the transferred CNN from new modules."""

    def transferred_parameters(self) -> list[nn.Parameter]: ...

    def new_parameters(self) -> list[nn.Parameter]: ...


def create_weak_optimizer(
    model: nn.Module,
    weight_decay: float,
    transferred_learning_rate: float,
    new_learning_rate: float,
) -> torch.optim.AdamW:
    """Build an AdamW with all parameters partitioned into exactly 2 groups."""
    _validate_positive_rates(transferred_learning_rate, new_learning_rate)
    if not hasattr(model, "transferred_parameters") or not hasattr(
        model, "new_parameters"
    ):
        raise TypeError("modelにはtransferred_parameters/new_parametersが必要です")
    grouped_model = cast(WeakFineTuningModel, model)
    transferred_ids = {id(value) for value in grouped_model.transferred_parameters()}
    new_ids = {id(value) for value in grouped_model.new_parameters()}
    if transferred_ids & new_ids:
        raise ValueError("transferred/new parameterが重複しています")

    groups: dict[str, list[nn.Parameter]] = {
        TRANSFERRED_CATEGORY: [],
        NEW_CATEGORY: [],
    }
    for name, parameter in model.named_parameters():
        if id(parameter) in transferred_ids:
            groups[TRANSFERRED_CATEGORY].append(parameter)
        elif id(parameter) in new_ids:
            groups[NEW_CATEGORY].append(parameter)
        else:
            raise ValueError(f"optimizerへ分類できないparameterがあります: {name}")

    return torch.optim.AdamW(
        [
            {
                "params": groups[TRANSFERRED_CATEGORY],
                "lr": transferred_learning_rate,
                "weight_decay": weight_decay,
                "category": TRANSFERRED_CATEGORY,
            },
            {
                "params": groups[NEW_CATEGORY],
                "lr": new_learning_rate,
                "weight_decay": weight_decay,
                "category": NEW_CATEGORY,
            },
        ]
    )


def create_weak_scheduler(
    optimizer: torch.optim.Optimizer,
    max_gt_passes: int,
    transferred_learning_rate: float,
    new_learning_rate: float,
    transferred_min_learning_rate: float,
    new_min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Restart-free cosine decay, one ratio per group, to independent minimums."""
    if max_gt_passes < 1:
        raise ValueError("max_gt_passesは1以上である必要があります")
    _validate_positive_rates(transferred_learning_rate, new_learning_rate)
    _validate_min_rate(transferred_min_learning_rate, transferred_learning_rate)
    _validate_min_rate(new_min_learning_rate, new_learning_rate)
    factors = {
        TRANSFERRED_CATEGORY: transferred_min_learning_rate / transferred_learning_rate,
        NEW_CATEGORY: new_min_learning_rate / new_learning_rate,
    }
    lambdas = []
    for group in optimizer.param_groups:
        category = group.get("category")
        if category not in factors:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        minimum_factor = factors[category]
        lambdas.append(
            lambda gt_pass, factor=minimum_factor: factor
            + (1.0 - factor)
            * (1.0 + math.cos(math.pi * min(gt_pass, max_gt_passes) / max_gt_passes))
            / 2.0
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return the current (transferred_lr, new_lr)."""
    rates: dict[str, set[float]] = {TRANSFERRED_CATEGORY: set(), NEW_CATEGORY: set()}
    for group in optimizer.param_groups:
        category = group.get("category")
        if category not in rates:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        rates[category].add(float(group["lr"]))
    if any(len(values) != 1 for values in rates.values()):
        raise ValueError("同category内のlearning rateが一致しません")
    return (
        next(iter(rates[TRANSFERRED_CATEGORY])),
        next(iter(rates[NEW_CATEGORY])),
    )


def _validate_positive_rates(*values: float) -> None:
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("learning rateは正の有限値である必要があります")


def _validate_min_rate(minimum: float, maximum: float) -> None:
    if not math.isfinite(minimum) or not 0 <= minimum <= maximum:
        raise ValueError("minimum learning rateは0以上かつ初期値以下が必要です")
