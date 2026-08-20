"""全アーム共通のAdamW parameter群と学習率制御。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn


class ParameterGroupedModel(Protocol):
    """backbone/head parameterを分離するmodel契約。"""

    def backbone_parameters(self) -> list[nn.Parameter]: ...

    def head_parameters(self) -> list[nn.Parameter]: ...

    def set_backbone_trainable(self, trainable: bool) -> None: ...


def create_optimizer(
    model: nn.Module,
    weight_decay: float,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.AdamW:
    """全parameterを重複なくbackbone/headへ分けたAdamWを返す。"""
    if any(
        not math.isfinite(value) or value <= 0
        for value in (backbone_learning_rate, head_learning_rate)
    ):
        raise ValueError("learning rateは正の有限値である必要があります")
    if not hasattr(model, "backbone_parameters") or not hasattr(
        model, "head_parameters"
    ):
        raise TypeError("modelにはbackbone_parameters/head_parametersが必要です")
    grouped_model = cast(ParameterGroupedModel, model)
    backbone_ids = {id(value) for value in grouped_model.backbone_parameters()}
    head_ids = {id(value) for value in grouped_model.head_parameters()}
    if backbone_ids & head_ids:
        raise ValueError("backbone/head parameterが重複しています")
    groups: dict[str, list[nn.Parameter]] = {"backbone": [], "head": []}
    for name, parameter in model.named_parameters():
        if id(parameter) in backbone_ids:
            groups["backbone"].append(parameter)
        elif id(parameter) in head_ids:
            groups["head"].append(parameter)
        else:
            raise ValueError(f"optimizerへ分類できないparameterがあります: {name}")
    return torch.optim.AdamW(
        [
            {
                "params": values,
                "lr": (
                    backbone_learning_rate
                    if category == "backbone"
                    else head_learning_rate
                ),
                "weight_decay": weight_decay,
                "category": category,
            }
            for category, values in groups.items()
            if values
        ]
    )


@dataclass(frozen=True)
class LearningRateController:
    """任意のfreeze/warmupをstep単位で適用する。"""

    steps_per_epoch: int
    freeze_backbone_epochs: int
    warmup_epochs: int
    warmup_start_factor: float
    backbone_learning_rate: float
    head_learning_rate: float

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epochは1以上である必要があります")
        if self.freeze_backbone_epochs < 0 or self.warmup_epochs < 0:
            raise ValueError("freeze/warmup epochは0以上である必要があります")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factorは0より大きく1以下が必要です")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.backbone_learning_rate, self.head_learning_rate)
        ):
            raise ValueError("learning rateは正の有限値である必要があります")

    def set_epoch_state(self, model: nn.Module, epoch_index: int) -> None:
        """backbone凍結状態をepoch境界で設定する。"""
        if not hasattr(model, "set_backbone_trainable"):
            raise TypeError("modelにはset_backbone_trainableが必要です")
        grouped_model = cast(ParameterGroupedModel, model)
        grouped_model.set_backbone_trainable(epoch_index >= self.freeze_backbone_epochs)

    def apply(
        self, optimizer: torch.optim.Optimizer, global_step: int
    ) -> tuple[float, float]:
        """freeze/warmup期間だけ明示LRを適用する。"""
        if global_step < 0:
            raise ValueError("global_stepは0以上である必要があります")
        freeze_steps = self.freeze_backbone_epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        if global_step >= freeze_steps + warmup_steps:
            return optimizer_learning_rates(optimizer)
        if global_step < freeze_steps:
            progress = global_step / max(freeze_steps - 1, 1)
            backbone_lr = 0.0
            head_lr = _linear(
                self.head_learning_rate * self.warmup_start_factor,
                self.head_learning_rate,
                progress,
            )
        else:
            progress = (global_step - freeze_steps) / max(warmup_steps - 1, 1)
            backbone_lr = _linear(
                self.backbone_learning_rate * self.warmup_start_factor,
                self.backbone_learning_rate,
                progress,
            )
            head_start = (
                self.head_learning_rate
                if freeze_steps
                else self.head_learning_rate * self.warmup_start_factor
            )
            head_lr = _linear(head_start, self.head_learning_rate, progress)
        for group in optimizer.param_groups:
            if group.get("category") == "backbone":
                group["lr"] = backbone_lr
            elif group.get("category") == "head":
                group["lr"] = head_lr
            else:
                raise ValueError("optimizer parameter groupのcategoryが不正です")
        return backbone_lr, head_lr


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    backbone_min_learning_rate: float,
    head_min_learning_rate: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """restartしないRSNA Type1 cosine schedulerを返す。"""
    if max_epochs < 1:
        raise ValueError("max_epochsは1以上である必要があります")
    if backbone_min_learning_rate != head_min_learning_rate:
        raise ValueError("minimum LRはbackbone/headで一致させます")
    if not math.isfinite(backbone_min_learning_rate) or backbone_min_learning_rate < 0:
        raise ValueError("minimum learning rateは0以上の有限値である必要があります")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=backbone_min_learning_rate,
    )


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """現在のbackbone/head LRを返す。"""
    rates: dict[str, set[float]] = {"backbone": set(), "head": set()}
    for group in optimizer.param_groups:
        category = group.get("category")
        if category not in rates:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        rates[category].add(float(group["lr"]))
    if any(len(values) != 1 for values in rates.values()):
        raise ValueError("同category内のlearning rateが一致しません")
    return next(iter(rates["backbone"])), next(iter(rates["head"]))


def _linear(start: float, end: float, progress: float) -> float:
    clipped = min(max(progress, 0.0), 1.0)
    return start + (end - start) * clipped
