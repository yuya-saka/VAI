"""Baseline 1のAdamWパラメータ群と段階的な学習率制御。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn


class _ParameterGroupedModel(Protocol):
    """異なる学習率の設定に必要なモデルinterface。"""

    def backbone_parameters(self) -> list[nn.Parameter]: ...

    def head_parameters(self) -> list[nn.Parameter]: ...

    def set_backbone_trainable(self, trainable: bool) -> None: ...


def _is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    """bias・正規化パラメータをweight decayの対象外にする。"""
    return name.endswith(".bias") or parameter.ndim <= 1


def create_optimizer(
    model: nn.Module,
    weight_decay: float,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.AdamW:
    """バックボーン・分類部とdecay・no-decayを分離したAdamWを作る。"""
    if any(
        not math.isfinite(learning_rate) or learning_rate <= 0
        for learning_rate in (backbone_learning_rate, head_learning_rate)
    ):
        raise ValueError("learning rateは正の有限値である必要があります")
    if not hasattr(model, "backbone_parameters") or not hasattr(
        model, "head_parameters"
    ):
        raise TypeError("modelにはbackbone_parameters/head_parametersが必要です")
    parameter_model = cast(_ParameterGroupedModel, model)
    backbone_ids = {
        id(parameter) for parameter in parameter_model.backbone_parameters()
    }
    head_ids = {id(parameter) for parameter in parameter_model.head_parameters()}
    if backbone_ids & head_ids:
        raise ValueError("backbone/head parameterが重複しています")

    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, parameter in model.named_parameters():
        if id(parameter) in backbone_ids:
            category = "backbone"
        elif id(parameter) in head_ids:
            category = "head"
        else:
            raise ValueError(f"optimizerへ分類できないparameterがあります: {name}")
        grouped[(category, not _is_no_decay(name, parameter))].append(parameter)

    parameter_groups: list[dict[str, object]] = []
    for (category, apply_decay), parameters in grouped.items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": (
                    backbone_learning_rate
                    if category == "backbone"
                    else head_learning_rate
                ),
                "weight_decay": weight_decay if apply_decay else 0.0,
                "category": category,
            }
        )
    return torch.optim.AdamW(parameter_groups)


@dataclass(frozen=True)
class LearningRateController:
    """ReduceLROnPlateau前の差分学習率warmupをstep単位で適用する。"""

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
            raise ValueError(
                "warmup_start_factorは0より大きく1以下である必要があります"
            )
        if any(
            not math.isfinite(learning_rate) or learning_rate <= 0
            for learning_rate in (
                self.backbone_learning_rate,
                self.head_learning_rate,
            )
        ):
            raise ValueError("learning rateは正の有限値である必要があります")

    def set_epoch_state(self, model: nn.Module, epoch_index: int) -> None:
        """設定された凍結期間をepoch境界で反映する。"""
        if not hasattr(model, "set_backbone_trainable"):
            raise TypeError("modelにはset_backbone_trainableが必要です")
        parameter_model = cast(_ParameterGroupedModel, model)
        parameter_model.set_backbone_trainable(
            epoch_index >= self.freeze_backbone_epochs
        )

    def apply(
        self, optimizer: torch.optim.Optimizer, global_step: int
    ) -> tuple[float, float]:
        """warmup中だけoptimizerの全groupへ指定stepの学習率を設定する。"""
        if global_step < 0:
            raise ValueError("global_stepは0以上である必要があります")
        freeze_steps = self.freeze_backbone_epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        if global_step >= freeze_steps + warmup_steps:
            return optimizer_learning_rates(optimizer)

        if global_step < freeze_steps:
            head_progress = global_step / max(freeze_steps - 1, 1)
            backbone_lr = 0.0
            head_lr = _linear(
                self.head_learning_rate * self.warmup_start_factor,
                self.head_learning_rate,
                head_progress,
            )
        else:
            warmup_step = global_step - freeze_steps
            warmup_progress = warmup_step / max(warmup_steps - 1, 1)
            backbone_lr = _linear(
                self.backbone_learning_rate * self.warmup_start_factor,
                self.backbone_learning_rate,
                warmup_progress,
            )
            head_start = (
                self.head_learning_rate
                if freeze_steps > 0
                else self.head_learning_rate * self.warmup_start_factor
            )
            head_lr = _linear(
                head_start,
                self.head_learning_rate,
                warmup_progress,
            )
        for parameter_group in optimizer.param_groups:
            category = parameter_group.get("category")
            if category == "backbone":
                parameter_group["lr"] = backbone_lr
            elif category == "head":
                parameter_group["lr"] = head_lr
            else:
                raise ValueError("optimizer parameter groupのcategoryが不正です")
        return backbone_lr, head_lr


def create_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float,
    patience: int,
    threshold: float,
    cooldown: int,
    backbone_min_learning_rate: float,
    head_min_learning_rate: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """val BCEを監視する差分minimum LR付きplateau schedulerを返す。"""
    minimum_learning_rates: list[float] = []
    for parameter_group in optimizer.param_groups:
        category = parameter_group.get("category")
        if category == "backbone":
            minimum_learning_rates.append(backbone_min_learning_rate)
        elif category == "head":
            minimum_learning_rates.append(head_min_learning_rate)
        else:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="rel",
        cooldown=cooldown,
        min_lr=minimum_learning_rates,
    )


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """optimizer groupからbackbone/headの現在LRを返す。"""
    category_rates: dict[str, set[float]] = {"backbone": set(), "head": set()}
    for parameter_group in optimizer.param_groups:
        category = parameter_group.get("category")
        if category not in category_rates:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        category_rates[category].add(float(parameter_group["lr"]))
    if any(len(rates) != 1 for rates in category_rates.values()):
        raise ValueError("同じparameter category内のlearning rateが一致しません")
    return (
        next(iter(category_rates["backbone"])),
        next(iter(category_rates["head"])),
    )


def _linear(start: float, end: float, progress: float) -> float:
    """0〜1へ切り詰めた線形補間を返す。"""
    clipped = min(max(progress, 0.0), 1.0)
    return start + (end - start) * clipped
