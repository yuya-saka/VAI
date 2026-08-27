"""fine-tuning用のAdamW parameter群とcosine scheduler。"""

from __future__ import annotations

import math
from typing import Protocol, cast

import torch
from torch import nn


class FineTuningModel(Protocol):
    """学習済み部分と新規region部分を分離するmodel契約。"""

    def pretrained_parameters(self) -> list[nn.Parameter]: ...

    def region_parameters(self) -> list[nn.Parameter]: ...


def create_finetuning_optimizer(
    model: nn.Module,
    weight_decay: float,
    pretrained_learning_rate: float,
    region_learning_rate: float,
) -> torch.optim.AdamW:
    """全parameterをpretrained/regionへ重複なく分けたAdamWを返す。"""
    _validate_positive_rates(pretrained_learning_rate, region_learning_rate)
    if not hasattr(model, "pretrained_parameters") or not hasattr(
        model, "region_parameters"
    ):
        raise TypeError("modelにはpretrained_parameters/region_parametersが必要です")
    grouped_model = cast(FineTuningModel, model)
    pretrained_ids = {id(value) for value in grouped_model.pretrained_parameters()}
    region_ids = {id(value) for value in grouped_model.region_parameters()}
    if pretrained_ids & region_ids:
        raise ValueError("pretrained/region parameterが重複しています")
    groups: dict[str, list[nn.Parameter]] = {"pretrained": [], "region": []}
    for name, parameter in model.named_parameters():
        if id(parameter) in pretrained_ids:
            groups["pretrained"].append(parameter)
        elif id(parameter) in region_ids:
            groups["region"].append(parameter)
        else:
            raise ValueError(f"optimizerへ分類できないparameterがあります: {name}")
    return torch.optim.AdamW(
        [
            {
                "params": groups["pretrained"],
                "lr": pretrained_learning_rate,
                "weight_decay": weight_decay,
                "category": "pretrained",
            },
            {
                "params": groups["region"],
                "lr": region_learning_rate,
                "weight_decay": weight_decay,
                "category": "region",
            },
        ]
    )


def create_finetuning_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    pretrained_learning_rate: float,
    region_learning_rate: float,
    pretrained_min_learning_rate: float,
    region_min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """各parameter群のLR比を保つrestartなしcosine schedulerを返す。"""
    if max_epochs < 1:
        raise ValueError("max_epochsは1以上である必要があります")
    _validate_positive_rates(pretrained_learning_rate, region_learning_rate)
    _validate_min_rate(pretrained_min_learning_rate, pretrained_learning_rate)
    _validate_min_rate(region_min_learning_rate, region_learning_rate)
    factors = {
        "pretrained": pretrained_min_learning_rate / pretrained_learning_rate,
        "region": region_min_learning_rate / region_learning_rate,
    }
    lambdas = []
    for group in optimizer.param_groups:
        category = group.get("category")
        if category not in factors:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        minimum_factor = factors[category]
        lambdas.append(
            lambda epoch, factor=minimum_factor: factor
            + (1.0 - factor)
            * (1.0 + math.cos(math.pi * min(epoch, max_epochs) / max_epochs))
            / 2.0
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """現在のpretrained/region LRを返す。"""
    rates: dict[str, set[float]] = {"pretrained": set(), "region": set()}
    for group in optimizer.param_groups:
        category = group.get("category")
        if category not in rates:
            raise ValueError("optimizer parameter groupのcategoryが不正です")
        rates[category].add(float(group["lr"]))
    if any(len(values) != 1 for values in rates.values()):
        raise ValueError("同category内のlearning rateが一致しません")
    return next(iter(rates["pretrained"])), next(iter(rates["region"]))


def _validate_positive_rates(*values: float) -> None:
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("learning rateは正の有限値である必要があります")


def _validate_min_rate(minimum: float, maximum: float) -> None:
    if not math.isfinite(minimum) or not 0 <= minimum <= maximum:
        raise ValueError("minimum learning rateは0以上かつ初期値以下が必要です")
