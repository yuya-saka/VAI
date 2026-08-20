"""two-stream学習で使うmodel状態context。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def batch_norm_eval(model: nn.Module) -> Iterator[None]:
    """BatchNormだけを一時的にevalへ切り替えて個別状態を復元する。"""
    modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    training_states = [module.training for module in modules]
    try:
        for module in modules:
            module.train(False)
        yield
    finally:
        for module, was_training in zip(modules, training_states, strict=True):
            module.train(was_training)
