"""λ・βの一回限りの初期勾配ノルム校正。"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor, nn

from fracture_detection.core.contexts import batch_norm_eval
from fracture_detection.core.rng import GlobalRngState

Batch = TypeVar("Batch")
LossPair = Callable[[Batch], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class CalibrationResult:
    """校正係数と監査用gradient norm。"""

    coefficient: float
    unclipped_coefficient: float
    clipped: bool
    whole_gradient_norms: tuple[float, ...]
    auxiliary_gradient_norms: tuple[float, ...]


def calibrate_gradient_weight(
    model: nn.Module,
    batches: Iterable[Batch],
    shared_parameters: Sequence[nn.Parameter],
    loss_pair: LossPair[Batch],
    *,
    expected_batches: int = 64,
    target_ratio: float = 0.5,
    epsilon: float = 1e-12,
    minimum: float = 1e-2,
    maximum: float = 1e2,
    optimizer: torch.optim.Optimizer | None = None,
) -> CalibrationResult:
    """重み付け前の2損失から固定校正係数を算出する。"""
    if expected_batches < 1:
        raise ValueError("expected_batchesは1以上である必要があります")
    if not 0 < target_ratio or not 0 < epsilon:
        raise ValueError("target_ratioとepsilonは正である必要があります")
    if not 0 < minimum <= maximum:
        raise ValueError("clip範囲が不正です")
    parameters = tuple(shared_parameters)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("shared_parametersには学習可能parameterが必要です")

    model_state = _clone_state(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer else None
    was_training = model.training
    rng_state = GlobalRngState.capture()
    whole_norms: list[float] = []
    auxiliary_norms: list[float] = []
    try:
        model.train(True)
        with batch_norm_eval(model):
            for batch_index, batch in enumerate(batches):
                if batch_index >= expected_batches:
                    break
                whole_loss, auxiliary_loss = loss_pair(batch)
                if whole_loss.ndim != 0 or auxiliary_loss.ndim != 0:
                    raise ValueError("校正対象lossはscalarである必要があります")
                if not torch.isfinite(whole_loss) or not torch.isfinite(auxiliary_loss):
                    raise FloatingPointError("校正lossが非有限値です")
                whole_gradients = torch.autograd.grad(
                    whole_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=False,
                )
                whole_norms.append(_gradient_l2_norm(whole_gradients))
                del whole_gradients
                auxiliary_gradients = torch.autograd.grad(
                    auxiliary_loss,
                    parameters,
                    allow_unused=False,
                )
                auxiliary_norms.append(_gradient_l2_norm(auxiliary_gradients))
                del auxiliary_gradients, whole_loss, auxiliary_loss
        if len(whole_norms) != expected_batches:
            raise ValueError(
                f"校正batch数が不足しています: {len(whole_norms)}/{expected_batches}"
            )
    finally:
        model.train(was_training)
        rng_state.restore()

    _assert_state_equal(model_state, model.state_dict(), "model")
    if optimizer is not None and optimizer_state is not None:
        if not _nested_equal(optimizer_state, optimizer.state_dict()):
            raise RuntimeError("校正後にoptimizer stateが変化しました")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("校正後にparameter.gradが残っています")

    log_ratios = np.log(
        (np.asarray(whole_norms, dtype=np.float64) + epsilon)
        / (np.asarray(auxiliary_norms, dtype=np.float64) + epsilon)
    )
    if not np.isfinite(log_ratios).all():
        raise FloatingPointError("校正gradient比が非有限値です")
    unclipped = target_ratio * math.exp(float(np.median(log_ratios)))
    coefficient = min(max(unclipped, minimum), maximum)
    return CalibrationResult(
        coefficient=coefficient,
        unclipped_coefficient=unclipped,
        clipped=coefficient != unclipped,
        whole_gradient_norms=tuple(whole_norms),
        auxiliary_gradient_norms=tuple(auxiliary_norms),
    )


def _gradient_l2_norm(gradients: Sequence[Tensor]) -> float:
    """複数parameterのgradientを連結したL2 normを返す。"""
    squared = torch.stack(
        [gradient.detach().float().pow(2).sum() for gradient in gradients]
    ).sum()
    value = float(torch.sqrt(squared))
    if not math.isfinite(value):
        raise FloatingPointError("校正gradient normが非有限値です")
    return value


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """model stateをGPU memoryを圧迫せずCPUへ複製する。"""
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _assert_state_equal(
    expected: Mapping[str, Tensor], actual: Mapping[str, Tensor], name: str
) -> None:
    """tensor stateの完全一致を検証する。"""
    if expected.keys() != actual.keys():
        raise RuntimeError(f"校正後に{name} stateのkeyが変化しました")
    changed = [
        key
        for key in expected
        if not torch.equal(expected[key], actual[key].detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"校正後に{name} stateが変化しました: {changed[:5]}")


def _nested_equal(left: Any, right: Any) -> bool:
    """optimizer stateの入れ子構造を比較する。"""
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)
