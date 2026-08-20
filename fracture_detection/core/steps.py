"""arm非依存のtwo-stream optimizer step。"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from fracture_detection.core.contexts import batch_norm_eval
from fracture_detection.core.contracts import ArmOutput, LossWeights
from fracture_detection.core.losses import (
    attention_rmse,
    broadcast_bce_loss,
    region_bce,
)
from fracture_detection.core.rng import TrainingRngStreams


class SharedParameterModel(Protocol):
    """勾配比監査に必要なmodel interface。"""

    def shared_parameters(self) -> list[nn.Parameter]: ...


@dataclass(frozen=True)
class ArmAdapter:
    """モデル出力と有効な補助損失を定義する。"""

    input_channels: int
    region_enabled: bool
    attention_enabled: bool
    legacy_tensor_output: bool = False

    def forward(self, model: nn.Module, inputs: Tensor) -> ArmOutput:
        """arm固有forwardを共通出力へ変換する。"""
        output = model(inputs)
        if self.legacy_tensor_output:
            if not isinstance(output, Tensor):
                raise TypeError("legacy modelはTensorを返す必要があります")
            return ArmOutput(whole_logits=output)
        if not isinstance(output, ArmOutput):
            raise TypeError("modelはArmOutputを返す必要があります")
        return output

    def shared_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """勾配監査対象parameterを返す。"""
        if hasattr(model, "shared_parameters"):
            return cast(SharedParameterModel, model).shared_parameters()
        encoder = getattr(model, "encoder", None)
        blocks = getattr(encoder, "blocks", None)
        if not isinstance(blocks, nn.Sequential) or len(blocks) <= 4:
            raise TypeError("modelにshared_parametersまたはencoder.blocks[4]が必要です")
        return list(blocks[4].parameters())


@dataclass(frozen=True)
class PreparedBatch:
    """deviceへ移動・正規化した学習batch。"""

    inputs: Tensor
    vertebra_targets: Tensor
    region_targets: Tensor | None
    region_target_valid: Tensor | None
    region_masks: Tensor | None


@dataclass(frozen=True)
class MixedNaturalBatch:
    """mixup後入力と2組のwhole target。"""

    inputs: Tensor
    targets_a: Tensor
    targets_b: Tensor
    region_masks: Tensor | None
    mixup_lambda: float
    mixed: bool


@dataclass(frozen=True)
class GradientNorms:
    """shared block上のraw/weighted勾配norm。"""

    whole: float
    region: float
    weighted_region: float
    attention: float
    weighted_attention: float


@dataclass(frozen=True)
class TrainStepResult:
    """1 optimizer stepの監査値。"""

    whole_loss: float
    region_loss: float
    attention_loss: float
    total_loss: float
    gradient_norm: float
    clipped: bool
    mixed: bool
    gradient_components: GradientNorms | None
    natural_seconds: float
    annotated_seconds: float
    optimizer_seconds: float


def prepare_batch(
    batch: Mapping[str, object], device: torch.device, input_channels: int
) -> PreparedBatch:
    """canonical batchをarm入力chへ選択し0〜1へ正規化する。"""
    inputs = _tensor(batch, "inputs")
    if inputs.ndim != 5 or inputs.shape[2] < input_channels:
        raise ValueError(f"canonical inputs shapeが不正です: {inputs.shape}")
    moved = inputs[:, :, :input_channels].to(device, non_blocking=True)
    if moved.dtype == torch.uint8:
        moved = moved.to(torch.float32).div(255.0)
    else:
        moved = moved.to(torch.float32)
    region_masks = _optional_tensor(batch, "region_masks")
    moved_region_masks: Tensor | None = None
    if region_masks is not None:
        moved_region_masks = region_masks.to(device, non_blocking=True)
        if moved_region_masks.dtype == torch.uint8:
            moved_region_masks = moved_region_masks.to(torch.float32).div(255.0)
        else:
            moved_region_masks = moved_region_masks.to(torch.float32)
    return PreparedBatch(
        inputs=moved,
        vertebra_targets=_tensor(batch, "vertebra_target").to(
            device, non_blocking=True
        ),
        region_targets=_move_optional(batch, "region_targets", device),
        region_target_valid=_move_optional(batch, "region_target_valid", device),
        region_masks=moved_region_masks,
    )


def mix_natural_batch(
    batch: PreparedBatch,
    probability: float,
    generator: torch.Generator,
) -> MixedNaturalBatch:
    """専用CPU generatorで発火・lambda・permutationを決める。"""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("mixup probabilityは0から1が必要です")
    use_mixup = float(torch.rand((), generator=generator)) < probability
    if not use_mixup:
        return MixedNaturalBatch(
            inputs=batch.inputs,
            targets_a=batch.vertebra_targets,
            targets_b=batch.vertebra_targets,
            region_masks=batch.region_masks,
            mixup_lambda=1.0,
            mixed=False,
        )
    mixup_lambda = float(torch.rand((), generator=generator))
    permutation_cpu = torch.randperm(batch.inputs.shape[0], generator=generator)
    permutation = permutation_cpu.to(batch.inputs.device)
    mixed_inputs = batch.inputs * mixup_lambda + batch.inputs[permutation] * (
        1.0 - mixup_lambda
    )
    mixed_masks = None
    if batch.region_masks is not None:
        mixed_masks = batch.region_masks * mixup_lambda + batch.region_masks[
            permutation
        ] * (1.0 - mixup_lambda)
    return MixedNaturalBatch(
        inputs=mixed_inputs,
        targets_a=batch.vertebra_targets,
        targets_b=batch.vertebra_targets[permutation],
        region_masks=mixed_masks,
        mixup_lambda=mixup_lambda,
        mixed=True,
    )


def train_step(
    model: nn.Module,
    adapter: ArmAdapter,
    natural_batch: Mapping[str, object],
    annotated_batch: Mapping[str, object] | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng_streams: TrainingRngStreams,
    loss_weights: LossWeights,
    *,
    pos_weight: float,
    mixup_probability: float,
    gradient_clip_norm: float,
    measure_gradient_components: bool = False,
    profile_timing: bool = False,
) -> TrainStepResult:
    """natural backward後にannotated backwardし、1回だけstepする。"""
    model.train()
    natural = prepare_batch(natural_batch, device, adapter.input_channels)
    mixed = mix_natural_batch(natural, mixup_probability, rng_streams.mixup)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    _synchronize(device, profile_timing)
    natural_started = time.perf_counter()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
    ):
        natural_output = adapter.forward(model, mixed.inputs)
        whole_loss = _mixup_whole_loss(
            natural_output.whole_logits,
            mixed,
            pos_weight,
        )
        attention_loss = _attention_loss(adapter, natural_output, mixed)
        natural_loss = whole_loss + loss_weights.attention * attention_loss
    if not torch.isfinite(natural_loss):
        raise FloatingPointError("natural lossが非有限値です")
    shared_parameters = tuple(
        parameter
        for parameter in adapter.shared_parameters(model)
        if parameter.requires_grad
    )
    whole_norm = math.nan
    attention_norm = math.nan
    if measure_gradient_components:
        whole_norm = gradient_l2_norm(whole_loss, shared_parameters, retain_graph=True)
        if adapter.attention_enabled:
            attention_norm = gradient_l2_norm(
                attention_loss, shared_parameters, retain_graph=True
            )
    torch.autograd.backward(natural_loss)
    _synchronize(device, profile_timing)
    natural_seconds = time.perf_counter() - natural_started if profile_timing else 0.0

    region_loss = natural_loss.new_zeros(())
    region_norm = math.nan
    annotated_seconds = 0.0
    if adapter.region_enabled:
        if annotated_batch is None:
            raise ValueError("region有効armにはannotated batchが必要です")
        annotated = prepare_batch(annotated_batch, device, adapter.input_channels)
        if annotated.region_targets is None or annotated.region_target_valid is None:
            raise ValueError("annotated batchにregion target/validが必要です")
        _synchronize(device, profile_timing)
        annotated_started = time.perf_counter()
        with (
            batch_norm_eval(model),
            rng_streams.annotated(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ),
        ):
            annotated_output = adapter.forward(model, annotated.inputs)
            if annotated_output.region_logits is None:
                raise ValueError("region有効armがregion logitsを返しません")
            region_loss = region_bce(
                annotated_output.region_logits,
                annotated.region_targets,
                annotated.region_target_valid,
            )
            weighted_region = loss_weights.region * region_loss
        if not torch.isfinite(weighted_region):
            raise FloatingPointError("annotated lossが非有限値です")
        if measure_gradient_components:
            region_norm = gradient_l2_norm(
                region_loss, shared_parameters, retain_graph=True
            )
        torch.autograd.backward(weighted_region)
        _synchronize(device, profile_timing)
        annotated_seconds = (
            time.perf_counter() - annotated_started if profile_timing else 0.0
        )

    _synchronize(device, profile_timing)
    optimizer_started = time.perf_counter()
    gradient_norm_tensor = clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not torch.isfinite(gradient_norm_tensor):
        raise FloatingPointError("gradient normが非有限値です")
    optimizer.step()
    _synchronize(device, profile_timing)
    optimizer_seconds = (
        time.perf_counter() - optimizer_started if profile_timing else 0.0
    )
    gradient_norm = float(gradient_norm_tensor)
    total_loss = (
        float(whole_loss.detach())
        + loss_weights.region * float(region_loss.detach())
        + loss_weights.attention * float(attention_loss.detach())
    )
    components = None
    if measure_gradient_components:
        components = GradientNorms(
            whole=whole_norm,
            region=region_norm,
            weighted_region=(
                loss_weights.region * region_norm
                if math.isfinite(region_norm)
                else math.nan
            ),
            attention=attention_norm,
            weighted_attention=(
                loss_weights.attention * attention_norm
                if math.isfinite(attention_norm)
                else math.nan
            ),
        )
    return TrainStepResult(
        whole_loss=float(whole_loss.detach()),
        region_loss=float(region_loss.detach()),
        attention_loss=float(attention_loss.detach()),
        total_loss=total_loss,
        gradient_norm=gradient_norm,
        clipped=gradient_norm > gradient_clip_norm,
        mixed=mixed.mixed,
        gradient_components=components,
        natural_seconds=natural_seconds,
        annotated_seconds=annotated_seconds,
        optimizer_seconds=optimizer_seconds,
    )


def gradient_l2_norm(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    """指定lossのshared parameter gradient L2 normを返す。"""
    if not parameters:
        raise ValueError("gradient計測対象parameterがありません")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    non_null = [value for value in gradients if value is not None]
    if not non_null:
        raise RuntimeError("lossがshared parametersへ接続されていません")
    squared = torch.stack(
        [value.detach().float().square().sum() for value in non_null]
    ).sum()
    norm = float(torch.sqrt(squared))
    if not math.isfinite(norm):
        raise FloatingPointError("gradient component normが非有限値です")
    return norm


def _mixup_whole_loss(
    logits: Tensor, mixed: MixedNaturalBatch, pos_weight: float
) -> Tensor:
    first = broadcast_bce_loss(logits, mixed.targets_a, pos_weight)
    if not mixed.mixed:
        return first
    second = broadcast_bce_loss(logits, mixed.targets_b, pos_weight)
    return mixed.mixup_lambda * first + (1.0 - mixed.mixup_lambda) * second


def _attention_loss(
    adapter: ArmAdapter,
    output: ArmOutput,
    mixed: MixedNaturalBatch,
) -> Tensor:
    if not adapter.attention_enabled:
        return output.whole_logits.sum() * 0.0
    if output.spatial_attention is None or mixed.region_masks is None:
        raise ValueError("attention有効armにattention map/maskがありません")
    return attention_rmse(output.spatial_attention, mixed.region_masks)


def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value


def _optional_tensor(batch: Mapping[str, object], key: str) -> Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value


def _move_optional(
    batch: Mapping[str, object], key: str, device: torch.device
) -> Tensor | None:
    value = _optional_tensor(batch, key)
    return value.to(device, non_blocking=True) if value is not None else None


def _synchronize(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
