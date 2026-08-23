"""低頻度detail streamに対応したtwo-stream optimizer step。

`core.steps.train_step`との違いは1点だけ: 元実装は
`adapter.region_enabled`が真なら**毎step**annotated batchを要求するが、
ここではannotated batchが渡された時だけregion backwardを行う
（渡されないstepはwhole lossのみ、backward 1回）。低頻度化そのものは
`training.schedule.region_step_schedule`が担い、呼び出し側が
annotated_batchをNoneにするかどうかを決める。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from fracture_detection.core.contexts import batch_norm_eval
from fracture_detection.core.contracts import LossWeights
from fracture_detection.core.losses import broadcast_bce_loss, region_bce
from fracture_detection.core.rng import TrainingRngStreams
from fracture_detection.core.steps import (
    ArmAdapter,
    GradientNorms,
    MixedNaturalBatch,
    TrainStepResult,
    gradient_l2_norm,
    mix_natural_batch,
    prepare_batch,
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
) -> TrainStepResult:
    """natural backward後、annotated_batchがある時だけregion backwardする。"""
    model.train()
    natural = prepare_batch(natural_batch, device, adapter.input_channels)
    mixed = mix_natural_batch(natural, mixup_probability, rng_streams.mixup)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
    ):
        natural_output = adapter.forward(model, mixed.inputs)
        whole_loss = _mixup_whole_loss(natural_output.whole_logits, mixed, pos_weight)
    if not torch.isfinite(whole_loss):
        raise FloatingPointError("natural lossが非有限値です")
    shared_parameters = tuple(
        parameter
        for parameter in adapter.shared_parameters(model)
        if parameter.requires_grad
    )
    whole_norm = math.nan
    if measure_gradient_components:
        whole_norm = gradient_l2_norm(whole_loss, shared_parameters, retain_graph=True)
    torch.autograd.backward(whole_loss)

    region_loss = whole_loss.new_zeros(())
    region_norm = math.nan
    if annotated_batch is not None:
        annotated = prepare_batch(annotated_batch, device, adapter.input_channels)
        if annotated.region_targets is None or annotated.region_target_valid is None:
            raise ValueError("annotated batchにregion target/validが必要です")
        with (
            batch_norm_eval(model),
            rng_streams.annotated(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ),
        ):
            annotated_output = adapter.forward(model, annotated.inputs)
            if annotated_output.region_logits is None:
                raise ValueError("modelがregion logitsを返しません")
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

    gradient_norm_tensor = clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not torch.isfinite(gradient_norm_tensor):
        raise FloatingPointError("gradient normが非有限値です")
    optimizer.step()
    gradient_norm = float(gradient_norm_tensor)
    total_loss = float(whole_loss.detach()) + loss_weights.region * float(
        region_loss.detach()
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
            attention=math.nan,
            weighted_attention=math.nan,
        )
    return TrainStepResult(
        whole_loss=float(whole_loss.detach()),
        region_loss=float(region_loss.detach()),
        attention_loss=0.0,
        total_loss=total_loss,
        gradient_norm=gradient_norm,
        clipped=gradient_norm > gradient_clip_norm,
        mixed=mixed.mixed,
        gradient_components=components,
        natural_seconds=0.0,
        annotated_seconds=0.0,
        optimizer_seconds=0.0,
    )


def _mixup_whole_loss(
    logits: Tensor, mixed: MixedNaturalBatch, pos_weight: float
) -> Tensor:
    first = broadcast_bce_loss(logits, mixed.targets_a, pos_weight)
    if not mixed.mixed:
        return first
    second = broadcast_bce_loss(logits, mixed.targets_b, pos_weight)
    return mixed.mixup_lambda * first + (1.0 - mixed.mixup_lambda) * second
