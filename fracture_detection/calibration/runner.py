"""reference arm上でouter fold別gradient校正を実行する。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from fracture_detection.common.calibration import (
    CalibrationResult,
    calibrate_gradient_weight,
)
from fracture_detection.common.canonical_dataset import CanonicalFractureDataset
from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.common.splits import split_nested_manifest
from fracture_detection.config.schema import apply_overrides
from fracture_detection.core.artifacts import canonical_sha256, normalized_config
from fracture_detection.core.factory import build_adapter, build_model
from fracture_detection.core.losses import (
    attention_rmse,
    broadcast_bce_loss,
    region_bce,
)
from fracture_detection.core.optimization import create_optimizer
from fracture_detection.core.rng import (
    GlobalRngState,
    global_rng_states_equal,
)
from fracture_detection.core.steps import ArmAdapter, prepare_batch
from fracture_detection.core.trainer import create_data_loader, set_seed


def calibrate_outer_fold(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    dataset_dir: Path,
    outer_fold: int,
    kind: str,
    device: torch.device,
    *,
    expected_batches: int = 64,
) -> tuple[CalibrationResult, str]:
    """1 outer foldのλまたはβを校正してconfig hashと返す。"""
    expected_arm = "baseline1_b" if kind == "lambda" else "proposed_b"
    if config["arm"]["name"] != expected_arm:
        raise ValueError(f"{kind} referenceは{expected_arm}が必要です")
    fold_config = apply_overrides(
        config, outer_fold=outer_fold, gpu_id=device.index or 0
    )
    set_seed(int(fold_config["data"]["random_seed"]), outer_fold)
    train_manifest, _, _ = split_nested_manifest(manifest, outer_fold)
    annotated_manifest = train_manifest[
        train_manifest["has_region_target"].astype(bool)
    ].reset_index(drop=True)
    natural_dataset = CanonicalFractureDataset(train_manifest, dataset_dir)
    seed = int(fold_config["data"]["random_seed"]) + outer_fold
    natural_loader = create_data_loader(
        natural_dataset,
        int(fold_config["training"]["natural_batch_size"]),
        0,
        seed,
        device,
        EpochShuffleSampler(natural_dataset, seed=seed),
    )
    annotated_loader = None
    if kind == "lambda":
        annotated_dataset = CanonicalFractureDataset(annotated_manifest, dataset_dir)
        annotated_loader = create_data_loader(
            annotated_dataset,
            1,
            0,
            seed + 10_000,
            device,
            AnnotatedCycleSampler(
                len(annotated_dataset), expected_batches, seed + 10_000
            ),
        )
    model = build_model(fold_config).to(device)
    adapter = build_adapter(fold_config)
    optimizer = create_optimizer(
        model,
        float(fold_config["training"]["weight_decay"]),
        float(fold_config["training"]["backbone_learning_rate"]),
        float(fold_config["training"]["head_learning_rate"]),
    )
    if kind == "lambda":
        if annotated_loader is None:
            raise AssertionError("lambda校正にはannotated loaderが必要です")
        batches: Iterable[object] = zip(natural_loader, annotated_loader, strict=False)
        loss_pair = _lambda_loss_pair(model, adapter, device, fold_config)
    else:
        batches = natural_loader
        loss_pair = _beta_loss_pair(model, adapter, device, fold_config)
    rng_before = GlobalRngState.capture()
    result = calibrate_gradient_weight(
        model,
        batches,
        adapter.shared_parameters(model),
        loss_pair,
        expected_batches=expected_batches,
        optimizer=optimizer,
    )
    rng_after = GlobalRngState.capture()
    if not global_rng_states_equal(rng_before, rng_after):
        raise RuntimeError("校正後にglobal RNG stateが変化しました")
    return result, canonical_sha256(normalized_config(fold_config))


def calibration_record(result: CalibrationResult) -> dict[str, object]:
    """CalibrationResultをraw artifact recordへ変換する。"""
    return asdict(result)


def _lambda_loss_pair(
    model: torch.nn.Module,
    adapter: ArmAdapter,
    device: torch.device,
    config: Mapping[str, Any],
) -> Callable[[object], tuple[Tensor, Tensor]]:
    def calculate(batch_pair: object) -> tuple[Tensor, Tensor]:
        if not isinstance(batch_pair, tuple) or len(batch_pair) != 2:
            raise TypeError("lambda calibration batch pairが不正です")
        natural_raw, annotated_raw = batch_pair
        if not isinstance(natural_raw, Mapping) or not isinstance(
            annotated_raw, Mapping
        ):
            raise TypeError("lambda calibration batchはmappingが必要です")
        natural = prepare_batch(
            cast(Mapping[str, object], natural_raw), device, adapter.input_channels
        )
        annotated = prepare_batch(
            cast(Mapping[str, object], annotated_raw),
            device,
            adapter.input_channels,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = adapter.forward(model, natural.inputs)
            annotated_output = adapter.forward(model, annotated.inputs)
            if (
                annotated_output.region_logits is None
                or annotated.region_targets is None
                or annotated.region_target_valid is None
            ):
                raise ValueError("lambda referenceにregion出力/targetがありません")
            whole = broadcast_bce_loss(
                output.whole_logits,
                natural.vertebra_targets,
                float(config["training"]["pos_weight"]),
            )
            region = region_bce(
                annotated_output.region_logits,
                annotated.region_targets,
                annotated.region_target_valid,
            )
        return whole, region

    return calculate


def _beta_loss_pair(
    model: torch.nn.Module,
    adapter: ArmAdapter,
    device: torch.device,
    config: Mapping[str, Any],
) -> Callable[[object], tuple[Tensor, Tensor]]:
    def calculate(raw_batch: object) -> tuple[Tensor, Tensor]:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("beta calibration batchが不正です")
        batch = prepare_batch(
            cast(Mapping[str, object], raw_batch), device, adapter.input_channels
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = adapter.forward(model, batch.inputs)
            if output.spatial_attention is None or batch.region_masks is None:
                raise ValueError("beta referenceにattention/maskがありません")
            whole = broadcast_bce_loss(
                output.whole_logits,
                batch.vertebra_targets,
                float(config["training"]["pos_weight"]),
            )
            attention = attention_rmse(output.spatial_attention, batch.region_masks)
        return whole, attention

    return calculate
