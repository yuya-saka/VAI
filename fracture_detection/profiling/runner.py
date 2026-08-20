"""同一条件20-stepのVRAM・時間・parameter profile。"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from fracture_detection.common.augmentation import build_canonical_augmentation
from fracture_detection.common.canonical_dataset import CanonicalFractureDataset
from fracture_detection.common.constants import FOLDS_CSV, INPUT_MANIFEST_CSV
from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.common.splits import split_nested_manifest
from fracture_detection.config.schema import apply_overrides
from fracture_detection.core.artifacts import (
    canonical_sha256,
    dependency_sha256,
    normalized_config,
    sha256_file,
    source_tree_sha256,
)
from fracture_detection.core.contracts import LossWeights
from fracture_detection.core.factory import build_adapter, build_model
from fracture_detection.core.optimization import create_optimizer
from fracture_detection.core.rng import TrainingRngStreams
from fracture_detection.core.steps import train_step
from fracture_detection.core.trainer import create_data_loader, set_seed

RESOURCE_LIMIT_BYTES = 49_000_000_000


def profile_arm(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    dataset_dir: Path,
    device: torch.device,
    *,
    outer_fold: int = 0,
    steps: int = 20,
    warmup_steps: int = 10,
) -> dict[str, object]:
    """指定armを20 step実行しwarmup後中央値を返す。"""
    if device.type != "cuda":
        raise RuntimeError("正式resource profileにはCUDAが必要です")
    if steps <= warmup_steps or warmup_steps < 0:
        raise ValueError("stepsはwarmup_stepsより大きい必要があります")
    fold_config = apply_overrides(
        config, outer_fold=outer_fold, gpu_id=device.index or 0
    )
    set_seed(int(fold_config["data"]["random_seed"]), outer_fold)
    train_manifest, _, _ = split_nested_manifest(manifest, outer_fold)
    annotated_manifest = train_manifest[
        train_manifest["has_region_target"].astype(bool)
    ].reset_index(drop=True)
    seed = int(fold_config["data"]["random_seed"]) + outer_fold
    natural_dataset = CanonicalFractureDataset(
        train_manifest,
        dataset_dir,
        build_canonical_augmentation(fold_config["augmentation"]),
        base_seed=int(fold_config["data"]["random_seed"]),
        outer_fold=outer_fold,
        stream="natural",
    )
    natural_loader = create_data_loader(
        natural_dataset,
        int(fold_config["training"]["natural_batch_size"]),
        int(fold_config["data"]["num_workers"]),
        seed,
        device,
        EpochShuffleSampler(natural_dataset, seed=seed, include_metadata=True),
    )
    adapter = build_adapter(fold_config)
    annotated_loader = None
    if adapter.region_enabled:
        annotated_dataset = CanonicalFractureDataset(
            annotated_manifest,
            dataset_dir,
            build_canonical_augmentation(fold_config["augmentation"]),
            base_seed=int(fold_config["data"]["random_seed"]),
            outer_fold=outer_fold,
            stream="annotated",
        )
        annotated_loader = create_data_loader(
            annotated_dataset,
            1,
            int(fold_config["data"]["num_workers"]),
            seed + 10_000,
            device,
            AnnotatedCycleSampler(
                len(annotated_dataset), steps, seed + 10_000, include_metadata=True
            ),
        )
    model = build_model(fold_config).to(device)
    optimizer = create_optimizer(
        model,
        float(fold_config["training"]["weight_decay"]),
        float(fold_config["training"]["backbone_learning_rate"]),
        float(fold_config["training"]["head_learning_rate"]),
    )
    streams = TrainingRngStreams(seed + 30_000, seed + 40_000)
    weights = LossWeights(
        region=1.0 if adapter.region_enabled else 0.0,
        attention=(0.0 if fold_config["arm"]["beta_mode"] == "zero" else 1.0),
    )
    natural_iterator = iter(natural_loader)
    annotated_iterator = (
        iter(annotated_loader) if annotated_loader is not None else None
    )
    torch.cuda.reset_peak_memory_stats(device)
    step_seconds: list[float] = []
    natural_seconds: list[float] = []
    annotated_seconds: list[float] = []
    optimizer_seconds: list[float] = []
    for _ in range(steps):
        natural_batch = next(natural_iterator)
        annotated_batch = (
            next(annotated_iterator) if annotated_iterator is not None else None
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = train_step(
            model,
            adapter,
            natural_batch,
            annotated_batch,
            optimizer,
            device,
            streams,
            weights,
            pos_weight=float(fold_config["training"]["pos_weight"]),
            mixup_probability=float(fold_config["training"]["mixup_probability"]),
            gradient_clip_norm=float("inf"),
            profile_timing=True,
        )
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        natural_seconds.append(result.natural_seconds)
        annotated_seconds.append(result.annotated_seconds)
        optimizer_seconds.append(result.optimizer_seconds)
    measured = slice(warmup_steps, None)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    properties = torch.cuda.get_device_properties(device)
    return {
        "protocol_version": "fracture-resource-profile-v1",
        "arm": str(fold_config["arm"]["name"]),
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
        "config_sha256": canonical_sha256(normalized_config(fold_config)),
        "outer_fold": outer_fold,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "natural_batch_size": int(fold_config["training"]["natural_batch_size"]),
        "annotated_batch_size": (
            int(fold_config["training"]["annotated_batch_size"])
            if adapter.region_enabled
            else 0
        ),
        "beta_mode": str(fold_config["arm"]["beta_mode"]),
        "attention_computed": adapter.attention_enabled,
        "parameters": parameter_breakdown(model),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "median_step_seconds": statistics.median(step_seconds[measured]),
        "median_natural_seconds": statistics.median(natural_seconds[measured]),
        "median_annotated_seconds": statistics.median(annotated_seconds[measured]),
        "median_optimizer_seconds": statistics.median(optimizer_seconds[measured]),
        "gpu": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
        },
        "resource_limit_bytes": RESOURCE_LIMIT_BYTES,
        "resource_gate_passed": peak_reserved <= RESOURCE_LIMIT_BYTES,
    }


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """全parameterとProposed主要component数を返す。"""
    result = {"total": sum(value.numel() for value in model.parameters())}
    for name in (
        "attention_modules",
        "region_branches",
        "whole_branch",
        "shared_blocks",
    ):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            result[name] = sum(value.numel() for value in module.parameters())
    temporal = sum(
        value.numel()
        for name, value in model.named_parameters()
        if name.startswith("lstm.")
        or ".lstm." in name
        or name.startswith(("whole_head.", "region_head."))
        or ".head." in name
    )
    result["temporal_and_heads"] = temporal
    return result
