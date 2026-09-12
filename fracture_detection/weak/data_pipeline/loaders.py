"""Outer-fold DataLoader construction: GT-pass train stream + natural eval.

Inner/outer evaluation reuses ``baseline0.training.trainer.create_data_loader``
unchanged (natural order, no sampler, single pass over every row). The GT-pass
training stream needs a ``batch_sampler`` (whole batches of pre-composed N/A/U
row indices, see ``sampling.GtPassBatchSampler``), which
``create_data_loader``'s ``batch_size``/``sampler`` signature cannot express,
so a small dedicated constructor is defined here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fracture_detection.baseline0.data.splits import split_nested_manifest
from fracture_detection.baseline0.training.trainer import (
    create_data_loader,
    seed_worker,
)
from fracture_detection.weak.data_pipeline.dataset import WeakBagDataset
from fracture_detection.weak.data_pipeline.groups import (
    group_row_indices,
    resolve_bag_groups,
)
from fracture_detection.weak.data_pipeline.sampling import GtPassBatchSampler


@dataclass(frozen=True)
class OuterFoldLoaders:
    """Train (GT-pass), inner, and outer natural-distribution loaders."""

    train_dataset: WeakBagDataset
    train_sampler: GtPassBatchSampler
    train_loader: DataLoader[Any]
    inner_loader: DataLoader[Any]
    outer_loader: DataLoader[Any]


def create_weak_train_loader(
    dataset: WeakBagDataset,
    batch_sampler: GtPassBatchSampler,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    """Build the GT-pass DataLoader driven by a whole-batch sampler."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def build_outer_fold_loaders(
    manifest: pd.DataFrame,
    outer_fold: int,
    dataset_dir: Path,
    negative_per_batch: int,
    annotated_per_batch: int,
    weak_per_batch: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    train_transform: A.ReplayCompose | None,
    eval_batch_size: int,
) -> OuterFoldLoaders:
    """Build the train GT-pass loader and natural inner/outer eval loaders."""
    train_manifest, inner_manifest, outer_manifest = split_nested_manifest(
        manifest, outer_fold
    )
    train_dataset = WeakBagDataset(
        train_manifest, dataset_dir=dataset_dir, transform=train_transform
    )
    resolved_train = resolve_bag_groups(train_manifest)
    indices = group_row_indices(resolved_train)
    train_sampler = GtPassBatchSampler(
        indices,
        negative_per_batch=negative_per_batch,
        annotated_per_batch=annotated_per_batch,
        weak_per_batch=weak_per_batch,
        seed=seed,
    )
    train_loader = create_weak_train_loader(
        train_dataset, train_sampler, num_workers=num_workers, seed=seed
    )

    inner_dataset = WeakBagDataset(inner_manifest, dataset_dir=dataset_dir)
    outer_dataset = WeakBagDataset(outer_manifest, dataset_dir=dataset_dir)
    inner_loader = create_data_loader(
        inner_dataset,
        eval_batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    outer_loader = create_data_loader(
        outer_dataset,
        eval_batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    return OuterFoldLoaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        train_loader=train_loader,
        inner_loader=inner_loader,
        outer_loader=outer_loader,
    )
