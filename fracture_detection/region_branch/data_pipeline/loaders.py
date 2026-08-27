"""outer foldごとのnatural stream + 3ソース補助loaderの構築。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fracture_detection.baseline0.data.sampling import EpochShuffleSampler
from fracture_detection.baseline0.data.splits import split_nested_manifest
from fracture_detection.baseline0.training.trainer import create_data_loader
from fracture_detection.region_branch.data_pipeline.dataset import RegionBranchDataset
from fracture_detection.region_branch.data_pipeline.pseudo_labels import (
    attach_teacher_scores,
    load_pseudo_scores,
    load_pseudo_temperatures,
)
from fracture_detection.region_branch.data_pipeline.sampling import create_source_loader
from fracture_detection.region_branch.data_pipeline.sources import (
    SourcePools,
    split_source_pools,
)


@dataclass(frozen=True)
class OuterFoldLoaders:
    """1 outer foldの学習に使うnatural stream + 3ソース補助DataLoader。"""

    natural: DataLoader[Any]
    human: DataLoader[Any]
    negative: DataLoader[Any]
    pseudo: DataLoader[Any]
    temperatures: torch.Tensor
    steps_per_epoch: int
    pools: SourcePools


def build_outer_fold_loaders(
    manifest: pd.DataFrame,
    outer_fold: int,
    dataset_dir: Path,
    pseudo_label_dir: Path,
    natural_batch_size: int,
    human_bags_per_batch: int,
    negative_bags_per_batch: int,
    pseudo_bags_per_batch: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    train_transform: A.ReplayCompose | None,
) -> OuterFoldLoaders:
    """train manifestからnatural stream loaderと3ソース補助loaderを作る。

    補助loaderのstep数はnatural streamの`steps_per_epoch`（=len(natural_loader)）
    へ固定する。mixupはnatural streamだけに適用するため、ここではtransformだけを
    3ソース共通で渡す（呼び出し側でmixupは適用しない）。
    """
    train_manifest, _inner, _outer = split_nested_manifest(manifest, outer_fold)
    pools = split_source_pools(train_manifest)
    pseudo_scores = load_pseudo_scores(pseudo_label_dir, outer_fold)
    temperatures = load_pseudo_temperatures(pseudo_label_dir, outer_fold)
    pseudo_manifest = attach_teacher_scores(pools.pseudo, pseudo_scores)

    natural_dataset = RegionBranchDataset(
        train_manifest, dataset_dir=dataset_dir, transform=train_transform
    )
    natural_sampler = EpochShuffleSampler(natural_dataset, seed=seed)
    natural_loader = create_data_loader(
        natural_dataset,
        natural_batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
        sampler=natural_sampler,
    )
    steps_per_epoch = len(natural_loader)

    human_loader = create_source_loader(
        RegionBranchDataset(
            pools.human, dataset_dir=dataset_dir, transform=train_transform
        ),
        human_bags_per_batch,
        steps_per_epoch,
        seed=seed + 10_000,
        num_workers=num_workers,
        device=device,
    )
    negative_loader = create_source_loader(
        RegionBranchDataset(
            pools.negative, dataset_dir=dataset_dir, transform=train_transform
        ),
        negative_bags_per_batch,
        steps_per_epoch,
        seed=seed + 20_000,
        num_workers=num_workers,
        device=device,
    )
    pseudo_loader = create_source_loader(
        RegionBranchDataset(
            pseudo_manifest, dataset_dir=dataset_dir, transform=train_transform
        ),
        pseudo_bags_per_batch,
        steps_per_epoch,
        seed=seed + 30_000,
        num_workers=num_workers,
        device=device,
    )
    return OuterFoldLoaders(
        natural=natural_loader,
        human=human_loader,
        negative=negative_loader,
        pseudo=pseudo_loader,
        temperatures=temperatures.to(device),
        steps_per_epoch=steps_per_epoch,
        pools=pools,
    )


def build_eval_loader(
    manifest_subset: pd.DataFrame,
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[Any]:
    """augmentationなしのinner/outer評価用DataLoaderを作る。"""
    dataset = RegionBranchDataset(
        manifest_subset, dataset_dir=dataset_dir, transform=None
    )
    return create_data_loader(
        dataset, batch_size, num_workers=num_workers, seed=seed, device=device
    )
