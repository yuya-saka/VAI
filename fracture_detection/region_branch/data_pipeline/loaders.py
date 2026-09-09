"""outer foldごとのnatural stream DataLoaderの構築。

`region`は`whole`と同じnatural-distribution stream一本を共有する
（baseline0と同一のbatch構成）。region forward時に同じbatchのwhole陽性bagだけを
抽出し、GT/pseudo統一targetへ条件付きregion lossを計算する。
"""

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
    attach_pseudo_targets,
    load_pseudo_region_targets,
    select_pseudo_targets_for_teacher,
    shuffle_pseudo_target_associations,
)

PSEUDO_ARMS = ("no_pseudo", "cam_soft", "cam_soft_shuffled")


@dataclass(frozen=True)
class OuterFoldLoaders:
    """1 outer foldの学習に使うnatural stream DataLoader。"""

    natural: DataLoader[Any]
    steps_per_epoch: int


def build_outer_fold_loaders(
    manifest: pd.DataFrame,
    outer_fold: int,
    dataset_dir: Path,
    pseudo_label_dir: Path | None,
    pseudo_arm: str,
    natural_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    train_transform: A.ReplayCompose | None,
) -> OuterFoldLoaders:
    """train manifestからnatural stream loaderを作る。

    `pseudo_arm=="no_pseudo"`ならCAM soft targetを結合せず、
    `RegionBranchDataset`は`require_pseudo_targets=False`で構築する（human-unknown
    cellはすべてpseudo_valid=Falseになる）。`cam_soft`/`cam_soft_shuffled`は
    fold一致teacherのCAM soft targetを結合し、`require_pseudo_targets=True`で
    欠落を即座にエラーにする。`cam_soft_shuffled`は固定seedでbag↔targetベクトルの
    対応だけを置換する（target値の集合・有効セル母集団は変えない）。
    """
    train_manifest, _inner, _outer = split_nested_manifest(manifest, outer_fold)

    if pseudo_arm == "no_pseudo":
        require_pseudo_targets = False
    elif pseudo_arm in ("cam_soft", "cam_soft_shuffled"):
        require_pseudo_targets = True
        if pseudo_label_dir is None:
            raise ValueError(f"pseudo_arm={pseudo_arm!r}にはpseudo_label_dirが必要です")
        all_pseudo_targets = load_pseudo_region_targets(pseudo_label_dir, outer_fold)
        pseudo_targets = select_pseudo_targets_for_teacher(
            all_pseudo_targets, outer_fold
        )
        if pseudo_arm == "cam_soft_shuffled":
            pseudo_targets = shuffle_pseudo_target_associations(
                pseudo_targets, seed=seed
            )
        train_manifest = attach_pseudo_targets(train_manifest, pseudo_targets)
    else:
        raise ValueError(
            f"unknown pseudo_arm: {pseudo_arm!r} (expected one of {PSEUDO_ARMS})"
        )

    natural_dataset = RegionBranchDataset(
        train_manifest,
        dataset_dir=dataset_dir,
        transform=train_transform,
        require_pseudo_targets=require_pseudo_targets,
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
    return OuterFoldLoaders(
        natural=natural_loader,
        steps_per_epoch=len(natural_loader),
    )


def build_eval_loader(
    manifest_subset: pd.DataFrame,
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    pseudo_targets: pd.DataFrame | None = None,
    require_pseudo_targets: bool = False,
) -> DataLoader[Any]:
    """augmentationなしの評価用DataLoaderを作る。"""
    if require_pseudo_targets and pseudo_targets is None:
        raise ValueError("require_pseudo_targets=Trueにはpseudo_targetsが必要です")
    resolved_manifest = (
        attach_pseudo_targets(manifest_subset, pseudo_targets)
        if pseudo_targets is not None
        else manifest_subset
    )
    dataset = RegionBranchDataset(
        resolved_manifest,
        dataset_dir=dataset_dir,
        transform=None,
        require_pseudo_targets=require_pseudo_targets,
    )
    return create_data_loader(
        dataset, batch_size, num_workers=num_workers, seed=seed, device=device
    )
