"""Configuration, data collection, splitting, and object factories."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import RSNARegionDataset, get_train_transforms
from .model import Stage2Model

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def set_seed(seed: int = 42) -> None:
    """Seed process-local random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker and limit nested threading."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the Stage2 YAML configuration, applying an optional `_base` parent.

    A config may set a top-level ``_base: <filename>`` (resolved relative to
    its own directory) to inherit from another config; its own keys are then
    deep-merged on top, so an ablation config only needs to list overrides.
    """
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).parent.parent / "config" / "config.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    base_name = config.pop("_base", None)
    if base_name is None:
        return config
    base_config = load_config(path.parent / str(base_name))
    return _deep_merge(base_config, config)


def save_effective_config(config: dict[str, Any], output_dir: Path) -> Path:
    """Persist the configuration after applying CLI overrides."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config.yaml"
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return output_path


def _load_exclusions(
    studies_path: Path | None,
    levels_path: Path | None,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Load excluded_studies.csv and excluded_levels.csv if they exist."""
    excluded_studies: set[str] = set()
    excluded_levels: set[tuple[str, str]] = set()
    if studies_path is not None and studies_path.exists():
        df = pd.read_csv(studies_path)
        excluded_studies = set(df["study_uid"].astype(str))
        print(f"[INFO] excluded_studies={len(excluded_studies)}")
    if levels_path is not None and levels_path.exists():
        df = pd.read_csv(levels_path)
        excluded_levels = set(
            zip(
                df["study_uid"].astype(str),
                df["vertebra"].astype(str),
                strict=True,
            )
        )
        print(f"[INFO] excluded_levels={len(excluded_levels)}")
    return excluded_studies, excluded_levels


def collect_items(
    dataset_dir: Path,
    csv_path: Path,
    excluded_studies_path: Path | None = None,
    excluded_levels_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect samples that contain all required Stage2 array files."""
    labels = pd.read_csv(csv_path)
    excluded_studies, excluded_levels = _load_exclusions(
        excluded_studies_path, excluded_levels_path
    )
    items: list[dict[str, Any]] = []
    missing = {
        "study": 0,
        "excluded_study": 0,
        "excluded_level": 0,
        "ct": 0,
        "vertebra_mask": 0,
        "region_mask": 0,
    }

    for _, row in labels.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        if study_uid in excluded_studies:
            missing["excluded_study"] += 1
            continue
        study_dir = dataset_dir / study_uid
        if not study_dir.exists():
            missing["study"] += 1
            continue
        for vertebra in VERTEBRAE:
            if (study_uid, vertebra) in excluded_levels:
                missing["excluded_level"] += 1
                continue
            level_dir = study_dir / vertebra
            ct_path = level_dir / "ct.npy"
            mask_path = level_dir / "vertebra_mask.npy"
            region_mask_path = level_dir / "region_4class.npy"
            if not ct_path.exists():
                missing["ct"] += 1
                continue
            if not mask_path.exists():
                missing["vertebra_mask"] += 1
                continue
            if not region_mask_path.exists():
                missing["region_mask"] += 1
                continue
            items.append(
                {
                    "study_uid": study_uid,
                    "vertebra": vertebra,
                    "label": int(row[vertebra]),
                    "ct_path": ct_path,
                    "mask_path": mask_path,
                    "region_mask_path": region_mask_path,
                }
            )

    positives = sum(int(item["label"]) for item in items)
    print(
        f"[INFO] Stage2 items={len(items)} positive={positives} "
        f"negative={len(items) - positives} missing={missing}"
    )
    return items


def _study_labels(items: list[dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for item in items:
        study_uid = str(item["study_uid"])
        labels[study_uid] = max(labels.get(study_uid, 0), int(item["label"]))
    return labels


def split_test_holdout(
    items: list[dict[str, Any]],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a fixed study-level holdout with Stage1-compatible semantics."""
    study_labels = _study_labels(items)
    study_uids = np.asarray(list(study_labels))
    labels = np.asarray([study_labels[study_uid] for study_uid in study_uids])
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_indices, test_indices = next(splitter.split(study_uids, labels))
    train_uids = set(study_uids[train_indices])
    test_uids = set(study_uids[test_indices])
    return (
        [item for item in items if item["study_uid"] in train_uids],
        [item for item in items if item["study_uid"] in test_uids],
    )


def split_items_cv(
    items: list[dict[str, Any]],
    n_splits: int = 5,
    val_fold: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples by study with stratification on study fracture status."""
    study_labels = _study_labels(items)
    study_uids = list(study_labels)
    labels = np.asarray([study_labels[study_uid] for study_uid in study_uids])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(study_uids, labels, groups=study_uids))
    train_indices, val_indices = splits[val_fold]
    train_uids = {study_uids[index] for index in train_indices}
    val_uids = {study_uids[index] for index in val_indices}
    return (
        [item for item in items if item["study_uid"] in train_uids],
        [item for item in items if item["study_uid"] in val_uids],
    )


def _worker_options(config: dict[str, Any]) -> dict[str, Any]:
    workers = int(config.get("num_workers", 4))
    if workers == 0:
        return {}
    return {
        "persistent_workers": bool(config.get("persistent_workers", True)),
        "prefetch_factor": int(config.get("prefetch_factor", 4)),
    }


def create_data_loaders(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    workers = int(training_config.get("num_workers", 4))
    batch_size = int(training_config.get("batch_size", 4))
    train_dataset = RSNARegionDataset(
        train_items,
        mode="train",
        transform=get_train_transforms(config.get("augmentation", {})),
        p_rand_order=float(training_config.get("p_rand_order", 0.2)),
        include_metadata=False,
    )
    val_dataset = RSNARegionDataset(
        val_items, mode="valid", include_metadata=True, p_rand_order=0.0
    )
    generator = torch.Generator().manual_seed(int(data_config.get("random_seed", 42)))
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        if dist.is_initialized()
        else None
    )
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": generator,
        **_worker_options(training_config),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def create_eval_data_loader(
    items: list[dict[str, Any]], config: dict[str, Any]
) -> DataLoader:
    """Create a metadata-bearing inference DataLoader."""
    training_config = config.get("training", {})
    return DataLoader(
        RSNARegionDataset(items, mode="valid", include_metadata=True, p_rand_order=0.0),
        batch_size=int(training_config.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=seed_worker,
        **_worker_options(training_config),
    )


def create_model_optimizer_scheduler(
    config: dict[str, Any], device: torch.device
) -> tuple[Stage2Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the model, differential-LR optimizer, and cosine scheduler."""
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    model = Stage2Model(
        backbone=str(model_config.get("backbone", "tf_efficientnetv2_s")),
        in_chans=int(data_config.get("in_channels", 6)),
        n_slices=int(data_config.get("n_slices", 15)),
        n_regions=int(data_config.get("n_regions", 4)),
        drop_rate=float(model_config.get("drop_rate", 0.0)),
        drop_path_rate=float(model_config.get("drop_path_rate", 0.0)),
        drop_rate_last=float(model_config.get("drop_rate_last", 0.3)),
        lstm_hidden=int(model_config.get("lstm_hidden", 256)),
        lstm_layers=int(model_config.get("lstm_layers", 2)),
        fpn_channels=int(model_config.get("fpn_channels", 256)),
        region_hidden=int(model_config.get("region_hidden", 256)),
        region_layers=int(model_config.get("region_layers", 2)),
        region_dropout=float(model_config.get("region_dropout", 0.3)),
        pretrained=bool(model_config.get("pretrained", True)),
        force_primary_fp32=bool(training_config.get("force_primary_fp32", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone_parameters(),
                "lr": float(training_config.get("backbone_learning_rate", 2.3e-5)),
            },
            {
                "params": model.head_parameters(),
                "lr": float(training_config.get("head_learning_rate", 2.3e-4)),
            },
        ],
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config.get("epochs", 75)),
        eta_min=float(training_config.get("eta_min", 2.3e-6)),
    )
    return model, optimizer, scheduler
