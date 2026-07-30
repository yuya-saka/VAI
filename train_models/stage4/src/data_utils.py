"""Stage4 data collection and frozen-fold helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train_models.stage2.src.data_utils import (
    _deep_merge,
    _worker_options,
    save_effective_config,
    seed_worker,
    set_seed,
)
from train_models.stage2.src.data_utils import (
    collect_items as collect_stage2_items,
)
from train_models.stage2.src.dataset import get_train_transforms
from train_models.stage3.src.data_utils import normalize_squeeze_excite_conv_strides

from .batch_sampler import Stage4StratifiedBatchSampler
from .dataset import RSNAStage4Dataset
from .model import Stage4Model
from .negative_sampler import NegativeRegionSampler
from .region_labels import (
    RegionLabelMap,
    load_region_labels,
    region_supervision_of,
)
from .stage4_folds import load_stage4_fold_map, split_by_stage4_fold

__all__ = [
    "collect_items",
    "create_data_loaders",
    "create_eval_data_loader",
    "create_model_optimizer_scheduler",
    "load_config",
    "load_region_labels",
    "load_stage4_fold_map",
    "save_effective_config",
    "set_seed",
    "split_by_stage4_fold",
]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a Stage4 YAML config with optional relative `_base` inheritance."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).parent.parent / "config" / "stage4_mixed.yaml"
    )
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    base = config.pop("_base", None)
    if base is None:
        return config
    return _deep_merge(load_config(path.parent / str(base)), config)


def collect_items(
    dataset_dir: Path,
    csv_path: Path,
    region_labels_path: Path,
    excluded_studies_path: Path | None = None,
    excluded_levels_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect Stage2 arrays and attach Stage4 region supervision fields."""
    items = collect_stage2_items(
        dataset_dir,
        csv_path,
        excluded_studies_path,
        excluded_levels_path,
    )
    region_labels: RegionLabelMap = load_region_labels(region_labels_path)
    zero_label = np.zeros(4, dtype=np.int8)
    enriched: list[dict[str, Any]] = []
    for item in items:
        key = (str(item["study_uid"]), str(item["vertebra"]))
        supervision = region_supervision_of(
            int(item["label"]),
            key,
            region_labels,
        )
        enriched.append(
            {
                **item,
                "region_label": region_labels.get(key, zero_label).copy(),
                "region_supervision": supervision,
            }
        )
    counts = {
        name: sum(item["region_supervision"] == name for item in enriched)
        for name in ("strong", "weak", "negative")
    }
    print(f"[INFO] Stage4 supervision={counts}", flush=True)
    return enriched


def create_data_loaders(
    train_items: list[dict[str, Any]],
    valid_items: list[dict[str, Any]],
    config: dict[str, Any],
    fold_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create Stage4 stratified training and full-fold validation loaders."""
    training = config.get("training", {})
    model = config.get("model", {})
    batch_size = int(training.get("batch_size", 8))
    strong_per_batch = int(training.get("strong_per_batch", 2))
    weak_per_batch = int(training.get("weak_per_batch", 2))
    negative_per_batch = int(training.get("negative_per_batch", 4))
    workers = int(training.get("num_workers", 4))
    region_mode = str(model.get("region_mode", "masked"))
    scramble_seed = int(model.get("scramble_seed", 42))
    strong_items = [
        item for item in train_items if item["region_supervision"] == "strong"
    ]
    negative_items = [
        item for item in train_items if item["region_supervision"] == "negative"
    ]
    negative_sampler = NegativeRegionSampler(
        strong_items,
        negative_items,
        fold_dir,
        seed=int(training.get("negative_sampler_seed", 42)),
        write_manifest=rank == 0,
    )
    train_dataset = RSNAStage4Dataset(
        train_items,
        mode="train",
        transform=get_train_transforms(config.get("augmentation", {})),
        p_rand_order=float(training.get("p_rand_order", 0.0)),
        region_mode=region_mode,
        scramble_seed=scramble_seed,
    )
    batch_sampler = Stage4StratifiedBatchSampler(
        train_items,
        negative_sampler,
        batch_size=batch_size,
        strong_per_batch=strong_per_batch,
        weak_per_batch=weak_per_batch,
        negative_per_batch=negative_per_batch,
        rank=rank,
        world_size=world_size,
        seed=int(config.get("data", {}).get("random_seed", 42)),
        positive_weight=float(training.get("positive_weight", 2.0)),
    )
    valid_dataset = RSNAStage4Dataset(
        valid_items,
        mode="valid",
        include_metadata=True,
        p_rand_order=0.0,
        region_mode=region_mode,
        scramble_seed=scramble_seed,
    )
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        **_worker_options(training),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        **common,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    strong_labels = np.stack([item["region_label"] for item in strong_items])
    from train_models.stage4.utils.losses import compute_region_pos_weight

    pos_weight = compute_region_pos_weight(
        strong_labels,
        n_negative_sampled=len(strong_items),
    )
    return train_loader, valid_loader, pos_weight


def create_eval_data_loader(
    items: list[dict[str, Any]],
    config: dict[str, Any],
) -> DataLoader:
    """Create a metadata-bearing Stage4 inference loader."""
    training = config.get("training", {})
    model = config.get("model", {})
    return DataLoader(
        RSNAStage4Dataset(
            items,
            mode="valid",
            include_metadata=True,
            p_rand_order=0.0,
            region_mode=str(model.get("region_mode", "masked")),
            scramble_seed=int(model.get("scramble_seed", 42)),
        ),
        batch_size=int(training.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=seed_worker,
        **_worker_options(training),
    )


def create_model_optimizer_scheduler(
    config: dict[str, Any],
    device: torch.device,
    class_prior: float,
) -> tuple[Stage4Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the Stage4 model and two-group AdamW optimizer."""
    data = config.get("data", {})
    model_config = config.get("model", {})
    training = config.get("training", {})
    model = Stage4Model(
        backbone=str(model_config.get("backbone", "tf_efficientnetv2_s")),
        in_chans=int(data.get("in_channels", 6)),
        n_slices=int(data.get("n_slices", 15)),
        n_regions=int(data.get("n_regions", 4)),
        fpn_channels=int(model_config.get("fpn_channels", 256)),
        region_hidden=int(model_config.get("region_hidden", 256)),
        region_layers=int(model_config.get("region_layers", 2)),
        pretrained=bool(model_config.get("pretrained", True)),
        region_mode=str(model_config.get("region_mode", "masked")),
        slice_agg=str(model_config.get("slice_agg", "tied_attention")),
        region_agg=str(model_config.get("region_agg", "normalized_smoothmax")),
        attention_temperature=float(model_config.get("attention_temperature", 1.0)),
        slice_lse_temperature=float(model_config.get("slice_lse_temperature", 1.0)),
        region_temperature=float(model_config.get("region_temperature", 1.0)),
        class_prior=class_prior,
    ).to(device, memory_format=torch.channels_last)
    normalize_squeeze_excite_conv_strides(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone_parameters(),
                "lr": float(training.get("backbone_learning_rate", 2.3e-4)),
            },
            {
                "params": model.head_parameters(),
                "lr": float(training.get("head_learning_rate", 2.3e-4)),
            },
        ],
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training.get("fixed_epochs", 75)),
        eta_min=float(training.get("eta_min", 2.3e-5)),
    )
    return model, optimizer, scheduler
