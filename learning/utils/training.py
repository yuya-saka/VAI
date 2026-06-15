"""Shared training utilities for fracture classification experiments."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader

from learning.src.dataset import FractureDataset
from learning.src.model import FractureResNet18
from learning.utils.sampler import make_weighted_sampler

Bag = dict[str, Any]
BagBatch = tuple[list[Tensor], list[int], list[str], list[str]]


def resolve_device(gpu_id: int) -> torch.device:
    """Select the configured CUDA device when available."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def build_model(
    training_config: dict[str, Any],
    device: torch.device,
    *,
    default_dropout: float,
) -> FractureResNet18:
    """Build the supported fracture-classification model."""
    backbone_name = training_config.get("backbone", "resnet18")
    if backbone_name != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    conditioning = training_config.get("vertebra_conditioning", "one_hot")
    if conditioning != "one_hot":
        raise ValueError(f"Unsupported vertebra_conditioning: {conditioning}")

    return FractureResNet18(
        dropout=training_config.get("dropout", default_dropout),
        pretrained=training_config.get("pretrained", True),
        freeze_batch_norm=training_config.get("freeze_batch_norm", True),
    ).to(device)


def build_data_loaders(
    train_bags: list[Bag],
    val_bags: list[Bag],
    training_config: dict[str, Any],
    augmentation_config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build bag-wise training and validation loaders."""
    prefetch = training_config.get("prefetch_to_ram", True)
    train_dataset = FractureDataset(
        train_bags,
        training=True,
        aug_cfg=augmentation_config,
        prefetch_to_ram=prefetch,
    )
    val_dataset = FractureDataset(
        val_bags,
        training=False,
        prefetch_to_ram=prefetch,
    )
    train_sampler = make_weighted_sampler([bag["label"] for bag in train_bags])
    loader_kwargs = {
        "batch_size": 1,
        "num_workers": training_config.get("num_workers", 4),
        "pin_memory": True,
        "collate_fn": collate_bags,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def make_optimizer(
    model: FractureResNet18,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create AdamW with separate backbone and head learning rates."""
    return torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": backbone_lr},
            {"params": model.head_parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def lr_scale(epoch: int, warmup_epochs: int) -> float:
    """Linearly warm up from 20% and then keep a constant rate."""
    if epoch >= warmup_epochs:
        return 1.0
    return 0.2 + 0.8 * (epoch / max(1, warmup_epochs - 1))


def compute_ranking_metrics(
    labels: list[int],
    probabilities: list[float],
) -> tuple[float, float]:
    """Compute AUPRC and AUROC, returning zeros for single-class input."""
    if len(set(labels)) <= 1:
        return 0.0, 0.0
    auprc = float(average_precision_score(labels, probabilities))
    auroc = float(roc_auc_score(labels, probabilities))
    return auprc, auroc


def collate_bags(batch: list[tuple[Tensor, int, str, str]]) -> BagBatch:
    """Keep variable-length bags as lists instead of stacking them."""
    stacks, labels, samples, vertebrae = zip(*batch, strict=True)
    return list(stacks), list(labels), list(samples), list(vertebrae)
