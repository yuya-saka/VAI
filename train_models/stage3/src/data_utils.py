"""Stage3 configuration, loader, and model factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from train_models.stage2.src.data_utils import (
    _deep_merge,
    _worker_options,
    collect_items,
    save_effective_config,
    seed_worker,
    set_seed,
    split_items_cv,
    split_test_holdout,
)
from train_models.stage2.src.dataset import get_train_transforms

from .dataset import RSNAStage3Dataset
from .model import Stage3Model

__all__ = [
    "collect_items",
    "create_data_loaders",
    "create_eval_data_loader",
    "create_model_optimizer_scheduler",
    "load_config",
    "save_effective_config",
    "set_seed",
    "split_items_cv",
    "split_test_holdout",
]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).parent.parent / "config" / "config.yaml"
    )
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    base = config.pop("_base", None)
    if base is None:
        return config
    return _deep_merge(load_config(path.parent / str(base)), config)


def create_data_loaders(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    training, model = config.get("training", {}), config.get("model", {})
    workers = int(training.get("num_workers", 4))
    common = {
        "batch_size": int(training.get("batch_size", 4)),
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        **_worker_options(training),
    }
    region_mode = str(model.get("region_mode", "masked"))
    seed = int(model.get("scramble_seed", 42))
    train = RSNAStage3Dataset(
        train_items,
        mode="train",
        transform=get_train_transforms(config.get("augmentation", {})),
        p_rand_order=float(training.get("p_rand_order", 0.0)),
        region_mode=region_mode,
        scramble_seed=seed,
    )
    valid = RSNAStage3Dataset(
        val_items,
        mode="valid",
        include_metadata=True,
        p_rand_order=0.0,
        region_mode=region_mode,
        scramble_seed=seed,
    )
    generator = torch.Generator().manual_seed(
        int(config.get("data", {}).get("random_seed", 42))
    )
    sampler = (
        DistributedSampler(
            train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        if dist.is_initialized()
        else None
    )
    return DataLoader(
        train,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=generator,
        **common,
    ), DataLoader(valid, shuffle=False, generator=generator, **common)


def create_eval_data_loader(
    items: list[dict[str, Any]], config: dict[str, Any]
) -> DataLoader:
    """Create a metadata-bearing Stage3 inference DataLoader."""
    training = config.get("training", {})
    model = config.get("model", {})
    region_mode = str(model.get("region_mode", "masked"))
    scramble_seed = int(model.get("scramble_seed", 42))
    return DataLoader(
        RSNAStage3Dataset(
            items,
            mode="valid",
            include_metadata=True,
            p_rand_order=0.0,
            region_mode=region_mode,
            scramble_seed=scramble_seed,
        ),
        batch_size=int(training.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=seed_worker,
        **_worker_options(training),
    )


@torch.no_grad()
def normalize_squeeze_excite_conv_strides(
    model: Stage3Model, probe_size: int = 64
) -> list[str]:
    """Reset squeeze-excite 1x1 conv weights to contiguous strides for DDP parity.

    After ``model.to(memory_format=channels_last)`` every 1x1 conv weight keeps a
    channels_last stride. Under CUDA/BF16 backward, only convs whose input is
    1x1-spatial (squeeze-excite gates) produce a contiguous-strided gradient;
    that mismatches the channels_last bucket DDP builds from the weight, emitting
    a benign stride warning. Spatial 1x1 convs (pointwise / FPN) keep
    channels_last gradients, so they must stay channels_last. ``contiguous()`` is
    a no-op for these ambiguous singleton-spatial tensors, so we clone into
    contiguous_format. A single probe forward identifies the squeeze-excite convs
    by their 1x1-spatial input. Must run after ``.to(channels_last)`` and before
    DDP construction.
    """
    device = next(model.parameters()).device
    squeeze_convs: set[str] = set()
    handles = []

    def make_hook(name: str):
        def hook(module: torch.nn.Module, inputs: tuple, output: object) -> None:
            if tuple(inputs[0].shape[-2:]) == (1, 1):
                squeeze_convs.add(name)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and tuple(module.weight.shape[-2:]) == (
            1,
            1,
        ):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    images = torch.randn(
        1, model.n_slices, model.in_chans, probe_size, probe_size, device=device
    )
    regions = torch.randint(
        0, model.n_regions + 1, (1, model.n_slices, probe_size, probe_size), device=device
    )
    model(images, regions)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    normalized: list[str] = []
    for name, module in model.named_modules():
        if name in squeeze_convs:
            weight = module.weight
            weight.set_(weight.detach().clone(memory_format=torch.contiguous_format))
            normalized.append(f"{name}.weight")
    return normalized


def create_model_optimizer_scheduler(
    config: dict[str, Any], device: torch.device, class_prior: float
) -> tuple[Stage3Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    data, model_cfg, training = (
        config.get("data", {}),
        config.get("model", {}),
        config.get("training", {}),
    )
    model = Stage3Model(
        backbone=str(model_cfg.get("backbone", "tf_efficientnetv2_s")),
        in_chans=int(data.get("in_channels", 6)),
        n_slices=int(data.get("n_slices", 15)),
        n_regions=int(data.get("n_regions", 4)),
        fpn_channels=int(model_cfg.get("fpn_channels", 256)),
        region_hidden=int(model_cfg.get("region_hidden", 256)),
        region_layers=int(model_cfg.get("region_layers", 2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        region_mode=str(model_cfg.get("region_mode", "masked")),
        slice_agg=str(model_cfg.get("slice_agg", "tied_attention")),
        region_agg=str(model_cfg.get("region_agg", "normalized_smoothmax")),
        attention_temperature=float(model_cfg.get("attention_temperature", 1.0)),
        slice_lse_temperature=float(model_cfg.get("slice_lse_temperature", 1.0)),
        region_temperature=float(model_cfg.get("region_temperature", 1.0)),
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
        T_max=int(training.get("epochs", 75)),
        eta_min=float(training.get("eta_min", 2.3e-5)),
    )
    return model, optimizer, scheduler
