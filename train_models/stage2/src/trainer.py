"""Training, validation, checkpointing, and inference for Stage2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from train_models.stage1.src.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
)
from train_models.stage2.utils.losses import (
    mixup_batch,
    region_any_probability,
    region_log_survival,
    stage2_loss,
)

from .data_utils import (
    create_data_loaders,
    create_eval_data_loader,
    create_model_optimizer_scheduler,
    split_items_cv,
)
from .evaluation import compute_metrics, compute_region_diagnostics, region_names_for
from .experiment import (
    append_jsonl,
    prune_training_jsonl,
    resolve_fold_paths,
    validate_resume_config,
)
from .model import STAGE2_ARCHITECTURE_VERSION, Stage2Model, Stage2Output


def _prepare_images(images: Tensor, device: torch.device) -> Tensor:
    """Transfer uint8 images and normalize them on the target device."""
    return images.to(device, non_blocking=True).float().div_(255.0)


def _prepare_regions(regions: Tensor, device: torch.device) -> Tensor:
    """Transfer integer region masks without normalization."""
    return regions.to(device, non_blocking=True)


def _base_model(model: torch.nn.Module) -> Stage2Model:
    """Unwrap DDP when checkpointing."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, Stage2Model):
        raise TypeError(f"unexpected model type: {type(model)}")
    return model


def _amp_settings(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[bool, torch.dtype]:
    """Resolve autocast enablement and dtype with an explicit BF16 safety check."""
    training_config = config.get("training", {})
    enabled = bool(training_config.get("use_amp", True)) and device.type == "cuda"
    dtype_name = str(training_config.get("amp_dtype", "bfloat16")).lower()
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported amp_dtype: {dtype_name}")
    dtype = dtype_by_name[dtype_name]
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "amp_dtype=bfloat16 requires a BF16-capable CUDA device; "
            "set training.use_amp=false for FP32"
        )
    return enabled, dtype


def _loss_for_targets(
    output: Stage2Output,
    targets: Tensor,
    positive_weight: float,
    region_loss_weight: float,
    primary_loss_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    return stage2_loss(
        output,
        targets,
        positive_weight=positive_weight,
        region_loss_weight=region_loss_weight,
        primary_loss_weight=primary_loss_weight,
    )


_SELECTION_PROBABILITY_KEYS = {"primary": "pred_prob", "region": "region_pred_prob"}


def _resolve_selection_probability_key(training_config: dict[str, Any]) -> str:
    """Map `training.selection_metric` to the prediction column it selects on.

    Checkpoint selection and early stopping must track whichever head is
    actually being trained (see ``primary_loss_weight``/``region_loss_weight``);
    a region-only run whose selection stayed on the untrained primary head
    would pick checkpoints and stop training on pure noise.
    """
    selection_metric = str(training_config.get("selection_metric", "primary"))
    if selection_metric not in _SELECTION_PROBABILITY_KEYS:
        raise ValueError(
            f"unsupported training.selection_metric: {selection_metric!r} "
            f"(expected one of {sorted(_SELECTION_PROBABILITY_KEYS)})"
        )
    return _SELECTION_PROBABILITY_KEYS[selection_metric]


def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    """Return parameter names whose gradients contain NaN or infinity."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


def _nonfinite_output_names(output: Stage2Output) -> list[str]:
    """Return model output names containing NaN or infinity."""
    return [
        name
        for name, value in output._asdict().items()
        if torch.is_tensor(value)
        and torch.is_floating_point(value)
        and not torch.isfinite(value).all()
    ]


_SUM_REDUCED_STAT_KEYS = (
    "loss_sum",
    "stage1_loss_sum",
    "region_loss_sum",
    "batch_count",
    "gradient_norm_sum",
    "successful_gradient_steps",
    "amp_skipped_steps",
)


def _all_reduce_epoch_stats(
    stats: dict[str, float], device: torch.device
) -> dict[str, float]:
    """Sum-reduce per-rank counters and max-reduce the peak gradient norm across ranks."""
    reduced = dict(stats)
    sums = torch.tensor(
        [stats[key] for key in _SUM_REDUCED_STAT_KEYS],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    for key, value in zip(_SUM_REDUCED_STAT_KEYS, sums.tolist(), strict=True):
        reduced[key] = value
    max_norm = torch.tensor(
        stats["max_gradient_norm"], dtype=torch.float64, device=device
    )
    dist.all_reduce(max_norm, op=dist.ReduceOp.MAX)
    reduced["max_gradient_norm"] = float(max_norm.item())
    return reduced


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    is_main: bool,
) -> dict[str, float]:
    """Train one epoch and return process-local optimization statistics."""
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    training_config = config.get("training", {})
    use_amp, amp_dtype = _amp_settings(config, device)
    mixup_probability = float(training_config.get("p_mixup", 0.2))
    positive_weight = float(training_config.get("positive_weight", 2.0))
    region_loss_weight = float(training_config.get("region_loss_weight", 0.5))
    primary_loss_weight = float(training_config.get("primary_loss_weight", 1.0))
    raw_gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)
    gradient_clip_norm = (
        float("inf")
        if raw_gradient_clip_norm is None
        else float(raw_gradient_clip_norm)
    )
    max_consecutive_amp_skips = int(training_config.get("max_consecutive_amp_skips", 8))
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_region_loss = 0.0
    total_gradient_norm = 0.0
    max_gradient_norm = 0.0
    successful_gradient_steps = 0
    skipped_amp_steps = 0
    consecutive_amp_skips = 0
    progress = tqdm(loader, desc=f"train {epoch}", disable=not is_main)

    for images, regions, targets in progress:
        images = _prepare_images(images, device)
        regions = _prepare_regions(regions, device)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_mixup = np.random.random() < mixup_probability
        if use_mixup:
            images, regions, targets_a, targets_b, lam = mixup_batch(
                images, regions, targets
            )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(images, regions)
            if use_mixup:
                loss_a, components_a = _loss_for_targets(
                    output, targets_a, positive_weight, region_loss_weight, primary_loss_weight
                )
                loss_b, components_b = _loss_for_targets(
                    output, targets_b, positive_weight, region_loss_weight, primary_loss_weight
                )
                loss = lam * loss_a + (1.0 - lam) * loss_b
                components = {
                    name: lam * components_a[name] + (1.0 - lam) * components_b[name]
                    for name in components_a
                }
            else:
                loss, components = _loss_for_targets(
                    output, targets, positive_weight, region_loss_weight, primary_loss_weight
                )

        if not torch.isfinite(loss):
            nonfinite_outputs = _nonfinite_output_names(output)
            nonfinite_components = [
                name
                for name, value in components.items()
                if not torch.isfinite(value).all()
            ]
            raise FloatingPointError(
                f"non-finite training loss: {loss.item()} "
                f"outputs={nonfinite_outputs} components={nonfinite_components}"
            )
        total_loss += float(loss.detach())
        total_stage1_loss += float(components["stage1_loss"].detach())
        total_region_loss += float(components["region_loss"].detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            nonfinite_names = _nonfinite_gradient_names(model)
            if not scaler.is_enabled():
                raise FloatingPointError(
                    f"non-finite gradient norm: {gradient_norm.item()} "
                    f"parameters={nonfinite_names[:20]}"
                )
            scaler.step(optimizer)
            scaler.update()
            skipped_amp_steps += 1
            consecutive_amp_skips += 1
            if consecutive_amp_skips > max_consecutive_amp_skips:
                raise FloatingPointError(
                    f"non-finite gradients persisted for "
                    f"{consecutive_amp_skips} AMP steps "
                    f"parameters={nonfinite_names[:20]}"
                )
            if is_main:
                progress.set_postfix(
                    {
                        "loss": f"{total_loss / (progress.n + 1):.4f}",
                        "amp_skips": skipped_amp_steps,
                        "amp_scale": f"{scaler.get_scale():.0f}",
                    }
                )
            continue
        scaler.step(optimizer)
        scaler.update()
        consecutive_amp_skips = 0
        gradient_norm_value = float(gradient_norm.detach())
        total_gradient_norm += gradient_norm_value
        max_gradient_norm = max(max_gradient_norm, gradient_norm_value)
        successful_gradient_steps += 1
        if is_main:
            completed_batches = progress.n + 1
            progress.set_postfix(
                {
                    "loss": f"{total_loss / completed_batches:.4f}",
                    "stage1": f"{total_stage1_loss / completed_batches:.4f}",
                    "region": f"{total_region_loss / completed_batches:.4f}",
                    "grad": f"{total_gradient_norm / completed_batches:.2f}",
                }
            )
    raw_stats = {
        "loss_sum": total_loss,
        "stage1_loss_sum": total_stage1_loss,
        "region_loss_sum": total_region_loss,
        "batch_count": float(max(len(loader), 1)),
        "gradient_norm_sum": total_gradient_norm,
        "successful_gradient_steps": float(successful_gradient_steps),
        "max_gradient_norm": max_gradient_norm,
        "amp_skipped_steps": float(skipped_amp_steps),
    }
    if dist.is_initialized():
        raw_stats = _all_reduce_epoch_stats(raw_stats, device)
    batch_count = max(raw_stats["batch_count"], 1.0)
    successful_steps = max(raw_stats["successful_gradient_steps"], 1.0)
    return {
        "loss": raw_stats["loss_sum"] / batch_count,
        "stage1_loss": raw_stats["stage1_loss_sum"] / batch_count,
        "region_loss": raw_stats["region_loss_sum"] / batch_count,
        "mean_gradient_norm": raw_stats["gradient_norm_sum"] / successful_steps,
        "max_gradient_norm": raw_stats["max_gradient_norm"],
        "amp_skipped_steps": raw_stats["amp_skipped_steps"],
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    description: str = "valid",
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a metadata-bearing loader and return predictions and diagnostics."""
    model.eval()
    training_config = config.get("training", {})
    use_amp, amp_dtype = _amp_settings(config, device)
    positive_weight = float(training_config.get("positive_weight", 2.0))
    region_loss_weight = float(training_config.get("region_loss_weight", 0.5))
    primary_loss_weight = float(training_config.get("primary_loss_weight", 1.0))
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_region_loss = 0.0
    predictions: list[dict[str, Any]] = []
    all_region_logits: list[np.ndarray] = []
    all_region_plane_valid: list[np.ndarray] = []
    all_plane_valid: list[np.ndarray] = []

    for images, regions, targets, study_uids, vertebrae in tqdm(
        loader, desc=description, disable=dist.is_initialized() and dist.get_rank() != 0
    ):
        images = _prepare_images(images, device)
        regions = _prepare_regions(regions, device)
        targets_device = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(images, regions)
            loss, components = _loss_for_targets(
                output, targets_device, positive_weight, region_loss_weight, primary_loss_weight
            )
        total_loss += float(loss)
        total_stage1_loss += float(components["stage1_loss"])
        total_region_loss += float(components["region_loss"])
        slice_logits = output.slice_logits.float()
        pred_prob = torch.sigmoid(slice_logits).mean(dim=1).cpu().numpy()

        region_pred_prob: np.ndarray | None = None
        region_evidence: dict[str, np.ndarray] = {}
        valid_plane_count: np.ndarray | None = None
        if output.region_logits is not None:
            region_logits = output.region_logits.float()
            plane_valid_f = output.plane_valid.to(region_logits.dtype)
            log_survival = region_log_survival(region_logits, output.region_plane_valid)
            p_any = region_any_probability(log_survival)
            region_pred_prob = (
                (
                    (p_any * plane_valid_f).sum(dim=1)
                    / plane_valid_f.sum(dim=1).clamp_min(1.0)
                )
                .cpu()
                .numpy()
            )
            region_probabilities = torch.sigmoid(region_logits)
            region_plane_valid_f = output.region_plane_valid.to(region_logits.dtype)
            region_names = region_names_for(region_logits.shape[-1])
            for region_index, region_name in enumerate(region_names):
                weights = region_plane_valid_f[..., region_index]
                evidence = (region_probabilities[..., region_index] * weights).sum(
                    dim=1
                ) / weights.sum(dim=1).clamp_min(1.0)
                region_evidence[region_name] = evidence.cpu().numpy()
            valid_plane_count = output.plane_valid.sum(dim=1).cpu().numpy()
            all_region_logits.append(region_logits.cpu().numpy())
            all_region_plane_valid.append(output.region_plane_valid.cpu().numpy())
            all_plane_valid.append(output.plane_valid.cpu().numpy())

        for index, study_uid in enumerate(study_uids):
            record: dict[str, Any] = {
                "study_uid": str(study_uid),
                "vertebra": str(vertebrae[index]),
                "label": int(targets[index].item()),
                "pred_prob": float(pred_prob[index]),
            }
            if region_pred_prob is not None:
                record["region_pred_prob"] = float(region_pred_prob[index])
                record["valid_plane_count"] = int(valid_plane_count[index])
                for region_name in region_evidence:
                    record[f"region_evidence_{region_name}"] = float(
                        region_evidence[region_name][index]
                    )
            predictions.append(record)

    diagnostics = (
        compute_region_diagnostics(
            np.concatenate(all_region_logits),
            np.concatenate(all_region_plane_valid),
            np.concatenate(all_plane_valid),
        )
        if all_region_logits
        else {}
    )
    batch_count = max(len(loader), 1)
    val_stats = {
        "loss": total_loss / batch_count,
        "stage1_loss": total_stage1_loss / batch_count,
        "region_loss": total_region_loss / batch_count,
    }
    return val_stats, predictions, diagnostics


def _metrics_from_predictions(
    predictions: list[dict[str, Any]], probability_key: str = "pred_prob"
) -> dict[str, Any]:
    return compute_metrics(
        np.asarray([record["label"] for record in predictions]),
        np.asarray([record[probability_key] for record in predictions]),
        np.asarray([record["study_uid"] for record in predictions]),
        np.asarray([record["vertebra"] for record in predictions]),
    )


def _verify_architecture_version(checkpoint: dict[str, Any], path: Path) -> None:
    checkpoint_version = checkpoint.get("architecture_version")
    if checkpoint_version != STAGE2_ARCHITECTURE_VERSION:
        raise ValueError(
            f"incompatible Stage2 checkpoint at {path}: "
            f"architecture_version={checkpoint_version!r}, "
            f"expected {STAGE2_ARCHITECTURE_VERSION!r}"
        )


def _save_best_model(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    best_auroc: float,
    config: dict[str, Any],
) -> None:
    """Save AUROC-best weights for inference; never used to resume training."""
    torch.save(
        {
            "model": _base_model(model).state_dict(),
            "epoch": epoch,
            "best_auroc": best_auroc,
            "config": config,
            "architecture_version": STAGE2_ARCHITECTURE_VERSION,
        },
        path,
    )


def _load_best_model(path: Path, model: torch.nn.Module) -> None:
    """Load AUROC-best weights for inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _verify_architecture_version(checkpoint, path)
    _base_model(model).load_state_dict(checkpoint["model"])


def _save_resume_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    next_epoch: int,
    best_auroc: float,
    config: dict[str, Any],
) -> None:
    """Save full training state for exact continuation, after ``scheduler.step()``."""
    torch.save(
        {
            "model": _base_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": next_epoch,
            "best_auroc": best_auroc,
            "config": config,
            "architecture_version": STAGE2_ARCHITECTURE_VERSION,
        },
        path,
    )


def _load_resume_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, dict[str, Any]]:
    """Restore model/optimizer/scheduler and return (next_epoch, best_auroc, saved_config)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _verify_architecture_version(checkpoint, path)
    _base_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["next_epoch"]),
        float(checkpoint.get("best_auroc", -np.inf)),
        checkpoint.get("config", {}),
    )


def train_one_fold(
    config: dict[str, Any],
    fold: int,
    items: list[dict[str, Any]],
    root: Path,
    device: torch.device,
    resume: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train one fold, reload the best checkpoint, and produce OOF predictions."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    selection_probability_key = _resolve_selection_probability_key(training_config)
    train_items, val_items = split_items_cv(
        items,
        n_splits=int(data_config.get("n_folds", 5)),
        val_fold=fold,
        seed=int(data_config.get("random_seed", 42)),
    )
    train_loader, val_loader = create_data_loaders(train_items, val_items, config)
    model, optimizer, scheduler = create_model_optimizer_scheduler(config, device)
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            # Validation only runs on rank 0 (see `is_main` branches below), so
            # per-forward buffer broadcasts would desync collective calls
            # against the other ranks and deadlock NCCL.
            broadcast_buffers=False,
        )
    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=float(training_config.get("amp_initial_scale", 4096.0)),
    )
    best_path, latest_path, fold_dir = resolve_fold_paths(config, fold, root)
    use_wandb, wandb_client = (
        initialize_wandb(config, fold) if is_main else (False, None)
    )
    start_epoch = 0
    best_auroc = -np.inf
    if resume and latest_path.exists():
        start_epoch, best_auroc, saved_config = _load_resume_state(
            latest_path, model, optimizer, scheduler
        )
        validate_resume_config(saved_config, config, start_epoch)
        if is_main:
            prune_training_jsonl(fold_dir / "training.jsonl", start_epoch)
    if is_main:
        print(
            f"[INFO] fold={fold} train_items={len(train_items)} "
            f"val_items={len(val_items)} train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} world_size="
            f"{dist.get_world_size() if dist.is_initialized() else 1} "
            f"amp={amp_dtype if amp_enabled else 'disabled'}",
            flush=True,
        )

    patience = 0
    epochs = int(training_config.get("epochs", 75))
    for epoch in range(start_epoch, epochs):
        train_stats = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch, is_main
        )
        train_loss = train_stats["loss"]
        stop = False
        if is_main:
            val_stats, predictions, diagnostics = evaluate(
                _base_model(model), val_loader, device, config
            )
            val_loss = val_stats["loss"]
            metrics = _metrics_from_predictions(predictions, selection_probability_key)
            auroc = float(metrics.get("auroc", float("nan")))
            improved = np.isfinite(auroc) and auroc > best_auroc
            append_jsonl(
                fold_dir / "training.jsonl",
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train": train_stats,
                    "val_loss": val_loss,
                    "val": val_stats,
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                },
            )
            if improved:
                best_auroc = auroc
                _save_best_model(best_path, model, epoch, best_auroc, config)
            if use_wandb and wandb_client is not None:
                log_wandb_epoch(
                    wandb_client,
                    epoch,
                    float(optimizer.param_groups[-1]["lr"]),
                    train_loss,
                    val_loss,
                    metrics,
                )
        scheduler.step()
        if is_main:
            _save_resume_state(
                latest_path, model, optimizer, scheduler, epoch + 1, best_auroc, config
            )
            patience = 0 if improved else patience + 1
            stop = patience >= int(training_config.get("early_stopping_patience", 15))
        if dist.is_initialized():
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
        if stop:
            break

    if not is_main:
        return {}, []
    if not best_path.exists():
        raise RuntimeError(f"best checkpoint was not created: {best_path}")
    _load_best_model(best_path, model)
    val_stats, predictions, diagnostics = evaluate(
        _base_model(model), val_loader, device, config, description="best"
    )
    primary_metrics = _metrics_from_predictions(predictions, "pred_prob")
    region_metrics = _metrics_from_predictions(predictions, "region_pred_prob")
    selection_metrics = (
        primary_metrics if selection_probability_key == "pred_prob" else region_metrics
    )
    metrics = {
        **selection_metrics,
        "primary": primary_metrics,
        "region": region_metrics,
        "val_loss": val_stats["loss"],
        "val_stage1_loss": val_stats["stage1_loss"],
        "val_region_loss": val_stats["region_loss"],
        "diagnostics": diagnostics,
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=True)
    if use_wandb and wandb_client is not None:
        finish_wandb(wandb_client, metrics)
    return metrics, predictions


@torch.inference_mode()
def predict_ensemble(
    config: dict[str, Any],
    items: list[dict[str, Any]],
    model_paths: list[Path],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Average primary and region-evidence probabilities across fold checkpoints."""
    loader = create_eval_data_loader(items, config)
    fold_predictions: list[list[dict[str, Any]]] = []
    for model_path in model_paths:
        model, _, _ = create_model_optimizer_scheduler(config, device)
        _load_best_model(model_path, model)
        _, predictions, _ = evaluate(
            model, loader, device, config, description=model_path.parent.name
        )
        fold_predictions.append(predictions)
        del model
    outputs: list[dict[str, Any]] = []
    # Region evidence column names depend on the model's region_mode (4
    # anatomical names for "masked", a single generic name for "global"), so
    # derive the averaged keys from the predictions actually produced rather
    # than a fixed region list.
    probability_keys = [
        key
        for key in fold_predictions[0][0]
        if key == "pred_prob"
        or key == "region_pred_prob"
        or key.startswith("region_evidence_")
    ]
    for index, item in enumerate(items):
        record: dict[str, Any] = {
            "study_uid": item["study_uid"],
            "vertebra": item["vertebra"],
            "label": int(item["label"]),
        }
        for key in probability_keys:
            record[key] = float(
                np.mean([fold[index][key] for fold in fold_predictions])
            )
        outputs.append(record)
    return outputs
