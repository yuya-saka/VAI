"""DDP/BF16 training, validation, checkpointing, and resume for Stage3."""

from __future__ import annotations

import hashlib
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

from train_models.stage2.utils.losses import mixup_batch
from train_models.stage3.utils.losses import stage3_loss

from .data_utils import (
    create_data_loaders,
    create_eval_data_loader,
    create_model_optimizer_scheduler,
    set_seed,
    split_items_cv,
)
from .evaluation import compute_prediction_metrics
from .experiment import (
    append_jsonl,
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    prune_training_jsonl,
    resolve_fold_paths,
    validate_resume_config,
)
from .model import STAGE3_ARCHITECTURE_VERSION, Stage3Model, Stage3Output

STAGE3_DATA_PROTOCOL_VERSION = 2


def _base_model(model: torch.nn.Module) -> Stage3Model:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, Stage3Model):
        raise TypeError(f"unexpected model type: {type(model)}")
    return model


def _images(images: Tensor, device: torch.device) -> Tensor:
    return images.to(device, non_blocking=True).float().div_(255.0)


def _amp_settings(
    config: dict[str, Any], device: torch.device
) -> tuple[bool, torch.dtype]:
    training = config.get("training", {})
    enabled = bool(training.get("use_amp", True)) and device.type == "cuda"
    name = str(training.get("amp_dtype", "bfloat16")).lower()
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in dtypes:
        raise ValueError(f"unsupported amp_dtype: {name!r}")
    dtype = dtypes[name]
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 AMP requires a BF16-capable CUDA device")
    return enabled, dtype


def _loss(
    output: Stage3Output,
    targets: Tensor,
    config: dict[str, Any],
    secondary: Tensor | None = None,
    mixup_lambda: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    training = config.get("training", {})
    return stage3_loss(
        output.vertebra_logit,
        output.instance_evidence_logits,
        output.region_plane_valid,
        output.vertebra_valid,
        targets,
        positive_weight=float(training.get("positive_weight", 2.0)),
        lambda_neg=float(training.get("lambda_neg", 0.1)),
        secondary_targets=secondary,
        mixup_lambda=mixup_lambda,
    )


def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


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
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    training = config.get("training", {})
    amp_enabled, amp_dtype = _amp_settings(config, device)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    progress = tqdm(loader, desc=f"train {epoch}", disable=not is_main)
    for images, regions, targets in progress:
        images = _images(images, device)
        regions = regions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_mixup = np.random.random() < float(training.get("p_mixup", 0.2))
        if use_mixup:
            images, regions, targets_a, targets_b, mixup_lambda = mixup_batch(
                images, regions, targets
            )
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(images, regions)
            if use_mixup:
                loss, parts = _loss(output, targets_a, config, targets_b, mixup_lambda)
            else:
                loss, parts = _loss(output, targets, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage3 training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw_clip = training.get("gradient_clip_norm", 1.0)
        clip = float("inf") if raw_clip is None else float(raw_clip)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(gradient_norm):
            names = _nonfinite_gradient_names(model)
            raise FloatingPointError(
                f"non-finite Stage3 gradient parameters={names[:20]}"
            )
        scaler.step(optimizer)
        scaler.update()
        totals += torch.tensor(
            [
                float(loss.detach()),
                float(parts["bag_loss"].detach()),
                float(parts["negative_instance_loss"].detach()),
                1.0,
            ],
            dtype=torch.float64,
            device=device,
        )
        if is_main:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[3]), 1.0)
    return {
        "loss": float(totals[0]) / count,
        "bag_loss": float(totals[1]) / count,
        "negative_instance_loss": float(totals[2]) / count,
    }


@torch.inference_mode()
def evaluate(
    model: Stage3Model,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    description: str = "valid",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    loss_total = 0.0
    predictions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for images, regions, targets, study_uids, vertebrae in tqdm(
        loader, desc=description
    ):
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(
                _images(images, device), regions.to(device, non_blocking=True)
            )
            loss, _ = _loss(output, targets.to(device), config)
        if not torch.isfinite(loss) or not torch.isfinite(output.vertebra_logit).all():
            raise FloatingPointError("non-finite Stage3 validation output")
        loss_total += float(loss)
        probabilities = torch.sigmoid(output.vertebra_logit.float()).cpu().numpy()
        for index, study_uid in enumerate(study_uids):
            valid = bool(output.vertebra_valid[index])
            predictions.append(
                {
                    "study_uid": str(study_uid),
                    "vertebra": str(vertebrae[index]),
                    "label": int(targets[index]),
                    "pred_prob": float(probabilities[index]),
                    "vertebra_valid": valid,
                }
            )
            evidence.append(
                {
                    "study_uid": str(study_uid),
                    "vertebra": str(vertebrae[index]),
                    "label": int(targets[index]),
                    "fold": -1,
                    "vertebra_logit": float(output.vertebra_logit[index]),
                    "probability": float(probabilities[index]),
                    "instance": output.instance_evidence_logits[index]
                    .float()
                    .cpu()
                    .numpy(),
                    "region": output.region_evidence_logits[index]
                    .float()
                    .cpu()
                    .numpy(),
                    "attention": output.slice_attention[index].float().cpu().numpy(),
                    "slice": output.slice_evidence_logits[index].float().cpu().numpy(),
                    "region_weights": output.region_pool_weights[index]
                    .float()
                    .cpu()
                    .numpy(),
                    "region_plane_valid": output.region_plane_valid[index]
                    .cpu()
                    .numpy(),
                    "region_valid": output.region_valid[index].cpu().numpy(),
                    "plane_valid": output.plane_valid[index].cpu().numpy(),
                    "vertebra_valid": valid,
                }
            )
    metrics = compute_prediction_metrics(predictions)
    return {"loss": loss_total / max(len(loader), 1), **metrics}, predictions, evidence


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    next_epoch: int,
    best_auroc: float,
    patience: int,
    config: dict[str, Any],
    data_manifest_sha256: str,
    full_state: bool,
) -> None:
    payload: dict[str, Any] = {
        "model": _base_model(model).state_dict(),
        "epoch": next_epoch - 1,
        "best_auroc": best_auroc,
        "config": config,
        "architecture_version": STAGE3_ARCHITECTURE_VERSION,
        "data_protocol_version": STAGE3_DATA_PROTOCOL_VERSION,
        "data_manifest_sha256": data_manifest_sha256,
    }
    if full_state:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": next_epoch,
                "patience": patience,
            }
        )
    torch.save(payload, path)


def _load_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_manifest_sha256: str,
) -> tuple[int, float, int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_protocol(checkpoint, path, expected_manifest_sha256)
    _base_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["next_epoch"]),
        float(checkpoint.get("best_auroc", -np.inf)),
        int(checkpoint.get("patience", 0)),
        checkpoint.get("config", {}),
    )


def _validate_checkpoint_protocol(
    checkpoint: dict[str, Any],
    path: Path,
    expected_manifest_sha256: str | None = None,
) -> None:
    if checkpoint.get("architecture_version") != STAGE3_ARCHITECTURE_VERSION:
        raise ValueError(f"incompatible Stage3 checkpoint architecture: {path}")
    if checkpoint.get("data_protocol_version") != STAGE3_DATA_PROTOCOL_VERSION:
        raise ValueError(
            f"incompatible Stage3 data protocol: {path}; retraining is required"
        )
    if (
        expected_manifest_sha256 is not None
        and checkpoint.get("data_manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError(f"Stage3 checkpoint data manifest mismatch: {path}")


def _load_best_model(
    path: Path,
    model: Stage3Model,
    expected_manifest_sha256: str | None = None,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_protocol(checkpoint, path, expected_manifest_sha256)
    model.load_state_dict(checkpoint["model"])


def _data_manifest_sha256(items: list[dict[str, Any]]) -> str:
    rows = sorted(
        (str(item["study_uid"]), str(item["vertebra"]), int(item["label"]))
        for item in items
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.inference_mode()
def predict_ensemble(
    config: dict[str, Any],
    items: list[dict[str, Any]],
    model_paths: list[tuple[int, Path]],
    device: torch.device,
    class_prior: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Predict a fixed holdout with fold checkpoints and retain fold evidence."""
    if not model_paths:
        raise ValueError("at least one Stage3 checkpoint is required")
    loader = create_eval_data_loader(items, config)
    fold_predictions: list[tuple[int, list[dict[str, Any]]]] = []
    test_evidence: list[dict[str, Any]] = []
    inference_config = {
        **config,
        "model": {**config.get("model", {}), "pretrained": False},
    }
    for fold, model_path in model_paths:
        model, _, _ = create_model_optimizer_scheduler(
            inference_config, device, class_prior
        )
        _load_best_model(model_path, model)
        _, predictions, evidence = evaluate(
            model, loader, device, config, description=f"test-fold{fold}"
        )
        fold_predictions.append((fold, predictions))
        test_evidence.extend({**record, "fold": fold} for record in evidence)
        del model

    ensemble_predictions: list[dict[str, Any]] = []
    per_fold_predictions: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        item_predictions = [predictions[index] for _, predictions in fold_predictions]
        ensemble_predictions.append(
            {
                "study_uid": item["study_uid"],
                "vertebra": item["vertebra"],
                "label": int(item["label"]),
                "pred_prob": float(
                    np.mean([record["pred_prob"] for record in item_predictions])
                ),
                "vertebra_valid": all(
                    bool(record["vertebra_valid"]) for record in item_predictions
                ),
            }
        )
        for fold, predictions in fold_predictions:
            per_fold_predictions.append({**predictions[index], "fold": fold})
    return ensemble_predictions, per_fold_predictions, test_evidence


def train_one_fold(
    config: dict[str, Any],
    fold: int,
    items: list[dict[str, Any]],
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    data = config.get("data", {})
    training = config.get("training", {})
    set_seed(int(data.get("random_seed", 42)) + rank)
    data_manifest_sha256 = _data_manifest_sha256(items)
    train_items, valid_items = split_items_cv(
        items,
        n_splits=int(data.get("n_folds", 5)),
        val_fold=fold,
        seed=int(data.get("random_seed", 42)),
    )
    class_prior = sum(int(item["label"]) for item in train_items) / max(
        len(train_items), 1
    )
    train_loader, valid_loader = create_data_loaders(train_items, valid_items, config)
    model, optimizer, scheduler = create_model_optimizer_scheduler(
        config, device, class_prior
    )
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=float(training.get("amp_initial_scale", 4096.0)),
    )
    best_path, latest_path, fold_dir = resolve_fold_paths(config, fold, root)
    use_wandb, wandb_run = initialize_wandb(config, fold) if is_main else (False, None)
    start_epoch, best_auroc, patience = 0, -np.inf, 0
    if resume and latest_path.exists():
        start_epoch, best_auroc, patience, saved_config = _load_resume(
            latest_path,
            model,
            optimizer,
            scheduler,
            data_manifest_sha256,
        )
        validate_resume_config(saved_config, config, start_epoch)
        if is_main:
            prune_training_jsonl(fold_dir / "training.jsonl", start_epoch)
    epochs = int(training.get("epochs", 75))
    for epoch in range(start_epoch, epochs):
        train_stats = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch, is_main
        )
        improved = False
        stop = False
        if is_main:
            metrics, predictions, evidence = evaluate(
                _base_model(model), valid_loader, device, config
            )
            auroc = float(metrics.get("auroc", float("nan")))
            improved = np.isfinite(auroc) and auroc > best_auroc
            if improved:
                best_auroc = auroc
                patience = 0
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best_auroc,
                    patience,
                    config,
                    data_manifest_sha256,
                    full_state=False,
                )
            else:
                patience += 1
            append_jsonl(
                fold_dir / "training.jsonl",
                {
                    "epoch": epoch,
                    "train": train_stats,
                    "valid": metrics,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                },
            )
            if use_wandb and wandb_run is not None:
                log_wandb_epoch(
                    wandb_run,
                    epoch,
                    train_stats,
                    metrics,
                    [group["lr"] for group in optimizer.param_groups],
                    patience,
                    best_auroc,
                )
            stop = patience >= int(training.get("early_stopping_patience", 15))
        scheduler.step()
        if is_main:
            _save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_auroc,
                patience,
                config,
                data_manifest_sha256,
                full_state=True,
            )
        if dist.is_initialized():
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
        if stop:
            break
    if not is_main:
        return {}, [], []
    if not best_path.exists():
        raise RuntimeError(f"best checkpoint was not created: {best_path}")
    _load_best_model(best_path, _base_model(model), data_manifest_sha256)
    metrics, predictions, evidence = evaluate(
        _base_model(model), valid_loader, device, config, description="best"
    )
    if use_wandb and wandb_run is not None:
        finish_wandb(wandb_run, metrics)
    return metrics, predictions, evidence
