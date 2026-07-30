"""Fixed-epoch DDP/BF16 training and validation for Stage4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_models.stage3.src.evaluation import compute_prediction_metrics
from train_models.stage3.utils.losses import weighted_bce
from train_models.stage4.utils.diagnostics import (
    DiagnosticHistory,
    gradient_alignment,
    pooling_diagnostics,
)
from train_models.stage4.utils.losses import (
    lambda_region_schedule,
    negative_instance_loss,
    region_loss,
    stratified_negative_instance_loss,
    stratified_vertebra_loss,
)

from .data_utils import (
    create_data_loaders,
    create_model_optimizer_scheduler,
    set_seed,
)
from .experiment import (
    append_jsonl,
    finish_wandb,
    initialize_wandb,
    resolve_fold_paths,
    validate_resume_config,
)
from .model import STAGE4_ARCHITECTURE_VERSION, Stage4Model, Stage4Output
from .stage4_folds import split_by_stage4_fold

STAGE4_DATA_PROTOCOL_VERSION = 2


def _base_model(model: torch.nn.Module) -> Stage4Model:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, Stage4Model):
        raise TypeError(f"unexpected model type: {type(model)}")
    return model


def _images(images: Tensor, device: torch.device) -> Tensor:
    return images.to(device, non_blocking=True).float().div_(255.0)


def _amp_settings(
    config: dict[str, Any],
    device: torch.device,
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


def _shared_encoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    encoder = _base_model(model).encoder
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return list(blocks[-1].parameters())
    parameters = list(encoder.parameters())
    return parameters[-1:]


def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


def _stage4_loss(
    output: Stage4Output,
    targets: Tensor,
    region_targets: Tensor,
    supervision_mask: Tensor,
    pos_weight: Tensor,
    config: dict[str, Any],
    epoch: int,
    population_counts: tuple[int, int, int, int] | None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    training = config.get("training", {})
    positive_weight = float(training.get("positive_weight", 2.0))
    if population_counts is None:
        bag_loss = weighted_bce(
            output.vertebra_logit[output.vertebra_valid],
            targets[output.vertebra_valid],
            positive_weight,
        )
        zero = bag_loss.detach() * 0.0
        bag_parts = {
            "strong_bag_loss": zero,
            "weak_bag_loss": zero,
            "negative_bag_loss": zero,
            "sampled_negative_bag_loss": zero,
            "other_negative_bag_loss": zero,
        }
        raw_negative_loss = negative_instance_loss(
            output.instance_evidence_logits,
            output.region_plane_valid,
            targets,
            output.vertebra_valid,
        )
    else:
        n_strong, n_weak, n_negative, n_sampled_negative = population_counts
        bag_loss, bag_parts = stratified_vertebra_loss(
            output.vertebra_logit,
            targets,
            output.vertebra_valid,
            supervision_mask,
            n_strong=n_strong,
            n_weak=n_weak,
            n_negative=n_negative,
            n_sampled_negative=n_sampled_negative,
            positive_weight=positive_weight,
        )
        raw_negative_loss = stratified_negative_instance_loss(
            output.instance_evidence_logits,
            output.region_plane_valid,
            targets,
            output.vertebra_valid,
            supervision_mask,
            n_negative=n_negative,
            n_sampled_negative=n_sampled_negative,
        )
    lambda_negative = float(training.get("lambda_neg", 0.05))
    weighted_negative_loss = lambda_negative * raw_negative_loss
    vertebra_total = bag_loss + weighted_negative_loss
    raw_region_loss = region_loss(
        output.region_evidence_logits,
        region_targets,
        supervision_mask,
        pos_weight,
    )
    lambda_region = float(training.get("lambda_region_scale", 1.0)) * (
        lambda_region_schedule(epoch)
    )
    weighted_region_loss = lambda_region * raw_region_loss
    total = vertebra_total + weighted_region_loss
    return total, {
        "bag_loss": bag_loss,
        **bag_parts,
        "negative_instance_loss": raw_negative_loss,
        "weighted_negative_loss": weighted_negative_loss,
        "region_loss": raw_region_loss,
        "weighted_region_loss": weighted_region_loss,
        "lambda_region": lambda_region,
        "positive_weight": positive_weight,
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    population_counts: tuple[int, int, int, int],
    pos_weight: Tensor,
    diagnostic_history: DiagnosticHistory,
    is_main: bool,
) -> tuple[dict[str, float], dict[str, float | bool] | None]:
    """Train one fixed-composition epoch and collect Stage4 diagnostics."""
    model.train()
    batch_sampler = loader.batch_sampler
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    training = config.get("training", {})
    if float(training.get("p_mixup", 0.0)) != 0.0:
        raise ValueError("Stage4 mixed supervision requires training.p_mixup=0")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    totals = torch.zeros(12, dtype=torch.float64, device=device)
    diagnostic_interval = int(training.get("diagnostic_interval_steps", 100))
    progress = tqdm(loader, desc=f"train {epoch}", disable=not is_main)
    for batch_index, batch in enumerate(progress):
        images, regions, targets, region_targets, supervision_mask = batch
        images = _images(images, device)
        regions = regions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        region_targets = region_targets.to(device, non_blocking=True)
        supervision_mask = supervision_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(images, regions)
            loss, parts = _stage4_loss(
                output,
                targets,
                region_targets,
                supervision_mask,
                pos_weight,
                config,
                epoch,
                population_counts,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage4 training loss")
        global_step = epoch * len(loader) + batch_index + 1
        if diagnostic_interval > 0 and global_step % diagnostic_interval == 0:
            alignment = gradient_alignment(
                parts["bag_loss"],
                parts["region_loss"],
                _shared_encoder_parameters(model),
                float(parts["lambda_region"]),
            )
            pooling = pooling_diagnostics(
                output.region_evidence_logits,
                output.region_pool_weights,
                targets,
                output.region_valid,
            )
            if is_main:
                diagnostic_history.add_step({**alignment, **pooling})

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw_clip = training.get("gradient_clip_norm", 1.0)
        clip = float("inf") if raw_clip is None else float(raw_clip)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(gradient_norm):
            names = _nonfinite_gradient_names(model)
            raise FloatingPointError(
                f"non-finite Stage4 gradient parameters={names[:20]}"
            )
        scaler.step(optimizer)
        scaler.update()
        totals += torch.tensor(
            [
                float(loss.detach()),
                float(parts["bag_loss"].detach()),
                float(parts["region_loss"].detach()),
                float(parts["weighted_region_loss"].detach()),
                float(parts["negative_instance_loss"].detach()),
                float(parts["weighted_negative_loss"].detach()),
                float(parts["strong_bag_loss"].detach()),
                float(parts["weak_bag_loss"].detach()),
                float(parts["negative_bag_loss"].detach()),
                float(parts["sampled_negative_bag_loss"].detach()),
                float(parts["other_negative_bag_loss"].detach()),
                1.0,
            ],
            dtype=torch.float64,
            device=device,
        )
        if is_main:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[11]), 1.0)
    stats = {
        "loss": float(totals[0]) / count,
        "vertebra_loss": float(totals[1]) / count,
        "region_loss": float(totals[2]) / count,
        "weighted_region_loss": float(totals[3]) / count,
        "negative_instance_loss": float(totals[4]) / count,
        "weighted_negative_loss": float(totals[5]) / count,
        "strong_bag_loss": float(totals[6]) / count,
        "weak_bag_loss": float(totals[7]) / count,
        "negative_bag_loss": float(totals[8]) / count,
        "sampled_negative_bag_loss": float(totals[9]) / count,
        "other_negative_bag_loss": float(totals[10]) / count,
        "lambda_region": float(training.get("lambda_region_scale", 1.0))
        * lambda_region_schedule(epoch),
    }
    diagnostics = (
        diagnostic_history.summarize_epoch()
        if is_main and diagnostic_history.step_records
        else None
    )
    return stats, diagnostics


@torch.inference_mode()
def evaluate(
    model: Stage4Model,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    pos_weight: Tensor,
    epoch: int,
    description: str = "valid",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all bags while retaining region outputs for strong bags."""
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    loss_total = 0.0
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=description):
        (
            images,
            regions,
            targets,
            region_targets,
            supervision_mask,
            study_uids,
            vertebrae,
        ) = batch
        device_targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(
                _images(images, device),
                regions.to(device, non_blocking=True),
            )
            loss, _ = _stage4_loss(
                output,
                device_targets,
                region_targets.to(device, non_blocking=True),
                supervision_mask.to(device, non_blocking=True),
                pos_weight,
                config,
                epoch,
                population_counts=None,
            )
        if not torch.isfinite(loss) or not torch.isfinite(output.vertebra_logit).all():
            raise FloatingPointError("non-finite Stage4 validation output")
        loss_total += float(loss)
        vertebra_probabilities = torch.sigmoid(output.vertebra_logit.float()).cpu()
        region_probabilities = torch.sigmoid(
            output.region_evidence_logits.float()
        ).cpu()
        for index, study_uid in enumerate(study_uids):
            record: dict[str, Any] = {
                "study_uid": str(study_uid),
                "vertebra": str(vertebrae[index]),
                "label": int(targets[index]),
                "pred_prob": float(vertebra_probabilities[index]),
                "vertebra_valid": bool(output.vertebra_valid[index]),
                "region_supervised": bool(supervision_mask[index]),
            }
            for region_index in range(4):
                record[f"region_target_r{region_index + 1}"] = int(
                    region_targets[index, region_index]
                )
                record[f"region_prob_r{region_index + 1}"] = float(
                    region_probabilities[index, region_index]
                )
                record[f"region_valid_r{region_index + 1}"] = bool(
                    output.region_valid[index, region_index]
                )
            predictions.append(record)
    metrics = compute_prediction_metrics(predictions)
    return {"loss": loss_total / max(len(loader), 1), **metrics}, predictions


def _data_manifest_sha256(items: list[dict[str, Any]]) -> str:
    rows = sorted(
        (
            str(item["study_uid"]),
            str(item["vertebra"]),
            int(item["label"]),
            str(item["region_supervision"]),
            tuple(int(value) for value in item["region_label"]),
        )
        for item in items
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    next_epoch: int,
    config: dict[str, Any],
    data_manifest_sha256: str,
    full_state: bool,
) -> None:
    payload: dict[str, Any] = {
        "model": _base_model(model).state_dict(),
        "epoch": next_epoch - 1,
        "config": config,
        "architecture_version": STAGE4_ARCHITECTURE_VERSION,
        "data_protocol_version": STAGE4_DATA_PROTOCOL_VERSION,
        "data_manifest_sha256": data_manifest_sha256,
    }
    if full_state:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": next_epoch,
            }
        )
    torch.save(payload, path)


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    path: Path,
    expected_manifest_sha256: str,
) -> None:
    if checkpoint.get("architecture_version") != STAGE4_ARCHITECTURE_VERSION:
        raise ValueError(f"incompatible Stage4 checkpoint architecture: {path}")
    if checkpoint.get("data_protocol_version") != STAGE4_DATA_PROTOCOL_VERSION:
        raise ValueError(f"incompatible Stage4 data protocol: {path}")
    if checkpoint.get("data_manifest_sha256") != expected_manifest_sha256:
        raise ValueError(f"Stage4 checkpoint data manifest mismatch: {path}")


def _load_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_manifest_sha256: str,
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, path, expected_manifest_sha256)
    _base_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["next_epoch"]), checkpoint.get("config", {})


def _load_final(
    path: Path,
    model: Stage4Model,
    expected_manifest_sha256: str,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, path, expected_manifest_sha256)
    model.load_state_dict(checkpoint["model"])


def train_one_fold(
    config: dict[str, Any],
    fold: int,
    items: list[dict[str, Any]],
    fold_map: dict[str, int],
    root: Path,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train exactly `fixed_epochs` and evaluate the final checkpoint."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0
    data = config.get("data", {})
    training = config.get("training", {})
    set_seed(int(data.get("random_seed", 42)) + rank)
    train_items, valid_items = split_by_stage4_fold(items, fold_map, fold)
    data_manifest_sha256 = _data_manifest_sha256(train_items)
    n_strong = sum(item["region_supervision"] == "strong" for item in train_items)
    n_weak = sum(item["region_supervision"] == "weak" for item in train_items)
    n_negative = sum(item["region_supervision"] == "negative" for item in train_items)
    population_counts = (n_strong, n_weak, n_negative, n_strong)
    class_prior = (n_strong + n_weak) / max(len(train_items), 1)
    final_path, latest_path, fold_dir = resolve_fold_paths(config, fold, root)
    train_loader, valid_loader, pos_weight = create_data_loaders(
        train_items,
        valid_items,
        config,
        fold_dir,
        rank=rank,
        world_size=world_size,
    )
    model, optimizer, scheduler = create_model_optimizer_scheduler(
        config,
        device,
        class_prior,
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
    use_wandb, wandb_run = initialize_wandb(config, fold) if is_main else (False, None)
    start_epoch = 0
    if resume and latest_path.exists():
        start_epoch, saved_config = _load_resume(
            latest_path,
            model,
            optimizer,
            scheduler,
            data_manifest_sha256,
        )
        validate_resume_config(saved_config, config, start_epoch)
    epochs = int(training.get("fixed_epochs", 75))
    validation_interval = int(training.get("validation_interval_epochs", epochs))
    if validation_interval < 1:
        raise ValueError("validation_interval_epochs must be positive")
    diagnostic_history = DiagnosticHistory()
    final_metrics: dict[str, Any] | None = None
    final_predictions: list[dict[str, Any]] | None = None
    for epoch in range(start_epoch, epochs):
        train_stats, diagnostics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            epoch,
            population_counts,
            pos_weight,
            diagnostic_history,
            is_main,
        )
        if is_main:
            should_validate = (
                epoch + 1
            ) % validation_interval == 0 or epoch + 1 == epochs
            if should_validate:
                valid_metrics, valid_predictions = evaluate(
                    _base_model(model),
                    valid_loader,
                    device,
                    config,
                    pos_weight,
                    epoch,
                )
                if epoch + 1 == epochs:
                    final_metrics = valid_metrics
                    final_predictions = valid_predictions
            else:
                valid_metrics = {}
            payload = {
                "epoch": epoch,
                "train": train_stats,
                "valid": valid_metrics,
                "diagnostics": diagnostics,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
            append_jsonl(fold_dir / "training.jsonl", payload)
            if use_wandb and wandb_run is not None:
                wandb_payload = {
                    "epoch": epoch,
                    **{f"train/{key}": value for key, value in train_stats.items()},
                    "valid/loss": valid_metrics.get("loss", float("nan")),
                    "valid/auroc": valid_metrics.get("auroc", float("nan")),
                    "valid/auprc": valid_metrics.get("auprc", float("nan")),
                }
                if diagnostics is not None:
                    wandb_payload.update(
                        {
                            f"diagnostics/{key}": value
                            for key, value in diagnostics.items()
                        }
                    )
                wandb_run.log(wandb_payload, step=epoch)
        scheduler.step()
        if is_main:
            _save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                config,
                data_manifest_sha256,
                full_state=True,
            )
            if epoch + 1 == epochs:
                _save_checkpoint(
                    final_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    config,
                    data_manifest_sha256,
                    full_state=False,
                )
        if dist.is_initialized():
            dist.barrier()
    if not is_main:
        return {}, []
    if not final_path.exists():
        raise RuntimeError(f"final checkpoint was not created: {final_path}")
    if final_metrics is None or final_predictions is None:
        _load_final(final_path, _base_model(model), data_manifest_sha256)
        final_metrics, final_predictions = evaluate(
            _base_model(model),
            valid_loader,
            device,
            config,
            pos_weight,
            epochs - 1,
            description="final",
        )
    if use_wandb and wandb_run is not None:
        finish_wandb(wandb_run, final_metrics)
    return final_metrics, final_predictions
