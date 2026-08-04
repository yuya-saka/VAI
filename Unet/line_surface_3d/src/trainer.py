"""1 foldの学習、検証、checkpoint選択。"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.losses import compute_plane_loss, warmup_weight
from ..utils.plane import centered_positions
from .data_utils import create_training_loaders
from .evaluation import evaluate, vertebra_indices
from .experiment import (
    append_epoch_metrics,
    fold_paths,
    initialize_wandb,
    save_json,
)
from .model import TinyUNet, reshape_slab_heatmaps

CHECKPOINT_PROTOCOL = "line_surface_3d_v2"


def build_model(config: dict[str, Any], device: torch.device) -> TinyUNet:
    """slab sizeから入出力channelを導出してモデルを作る。"""
    slab_size = int(config["data"]["slab_size"])
    model_config = config.get("model", {})
    use_conditioning = bool(model_config.get("use_vertebra_conditioning", False))
    model = TinyUNet(
        in_channels=2 * slab_size,
        out_channels=4 * slab_size,
        features=tuple(model_config.get("features", [16, 32, 64, 128])),
        dropout=float(model_config.get("dropout", 0.0)),
        num_vertebra=(
            int(model_config.get("num_vertebra", 7)) if use_conditioning else 0
        ),
    )
    return model.to(device)


def build_optimizer_scheduler(
    model: nn.Module,
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, Any]:
    """AdamWとReduceLROnPlateauを作る。"""
    training_config = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 5e-4)),
        weight_decay=float(training_config.get("weight_decay", 2e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(training_config.get("lr_patience", 8)),
        factor=float(training_config.get("lr_factor", 0.5)),
    )
    return optimizer, scheduler


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """1 epochを学習し、損失平均を返す。"""
    model.train()
    training_config = config.get("training", {})
    plane_config = config.get("loss", {}).get("plane", {})
    slab_size = int(config["data"]["slab_size"])
    image_size = int(config["data"]["image_size"])
    geometry_weight = warmup_weight(
        epoch,
        int(plane_config.get("warmup_start_epoch", 0)),
        int(plane_config.get("warmup_epochs", 0)),
    )
    grad_clip = float(training_config.get("grad_clip", 1.0))
    log_interval_steps = max(1, int(training_config.get("log_interval_steps", 10)))
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    total_steps = len(loader) if isinstance(loader, Sized) else None
    total_steps_text = str(total_steps) if total_steps is not None else "?"
    started_at = time.time()
    print(
        f"[TRAIN] epoch={epoch:03d} start steps={total_steps_text} device={device}",
        flush=True,
    )
    sums = {
        "total": 0.0,
        "heatmap": 0.0,
        "angle": 0.0,
        "rho": 0.0,
        "tilt": 0.0,
    }
    positions = centered_positions(slab_size, device, torch.float32)
    step_count = 0
    for batch in loader:
        images = batch["image"].to(device).float()
        targets = batch["heatmaps"].to(device).float()
        label_mask = batch["label_mask"].to(device).bool()
        line_params_gt = batch["line_params_gt"].to(device).float()
        gt_slope = batch["plane_slope_gt"].to(device).float()
        gt_reliable = batch["plane_reliable"].to(device).bool()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images, vertebra_indices(batch, device))
            predictions = torch.sigmoid(reshape_slab_heatmaps(logits, slab_size))
        # 幾何は数値安定性のためFP32で計算する
        loss_output = compute_plane_loss(
            predictions.float(),
            targets,
            label_mask,
            line_params_gt,
            gt_slope,
            gt_reliable,
            positions,
            image_size,
            plane_config,
            geometry_weight,
        )
        scaler.scale(loss_output.total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        sums["total"] += float(loss_output.total.detach())
        sums["heatmap"] += float(loss_output.heatmap.detach())
        sums["angle"] += float(loss_output.angle.detach())
        sums["rho"] += float(loss_output.rho.detach())
        sums["tilt"] += float(loss_output.tilt.detach())
        step_count += 1
        if (
            step_count == 1
            or step_count % log_interval_steps == 0
            or step_count == total_steps
        ):
            print(
                f"[TRAIN] epoch={epoch:03d} "
                f"step={step_count}/{total_steps_text} "
                f"loss={float(loss_output.total.detach()):.6f} "
                f"avg={sums['total'] / step_count:.6f} "
                f"elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )
    denominator = max(1, step_count)
    return {
        "train_loss": sums["total"] / denominator,
        "train_heatmap_loss": sums["heatmap"] / denominator,
        "train_angle_loss": sums["angle"] / denominator,
        "train_rho_loss": sums["rho"] / denominator,
        "train_tilt_loss": sums["tilt"] / denominator,
        "geometry_weight": geometry_weight,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> None:
    """再現性情報を含むbest checkpointを保存する。"""
    torch.save(
        {
            "protocol": CHECKPOINT_PROTOCOL,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation": validation_metrics,
            "manifest_hashes": manifest_hashes,
            "slab_size": int(config["data"]["slab_size"]),
            "channel_order": "slice_major_ct_mask",
        },
        path,
    )


def train_one_fold(config: dict[str, Any]) -> dict[str, Any]:
    """設定されたfoldを学習し、test指標を返す。"""
    fold = int(config["data"]["test_fold"])
    paths = fold_paths(config, fold)
    for output_key in ("checkpoint", "metrics", "test_metrics"):
        paths[output_key].unlink(missing_ok=True)
    train_loader, validation_loader, test_loader, manifest_hashes = (
        create_training_loaders(config)
    )
    save_json(
        paths["manifest"],
        {
            "protocol": CHECKPOINT_PROTOCOL,
            "hashes": manifest_hashes,
            "counts": {
                "train": len(train_loader.dataset),
                "validation": len(validation_loader.dataset),
                "test": len(test_loader.dataset),
            },
        },
    )
    training_config = config.get("training", {})
    gpu_id = int(training_config.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    wandb = initialize_wandb(config, fold)
    epochs = int(training_config.get("epochs", 200))
    patience = int(training_config.get("early_stopping_patience", 15))
    selection_metric = str(
        training_config.get("selection_metric", "plane_combined_error_px")
    )
    early_stopping_metric = str(
        training_config.get("early_stopping_metric", "val_loss_mse")
    )
    best_selection_value = math.inf
    best_early_stopping_value = math.inf
    epochs_without_improvement = 0

    print(
        f"[START] fold={fold} device={device} "
        f"train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} "
        f"test={len(test_loader.dataset)} "
        f"batch_size={int(training_config.get('batch_size', 1))}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        started_at = time.time()
        train_metrics = _train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            config,
            epoch,
            scaler,
        )
        print(f"[VALIDATION] epoch={epoch:03d} start", flush=True)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            config,
        )
        scheduler.step(validation_metrics["val_loss_mse"])
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started_at,
            **train_metrics,
            **validation_metrics,
        }
        append_epoch_metrics(paths["metrics"], epoch_metrics)
        if wandb is not None:
            wandb.log(epoch_metrics, step=epoch)
        current_selection_value = float(validation_metrics[selection_metric])
        print(
            f"[fold={fold} epoch={epoch:03d}] "
            f"train={train_metrics['train_loss']:.6f} "
            f"val_mse={validation_metrics['val_loss_mse']:.6f} "
            f"line_angle={validation_metrics['line_angle_error_deg']:.3f} "
            f"line_rho={validation_metrics['line_rho_error_px']:.3f} "
            f"tilt={validation_metrics['tilt_error_px_per_slice']:.4f} "
            f"sign={validation_metrics['tilt_sign_accuracy']:.3f} "
            f"combined={validation_metrics['plane_combined_error_px']:.3f}",
            flush=True,
        )
        if (
            math.isfinite(current_selection_value)
            and current_selection_value < best_selection_value - 1e-8
        ):
            best_selection_value = current_selection_value
            _save_checkpoint(
                paths["checkpoint"],
                model,
                optimizer,
                config,
                epoch,
                validation_metrics,
                manifest_hashes,
            )

        current_early_stopping_value = float(validation_metrics[early_stopping_metric])
        if (
            math.isfinite(current_early_stopping_value)
            and current_early_stopping_value < best_early_stopping_value - 1e-8
        ):
            best_early_stopping_value = current_early_stopping_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[EARLY STOP] {early_stopping_metric}が"
                    f"{patience} epoch改善なし "
                    f"(best={best_early_stopping_value:.6f})",
                    flush=True,
                )
                break

    if not paths["checkpoint"].exists():
        raise RuntimeError("best checkpointが保存されませんでした")
    checkpoint = torch.load(
        paths["checkpoint"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, config)
    save_json(paths["test_metrics"], test_metrics)
    if wandb is not None:
        for key, value in test_metrics.items():
            wandb.run.summary[f"test/{key}"] = value
        wandb.finish()
    return test_metrics
