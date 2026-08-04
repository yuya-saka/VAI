"""1 foldの学習、段階的幾何損失、checkpoint選択。"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .data_utils import create_data_loaders, set_seed
from .evaluation import evaluate, vertebra_indices
from .experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    update_best_summary,
)
from .inference import predict_lines_and_save, save_examples
from .losses import compute_loss
from .model import SliceSharedUNet

CHECKPOINT_PROTOCOL = "line-2p5d-local-consistency-v1"


def _output_paths(config: dict[str, Any], fold: int) -> dict[str, Path]:
    """phase・実験名・foldから成果物pathを作る。"""
    experiment = config["experiment"]
    unet_dir = Path(__file__).resolve().parents[2]
    experiment_root = (
        unet_dir / "outputs" / str(experiment["phase"]) / str(experiment["name"])
    )
    root = experiment_root / f"fold_{fold}"
    root.mkdir(parents=True, exist_ok=True)
    visualization_value = config.get("evaluation", {}).get("visualization_dir")
    visualization_root = (
        Path(visualization_value) if visualization_value else experiment_root / "vis"
    )
    visualization_fold = visualization_root / f"fold{fold}"
    return {
        "root": root,
        "checkpoint": root / "best.pt",
        "metrics": root / "metrics.jsonl",
        "test_metrics": root / "test_metrics.json",
        "manifest": root / "manifest.json",
        "config": root / "effective_config.yaml",
        "validation_visualization": visualization_fold / "val",
        "test_visualization": visualization_fold / "test",
        "line_output": visualization_fold / "test_lines",
        "line_summary": root / "line_summary.json",
    }


def _write_json(path: Path, values: dict[str, Any]) -> None:
    """JSONをNaN許容で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, values: dict[str, Any]) -> None:
    """epoch指標を1行追記する。"""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(values, ensure_ascii=False, allow_nan=True) + "\n")


def build_model(config: dict[str, Any], device: torch.device) -> SliceSharedUNet:
    """設定から共有2D+z融合モデルを構築する。"""
    model_config = config.get("model", {})
    use_conditioning = bool(model_config.get("use_vertebra_conditioning", True))
    model = SliceSharedUNet(
        in_channels_per_slice=2,
        out_channels_per_slice=4,
        features=tuple(model_config.get("features", [16, 32, 64, 128])),
        dropout=float(model_config.get("dropout", 0.08)),
        temporal_blocks=int(model_config.get("temporal_blocks", 1)),
        num_vertebra=(
            int(model_config.get("num_vertebra", 7)) if use_conditioning else 0
        ),
    )
    return model.to(device)


def _build_optimizer_scheduler(
    model: nn.Module,
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """AdamWとheatmap MSE基準schedulerを構築する。"""
    training_config = config["training"]
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
    """1 epochを学習し、各損失の平均を返す。"""
    model.train()
    training_config = config["training"]
    loss_config = config.get("loss", {})
    image_size = int(config["data"]["image_size"])
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    grad_clip = float(training_config.get("grad_clip", 1.0))
    log_interval = max(1, int(training_config.get("log_interval_steps", 10)))
    total_steps = len(loader) if isinstance(loader, Sized) else None
    sums = {
        "total": 0.0,
        "heatmap": 0.0,
        "mse": 0.0,
        "angle": 0.0,
        "position": 0.0,
    }
    geometry_weight = 0.0
    step_count = 0
    started_at = time.time()
    for batch in loader:
        images = batch["image"].to(device).float()
        targets = batch["heatmaps"].to(device).float()
        context_valid = batch["context_valid"].to(device).bool()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images, vertebra_indices(batch, device))
            predictions = torch.sigmoid(logits)
        loss_output = compute_loss(
            predictions.float(),
            targets,
            context_valid,
            loss_config,
            epoch,
            image_size,
        )
        scaler.scale(loss_output.total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        sums["total"] += float(loss_output.total.detach())
        sums["heatmap"] += float(loss_output.heatmap.detach())
        sums["mse"] += float(loss_output.heatmap_mse.detach())
        sums["angle"] += float(loss_output.angle_consistency.detach())
        sums["position"] += float(loss_output.position_consistency.detach())
        geometry_weight = loss_output.geometry_weight
        step_count += 1
        if (
            step_count == 1
            or step_count % log_interval == 0
            or step_count == total_steps
        ):
            print(
                f"[TRAIN] epoch={epoch:03d} step={step_count}/{total_steps or '?'} "
                f"loss={float(loss_output.total.detach()):.6f} "
                f"geometry={geometry_weight:.3f} elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )
    denominator = max(1, step_count)
    return {
        "train_loss": sums["total"] / denominator,
        "train_heatmap_loss": sums["heatmap"] / denominator,
        "train_heatmap_mse": sums["mse"] / denominator,
        "train_angle_consistency": sums["angle"] / denominator,
        "train_position_consistency": sums["position"] / denominator,
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
            "context_offsets": tuple(config["data"]["context_offsets"]),
        },
        path,
    )


def train_one_fold(config: dict[str, Any]) -> dict[str, Any]:
    """設定されたfoldを学習し、test指標を保存する。"""
    fold = int(config["data"]["test_fold"])
    seed = int(config["data"].get("random_seed", 42)) + fold
    set_seed(seed)
    paths = _output_paths(config, fold)
    for key in ("checkpoint", "metrics", "test_metrics"):
        paths[key].unlink(missing_ok=True)
    paths["config"].write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    train_loader, validation_loader, test_loader, manifest_hashes = create_data_loaders(
        config
    )
    _write_json(
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

    training_config = config["training"]
    gpu_id = int(training_config.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    wandb_enabled, wandb_module = initialize_wandb(config, fold)
    model = build_model(config, device)
    optimizer, scheduler = _build_optimizer_scheduler(model, config)
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epochs = int(training_config.get("epochs", 150))
    patience = int(training_config.get("early_stopping_patience", 15))
    minimum_epoch = int(training_config.get("early_stopping_min_epoch", 1))
    selection_metric = str(
        training_config.get("selection_metric", "line_combined_error_px")
    )
    best_selection = math.inf
    has_finite_selection = False
    best_heatmap_mse = math.inf
    epochs_without_improvement = 0

    print(
        f"[START] fold={fold} device={device} train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} test={len(test_loader.dataset)}",
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
        validation_metrics = evaluate(model, validation_loader, device, config)
        scheduler.step(validation_metrics["val_heatmap_mse"])
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started_at,
            **train_metrics,
            **validation_metrics,
        }
        _append_jsonl(paths["metrics"], epoch_metrics)
        if wandb_enabled and wandb_module is not None:
            log_wandb_epoch(wandb_module, epoch, epoch_metrics)
        selection_value = float(validation_metrics[selection_metric])
        heatmap_mse = float(validation_metrics["val_heatmap_mse"])
        heatmap_improved = heatmap_mse < best_heatmap_mse - 1e-8
        print(
            f"[fold={fold} epoch={epoch:03d}] "
            f"heatmap={validation_metrics['val_heatmap_mse']:.6f} "
            f"angle={validation_metrics['line_angle_error_deg']:.3f} "
            f"rho={validation_metrics['line_rho_error_px']:.3f} "
            f"collapse={validation_metrics['heatmap_collapse_rate']:.3f} "
            f"geometry={train_metrics['geometry_weight']:.3f}",
            flush=True,
        )
        selection_improved = math.isfinite(selection_value) and (
            not has_finite_selection or selection_value < best_selection - 1e-8
        )
        if selection_improved or (not has_finite_selection and heatmap_improved):
            if selection_improved:
                best_selection = selection_value
                has_finite_selection = True
            _save_checkpoint(
                paths["checkpoint"],
                model,
                optimizer,
                config,
                epoch,
                validation_metrics,
                manifest_hashes,
            )
            if wandb_enabled and wandb_module is not None:
                checkpoint_metric = (
                    selection_metric if selection_improved else "val_heatmap_mse"
                )
                checkpoint_value = (
                    selection_value if selection_improved else heatmap_mse
                )
                update_best_summary(
                    wandb_module,
                    epoch,
                    checkpoint_metric,
                    checkpoint_value,
                    validation_metrics,
                )

        if heatmap_improved:
            best_heatmap_mse = heatmap_mse
            epochs_without_improvement = 0
        elif epoch >= minimum_epoch:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[EARLY STOP] epoch={epoch} heatmap MSEが{patience}回改善なし",
                    flush=True,
                )
                break

    if not paths["checkpoint"].exists():
        raise RuntimeError("best checkpointが保存されませんでした")
    checkpoint = torch.load(
        paths["checkpoint"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, config)
    _write_json(paths["test_metrics"], test_metrics)
    visualization_samples = int(
        config.get("evaluation", {}).get("visualization_samples", 16)
    )
    print("[INFO] 可視化画像を保存します", flush=True)
    save_examples(
        model,
        validation_loader,
        device,
        paths["validation_visualization"],
        n_save=visualization_samples,
        tag="VAL",
    )
    save_examples(
        model,
        test_loader,
        device,
        paths["test_visualization"],
        n_save=visualization_samples,
        tag="TEST",
    )
    line_summary = predict_lines_and_save(
        config,
        model,
        test_loader,
        device,
        paths["line_output"],
    )
    _write_json(paths["line_summary"], line_summary)
    print(f"[INFO] 可視化保存先: {paths['line_output'].parent}", flush=True)
    if wandb_enabled and wandb_module is not None:
        finish_wandb(wandb_module, test_metrics, line_summary)
    return test_metrics
