"""Baseline 0のfold単位の学習・検証・checkpoint制御。"""

from __future__ import annotations

import csv
import json
import random
import time
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from fracture_detection.baseline0.modeling.losses import (
    bag_probabilities,
    broadcast_bce_loss,
)
from fracture_detection.baseline0.training.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    update_best_prauc_summary,
    update_best_summary,
)
from fracture_detection.baseline0.training.optimization import (
    LearningRateController,
    create_cosine_scheduler,
    create_optimizer,
)
from fracture_detection.common.metrics import (
    safe_auroc,
    safe_average_precision,
    select_f1_threshold,
    threshold_metrics,
)
from fracture_detection.common.sampling import EpochShuffleSampler

Batch = dict[str, object]


@dataclass(frozen=True)
class FoldTrainingResult:
    """1 outer fold学習の再利用可能な要約。"""

    best_epoch: int
    best_metrics: dict[str, float]
    best_prauc_epoch: int
    best_prauc_metrics: dict[str, float]
    stopped_epoch: int
    outer_predictions: pd.DataFrame
    outer_prauc_predictions: pd.DataFrame


@dataclass(frozen=True)
class CheckpointEvaluation:
    """1 checkpointへval閾値決定とouter固定適用を行った結果。"""

    validation_metrics: dict[str, float]
    validation_threshold_metrics: dict[str, float | int]
    outer_metrics: dict[str, float]
    outer_threshold_metrics: dict[str, float | int]
    validation_predictions: pd.DataFrame
    outer_predictions: pd.DataFrame


def set_seed(seed: int, fold: int) -> None:
    """foldごとの決定的な乱数状態を設定する。"""
    resolved_seed = seed + fold
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader workerのnumpy/random seedを固定する。"""
    cv2.setNumThreads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_data_loader(
    dataset: torch.utils.data.Dataset[dict[str, Tensor | str]],
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader[Any]:
    """明示samplerと独立worker seedを使うDataLoaderを作る。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def train_fold(
    model: nn.Module,
    train_loader: DataLoader[Any],
    inner_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    config: dict[str, Any],
    outer_fold: int,
    fold_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> FoldTrainingResult:
    """innerでcheckpointを選択し、outerを一度だけ推論する。"""
    training = config["training"]
    max_epochs = int(training["max_epochs"])
    min_epoch = int(training["min_epoch"])
    patience = int(training["early_stopping_patience"])
    pos_weight = float(training["pos_weight"])
    train_rows = _dataset_length(train_loader)
    inner_rows = _dataset_length(inner_loader)
    outer_rows = _dataset_length(outer_loader)
    if not isinstance(train_loader.sampler, EpochShuffleSampler):
        raise TypeError("train loaderにはEpochShuffleSamplerが必要です")
    print(
        f"[outer {outer_fold}] 学習を初期化しています: device={device}, "
        f"train={train_rows:,} bag/{len(train_loader)} batch, "
        f"val={inner_rows:,} bag/{len(inner_loader)} batch, "
        f"outer={outer_rows:,} bag/{len(outer_loader)} batch",
        flush=True,
    )
    optimizer = create_optimizer(
        model,
        float(training["weight_decay"]),
        float(training["backbone_learning_rate"]),
        float(training["head_learning_rate"]),
    )
    controller = LearningRateController(
        steps_per_epoch=len(train_loader),
        freeze_backbone_epochs=int(training["freeze_backbone_epochs"]),
        warmup_epochs=int(training["warmup_epochs"]),
        warmup_start_factor=float(training["warmup_start_factor"]),
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        head_learning_rate=float(training["head_learning_rate"]),
    )
    scheduler = create_cosine_scheduler(
        optimizer,
        max_epochs=max_epochs,
        backbone_min_learning_rate=float(training["backbone_min_learning_rate"]),
        head_min_learning_rate=float(training["head_min_learning_rate"]),
    )
    model.to(device)
    # RSNA Stage1と同じく、nullは「clipしない」を表す実質無限大として扱う。
    raw_gradient_clip_norm = training["gradient_clip_norm"]
    gradient_clip_norm = (
        float("inf")
        if raw_gradient_clip_norm is None
        else float(raw_gradient_clip_norm)
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pt"
    best_prauc_path = fold_dir / "best_val_prauc_model.pt"
    last_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "history.csv"
    log_path = fold_dir / "training.log"

    (
        start_epoch,
        global_step,
        best_epoch,
        best_metrics,
        best_prauc_epoch,
        best_prauc_metrics,
        early_stopping_best_loss,
        no_improvement,
    ) = _resume_state(model, optimizer, scheduler, last_path, device, config, resume)
    history_rows = _load_history(history_path) if resume else []
    print(f"[outer {outer_fold}] W&Bを初期化しています", flush=True)
    wandb_module = initialize_wandb(config, outer_fold)
    print(f"[outer {outer_fold}] W&B初期化処理が完了しました", flush=True)
    latest_metrics: dict[str, float] | None = None
    stopped_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            print(
                f"[outer {outer_fold}] epoch {epoch}/{max_epochs}を開始します。"
                "最初のbatchではworker起動とprefetchを行います",
                flush=True,
            )
            start_time = time.monotonic()
            _set_loader_epoch(train_loader, epoch - 1)
            controller.set_epoch_state(model, epoch - 1)
            train_metrics, global_step, backbone_lr, head_lr = _train_epoch(
                model,
                train_loader,
                optimizer,
                controller,
                global_step,
                device,
                gradient_clip_norm,
                pos_weight,
                float(training["mixup_probability"]),
                f"outer{outer_fold} epoch{epoch}/{max_epochs} 学習",
            )
            validation_metrics, _ = evaluate(
                model,
                inner_loader,
                device,
                pos_weight,
                f"outer{outer_fold} epoch{epoch}/{max_epochs} val検証",
            )
            latest_metrics = validation_metrics
            stopped_epoch = epoch
            scheduler.step()

            eligible = epoch >= min_epoch
            current_auroc = validation_metrics["auroc"]
            checkpoint_improved = (
                eligible
                and np.isfinite(current_auroc)
                and current_auroc > best_metrics["auroc"]
            )
            current_prauc = validation_metrics["average_precision"]
            prauc_checkpoint_improved = (
                eligible
                and np.isfinite(current_prauc)
                and current_prauc > best_prauc_metrics["average_precision"]
            )
            early_stopping_improved = False
            if eligible:
                (
                    early_stopping_best_loss,
                    no_improvement,
                    early_stopping_improved,
                ) = _update_early_stopping(
                    validation_metrics["loss"],
                    early_stopping_best_loss,
                    no_improvement,
                )

            if checkpoint_improved:
                best_epoch = epoch
                best_metrics = validation_metrics
            if prauc_checkpoint_improved:
                best_prauc_epoch = epoch
                best_prauc_metrics = validation_metrics

            if checkpoint_improved:
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_epoch,
                    best_metrics,
                    best_prauc_epoch,
                    best_prauc_metrics,
                    early_stopping_best_loss,
                    no_improvement,
                    checkpoint_role="best_val_auroc",
                )
                if wandb_module is not None:
                    update_best_summary(wandb_module, epoch, validation_metrics)
            if prauc_checkpoint_improved:
                _save_checkpoint(
                    best_prauc_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_epoch,
                    best_metrics,
                    best_prauc_epoch,
                    best_prauc_metrics,
                    early_stopping_best_loss,
                    no_improvement,
                    checkpoint_role="best_val_prauc",
                )
                if wandb_module is not None:
                    update_best_prauc_summary(wandb_module, epoch, validation_metrics)

            _save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                global_step,
                best_epoch,
                best_metrics,
                best_prauc_epoch,
                best_prauc_metrics,
                early_stopping_best_loss,
                no_improvement,
                checkpoint_role="last",
            )
            elapsed = time.monotonic() - start_time
            row = {
                "epoch": epoch,
                "train_bce": train_metrics["loss"],
                "train_grad_norm": train_metrics["grad_norm"],
                "train_gradient_clip_fraction": train_metrics["clip_fraction"],
                "train_mixup_fraction": train_metrics["mixup_fraction"],
                "val_bce": validation_metrics["loss"],
                "val_auroc": validation_metrics["auroc"],
                "val_prauc": validation_metrics["average_precision"],
                "val_precision_at_0_5": validation_metrics["precision_at_0_5"],
                "val_recall_at_0_5": validation_metrics["recall_at_0_5"],
                "val_f1_at_0_5": validation_metrics["f1_at_0_5"],
                "val_f1_optimal_threshold": validation_metrics["f1_optimal_threshold"],
                "val_precision_at_f1_optimal": validation_metrics[
                    "precision_at_f1_optimal"
                ],
                "val_recall_at_f1_optimal": validation_metrics["recall_at_f1_optimal"],
                "val_f1_optimal": validation_metrics["f1_optimal"],
                "val_negative_score_mean": validation_metrics["negative_score_mean"],
                "val_positive_score_mean": validation_metrics["positive_score_mean"],
                "val_score_gap": validation_metrics["score_gap"],
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "train_data_wait_seconds": train_metrics["data_wait_seconds"],
                "train_compute_seconds": train_metrics["compute_seconds"],
                "epoch_seconds": elapsed,
                "is_best_val_auroc": checkpoint_improved,
                "is_best_val_prauc": prauc_checkpoint_improved,
                "early_stopping_improved": early_stopping_improved,
                "early_stopping_best_bce": early_stopping_best_loss,
                "early_stopping_bad_epochs": no_improvement,
            }
            history_rows.append(row)
            _append_history(history_path, history_rows)
            _append_log(log_path, row)
            if wandb_module is not None:
                log_wandb_epoch(
                    wandb_module,
                    epoch,
                    train_metrics,
                    validation_metrics,
                    backbone_lr,
                    head_lr,
                    elapsed,
                    early_stopping_best_loss,
                    no_improvement,
                )
            if eligible and no_improvement >= patience:
                break
    finally:
        finish_wandb(
            wandb_module,
            stopped_epoch,
            latest_metrics,
            train_rows,
            inner_rows,
        )

    if best_epoch < min_epoch or not best_path.is_file():
        raise RuntimeError("min_epoch以降に有効なbest checkpointを保存できませんでした")
    if best_prauc_epoch < min_epoch or not best_prauc_path.is_file():
        raise RuntimeError(
            "min_epoch以降に有効なPR-AUC checkpointを保存できませんでした"
        )
    outer_prediction_path = fold_dir / "outer_predictions.csv"
    outer_prauc_prediction_path = fold_dir / "outer_predictions_prauc_checkpoint.csv"
    if outer_prediction_path.exists() or outer_prauc_prediction_path.exists():
        raise RuntimeError("outer予測が既に存在するため再推論を拒否しました")

    auroc_evaluation = _evaluate_checkpoint(
        model,
        best_path,
        expected_role="best_val_auroc",
        checkpoint_label="AUROC-best",
        validation_loader=inner_loader,
        outer_loader=outer_loader,
        device=device,
        pos_weight=pos_weight,
        outer_fold=outer_fold,
    )
    prauc_evaluation = _evaluate_checkpoint(
        model,
        best_prauc_path,
        expected_role="best_val_prauc",
        checkpoint_label="PR-AUC-best",
        validation_loader=inner_loader,
        outer_loader=outer_loader,
        device=device,
        pos_weight=pos_weight,
        outer_fold=outer_fold,
    )
    _atomic_write_csv(
        auroc_evaluation.validation_predictions,
        fold_dir / "val_predictions.csv",
    )
    _atomic_write_csv(auroc_evaluation.outer_predictions, outer_prediction_path)
    _atomic_write_csv(
        prauc_evaluation.validation_predictions,
        fold_dir / "val_predictions_prauc_checkpoint.csv",
    )
    _atomic_write_csv(
        prauc_evaluation.outer_predictions,
        outer_prauc_prediction_path,
    )
    (fold_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "val_fold": int(config["runtime"]["inner_fold"]),
                "stopped_epoch": stopped_epoch,
                "primary_checkpoint": "best_val_auroc",
                "auroc_checkpoint": _checkpoint_metrics_summary(
                    best_epoch,
                    auroc_evaluation,
                ),
                "prauc_checkpoint": _checkpoint_metrics_summary(
                    best_prauc_epoch,
                    prauc_evaluation,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return FoldTrainingResult(
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_prauc_epoch=best_prauc_epoch,
        best_prauc_metrics=best_prauc_metrics,
        stopped_epoch=stopped_epoch,
        outer_predictions=auroc_evaluation.outer_predictions,
        outer_prauc_predictions=prauc_evaluation.outer_predictions,
    )


def _evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    expected_role: str,
    checkpoint_label: str,
    validation_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    outer_fold: int,
) -> CheckpointEvaluation:
    """checkpointのval最適F1閾値を同じcheckpointのouterへ固定適用する。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_role") != expected_role:
        raise ValueError(
            f"{checkpoint_label} checkpoint roleが不正です: "
            f"{checkpoint.get('checkpoint_role')}"
        )
    model.load_state_dict(checkpoint["model"])
    print(
        f"[outer {outer_fold}] {checkpoint_label} checkpointのval閾値を決定します",
        flush=True,
    )
    validation_metrics, validation_predictions = evaluate(
        model,
        validation_loader,
        device,
        pos_weight,
        f"outer{outer_fold} {checkpoint_label} val検証",
    )
    selected_threshold_metrics = select_f1_threshold(
        validation_predictions["vertebra_target"].to_numpy(),
        validation_predictions["vertebra_score"].to_numpy(),
    )
    threshold = float(selected_threshold_metrics["threshold"])
    validation_predictions = _apply_decision_threshold(
        validation_predictions,
        threshold,
    )
    print(
        f"[outer {outer_fold}] {checkpoint_label} val閾値={threshold:.6f}, "
        f"precision={selected_threshold_metrics['precision']:.6f}, "
        f"recall={selected_threshold_metrics['recall']:.6f}, "
        f"F1={selected_threshold_metrics['f1']:.6f}",
        flush=True,
    )
    print(
        f"[outer {outer_fold}] outerを1回だけ推論しています "
        f"({checkpoint_label}, val閾値を固定適用)",
        flush=True,
    )
    outer_metrics, outer_predictions = evaluate(
        model,
        outer_loader,
        device,
        pos_weight,
        f"outer{outer_fold} {checkpoint_label} outer推論",
        include_f1_optimal=False,
    )
    outer_predictions = _apply_decision_threshold(outer_predictions, threshold)
    outer_threshold_metrics = threshold_metrics(
        outer_predictions["vertebra_target"].to_numpy(),
        outer_predictions["vertebra_score"].to_numpy(),
        threshold,
    )
    print(
        f"[outer {outer_fold}] {checkpoint_label} outer: "
        f"AUROC={outer_metrics['auroc']:.6f}, "
        f"PR-AUC={outer_metrics['average_precision']:.6f}, "
        f"precision={outer_threshold_metrics['precision']:.6f}, "
        f"recall={outer_threshold_metrics['recall']:.6f}, "
        f"F1={outer_threshold_metrics['f1']:.6f}",
        flush=True,
    )
    return CheckpointEvaluation(
        validation_metrics=validation_metrics,
        validation_threshold_metrics=selected_threshold_metrics,
        outer_metrics=outer_metrics,
        outer_threshold_metrics=outer_threshold_metrics,
        validation_predictions=validation_predictions,
        outer_predictions=outer_predictions,
    )


def _apply_decision_threshold(
    predictions: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """予測表へ固定閾値と二値判定を追加したcopyを返す。"""
    return predictions.assign(
        decision_threshold=threshold,
        vertebra_prediction=(
            predictions["vertebra_score"].to_numpy() >= threshold
        ).astype(np.int8),
    )


def _checkpoint_metrics_summary(
    checkpoint_epoch: int,
    evaluation: CheckpointEvaluation,
) -> dict[str, Any]:
    """fold metrics用にcheckpointのval・outer指標をまとめる。"""
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "threshold_selection": "maximum_val_f1",
        "threshold_tie_break": "highest_threshold",
        "validation": {
            **evaluation.validation_metrics,
            **evaluation.validation_threshold_metrics,
        },
        "outer": {
            **evaluation.outer_metrics,
            **evaluation.outer_threshold_metrics,
        },
    }


def _resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_path: Path,
    device: torch.device,
    config: dict[str, Any],
    resume: bool,
) -> tuple[int, int, int, dict[str, float], int, dict[str, float], float, int]:
    """必要時に最後のcheckpointを復元し、学習再開状態を返す。"""
    default_metrics = {
        "loss": float("inf"),
        "auroc": float("-inf"),
        "average_precision": float("nan"),
    }
    if not resume:
        default_prauc_metrics = {
            **default_metrics,
            "average_precision": float("-inf"),
        }
        return 1, 0, 0, default_metrics, 0, default_prauc_metrics, float("inf"), 0
    if not last_path.is_file():
        raise FileNotFoundError(f"resume対象checkpointがありません: {last_path}")
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("checkpointの実効configが現在のconfigと一致しません")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        int(checkpoint["best_epoch"]),
        {key: float(value) for key, value in checkpoint["best_metrics"].items()},
        int(checkpoint["best_prauc_epoch"]),
        {key: float(value) for key, value in checkpoint["best_prauc_metrics"].items()},
        float(checkpoint["early_stopping_best_loss"]),
        int(checkpoint["no_improvement"]),
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    best_prauc_epoch: int,
    best_prauc_metrics: dict[str, float],
    early_stopping_best_loss: float,
    no_improvement: int,
    checkpoint_role: str,
) -> None:
    """学習再開に必要な状態をアトミックに保存する。"""
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_prauc_epoch": best_prauc_epoch,
            "best_prauc_metrics": best_prauc_metrics,
            "early_stopping_best_loss": early_stopping_best_loss,
            "no_improvement": no_improvement,
            "checkpoint_role": checkpoint_role,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor | list[str]]],
    optimizer: torch.optim.Optimizer,
    controller: LearningRateController,
    global_step: int,
    device: torch.device,
    gradient_clip_norm: float,
    pos_weight: float,
    mixup_probability: float,
    progress_description: str,
) -> tuple[dict[str, float], int, float, float]:
    """1 epoch学習し、平均BCE・勾配ノルム・最後の学習率を返す。"""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_rows = 0
    clipped_batches = 0
    mixed_batches = 0
    batch_count = 0
    backbone_lr = 0.0
    head_lr = 0.0
    data_wait_seconds = 0.0
    compute_seconds = 0.0
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    wait_started = time.monotonic()
    for batch in progress:
        batch_received = time.monotonic()
        data_wait_seconds += batch_received - wait_started
        compute_started = batch_received
        backbone_lr, head_lr = controller.apply(optimizer, global_step)
        inputs, targets = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        use_mixup = (
            mixup_probability > 0.0 and float(torch.rand(1).item()) < mixup_probability
        )
        if use_mixup:
            inputs, targets_a, targets_b, mixup_lambda = _mixup_batch(inputs, targets)
            mixed_batches += 1
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
            if use_mixup:
                loss = mixup_lambda * broadcast_bce_loss(
                    logits, targets_a, pos_weight
                ) + (1.0 - mixup_lambda) * broadcast_bce_loss(
                    logits, targets_b, pos_weight
                )
            else:
                loss = broadcast_bce_loss(logits, targets, pos_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("学習lossが非有限値です")
        torch.autograd.backward(loss)
        gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("学習gradientが非有限値です")
        if float(gradient_norm) > gradient_clip_norm:
            clipped_batches += 1
        optimizer.step()
        batch_rows = len(targets)
        total_loss += float(loss.detach())
        total_grad_norm += float(gradient_norm) * batch_rows
        total_rows += batch_rows
        batch_count += 1
        global_step += 1
        progress.set_postfix(bce=f"{total_loss / batch_count:.4f}")
        compute_seconds += time.monotonic() - compute_started
        wait_started = time.monotonic()
    if total_rows == 0:
        raise ValueError("train loaderが空です")
    return (
        {
            "loss": total_loss / batch_count,
            "grad_norm": total_grad_norm / total_rows,
            "clip_fraction": clipped_batches / batch_count,
            "mixup_fraction": mixed_batches / batch_count,
            "data_wait_seconds": data_wait_seconds,
            "compute_seconds": compute_seconds,
        },
        global_step,
        backbone_lr,
        head_lr,
    )


def _mixup_batch(
    inputs: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Stage1と同じ一様lambdaと共有batch permutationでmixupする。"""
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixup_lambda = float(np.random.uniform(0.0, 1.0))
    mixed_inputs = inputs * mixup_lambda + inputs[indices] * (1.0 - mixup_lambda)
    return mixed_inputs, targets, targets[indices], mixup_lambda


def _update_early_stopping(
    current_loss: float,
    best_loss: float,
    bad_epochs: int,
) -> tuple[float, int, bool]:
    """Update validation-loss early-stopping state."""
    if not np.isfinite(current_loss):
        raise FloatingPointError("validation lossが非有限値です")
    if current_loss < best_loss:
        return current_loss, 0, True
    return best_loss, bad_epochs + 1, False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    progress_description: str,
    include_f1_optimal: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """検証BCE、AUROC、APと個票予測を返す。"""
    model.eval()
    total_loss = 0.0
    total_rows = 0
    batch_count = 0
    records: list[dict[str, float | int | str]] = []
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    for batch in progress:
        inputs, targets = _batch_tensors(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
            loss = broadcast_bce_loss(logits, targets, pos_weight)
        probabilities = bag_probabilities(logits).float().cpu().numpy()
        target_values = targets.float().cpu().numpy()
        folds = _batch_folds(batch)
        study_ids = _batch_strings(batch, "study_id")
        levels = _batch_strings(batch, "level")
        for study_id, level, fold, target, probability in zip(
            study_ids, levels, folds, target_values, probabilities, strict=True
        ):
            records.append(
                {
                    "study_id": study_id,
                    "level": level,
                    "fold": fold,
                    "vertebra_target": int(target),
                    "vertebra_score": float(probability),
                }
            )
        total_loss += float(loss)
        total_rows += len(targets)
        batch_count += 1
        progress.set_postfix(bce=f"{total_loss / batch_count:.4f}")
    if total_rows == 0:
        raise ValueError("validation loaderが空です")
    predictions = pd.DataFrame(records)
    targets = predictions["vertebra_target"].to_numpy()
    scores = predictions["vertebra_score"].to_numpy()
    negative_score_mean = float(scores[targets == 0].mean())
    positive_score_mean = float(scores[targets == 1].mean())
    fixed_threshold_metrics = threshold_metrics(targets, scores, threshold=0.5)
    metrics = {
        "loss": total_loss / batch_count,
        "auroc": safe_auroc(targets, scores),
        "average_precision": safe_average_precision(targets, scores),
        "negative_score_mean": negative_score_mean,
        "positive_score_mean": positive_score_mean,
        "score_gap": positive_score_mean - negative_score_mean,
        "precision_at_0_5": float(fixed_threshold_metrics["precision"]),
        "recall_at_0_5": float(fixed_threshold_metrics["recall"]),
        "f1_at_0_5": float(fixed_threshold_metrics["f1"]),
    }
    if include_f1_optimal:
        optimal_threshold_metrics = select_f1_threshold(targets, scores)
        metrics.update(
            {
                "f1_optimal_threshold": float(optimal_threshold_metrics["threshold"]),
                "precision_at_f1_optimal": float(
                    optimal_threshold_metrics["precision"]
                ),
                "recall_at_f1_optimal": float(optimal_threshold_metrics["recall"]),
                "f1_optimal": float(optimal_threshold_metrics["f1"]),
            }
        )
    return metrics, predictions


def _batch_tensors(batch: Batch, device: torch.device) -> tuple[Tensor, Tensor]:
    """batchからdevice上の入力と椎体ターゲットを取得する。"""
    inputs = batch["inputs"]
    targets = batch["vertebra_target"]
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("batchのinputs/vertebra_targetはTensorである必要があります")
    moved_inputs = inputs.to(device, non_blocking=True)
    if moved_inputs.dtype == torch.uint8:
        moved_inputs = moved_inputs.to(dtype=torch.float32).div_(255.0)
    return moved_inputs, targets.to(device, non_blocking=True)


def _dataset_length(loader: DataLoader[Any]) -> int:
    """DataLoaderのDatasetがSizedであることを確認して件数を返す。"""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("DataLoaderのdatasetはSizedである必要があります")
    return len(dataset)


def _set_loader_epoch(loader: DataLoader[Any], epoch: int) -> None:
    """natural samplerへepochを伝え、arm間の入力順序を固定する。"""
    sampler = loader.sampler
    if not isinstance(sampler, EpochShuffleSampler):
        raise TypeError("train loaderにはEpochShuffleSamplerが必要です")
    sampler.set_epoch(epoch)


def _batch_strings(batch: Batch, key: str) -> list[str]:
    """DataLoaderのcollate後の文字列列をリストへ変換する。"""
    values = batch[key]
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"batchの{key}は文字列listである必要があります")
    return values


def _batch_folds(batch: Batch) -> list[int]:
    """DataLoaderのcollate後のfold Tensorを整数リストへ変換する。"""
    folds = batch["fold"]
    if not isinstance(folds, Tensor):
        raise TypeError("batchのfoldはTensorである必要があります")
    values = folds.cpu().tolist()
    if not isinstance(values, list) or not all(
        isinstance(value, int) for value in values
    ):
        raise TypeError("batchのfold値は整数listである必要があります")
    return cast(list[int], values)


def _append_history(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    """蓄積したepoch履歴をCSVへ安全に更新する。"""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """予測CSVを一時ファイル経由で置換する。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _load_history(path: Path) -> list[dict[str, float | int | bool]]:
    """再開時に既存epoch履歴を読み戻す。"""
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return cast(list[dict[str, float | int | bool]], frame.to_dict(orient="records"))


def _append_log(path: Path, row: dict[str, float | int | bool]) -> None:
    """コンソールと同じ主要epoch情報をローカルログへ追記する。"""
    line = (
        f"epoch={row['epoch']} train_bce={row['train_bce']:.6f} "
        f"clip_fraction={row['train_gradient_clip_fraction']:.3f} "
        f"mixup_fraction={row['train_mixup_fraction']:.3f} "
        f"val_bce={row['val_bce']:.6f} val_auroc={row['val_auroc']:.6f} "
        f"val_prauc={row['val_prauc']:.6f} "
        f"val_precision@0.5={row['val_precision_at_0_5']:.6f} "
        f"val_recall@0.5={row['val_recall_at_0_5']:.6f} "
        f"val_f1@0.5={row['val_f1_at_0_5']:.6f} "
        f"val_f1_optimal={row['val_f1_optimal']:.6f} "
        f"val_f1_threshold={row['val_f1_optimal_threshold']:.6f} "
        f"val_score_gap={row['val_score_gap']:.6f} "
        f"backbone_lr={row['backbone_lr']:.3e} head_lr={row['head_lr']:.3e} "
        f"data_wait={row['train_data_wait_seconds']:.2f}s "
        f"compute={row['train_compute_seconds']:.2f}s "
        f"seconds={row['epoch_seconds']:.2f} "
        f"auroc_best={row['is_best_val_auroc']} "
        f"prauc_best={row['is_best_val_prauc']} "
        f"stop_best_bce={row['early_stopping_best_bce']:.6f} "
        f"stop_bad_epochs={row['early_stopping_bad_epochs']}"
    )
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
