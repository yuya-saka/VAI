"""Baseline 1のfold単位の学習・検証・checkpoint制御。"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fracture_detection.baseline1.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    update_best_summary,
)
from fracture_detection.baseline1.losses import bag_probabilities, broadcast_bce_loss
from fracture_detection.baseline1.optimization import (
    LearningRateController,
    create_optimizer,
    create_plateau_scheduler,
    optimizer_learning_rates,
)
from fracture_detection.common.metrics import safe_auroc, safe_average_precision

Batch = dict[str, object]


@dataclass(frozen=True)
class FoldTrainingResult:
    """1 fold学習の再利用可能な要約。"""

    best_epoch: int
    best_metrics: dict[str, float]
    stopped_epoch: int
    predictions: pd.DataFrame


def set_seed(seed: int, fold: int) -> None:
    """foldごとの決定的な乱数状態を設定する。"""
    resolved_seed = seed + fold
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader workerのnumpy/random seedを固定する。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_data_loader(
    dataset: torch.utils.data.Dataset[dict[str, Tensor | str]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[Any]:
    """通常のshuffleだけを使う決定的なDataLoaderを作る。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
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
    validation_loader: DataLoader[Any],
    config: dict[str, Any],
    fold: int,
    fold_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> FoldTrainingResult:
    """1 foldを学習し、最良checkpointの検証予測を返す。"""
    training = config["training"]
    max_epochs = int(training["max_epochs"])
    min_epoch = int(training["min_epoch"])
    patience = int(training["early_stopping_patience"])
    pos_weight = float(training["pos_weight"])
    train_rows = _dataset_length(train_loader)
    validation_rows = _dataset_length(validation_loader)
    print(
        f"[fold {fold}] 学習を初期化しています: device={device}, "
        f"train={train_rows:,} bag/{len(train_loader)} batch, "
        f"validation={validation_rows:,} bag/{len(validation_loader)} batch",
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
    scheduler = create_plateau_scheduler(
        optimizer,
        factor=float(training["plateau_factor"]),
        patience=int(training["plateau_patience"]),
        threshold=float(training["plateau_threshold"]),
        cooldown=int(training["plateau_cooldown"]),
        backbone_min_learning_rate=float(training["backbone_min_learning_rate"]),
        head_min_learning_rate=float(training["head_min_learning_rate"]),
    )
    model.to(device)
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pt"
    last_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "history.csv"
    log_path = fold_dir / "training.log"

    start_epoch, global_step, best_epoch, best_metrics, no_improvement = _resume_state(
        model, optimizer, scheduler, last_path, device, resume
    )
    history_rows = _load_history(history_path) if resume else []
    print(f"[fold {fold}] W&Bを初期化しています", flush=True)
    wandb_module = initialize_wandb(config, fold)
    print(f"[fold {fold}] W&B初期化処理が完了しました", flush=True)
    latest_metrics: dict[str, float] | None = None
    stopped_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            print(
                f"[fold {fold}] epoch {epoch}/{max_epochs}を開始します。"
                "最初のbatchではworker起動とprefetchを行います",
                flush=True,
            )
            start_time = time.monotonic()
            controller.set_epoch_state(model, epoch - 1)
            train_metrics, global_step, backbone_lr, head_lr = _train_epoch(
                model,
                train_loader,
                optimizer,
                controller,
                global_step,
                device,
                float(training["gradient_clip_norm"]),
                pos_weight,
                f"fold{fold} epoch{epoch}/{max_epochs} 学習",
            )
            validation_metrics, _ = evaluate(
                model,
                validation_loader,
                device,
                pos_weight,
                f"fold{fold} epoch{epoch}/{max_epochs} 検証",
            )
            latest_metrics = validation_metrics
            stopped_epoch = epoch
            warmup_end = int(training["freeze_backbone_epochs"]) + int(
                training["warmup_epochs"]
            )
            if epoch >= warmup_end:
                scheduler.step(validation_metrics["loss"])
            backbone_lr, head_lr = optimizer_learning_rates(optimizer)

            eligible = epoch >= min_epoch
            current_auroc = validation_metrics["auroc"]
            improved = (
                eligible
                and np.isfinite(current_auroc)
                and current_auroc > best_metrics["auroc"]
            )
            if improved:
                best_epoch = epoch
                best_metrics = validation_metrics
                no_improvement = 0
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
                    no_improvement,
                )
                if wandb_module is not None:
                    update_best_summary(wandb_module, epoch, validation_metrics)
            elif eligible:
                no_improvement += 1

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
                no_improvement,
            )
            elapsed = time.monotonic() - start_time
            row = {
                "epoch": epoch,
                "train_bce": train_metrics["loss"],
                "train_grad_norm": train_metrics["grad_norm"],
                "train_gradient_clip_fraction": train_metrics["clip_fraction"],
                "val_bce": validation_metrics["loss"],
                "val_auroc": validation_metrics["auroc"],
                "val_average_precision": validation_metrics["average_precision"],
                "val_negative_score_mean": validation_metrics["negative_score_mean"],
                "val_positive_score_mean": validation_metrics["positive_score_mean"],
                "val_score_gap": validation_metrics["score_gap"],
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "epoch_seconds": elapsed,
                "is_best": improved,
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
                )
            if eligible and no_improvement >= patience:
                break
    finally:
        finish_wandb(
            wandb_module,
            stopped_epoch,
            latest_metrics,
            train_rows,
            validation_rows,
        )

    if best_epoch < min_epoch or not best_path.is_file():
        raise RuntimeError("min_epoch以降に有効なbest checkpointを保存できませんでした")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"[fold {fold}] 最良checkpointを再評価しています", flush=True)
    final_metrics, predictions = evaluate(
        model,
        validation_loader,
        device,
        pos_weight,
        f"fold{fold} 最良checkpoint検証",
    )
    predictions.to_csv(fold_dir / "val_predictions.csv", index=False)
    (fold_dir / "fold_metrics.json").write_text(
        pd.Series(
            {
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_val_auroc": best_metrics["auroc"],
                "best_val_average_precision": best_metrics["average_precision"],
                "best_val_bce": best_metrics["loss"],
                "checkpoint_val_auroc": final_metrics["auroc"],
                "checkpoint_val_average_precision": final_metrics["average_precision"],
                "checkpoint_val_bce": final_metrics["loss"],
            }
        ).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FoldTrainingResult(best_epoch, best_metrics, stopped_epoch, predictions)


def _resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    last_path: Path,
    device: torch.device,
    resume: bool,
) -> tuple[int, int, int, dict[str, float], int]:
    """必要時に最後のcheckpointを復元し、学習再開状態を返す。"""
    default_metrics = {
        "loss": float("inf"),
        "auroc": float("-inf"),
        "average_precision": float("nan"),
    }
    if not resume:
        return 1, 0, 0, default_metrics, 0
    if not last_path.is_file():
        raise FileNotFoundError(f"resume対象checkpointがありません: {last_path}")
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        int(checkpoint["best_epoch"]),
        {key: float(value) for key, value in checkpoint["best_metrics"].items()},
        int(checkpoint["no_improvement"]),
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    no_improvement: int,
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
            "no_improvement": no_improvement,
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
    progress_description: str,
) -> tuple[dict[str, float], int, float, float]:
    """1 epoch学習し、平均BCE・勾配ノルム・最後の学習率を返す。"""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_rows = 0
    clipped_batches = 0
    batch_count = 0
    backbone_lr = 0.0
    head_lr = 0.0
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    for batch in progress:
        backbone_lr, head_lr = controller.apply(optimizer, global_step)
        inputs, targets = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
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
        total_loss += float(loss.detach()) * batch_rows
        total_grad_norm += float(gradient_norm) * batch_rows
        total_rows += batch_rows
        batch_count += 1
        global_step += 1
        progress.set_postfix(bce=f"{total_loss / total_rows:.4f}")
    if total_rows == 0:
        raise ValueError("train loaderが空です")
    return (
        {
            "loss": total_loss / total_rows,
            "grad_norm": total_grad_norm / total_rows,
            "clip_fraction": clipped_batches / batch_count,
        },
        global_step,
        backbone_lr,
        head_lr,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    progress_description: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """検証BCE、AUROC、APと個票予測を返す。"""
    model.eval()
    total_loss = 0.0
    total_rows = 0
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
        total_loss += float(loss) * len(targets)
        total_rows += len(targets)
        progress.set_postfix(bce=f"{total_loss / total_rows:.4f}")
    if total_rows == 0:
        raise ValueError("validation loaderが空です")
    predictions = pd.DataFrame(records)
    targets = predictions["vertebra_target"].to_numpy()
    scores = predictions["vertebra_score"].to_numpy()
    negative_score_mean = float(scores[targets == 0].mean())
    positive_score_mean = float(scores[targets == 1].mean())
    return (
        {
            "loss": total_loss / total_rows,
            "auroc": safe_auroc(targets, scores),
            "average_precision": safe_average_precision(targets, scores),
            "negative_score_mean": negative_score_mean,
            "positive_score_mean": positive_score_mean,
            "score_gap": positive_score_mean - negative_score_mean,
        },
        predictions,
    )


def _batch_tensors(batch: Batch, device: torch.device) -> tuple[Tensor, Tensor]:
    """batchからdevice上の入力と椎体ターゲットを取得する。"""
    inputs = batch["inputs"]
    targets = batch["vertebra_target"]
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("batchのinputs/vertebra_targetはTensorである必要があります")
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _dataset_length(loader: DataLoader[Any]) -> int:
    """DataLoaderのDatasetがSizedであることを確認して件数を返す。"""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("DataLoaderのdatasetはSizedである必要があります")
    return len(dataset)


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
        f"val_bce={row['val_bce']:.6f} val_auroc={row['val_auroc']:.6f} "
        f"val_ap={row['val_average_precision']:.6f} "
        f"val_score_gap={row['val_score_gap']:.6f} "
        f"backbone_lr={row['backbone_lr']:.3e} head_lr={row['head_lr']:.3e} "
        f"seconds={row['epoch_seconds']:.2f} best={row['is_best']}"
    )
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
