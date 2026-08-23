"""全アーム共通のnested fold学習・resume・outer guard。"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from fracture_detection.common.constants import REGION_COLUMNS
from fracture_detection.common.metrics import (
    safe_auroc,
    safe_average_precision,
    select_f1_threshold,
    threshold_metrics,
)
from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.core.contracts import LossWeights
from fracture_detection.core.losses import bag_probabilities, broadcast_bce_loss
from fracture_detection.core.optimization import (
    LearningRateController,
    create_cosine_scheduler,
    create_optimizer,
)
from fracture_detection.core.rng import (
    TrainingRngStreams,
    checkpoint_rng_state,
    restore_checkpoint_rng_state,
)
from fracture_detection.core.steps import (
    ArmAdapter,
    GradientNorms,
    prepare_batch,
    train_step,
)
from fracture_detection.core.wandb import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
)


@dataclass(frozen=True)
class FoldTrainingResult:
    """1 outer foldの学習結果。"""

    best_epoch: int
    best_prauc_epoch: int
    stopped_epoch: int
    outer_predictions: pd.DataFrame
    outer_prauc_predictions: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResult:
    """検証指標とbag予測。"""

    metrics: dict[str, float]
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ResumeState:
    """epoch境界resumeに必要なscalar状態。"""

    start_epoch: int
    global_step: int
    best_epoch: int
    best_auroc: float
    best_prauc_epoch: int
    best_prauc: float
    early_stopping_best_loss: float
    bad_epochs: int


def set_seed(seed: int, outer_fold: int) -> None:
    """foldごとのglobal RNGを初期化する。"""
    resolved = seed + outer_fold
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader workerの非augmentation RNGを初期化する。"""
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
    sampler: Sampler[Any] | None = None,
) -> DataLoader[Any]:
    """明示samplerと独立worker generatorを持つDataLoaderを返す。"""
    generator = torch.Generator(device="cpu")
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
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **options)


def train_fold(
    model: nn.Module,
    adapter: ArmAdapter,
    natural_loader: DataLoader[Any],
    annotated_loader: DataLoader[Any] | None,
    validation_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    config: dict[str, Any],
    outer_fold: int,
    fold_dir: Path,
    device: torch.device,
    loss_weights: LossWeights,
    *,
    resume: bool = False,
    max_steps_per_epoch: int | None = None,
    run_outer_inference: bool = True,
) -> FoldTrainingResult:
    """innerでcheckpoint選択後、outerを各checkpoint1回だけ推論する。"""
    training = cast(dict[str, Any], config["training"])
    if not isinstance(natural_loader.sampler, EpochShuffleSampler):
        raise TypeError("natural loaderにはEpochShuffleSamplerが必要です")
    if adapter.region_enabled and (
        annotated_loader is None
        or not isinstance(annotated_loader.sampler, AnnotatedCycleSampler)
    ):
        raise TypeError("region有効armにはAnnotatedCycleSamplerが必要です")
    optimizer = create_optimizer(
        model,
        float(training["weight_decay"]),
        float(training["backbone_learning_rate"]),
        float(training["head_learning_rate"]),
    )
    max_epochs = int(training["max_epochs"])
    controller = LearningRateController(
        steps_per_epoch=(
            min(len(natural_loader), max_steps_per_epoch)
            if max_steps_per_epoch is not None
            else len(natural_loader)
        ),
        freeze_backbone_epochs=int(training["freeze_backbone_epochs"]),
        warmup_epochs=int(training["warmup_epochs"]),
        warmup_start_factor=float(training["warmup_start_factor"]),
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        head_learning_rate=float(training["head_learning_rate"]),
    )
    scheduler = create_cosine_scheduler(
        optimizer,
        max_epochs,
        float(training["backbone_min_learning_rate"]),
        float(training["head_min_learning_rate"]),
    )
    base_seed = int(config["data"]["random_seed"]) + outer_fold
    rng_streams = TrainingRngStreams(
        mixup_seed=base_seed + 30_000,
        annotated_seed=base_seed + 40_000,
    )
    model.to(device)
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "best": fold_dir / "best_model.pt",
        "prauc": fold_dir / "best_val_prauc_model.pt",
        "last": fold_dir / "last_checkpoint.pt",
        "history": fold_dir / "history.csv",
    }
    state = _resume_state(
        model,
        optimizer,
        scheduler,
        rng_streams,
        paths["last"],
        device,
        config,
        resume,
    )
    history = _load_history(paths["history"]) if resume else []
    wandb_run = initialize_wandb(config, outer_fold, fold_dir)
    stopped_epoch = state.start_epoch - 1
    current = state
    print(
        f"[outer {outer_fold}] train={_dataset_length(natural_loader):,}, "
        f"val={_dataset_length(validation_loader):,}, "
        f"outer={_dataset_length(outer_loader):,}, device={device}",
        flush=True,
    )
    for epoch in range(state.start_epoch, max_epochs + 1):
        started = time.monotonic()
        _set_epoch(natural_loader, epoch - 1)
        if annotated_loader is not None:
            _set_epoch(annotated_loader, epoch - 1)
        controller.set_epoch_state(model, epoch - 1)
        train_metrics, global_step, backbone_lr, head_lr = _train_epoch(
            model,
            adapter,
            natural_loader,
            annotated_loader,
            optimizer,
            controller,
            current.global_step,
            device,
            rng_streams,
            loss_weights,
            training,
            max_steps=max_steps_per_epoch,
            description=f"outer{outer_fold} epoch{epoch}/{max_epochs}",
        )
        validation = evaluate(
            model,
            adapter,
            validation_loader,
            device,
            float(training["pos_weight"]),
            f"outer{outer_fold} epoch{epoch} val",
        )
        scheduler.step()
        eligible = epoch >= int(training["min_epoch"])
        improved_auroc = (
            eligible
            and math.isfinite(validation.metrics["auroc"])
            and validation.metrics["auroc"] > current.best_auroc
        )
        improved_prauc = (
            eligible
            and math.isfinite(validation.metrics["average_precision"])
            and validation.metrics["average_precision"] > current.best_prauc
        )
        best_epoch = epoch if improved_auroc else current.best_epoch
        best_auroc = (
            validation.metrics["auroc"] if improved_auroc else current.best_auroc
        )
        best_prauc_epoch = epoch if improved_prauc else current.best_prauc_epoch
        best_prauc = (
            validation.metrics["average_precision"]
            if improved_prauc
            else current.best_prauc
        )
        best_loss, bad_epochs, stop_improved = _early_stopping(
            validation.metrics["loss"],
            current.early_stopping_best_loss,
            current.bad_epochs,
            eligible,
        )
        current = ResumeState(
            start_epoch=epoch + 1,
            global_step=global_step,
            best_epoch=best_epoch,
            best_auroc=best_auroc,
            best_prauc_epoch=best_prauc_epoch,
            best_prauc=best_prauc,
            early_stopping_best_loss=best_loss,
            bad_epochs=bad_epochs,
        )
        if improved_auroc:
            _save_checkpoint(
                paths["best"],
                model,
                optimizer,
                scheduler,
                rng_streams,
                config,
                current,
                epoch,
                "best_val_auroc",
            )
        if improved_prauc:
            _save_checkpoint(
                paths["prauc"],
                model,
                optimizer,
                scheduler,
                rng_streams,
                config,
                current,
                epoch,
                "best_val_prauc",
            )
        _save_checkpoint(
            paths["last"],
            model,
            optimizer,
            scheduler,
            rng_streams,
            config,
            current,
            epoch,
            "last",
        )
        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation.metrics.items()},
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "epoch_seconds": time.monotonic() - started,
            "is_best_val_auroc": improved_auroc,
            "is_best_val_prauc": improved_prauc,
            "early_stopping_improved": stop_improved,
            "early_stopping_best_bce": best_loss,
            "early_stopping_bad_epochs": bad_epochs,
        }
        history.append(row)
        _write_history(paths["history"], history)
        log_wandb_epoch(wandb_run, row)
        stopped_epoch = epoch
        print(
            f"[outer {outer_fold}] epoch={epoch} "
            f"train={train_metrics['total_loss']:.6f} "
            f"val_bce={validation.metrics['loss']:.6f} "
            f"val_auroc={validation.metrics['auroc']:.6f}",
            flush=True,
        )
        if eligible and bad_epochs >= int(training["early_stopping_patience"]):
            break
        if max_steps_per_epoch is not None:
            break
    finish_wandb(wandb_run)
    if current.best_epoch < int(training["min_epoch"]) or not paths["best"].is_file():
        raise RuntimeError("有効なAUROC-best checkpointがありません")
    if (
        current.best_prauc_epoch < int(training["min_epoch"])
        or not paths["prauc"].is_file()
    ):
        raise RuntimeError("有効なPR-AUC-best checkpointがありません")
    if not run_outer_inference:
        return FoldTrainingResult(
            best_epoch=current.best_epoch,
            best_prauc_epoch=current.best_prauc_epoch,
            stopped_epoch=stopped_epoch,
            outer_predictions=pd.DataFrame(),
            outer_prauc_predictions=pd.DataFrame(),
        )
    output_paths = {
        "best": fold_dir / "outer_predictions.csv",
        "prauc": fold_dir / "outer_predictions_prauc_checkpoint.csv",
    }
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError("outer予測が既に存在するため再推論を拒否しました")
    best_outer = _evaluate_checkpoint(
        model,
        adapter,
        paths["best"],
        "best_val_auroc",
        validation_loader,
        outer_loader,
        device,
        float(training["pos_weight"]),
        outer_fold,
    )
    prauc_outer = _evaluate_checkpoint(
        model,
        adapter,
        paths["prauc"],
        "best_val_prauc",
        validation_loader,
        outer_loader,
        device,
        float(training["pos_weight"]),
        outer_fold,
    )
    _atomic_csv(best_outer, output_paths["best"])
    _atomic_csv(prauc_outer, output_paths["prauc"])
    return FoldTrainingResult(
        best_epoch=current.best_epoch,
        best_prauc_epoch=current.best_prauc_epoch,
        stopped_epoch=stopped_epoch,
        outer_predictions=best_outer,
        outer_prauc_predictions=prauc_outer,
    )


def _train_epoch(
    model: nn.Module,
    adapter: ArmAdapter,
    natural_loader: DataLoader[Any],
    annotated_loader: DataLoader[Any] | None,
    optimizer: torch.optim.Optimizer,
    controller: LearningRateController,
    global_step: int,
    device: torch.device,
    rng_streams: TrainingRngStreams,
    loss_weights: LossWeights,
    training: Mapping[str, Any],
    *,
    max_steps: int | None,
    description: str,
) -> tuple[dict[str, float], int, float, float]:
    model.train()
    natural_steps = (
        len(natural_loader)
        if max_steps is None
        else min(len(natural_loader), max_steps)
    )
    annotated_iterator = (
        iter(annotated_loader) if annotated_loader is not None else None
    )
    totals: defaultdict[str, float] = defaultdict(float)
    visits: Counter[str] = Counter()
    measured: list[GradientNorms] = []
    measure_steps = {0, 126, 252, 378}
    backbone_lr = 0.0
    head_lr = 0.0
    progress = tqdm(natural_loader, total=natural_steps, desc=description, leave=False)
    for step_index, natural_batch in enumerate(progress):
        if step_index >= natural_steps:
            break
        annotated_batch = None
        if annotated_iterator is not None:
            try:
                annotated_batch = next(annotated_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "annotated loaderがnatural step数より短いです"
                ) from error
            for study_id, level in zip(
                _strings(annotated_batch, "study_id"),
                _strings(annotated_batch, "level"),
                strict=True,
            ):
                visits[f"{study_id}/{level}"] += 1
        backbone_lr, head_lr = controller.apply(optimizer, global_step)
        clip_value = training["gradient_clip_norm"]
        result = train_step(
            model,
            adapter,
            natural_batch,
            annotated_batch,
            optimizer,
            device,
            rng_streams,
            loss_weights,
            pos_weight=float(training["pos_weight"]),
            mixup_probability=float(training["mixup_probability"]),
            gradient_clip_norm=(
                float("inf") if clip_value is None else float(clip_value)
            ),
            measure_gradient_components=step_index in measure_steps,
        )
        for key, value in {
            "whole_loss": result.whole_loss,
            "region_loss": result.region_loss,
            "attention_loss": result.attention_loss,
            "total_loss": result.total_loss,
            "gradient_norm": result.gradient_norm,
            "clipped": float(result.clipped),
            "mixed": float(result.mixed),
        }.items():
            totals[key] += value
        if result.gradient_components is not None:
            measured.append(result.gradient_components)
        global_step += 1
        progress.set_postfix(loss=f"{totals['total_loss'] / (step_index + 1):.4f}")
    metrics = {key: float(value / natural_steps) for key, value in totals.items()}
    metrics.update(_gradient_medians(measured))
    if annotated_loader is not None:
        annotated_rows = _dataset_length(annotated_loader)
        visit_values = list(visits.values())
        metrics.update(
            region_optimizer_steps=float(natural_steps),
            region_passes=float(natural_steps / annotated_rows),
            region_unique_bags=float(len(visits)),
            region_visits_min=float(min(visit_values)),
            region_visits_median=float(np.median(visit_values)),
            region_visits_max=float(max(visit_values)),
        )
    return metrics, global_step, backbone_lr, head_lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    adapter: ArmAdapter,
    loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    description: str,
) -> EvaluationResult:
    """whole指標とregion列を持つ個票予測を返す。"""
    model.eval()
    total_loss = 0.0
    records: list[dict[str, float | int | str | bool]] = []
    batches = 0
    for batch in tqdm(loader, desc=description, leave=False):
        prepared = prepare_batch(batch, device, adapter.input_channels)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = adapter.forward(model, prepared.inputs)
            loss = broadcast_bce_loss(
                output.whole_logits, prepared.vertebra_targets, pos_weight
            )
        whole_scores = bag_probabilities(output.whole_logits).float().cpu()
        region_scores = (
            output.region_logits.sigmoid().mean(dim=1).float().cpu()
            if output.region_logits is not None
            else None
        )
        region_targets = (
            prepared.region_targets.float().cpu()
            if prepared.region_targets is not None
            else None
        )
        region_valid = (
            prepared.region_target_valid.bool().cpu()
            if prepared.region_target_valid is not None
            else None
        )
        folds = _tensor(batch, "fold").tolist()
        raw_has_region_target = batch.get("has_region_target")
        if raw_has_region_target is None:
            has_region_targets = (
                region_valid.any(dim=1).tolist()
                if region_valid is not None
                else [False] * len(folds)
            )
        elif isinstance(raw_has_region_target, Tensor):
            has_region_targets = raw_has_region_target.bool().tolist()
        else:
            raise TypeError("batchのhas_region_targetはTensorである必要があります")
        study_ids = _strings(batch, "study_id")
        levels = _strings(batch, "level")
        for index, (study_id, level, fold) in enumerate(
            zip(study_ids, levels, folds, strict=True)
        ):
            record: dict[str, float | int | str | bool] = {
                "study_id": study_id,
                "level": level,
                "fold": int(fold),
                "vertebra_target": int(prepared.vertebra_targets[index].cpu()),
                "vertebra_score": float(whole_scores[index]),
                "has_region_target": bool(has_region_targets[index]),
            }
            if region_scores is not None and region_targets is not None:
                for region_index, column in enumerate(REGION_COLUMNS):
                    record[f"{column}_score"] = float(
                        region_scores[index, region_index]
                    )
                    record[f"{column}_target"] = float(
                        region_targets[index, region_index]
                    )
                    if region_valid is not None:
                        record[f"{column}_target_valid"] = bool(
                            region_valid[index, region_index]
                        )
            records.append(record)
        total_loss += float(loss)
        batches += 1
    if not records:
        raise ValueError("evaluation loaderが空です")
    frame = pd.DataFrame(records)
    targets = frame["vertebra_target"].to_numpy()
    scores = frame["vertebra_score"].to_numpy()
    fixed = threshold_metrics(targets, scores, 0.5)
    optimal = select_f1_threshold(targets, scores)
    negative = float(scores[targets == 0].mean())
    positive = float(scores[targets == 1].mean())
    return EvaluationResult(
        metrics={
            "loss": total_loss / batches,
            "auroc": safe_auroc(targets, scores),
            "average_precision": safe_average_precision(targets, scores),
            "precision_at_0_5": float(fixed["precision"]),
            "recall_at_0_5": float(fixed["recall"]),
            "f1_at_0_5": float(fixed["f1"]),
            "f1_optimal_threshold": float(optimal["threshold"]),
            "f1_optimal": float(optimal["f1"]),
            "negative_score_mean": negative,
            "positive_score_mean": positive,
            "score_gap": positive - negative,
        },
        predictions=frame,
    )


def _evaluate_checkpoint(
    model: nn.Module,
    adapter: ArmAdapter,
    checkpoint_path: Path,
    expected_role: str,
    validation_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    outer_fold: int,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_role") != expected_role:
        raise ValueError("checkpoint roleが不正です")
    model.load_state_dict(checkpoint["model"])
    validation = evaluate(
        model,
        adapter,
        validation_loader,
        device,
        pos_weight,
        f"outer{outer_fold} {expected_role} val threshold",
    )
    threshold = float(
        select_f1_threshold(
            validation.predictions["vertebra_target"].to_numpy(),
            validation.predictions["vertebra_score"].to_numpy(),
        )["threshold"]
    )
    outer = evaluate(
        model,
        adapter,
        outer_loader,
        device,
        pos_weight,
        f"outer{outer_fold} {expected_role} outer",
    )
    predictions = outer.predictions.assign(
        decision_threshold=threshold,
        vertebra_prediction=(
            outer.predictions["vertebra_score"].to_numpy() >= threshold
        ).astype(np.int8),
    )
    checkpoint_config = checkpoint.get("config")
    manifest_hash = (
        checkpoint_config.get("frozen_manifest_sha256")
        if isinstance(checkpoint_config, Mapping)
        else None
    )
    if isinstance(manifest_hash, str):
        predictions = predictions.assign(frozen_manifest_sha256=manifest_hash)
    return predictions


def _resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_streams: TrainingRngStreams,
    path: Path,
    device: torch.device,
    config: dict[str, Any],
    resume: bool,
) -> ResumeState:
    if not resume:
        return ResumeState(1, 0, 0, float("-inf"), 0, float("-inf"), float("inf"), 0)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpointがありません: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("checkpointの実効configが一致しません")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_checkpoint_rng_state(checkpoint["rng"], rng_streams)
    saved = cast(dict[str, Any], checkpoint["resume_state"])
    return ResumeState(**saved)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_streams: TrainingRngStreams,
    config: dict[str, Any],
    state: ResumeState,
    epoch: int,
    role: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": checkpoint_rng_state(rng_streams),
            "config": config,
            "resume_state": asdict(state),
            "epoch": epoch,
            "checkpoint_role": role,
        },
        temporary,
    )
    temporary.replace(path)


def _early_stopping(
    current_loss: float,
    best_loss: float,
    bad_epochs: int,
    eligible: bool,
) -> tuple[float, int, bool]:
    if not math.isfinite(current_loss):
        raise FloatingPointError("validation lossが非有限値です")
    if not eligible:
        return best_loss, bad_epochs, False
    if current_loss < best_loss:
        return current_loss, 0, True
    return best_loss, bad_epochs + 1, False


def _gradient_medians(values: list[GradientNorms]) -> dict[str, float]:
    keys = tuple(GradientNorms.__dataclass_fields__)
    result: dict[str, float] = {}
    for key in keys:
        finite = [
            getattr(value, key)
            for value in values
            if math.isfinite(getattr(value, key))
        ]
        result[f"gradient_{key}_median"] = (
            float(np.median(finite)) if finite else math.nan
        )
    return result


def _set_epoch(loader: DataLoader[Any], epoch: int) -> None:
    sampler = loader.sampler
    if not hasattr(sampler, "set_epoch"):
        raise TypeError("samplerにset_epochが必要です")
    sampler.set_epoch(epoch)


def _dataset_length(loader: DataLoader[Any]) -> int:
    if not isinstance(loader.dataset, Sized):
        raise TypeError("datasetはSizedである必要があります")
    return len(loader.dataset)


def _strings(batch: Mapping[str, object], key: str) -> list[str]:
    values = batch.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"batchの{key}は文字列listである必要があります")
    return values


def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batchの{key}はTensorである必要があります")
    return value


def _write_history(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_history(path: Path) -> list[dict[str, float | int | bool]]:
    if not path.is_file():
        return []
    return cast(
        list[dict[str, float | int | bool]],
        pd.read_csv(path).to_dict(orient="records"),
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_fold_summary(result: FoldTrainingResult, path: Path) -> None:
    """fold結果の小さなJSON要約をatomic保存する。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_prauc_epoch": result.best_prauc_epoch,
                "stopped_epoch": result.stopped_epoch,
                "outer_rows": len(result.outer_predictions),
                "outer_inference_complete": not result.outer_predictions.empty,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
