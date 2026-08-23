"""低頻度detail streamに対応したnested fold学習ループ。

`core.trainer.train_fold`との違いは1点だけ: natural stepの**毎回**annotated
batchを消費する代わりに、1 epoch=annotated datasetを1周させ、その
step数を`training.schedule.region_step_schedule`でnatural step列へ
均等配置する（RSNA修正方針§4）。加えて、region branchが学習できているかを
val/train双方のregion APで毎epoch記録し（§10）、val region AP最良の
`best_region.pt`を**診断専用**として保存する（outer推論には使わない。
outer推論・primary checkpointは既存アームと同じくval AUROC-best）。
`create_data_loader`・`evaluate`・`set_seed`はarchitecture非依存のため
`core.trainer`から直接再利用する。
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.core.contracts import LossWeights
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
from fracture_detection.core.steps import ArmAdapter, GradientNorms
from fracture_detection.core.trainer import evaluate
from fracture_detection.core.wandb import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
)
from fracture_detection.mtl_type2.training.diagnostics import (
    region_average_precision,
    region_predictions,
)
from fracture_detection.mtl_type2.training.schedule import region_step_schedule
from fracture_detection.mtl_type2.training.steps import train_step


@dataclass(frozen=True)
class FoldTrainingResult:
    """1 outer foldの学習結果。"""

    best_epoch: int
    best_region_epoch: int
    stopped_epoch: int
    outer_predictions: pd.DataFrame


@dataclass(frozen=True)
class ResumeState:
    """epoch境界resumeに必要なscalar状態。"""

    start_epoch: int
    global_step: int
    best_epoch: int
    best_auroc: float
    best_region_epoch: int
    best_region_ap: float
    early_stopping_best_loss: float
    bad_epochs: int


def train_fold(
    model: nn.Module,
    adapter: ArmAdapter,
    natural_loader: DataLoader[Any],
    annotated_loader: DataLoader[Any],
    annotated_train_eval_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    config: dict[str, Any],
    outer_fold: int,
    fold_dir: Path,
    device: torch.device,
    loss_weights: LossWeights,
    *,
    resume: bool = False,
) -> FoldTrainingResult:
    """毎epoch annotated datasetを1周させ、region APを診断記録しつつ学習する。"""
    training = cast(dict[str, Any], config["training"])
    if not isinstance(natural_loader.sampler, EpochShuffleSampler):
        raise TypeError("natural loaderにはEpochShuffleSamplerが必要です")
    if not isinstance(annotated_loader.sampler, AnnotatedCycleSampler):
        raise TypeError("annotated loaderにはAnnotatedCycleSamplerが必要です")
    optimizer = create_optimizer(
        model,
        float(training["weight_decay"]),
        float(training["backbone_learning_rate"]),
        float(training["head_learning_rate"]),
    )
    max_epochs = int(training["max_epochs"])
    controller = LearningRateController(
        steps_per_epoch=len(natural_loader),
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
        "region": fold_dir / "best_region.pt",
        "last": fold_dir / "last_checkpoint.pt",
        "history": fold_dir / "history.csv",
    }
    state = _resume_state(
        model, optimizer, scheduler, rng_streams, paths["last"], device, config, resume
    )
    history = _load_history(paths["history"]) if resume else []
    wandb_run = initialize_wandb(config, outer_fold, fold_dir)
    stopped_epoch = state.start_epoch - 1
    current = state
    print(
        f"[outer {outer_fold}] train={_dataset_length(natural_loader):,}, "
        f"annotated={_dataset_length(annotated_loader):,} "
        f"({len(annotated_loader)} step/epoch), "
        f"val={_dataset_length(validation_loader):,}, "
        f"outer={_dataset_length(outer_loader):,}, "
        f"device={device}",
        flush=True,
    )
    for epoch in range(state.start_epoch, max_epochs + 1):
        started = time.monotonic()
        _set_epoch(natural_loader, epoch - 1)
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
        train_region_predictions = region_predictions(
            model, adapter, annotated_train_eval_loader, device
        )
        val_region_ap = region_average_precision(validation.predictions)
        train_region_ap = region_average_precision(train_region_predictions)
        scheduler.step()
        eligible = epoch >= int(training["min_epoch"])
        improved_auroc = (
            eligible
            and math.isfinite(validation.metrics["auroc"])
            and validation.metrics["auroc"] > current.best_auroc
        )
        improved_region = (
            eligible
            and math.isfinite(val_region_ap["macro"])
            and val_region_ap["macro"] > current.best_region_ap
        )
        best_epoch = epoch if improved_auroc else current.best_epoch
        best_auroc = (
            validation.metrics["auroc"] if improved_auroc else current.best_auroc
        )
        best_region_epoch = epoch if improved_region else current.best_region_epoch
        best_region_ap = (
            val_region_ap["macro"] if improved_region else current.best_region_ap
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
            best_region_epoch=best_region_epoch,
            best_region_ap=best_region_ap,
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
        if improved_region:
            _save_checkpoint(
                paths["region"],
                model,
                optimizer,
                scheduler,
                rng_streams,
                config,
                current,
                epoch,
                "best_region_ap_diagnostic",
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
            **{f"val_region_ap_{key}": value for key, value in val_region_ap.items()},
            **{
                f"train_region_ap_{key}": value
                for key, value in train_region_ap.items()
            },
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "epoch_seconds": time.monotonic() - started,
            "is_best_val_auroc": improved_auroc,
            "is_best_region_ap": improved_region,
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
            f"val_auroc={validation.metrics['auroc']:.6f} "
            f"val_region_ap_macro={val_region_ap['macro']:.4f} "
            f"train_region_ap_macro={train_region_ap['macro']:.4f}",
            flush=True,
        )
        if eligible and bad_epochs >= int(training["early_stopping_patience"]):
            break
    finish_wandb(wandb_run)
    if current.best_epoch < int(training["min_epoch"]) or not paths["best"].is_file():
        raise RuntimeError("有効なAUROC-best checkpointがありません")
    output_path = fold_dir / "outer_predictions.csv"
    if output_path.exists():
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
    _atomic_csv(best_outer, output_path)
    return FoldTrainingResult(
        best_epoch=current.best_epoch,
        best_region_epoch=current.best_region_epoch,
        stopped_epoch=stopped_epoch,
        outer_predictions=best_outer,
    )


def _train_epoch(
    model: nn.Module,
    adapter: ArmAdapter,
    natural_loader: DataLoader[Any],
    annotated_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    controller: LearningRateController,
    global_step: int,
    device: torch.device,
    rng_streams: TrainingRngStreams,
    loss_weights: LossWeights,
    training: Mapping[str, Any],
    *,
    description: str,
) -> tuple[dict[str, float], int, float, float]:
    model.train()
    natural_steps = len(natural_loader)
    annotated_steps = len(annotated_loader)
    region_steps = set(region_step_schedule(natural_steps, annotated_steps))
    annotated_iterator = iter(annotated_loader)
    totals: defaultdict[str, float] = defaultdict(float)
    visits: Counter[str] = Counter()
    measured: list[GradientNorms] = []
    measure_steps = {0, natural_steps // 4, natural_steps // 2, 3 * natural_steps // 4}
    backbone_lr = 0.0
    head_lr = 0.0
    progress = tqdm(natural_loader, total=natural_steps, desc=description, leave=False)
    for step_index, natural_batch in enumerate(progress):
        if step_index >= natural_steps:
            break
        annotated_batch = None
        if step_index in region_steps:
            try:
                annotated_batch = next(annotated_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "annotated loaderがscheduleより早く尽きました"
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
    remaining = list(annotated_iterator)
    if remaining:
        raise RuntimeError(
            f"annotated loaderが{len(remaining)} batch未消費のまま epochが終わりました"
        )
    # region_lossはregion_steps回（annotated_steps）だけ非ゼロで、それ以外の
    # natural stepでは0を加算している。natural_stepsで割ると
    # 「region更新1回あたりの平均loss」が実際の~natural_steps/annotated_steps倍
    # 薄まって表示されるため、region_lossだけannotated_stepsで割る。
    metrics = {
        key: float(value / natural_steps)
        for key, value in totals.items()
        if key != "region_loss"
    }
    metrics["region_loss"] = float(totals["region_loss"] / annotated_steps)
    metrics.update(_gradient_medians(measured))
    visit_values = list(visits.values())
    metrics.update(
        region_optimizer_steps=float(annotated_steps),
        region_unique_bags=float(len(visits)),
        region_visits_min=float(min(visit_values)) if visit_values else math.nan,
        region_visits_max=float(max(visit_values)) if visit_values else math.nan,
    )
    return metrics, global_step, backbone_lr, head_lr


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
    from fracture_detection.common.metrics import select_f1_threshold

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
    return outer.predictions.assign(
        decision_threshold=threshold,
        vertebra_prediction=(
            outer.predictions["vertebra_score"].to_numpy() >= threshold
        ).astype(np.int8),
    )


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
    current_loss: float, best_loss: float, bad_epochs: int, eligible: bool
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


def _dataset_length(loader: DataLoader[Any]) -> int:
    if not isinstance(loader.dataset, Sized):
        raise TypeError("datasetはSizedである必要があります")
    return len(loader.dataset)


def _set_epoch(loader: DataLoader[Any], epoch: int) -> None:
    sampler = loader.sampler
    if not hasattr(sampler, "set_epoch"):
        raise TypeError("samplerにset_epochが必要です")
    sampler.set_epoch(epoch)


def _strings(batch: Mapping[str, object], key: str) -> list[str]:
    values = batch.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"batchの{key}は文字列listである必要があります")
    return values


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
        list[dict[str, float | int | bool]], pd.read_csv(path).to_dict(orient="records")
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
                "best_region_epoch": result.best_region_epoch,
                "stopped_epoch": result.stopped_epoch,
                "outer_rows": len(result.outer_predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
