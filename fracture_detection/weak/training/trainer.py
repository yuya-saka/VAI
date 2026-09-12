"""GT-pass training loop: N/A/U loss, checkpoint, resume, source-level logs.

One "epoch" is one GT-pass (one full traversal of the annotated-positive
group, see ``data_pipeline.sampling.GtPassBatchSampler``). There is exactly
one checkpoint track, selected by inner conditional-localization region
macro AP (``fracture_detection/REGION_MIL_DESIGN.md`` section 8) -- unlike
Baseline 0's separate AUROC/PR-AUC checkpoints, this model has one primary
selection criterion and reports whole AP/AUROC alongside it for the same
checkpoint. There is no mixup (see ``fracture_detection/weak/README.md`` for
why it cannot be applied to this model) and no per-step LR
freeze/warmup controller -- the cosine schedule steps once per GT-pass.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fracture_detection.baseline0.training.trainer import seed_worker, set_seed
from fracture_detection.weak.data_pipeline.batching import batch_tensors
from fracture_detection.weak.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.weak.data_pipeline.groups import (
    ANNOTATED_GROUP,
    GROUP_IDS,
    NEGATIVE_GROUP,
    WEAK_GROUP,
    group_name_by_id,
)
from fracture_detection.weak.data_pipeline.loaders import OuterFoldLoaders
from fracture_detection.weak.evaluation.metrics import (
    evaluate_conditional_localization,
    evaluate_whole_detection,
)
from fracture_detection.weak.modeling.losses import (
    NOISY_OR_AGGREGATION,
    compute_weak_losses,
    mean_or_nan,
    mixture_objective,
    whole_probability,
)
from fracture_detection.weak.modeling.model import WeakModelOutput, WeakRegionMilModel
from fracture_detection.weak.training.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_pass,
    update_best_summary,
)
from fracture_detection.weak.training.monitoring import compute_pass_diagnostics
from fracture_detection.weak.training.optimization import (
    create_weak_optimizer,
    create_weak_scheduler,
    optimizer_learning_rates,
)

__all__ = ["set_seed", "seed_worker", "FoldTrainingResult", "train_fold", "evaluate"]

CHECKPOINT_ROLE = "best_inner_region_macro_ap"
LOGIT_COLUMNS = tuple(f"z_{index}" for index in range(1, len(REGION_COLUMNS) + 1))
REGION_KEYS = tuple(f"r{index}" for index in range(1, len(REGION_COLUMNS) + 1))


@dataclass(frozen=True)
class FoldTrainingResult:
    """Reusable summary of one outer fold's training."""

    best_gt_pass: int
    best_metrics: dict[str, float]
    stopped_gt_pass: int
    outer_predictions: pd.DataFrame


def train_fold(
    model: WeakRegionMilModel,
    loaders: OuterFoldLoaders,
    config: dict[str, Any],
    outer_fold: int,
    fold_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> FoldTrainingResult:
    """Run GT-passes, select by inner region macro AP, infer outer once."""
    training = config["training"]
    max_gt_passes = int(training["max_gt_passes"])
    min_gt_passes = int(training["min_gt_passes"])
    patience = int(training["patience_gt_passes"])
    loss_config = config["loss"]
    beta = float(loss_config["beta"])
    whole_aggregation = str(loss_config.get("whole_aggregation", NOISY_OR_AGGREGATION))
    raw_lse_temperature = loss_config.get("lse_temperature")
    lse_temperature = (
        None if raw_lse_temperature is None else float(raw_lse_temperature)
    )
    gradient_clip_norm = (
        float("inf")
        if training["gradient_clip_norm"] is None
        else float(training["gradient_clip_norm"])
    )

    optimizer = create_weak_optimizer(
        model,
        float(training["weight_decay"]),
        float(training["transferred_learning_rate"]),
        float(training["new_learning_rate"]),
    )
    scheduler = create_weak_scheduler(
        optimizer,
        max_gt_passes=max_gt_passes,
        transferred_learning_rate=float(training["transferred_learning_rate"]),
        new_learning_rate=float(training["new_learning_rate"]),
        transferred_min_learning_rate=float(training["transferred_min_learning_rate"]),
        new_min_learning_rate=float(training["new_min_learning_rate"]),
    )
    model.to(device)

    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_region.pt"
    last_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "history.csv"
    pass_log_path = fold_dir / "pass_log.csv"
    diagnostics_path = fold_dir / "diagnostics.csv"

    (
        start_pass,
        best_pass,
        best_metrics,
        no_improvement,
    ) = _resume_state(model, optimizer, scheduler, last_path, device, config, resume)
    history_rows = _load_rows(history_path) if resume else []
    pass_log_rows = _load_rows(pass_log_path) if resume else []
    diagnostic_rows = _load_rows(diagnostics_path) if resume else []
    wandb_module = initialize_wandb(config, outer_fold)
    latest_metrics: dict[str, float] | None = None
    stopped_pass = start_pass - 1

    try:
        for gt_pass in range(start_pass, max_gt_passes + 1):
            start_time = time.monotonic()
            loaders.train_sampler.set_pass(gt_pass - 1)
            train_metrics = _train_pass(
                model,
                loaders.train_loader,
                optimizer,
                device,
                gradient_clip_norm,
                beta,
                f"outer{outer_fold} pass{gt_pass}/{max_gt_passes} 学習",
                whole_aggregation,
                lse_temperature,
            )
            validation_metrics, inner_predictions = evaluate(
                model,
                loaders.inner_loader,
                device,
                beta,
                f"outer{outer_fold} pass{gt_pass}/{max_gt_passes} val検証",
                whole_aggregation,
                lse_temperature,
            )
            train_metrics = {
                **train_metrics,
                "objective": _objective(train_metrics, loaders, beta),
            }
            validation_metrics = {
                **validation_metrics,
                "objective": _objective(validation_metrics, loaders, beta),
            }
            latest_metrics = validation_metrics
            stopped_pass = gt_pass
            scheduler.step()
            transferred_lr, new_lr = optimizer_learning_rates(optimizer)

            eligible = gt_pass >= min_gt_passes
            current_ap = validation_metrics["region_macro_ap"]
            improved = (
                eligible
                and np.isfinite(current_ap)
                and current_ap > best_metrics["region_macro_ap"]
            )
            no_improvement = 0 if improved else no_improvement + (1 if eligible else 0)
            if improved:
                best_pass = gt_pass
                best_metrics = validation_metrics
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    gt_pass,
                    best_pass,
                    best_metrics,
                    no_improvement,
                    CHECKPOINT_ROLE,
                )
                if wandb_module is not None:
                    update_best_summary(wandb_module, gt_pass, validation_metrics)
            _save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                gt_pass,
                best_pass,
                best_metrics,
                no_improvement,
                "last",
            )

            elapsed = time.monotonic() - start_time
            pass_composition = loaders.train_sampler.pass_composition()
            history_rows.append(
                {
                    "gt_pass": gt_pass,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"val_{key}": value for key, value in validation_metrics.items()
                    },
                    "transferred_lr": transferred_lr,
                    "new_lr": new_lr,
                    "is_best": improved,
                    "no_improvement": no_improvement,
                    "gt_pass_seconds": elapsed,
                }
            )
            diagnostic_rows.append(_diagnostic_row(gt_pass, inner_predictions))
            pass_log_rows.append(
                {
                    "gt_pass": gt_pass,
                    "n_annotated_presented": pass_composition.n_annotated_presented,
                    "n_negative_presented": pass_composition.n_negative_presented,
                    "n_weak_presented": pass_composition.n_weak_presented,
                    "negative_cycle_count": _cycle_count(
                        gt_pass,
                        pass_composition.n_negative_presented,
                        len(loaders.train_sampler.negative_indices),
                    ),
                    "weak_cycle_count": _cycle_count(
                        gt_pass,
                        pass_composition.n_weak_presented,
                        len(loaders.train_sampler.weak_indices),
                    ),
                }
            )
            _write_rows(history_path, history_rows)
            _write_rows(pass_log_path, pass_log_rows)
            _write_rows(diagnostics_path, diagnostic_rows)
            if wandb_module is not None:
                log_wandb_pass(
                    wandb_module,
                    gt_pass,
                    train_metrics,
                    validation_metrics,
                    transferred_lr,
                    new_lr,
                    elapsed,
                )
            if eligible and no_improvement >= patience:
                break
    finally:
        finish_wandb(wandb_module, stopped_pass, latest_metrics, 0, 0)

    if best_pass < min_gt_passes or not best_path.is_file():
        raise RuntimeError("min_gt_passes以降に有効なcheckpointを保存できませんでした")

    outer_prediction_path = fold_dir / "outer_predictions.csv"
    if outer_prediction_path.exists():
        raise RuntimeError("outer予測が既に存在するため再推論を拒否しました")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_role") != CHECKPOINT_ROLE:
        raise ValueError(
            f"best checkpoint roleが不正です: {checkpoint.get('checkpoint_role')}"
        )
    model.load_state_dict(checkpoint["model"])
    _, outer_predictions = evaluate(
        model,
        loaders.outer_loader,
        device,
        beta,
        f"outer{outer_fold} outer推論",
        whole_aggregation,
        lse_temperature,
    )
    _atomic_write_csv(outer_predictions, outer_prediction_path)
    (fold_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "best_gt_pass": best_pass,
                "best_metrics": best_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return FoldTrainingResult(
        best_gt_pass=best_pass,
        best_metrics=best_metrics,
        stopped_gt_pass=stopped_pass,
        outer_predictions=outer_predictions,
    )


def _train_pass(
    model: WeakRegionMilModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    beta: float,
    progress_description: str,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> dict[str, float]:
    """Run one GT-pass; return group-wise loss sums/counts and CNN grad norm."""
    model.train()
    negative_loss = annotated_loss = weak_loss = 0.0
    n_negative = n_annotated = n_weak = dropped_bags = 0
    total_grad_norm = 0.0
    total_cnn_grad_norm = 0.0
    batch_count = 0
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    for batch in progress:
        batch_tensor = batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch_tensor.inputs, batch_tensor.region_mask)
            losses = compute_weak_losses(
                output.region_logits,
                output.region_observed,
                batch_tensor.group_id,
                batch_tensor.region_target,
                beta,
                whole_aggregation,
                lse_temperature,
            )
        if not torch.isfinite(losses.total):
            raise FloatingPointError("学習lossが非有限値です")
        torch.autograd.backward(losses.total)
        cnn_grad_norm = clip_grad_norm_(model.transferred_parameters(), float("inf"))
        gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("学習gradientが非有限値です")
        optimizer.step()

        negative_loss += float(losses.negative_sum.detach())
        annotated_loss += float(losses.annotated_sum.detach())
        weak_loss += float(losses.weak_sum.detach())
        n_negative += losses.n_negative
        n_annotated += losses.n_annotated
        n_weak += losses.n_weak
        dropped_bags += losses.dropped_bags
        total_grad_norm += float(gradient_norm)
        total_cnn_grad_norm += float(cnn_grad_norm)
        batch_count += 1
        progress.set_postfix(loss=f"{float(losses.total.detach()):.4f}")
    if batch_count == 0:
        raise ValueError("train loaderが空です")
    return {
        "negative_loss": mean_or_nan(negative_loss, n_negative),
        "annotated_loss": mean_or_nan(annotated_loss, n_annotated),
        "weak_loss": mean_or_nan(weak_loss, n_weak),
        "dropped_bags": float(dropped_bags),
        "grad_norm": total_grad_norm / batch_count,
        "cnn_grad_norm": total_cnn_grad_norm / batch_count,
    }


@torch.no_grad()
def evaluate(
    model: WeakRegionMilModel,
    loader: DataLoader[Any],
    device: torch.device,
    beta: float,
    progress_description: str,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run one natural-distribution pass; return per-pass metrics and predictions.

    Populations, never pooled into one number:
    - region_*: annotated-positive bags only, all four GT cells.
    - whole_*: every bag's p_whole, natural distribution.
    - negative/annotated/weak_loss: per-bag mean of that group's own loss term
      (same definition as the train columns), not weighted by beta. The
      weak-positive term is -log(p_whole) on bags without region GT.
    """
    model.eval()
    records: list[dict[str, Any]] = []
    group_sums = {"negative": 0.0, "annotated": 0.0, "weak": 0.0}
    group_counts = {"negative": 0, "annotated": 0, "weak": 0}
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    for batch in progress:
        batch_tensor = batch_tensors(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch_tensor.inputs, batch_tensor.region_mask)
            losses = compute_weak_losses(
                output.region_logits,
                output.region_observed,
                batch_tensor.group_id,
                batch_tensor.region_target,
                beta,
                whole_aggregation,
                lse_temperature,
            )
        group_sums["negative"] += float(losses.negative_sum)
        group_sums["annotated"] += float(losses.annotated_sum)
        group_sums["weak"] += float(losses.weak_sum)
        group_counts["negative"] += losses.n_negative
        group_counts["annotated"] += losses.n_annotated
        group_counts["weak"] += losses.n_weak
        records.extend(
            _predictions_records(batch, output, whole_aggregation, lse_temperature)
        )
    if not records:
        raise ValueError("評価loaderが空です")
    predictions = pd.DataFrame(records)
    observed = predictions[predictions["region_observed_all"]]
    if observed.empty:
        raise ValueError("観測済みbagが1件もありません")
    localization = evaluate_conditional_localization(observed, n_bootstrap=1)
    whole = evaluate_whole_detection(predictions, n_bootstrap=1)
    region_quality = localization["probability_quality"]
    metrics: dict[str, float] = {
        "region_macro_ap": localization["macro_average_precision"],
        **{
            f"region_ap_{key}": localization["regions"][column]["average_precision"]
            for column, key in zip(REGION_COLUMNS, REGION_KEYS, strict=True)
        },
        "region_bce_annotated": _region_mean(region_quality, "bce"),
        **{
            f"region_bce_annotated_{key}": region_quality[column]["bce"]
            for column, key in zip(REGION_COLUMNS, REGION_KEYS, strict=True)
        },
        "region_brier_annotated": _region_mean(region_quality, "brier"),
        "region_ece_annotated": _region_mean(region_quality, "ece"),
        "whole_average_precision": whole["average_precision"],
        "whole_auroc": whole["auroc"],
        "whole_bce": whole["probability_quality"]["bce"],
        "whole_brier": whole["probability_quality"]["brier"],
        "whole_ece": whole["probability_quality"]["ece"],
        **{
            f"{group}_loss": mean_or_nan(group_sums[group], group_counts[group])
            for group in group_sums
        },
        "n_dropped_unobserved": int((~predictions["region_observed_all"]).sum()),
    }
    return metrics, predictions


def _region_mean(region_quality: dict[str, dict[str, float]], name: str) -> float:
    """Mean of one quality metric over the four regions (equal cell counts)."""
    return float(np.mean([region_quality[column][name] for column in REGION_COLUMNS]))


def _objective(
    metrics: dict[str, float], loaders: OuterFoldLoaders, beta: float
) -> float:
    """Training-composition-weighted N/A/U loss, see ``mixture_objective``."""
    sampler = loaders.train_sampler
    return mixture_objective(
        metrics["negative_loss"],
        metrics["annotated_loss"],
        metrics["weak_loss"],
        sampler.negative_per_batch,
        sampler.annotated_per_batch,
        sampler.weak_per_batch,
        beta,
    )


def _diagnostic_row(gt_pass: int, predictions: pd.DataFrame) -> dict[str, Any]:
    """q-shape, correlation, argmax, and false-positive diagnostics for one pass."""
    observed = predictions[predictions["region_observed_all"]]
    record = compute_pass_diagnostics(
        gt_pass=gt_pass,
        region_logits=observed[list(LOGIT_COLUMNS)].to_numpy(dtype=np.float64),
        group_id=observed["bag_group"].map(GROUP_IDS).to_numpy(),
        negative_group_id=GROUP_IDS[NEGATIVE_GROUP],
        positive_group_ids=(GROUP_IDS[ANNOTATED_GROUP], GROUP_IDS[WEAK_GROUP]),
    )
    return record.as_row()


def _predictions_records(
    batch: dict[str, Any],
    output: WeakModelOutput,
    whole_aggregation: str = NOISY_OR_AGGREGATION,
    lse_temperature: float | None = None,
) -> list[dict[str, Any]]:
    """Build one prediction row per bag from a model output and its raw batch."""
    region_logits = output.region_logits.float().cpu().numpy()
    region_observed = output.region_observed.cpu().numpy()
    q = 1.0 / (1.0 + np.exp(-region_logits))
    p_whole = (
        whole_probability(output.region_logits, whole_aggregation, lse_temperature)
        .float()
        .cpu()
        .numpy()
    )
    region_target = batch["region_target"].cpu().numpy()
    vertebra_target = batch["vertebra_target"].cpu().numpy()
    group_id = batch["group_id"].cpu().numpy()
    study_ids = batch["study_id"]
    levels = batch["level"]
    folds = batch["fold"].cpu().tolist()

    records = []
    for index in range(len(study_ids)):
        records.append(
            {
                "study_id": study_ids[index],
                "level": levels[index],
                "fold": folds[index],
                "bag_group": group_name_by_id(int(group_id[index])),
                "vertebra_target": int(vertebra_target[index]),
                "region_1": float(region_target[index, 0]),
                "region_2": float(region_target[index, 1]),
                "region_3": float(region_target[index, 2]),
                "region_4": float(region_target[index, 3]),
                "q_1": float(q[index, 0]),
                "q_2": float(q[index, 1]),
                "q_3": float(q[index, 2]),
                "q_4": float(q[index, 3]),
                "z_1": float(region_logits[index, 0]),
                "z_2": float(region_logits[index, 1]),
                "z_3": float(region_logits[index, 2]),
                "z_4": float(region_logits[index, 3]),
                "p_whole": float(p_whole[index]),
                "region_observed_all": bool(region_observed[index].all()),
            }
        )
    return records


def _cycle_count(gt_pass: int, presented_per_pass: int, pool_size: int) -> float:
    """Approximate number of full cycles through an N/U pool after this pass."""
    if pool_size == 0:
        return float("nan")
    return (gt_pass * presented_per_pass) / pool_size


def _resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_path: Path,
    device: torch.device,
    config: dict[str, Any],
    resume: bool,
) -> tuple[int, int, dict[str, float], int]:
    """Restore the last checkpoint if resuming; otherwise start fresh."""
    default_metrics = {"region_macro_ap": float("-inf")}
    if not resume:
        return 1, 0, default_metrics, 0
    if not last_path.is_file():
        raise FileNotFoundError(f"resume対象checkpointがありません: {last_path}")
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("checkpointの実効configが現在のconfigと一致しません")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["gt_pass"]) + 1,
        int(checkpoint["best_gt_pass"]),
        {key: float(value) for key, value in checkpoint["best_metrics"].items()},
        int(checkpoint["no_improvement"]),
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    gt_pass: int,
    best_gt_pass: int,
    best_metrics: dict[str, float],
    no_improvement: int,
    checkpoint_role: str,
) -> None:
    """Atomically save all state needed to resume training."""
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "gt_pass": gt_pass,
            "best_gt_pass": best_gt_pass,
            "best_metrics": best_metrics,
            "no_improvement": no_improvement,
            "checkpoint_role": checkpoint_role,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
