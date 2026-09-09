"""region_branchのfold単位の学習・検証・checkpoint制御。

1 stepはnatural batchをencoderへ1回だけforwardし、mixupが発火しなかったstepでは
同じforwardからwhole pathとwhole陽性bagのregion pathを計算して1回だけbackwardする。
mixup時はwhole pathだけを計算する。目的関数は
`L = L_whole + lambda_k*L_region_conditional`。regionは同じnatural batchから
whole陽性bagだけを抽出し、GT/pseudo統一targetへplain macro BCEを計算する。
`best_region.pt`とearly stoppingはinner validationのentropy-centered統一lossを
最小化し、hard-GT macro AP/AUROCは独立した評価値として保存する。
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
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fracture_detection.baseline0.data.sampling import EpochShuffleSampler
from fracture_detection.baseline0.evaluation.metrics import (
    safe_auroc,
    safe_average_precision,
)
from fracture_detection.baseline0.modeling.losses import (
    bag_probabilities,
    broadcast_bce_loss,
)
from fracture_detection.region_branch.data_pipeline.batching import batch_tensors
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.region_branch.data_pipeline.loaders import OuterFoldLoaders
from fracture_detection.region_branch.modeling.losses import (
    compute_conditional_region_losses,
    region_bag_logits,
)
from fracture_detection.region_branch.modeling.model import RegionBranchModel
from fracture_detection.region_branch.training.calibration import CalibrationResult
from fracture_detection.region_branch.training.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
)
from fracture_detection.region_branch.training.monitoring import (
    CollapseMonitor,
    compute_diagnostics,
)
from fracture_detection.region_branch.training.optimization import (
    create_finetuning_optimizer,
    create_finetuning_scheduler,
    optimizer_learning_rates,
)

PROBABILITY_EPSILON = 1e-6
TORCH_COMPILE_MODE = "default"
Batch = dict[str, Any]


@dataclass(frozen=True)
class FoldTrainingResult:
    """1 outer fold学習の再利用可能な要約。"""

    best_region_epoch: int
    best_region_val_metrics: dict[str, float]
    best_whole_epoch: int
    best_whole_val_metrics: dict[str, float]
    stopped_epoch: int
    outer_predictions: pd.DataFrame


def train_fold(
    model: RegionBranchModel,
    loaders: OuterFoldLoaders,
    inner_loader: DataLoader[Any],
    outer_loader: DataLoader[Any],
    diagnostic_loader: DataLoader[Any],
    config: dict[str, Any],
    calibration: CalibrationResult,
    outer_fold: int,
    fold_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> FoldTrainingResult:
    """1 outer foldを学習し、region/whole best checkpointでouterを一度ずつ推論する。

    natural stream・inner/outer/diagnostic評価loaderはすべて呼び出し側（CLI）が
    構築して渡す。ここではmanifestを読まない（Baseline 0のtrain_foldと同じ
    責務分離。合成データでのunit testを可能にする）。
    """
    training = config["training"]
    region = config["region"]
    active_regions = tuple(int(value) for value in region["active_regions"])
    pseudo_arm = str(region["pseudo_arm"])
    max_epochs = int(training["max_epochs"])
    min_epoch = int(training["min_epoch"])
    patience = int(training["early_stopping_patience"])
    pos_weight = float(training["pos_weight"])
    lambda_value = float(calibration.lambda_)

    print(
        f"[outer {outer_fold}] 学習を初期化しています: device={device}, "
        f"active_regions={active_regions}, pseudo_arm={pseudo_arm}, "
        f"lambda={lambda_value:.6f}, steps/epoch={loaders.steps_per_epoch}",
        flush=True,
    )

    model.to(device)
    if _configure_compiled_training(model, device):
        print(
            f"[outer {outer_fold}] torch.compileを設定しました: "
            f"mode={TORCH_COMPILE_MODE}, dynamic=False",
            flush=True,
        )
    optimizer = create_finetuning_optimizer(
        model,
        float(training["weight_decay"]),
        float(training["pretrained_learning_rate"]),
        float(training["region_learning_rate"]),
    )
    scheduler = create_finetuning_scheduler(
        optimizer,
        max_epochs=max_epochs,
        pretrained_learning_rate=float(training["pretrained_learning_rate"]),
        region_learning_rate=float(training["region_learning_rate"]),
        pretrained_min_learning_rate=float(training["pretrained_min_learning_rate"]),
        region_min_learning_rate=float(training["region_min_learning_rate"]),
    )
    raw_gradient_clip_norm = training["gradient_clip_norm"]
    gradient_clip_norm = (
        float("inf")
        if raw_gradient_clip_norm is None
        else float(raw_gradient_clip_norm)
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    best_region_path = fold_dir / "best_region.pt"
    best_whole_path = fold_dir / "best_whole.pt"
    last_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "history.csv"
    diagnostics_path = fold_dir / "diagnostics.csv"
    log_path = fold_dir / "training.log"

    (
        start_epoch,
        global_step,
        best_region_epoch,
        best_region_loss,
        best_region_val_metrics,
        best_whole_epoch,
        best_whole_loss,
        best_whole_val_metrics,
        early_stopping_best_region_loss,
        no_improvement,
    ) = _resume_state(model, optimizer, scheduler, last_path, device, config, resume)
    history_rows = _load_rows(history_path) if resume else []
    diagnostic_rows = _load_rows(diagnostics_path) if resume else []
    wandb_module = initialize_wandb(config, outer_fold)
    collapse_monitor = CollapseMonitor()
    stopped_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            start_time = time.monotonic()
            _set_loaders_epoch(loaders, epoch - 1)
            print(
                f"[outer {outer_fold}] epoch {epoch}/{max_epochs}を開始します"
                + (
                    "（初回batchはworker起動・torch.compileで時間がかかります）"
                    if epoch == start_epoch
                    else ""
                ),
                flush=True,
            )
            train_metrics, global_step, pretrained_lr, region_lr = _train_epoch(
                model,
                loaders,
                optimizer,
                global_step,
                device,
                gradient_clip_norm,
                pos_weight,
                float(training["mixup_probability"]),
                active_regions,
                lambda_value,
                f"outer{outer_fold} epoch{epoch}/{max_epochs} 学習",
            )
            val_metrics, _ = evaluate(
                model,
                inner_loader,
                device,
                pos_weight,
                active_regions,
                f"outer{outer_fold} epoch{epoch}/{max_epochs} val検証",
            )
            stopped_epoch = epoch
            scheduler.step()

            (
                region_array,
                whole_array,
                pseudo_target_array,
                pseudo_valid_array,
            ) = _collect_diagnostic_arrays(
                model, diagnostic_loader, active_regions, device
            )
            diagnostic_record = compute_diagnostics(
                epoch,
                region_array,
                whole_array,
                pseudo_target_array,
                pseudo_valid_array,
                active_regions,
            )
            collapse_alarm = collapse_monitor.update(diagnostic_record)
            diagnostic_rows.append(diagnostic_record.as_row())
            _write_rows(diagnostics_path, diagnostic_rows)

            eligible = epoch >= min_epoch
            current_region_loss = val_metrics["region_centered_loss"]
            region_improved = (
                eligible
                and np.isfinite(current_region_loss)
                and current_region_loss < best_region_loss
            )
            if region_improved:
                best_region_epoch = epoch
                best_region_loss = current_region_loss
                best_region_val_metrics = val_metrics
                _save_checkpoint(
                    best_region_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_region_epoch,
                    best_region_loss,
                    best_region_val_metrics,
                    best_whole_epoch,
                    best_whole_loss,
                    best_whole_val_metrics,
                    early_stopping_best_region_loss,
                    no_improvement,
                    checkpoint_role="best_region",
                )

            current_whole_loss = val_metrics["whole"]
            whole_improved = (
                eligible
                and np.isfinite(current_whole_loss)
                and current_whole_loss < best_whole_loss
            )
            if whole_improved:
                best_whole_epoch = epoch
                best_whole_loss = current_whole_loss
                best_whole_val_metrics = val_metrics
                _save_checkpoint(
                    best_whole_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_region_epoch,
                    best_region_loss,
                    best_region_val_metrics,
                    best_whole_epoch,
                    best_whole_loss,
                    best_whole_val_metrics,
                    early_stopping_best_region_loss,
                    no_improvement,
                    checkpoint_role="best_whole",
                )

            early_stopping_improved = False
            if eligible:
                (
                    early_stopping_best_region_loss,
                    no_improvement,
                    early_stopping_improved,
                ) = _update_early_stopping(
                    current_region_loss,
                    early_stopping_best_region_loss,
                    no_improvement,
                )

            _save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                global_step,
                best_region_epoch,
                best_region_loss,
                best_region_val_metrics,
                best_whole_epoch,
                best_whole_loss,
                best_whole_val_metrics,
                early_stopping_best_region_loss,
                no_improvement,
                checkpoint_role="last",
            )

            elapsed = time.monotonic() - start_time
            row: dict[str, Any] = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "pretrained_lr": pretrained_lr,
                "region_lr": region_lr,
                "epoch_seconds": elapsed,
                "is_best_region": region_improved,
                "is_best_whole": whole_improved,
                "early_stopping_improved": early_stopping_improved,
                "early_stopping_best_region_loss": early_stopping_best_region_loss,
                "early_stopping_bad_epochs": no_improvement,
                "collapse_alarm": collapse_alarm,
            }
            history_rows.append(row)
            _write_rows(history_path, history_rows)
            _append_log(log_path, row)
            if wandb_module is not None:
                log_wandb_epoch(
                    wandb_module, epoch, {**row, **diagnostic_record.as_row()}
                )

            if collapse_alarm:
                (fold_dir / "collapse_alarm.json").write_text(
                    json.dumps(
                        {"epoch": epoch, "diagnostics": diagnostic_record.as_row()},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"[outer {outer_fold}] epoch {epoch}: collapse alarmが3epoch連続で"
                    "確定しました。係数を調整せず、事前定義した失敗として停止します"
                )
            if eligible and no_improvement >= patience:
                break
    finally:
        finish_wandb(wandb_module, stopped_epoch)

    if best_region_epoch < min_epoch or not best_region_path.is_file():
        raise RuntimeError(
            "min_epoch以降に有効なbest region checkpointを保存できませんでした"
        )
    if best_whole_epoch < min_epoch or not best_whole_path.is_file():
        raise RuntimeError(
            "min_epoch以降に有効なbest whole checkpointを保存できませんでした"
        )
    outer_prediction_path = fold_dir / "outer_predictions.csv"
    if outer_prediction_path.exists():
        raise RuntimeError("outer予測が既に存在するため再推論を拒否しました")

    region_checkpoint = torch.load(
        best_region_path, map_location=device, weights_only=False
    )
    if region_checkpoint.get("checkpoint_role") != "best_region":
        raise ValueError(
            f"best region checkpoint roleが不正です: {region_checkpoint.get('checkpoint_role')}"
        )
    model.load_state_dict(region_checkpoint["model"])
    print(
        f"[outer {outer_fold}] outerを推論しています"
        "(region, val unified centered lossが最良のcheckpoint)",
        flush=True,
    )
    region_metrics, region_predictions = evaluate(
        model,
        outer_loader,
        device,
        pos_weight,
        active_regions,
        f"outer{outer_fold} outer推論(region)",
    )

    whole_checkpoint = torch.load(
        best_whole_path, map_location=device, weights_only=False
    )
    if whole_checkpoint.get("checkpoint_role") != "best_whole":
        raise ValueError(
            f"best whole checkpoint roleが不正です: {whole_checkpoint.get('checkpoint_role')}"
        )
    model.load_state_dict(whole_checkpoint["model"])
    print(
        f"[outer {outer_fold}] outerを推論しています(whole, val wholeが最良のcheckpoint)",
        flush=True,
    )
    whole_metrics, whole_predictions = evaluate(
        model,
        outer_loader,
        device,
        pos_weight,
        active_regions,
        f"outer{outer_fold} outer推論(whole)",
    )

    if len(region_predictions) != len(whole_predictions) or not (
        region_predictions[["study_id", "level"]]
        .reset_index(drop=True)
        .equals(whole_predictions[["study_id", "level"]].reset_index(drop=True))
    ):
        raise ValueError("region推論とwhole推論のbag集合が一致しません")
    if not region_predictions["vertebra_target"].equals(
        whole_predictions["vertebra_target"]
    ):
        raise ValueError("region推論とwhole推論のvertebra_targetが一致しません")

    region_columns_to_copy = [
        column
        for name in REGION_COLUMNS
        for column in (
            f"{name}_target",
            f"{name}_target_valid",
            f"{name}_conditional_score",
        )
    ]
    outer_predictions = whole_predictions[
        ["study_id", "level", "fold", "vertebra_target", "vertebra_score"]
    ].merge(
        region_predictions[["study_id", "level", *region_columns_to_copy]],
        on=["study_id", "level"],
        validate="one_to_one",
    )
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        outer_predictions[f"{column}_score"] = (
            outer_predictions["vertebra_score"]
            * outer_predictions[f"{column}_conditional_score"]
        )
    final_region_outer_metrics = _hard_region_metrics(outer_predictions, active_regions)
    _atomic_write_csv(outer_predictions, outer_prediction_path)
    (fold_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "val_fold": int(config["runtime"]["inner_fold"]),
                "active_regions": list(active_regions),
                "pseudo_arm": pseudo_arm,
                "lambda": lambda_value,
                "stopped_epoch": stopped_epoch,
                "best_region_epoch": best_region_epoch,
                "best_region_loss": best_region_loss,
                "best_region_val_metrics": best_region_val_metrics,
                "best_whole_epoch": best_whole_epoch,
                "best_whole_loss": best_whole_loss,
                "best_whole_val_metrics": best_whole_val_metrics,
                "region_checkpoint_outer_metrics": region_metrics,
                "region_outer_metrics": final_region_outer_metrics,
                "whole_outer_metrics": whole_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return FoldTrainingResult(
        best_region_epoch=best_region_epoch,
        best_region_val_metrics=best_region_val_metrics,
        best_whole_epoch=best_whole_epoch,
        best_whole_val_metrics=best_whole_val_metrics,
        stopped_epoch=stopped_epoch,
        outer_predictions=outer_predictions,
    )


def _train_epoch(
    model: RegionBranchModel,
    loaders: Any,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    device: torch.device,
    gradient_clip_norm: float,
    pos_weight: float,
    mixup_probability: float,
    active_regions: tuple[int, ...],
    lambda_value: float,
    progress_description: str,
) -> tuple[dict[str, float], int, float, float]:
    """1 epoch学習し、平均損失群・勾配ノルム・最後の学習率を返す。"""
    model.train()
    sums: dict[str, float] = {
        "whole_loss": 0.0,
        "region_loss": 0.0,
        "region_centered_loss": 0.0,
        "total_loss": 0.0,
        "grad_norm": 0.0,
        "clip_fraction": 0.0,
        "mixup_fraction": 0.0,
        "region_skip_fraction": 0.0,
    }
    batch_count = 0
    pretrained_lr, region_lr = optimizer_learning_rates(optimizer)
    active = list(active_regions)
    progress = tqdm(
        loaders.natural,
        total=loaders.steps_per_epoch,
        desc=progress_description,
        leave=False,
        dynamic_ncols=True,
    )
    for natural_batch in progress:
        optimizer.zero_grad(set_to_none=True)

        nbt = batch_tensors(natural_batch, device)
        use_mixup = (
            mixup_probability > 0.0 and float(torch.rand(1).item()) < mixup_probability
        )
        whole_inputs = nbt.inputs
        if use_mixup:
            whole_inputs, targets_a, targets_b, mixup_lambda = _mixup_batch(
                nbt.inputs, nbt.vertebra_target
            )
        positive = nbt.vertebra_target.eq(1.0)
        run_region = not use_mixup and bool(positive.any())
        positive_indices = positive.nonzero(as_tuple=False).flatten()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                whole_inputs,
                nbt.region_mask,
                need_whole=True,
                need_region=run_region,
                region_sample_indices=positive_indices if run_region else None,
            )
            if output.whole_plane_logits is None:
                raise RuntimeError("whole_plane_logitsが計算されませんでした")
            if use_mixup:
                l_whole = mixup_lambda * broadcast_bce_loss(
                    output.whole_plane_logits, targets_a, pos_weight
                ) + (1.0 - mixup_lambda) * broadcast_bce_loss(
                    output.whole_plane_logits, targets_b, pos_weight
                )
            else:
                l_whole = broadcast_bce_loss(
                    output.whole_plane_logits, nbt.vertebra_target, pos_weight
                )

        loss = l_whole
        if run_region:
            if output.region_plane_logits is None or output.region_plane_valid is None:
                raise RuntimeError("region_plane_logitsが計算されませんでした")
            bag_logits, cell_valid = region_bag_logits(
                output.region_plane_logits, output.region_plane_valid
            )
            effective_target_valid = (
                cell_valid & nbt.region_target_valid[positive][:, active]
            )
            region_losses = compute_conditional_region_losses(
                bag_logits,
                nbt.region_target[positive][:, active],
                effective_target_valid,
                nbt.vertebra_target[positive],
            )
            if region_losses.valid_cells > 0:
                weighted_region_loss = lambda_value * region_losses.bce
                loss = loss + weighted_region_loss
                region_ran = True
                region_loss_value = float(region_losses.bce.detach())
                region_centered_value = float(region_losses.centered.detach())
                region_total_value = float(weighted_region_loss.detach())
            else:
                region_ran = False
                region_loss_value = 0.0
                region_centered_value = 0.0
                region_total_value = 0.0
        else:
            region_ran = False
            region_loss_value = 0.0
            region_centered_value = 0.0
            region_total_value = 0.0

        _backward_finite_loss(loss, "total loss")

        gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("学習gradientが非有限値です")
        if float(gradient_norm) > gradient_clip_norm:
            sums["clip_fraction"] += 1.0
        optimizer.step()

        sums["whole_loss"] += float(l_whole.detach())
        sums["region_loss"] += region_loss_value
        sums["region_centered_loss"] += region_centered_value
        sums["total_loss"] += float(l_whole.detach()) + region_total_value
        sums["grad_norm"] += float(gradient_norm)
        if use_mixup:
            sums["mixup_fraction"] += 1.0
        if not region_ran:
            sums["region_skip_fraction"] += 1.0
        batch_count += 1
        global_step += 1
        progress.set_postfix(total=f"{sums['total_loss'] / batch_count:.4f}")

    if batch_count == 0:
        raise ValueError("train loaderが空です")
    metrics = {key: value / batch_count for key, value in sums.items()}
    return metrics, global_step, pretrained_lr, region_lr


def _backward_finite_loss(loss: Tensor, name: str) -> None:
    """有限なscalar lossをbackwardし、既存gradientへ加算する。"""
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{name}が非有限値です")
    torch.autograd.backward(loss)


def _configure_compiled_training(
    model: RegionBranchModel, device: torch.device
) -> bool:
    """CUDA学習modelを実測済みのTorchInductor default modeでcompileする。"""
    if device.type != "cuda":
        return False
    model.compile(mode=TORCH_COMPILE_MODE, dynamic=False)
    return True


def _mixup_batch(
    inputs: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Baseline 0と同じ一様lambdaと共有batch permutationでmixupする。"""
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixup_lambda = float(np.random.uniform(0.0, 1.0))
    mixed_inputs = inputs * mixup_lambda + inputs[indices] * (1.0 - mixup_lambda)
    return mixed_inputs, targets, targets[indices], mixup_lambda


@torch.no_grad()
def evaluate(
    model: RegionBranchModel,
    loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    active_regions: tuple[int, ...],
    progress_description: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """foldを1回通し、whole/統一region損失・AUROC/APと個票予測を返す。"""
    was_training = model.training
    model.eval()
    total_whole_loss = 0.0
    batch_count = 0
    active = list(active_regions)
    records: list[dict[str, Any]] = []
    region_logit_chunks: list[Tensor] = []
    region_target_chunks: list[Tensor] = []
    region_valid_chunks: list[Tensor] = []
    vertebra_target_chunks: list[Tensor] = []
    progress = tqdm(loader, desc=progress_description, leave=False, dynamic_ncols=True)
    for batch in progress:
        bt = batch_tensors(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(bt.inputs, bt.region_mask, need_whole=True, need_region=True)
            if output.whole_plane_logits is None:
                raise RuntimeError("whole_plane_logitsが計算されませんでした")
            whole_loss = broadcast_bce_loss(
                output.whole_plane_logits, bt.vertebra_target, pos_weight
            )
        if output.region_plane_logits is None or output.region_plane_valid is None:
            raise RuntimeError("region_plane_logitsが計算されませんでした")
        bag_logits, cell_valid = region_bag_logits(
            output.region_plane_logits, output.region_plane_valid
        )
        effective_target_valid = cell_valid & bt.region_target_valid[:, active]
        region_logit_chunks.append(bag_logits.float().cpu())
        region_target_chunks.append(bt.region_target[:, active].float().cpu())
        region_valid_chunks.append(effective_target_valid.cpu())
        vertebra_target_chunks.append(bt.vertebra_target.float().cpu())
        total_whole_loss += float(whole_loss)
        batch_count += 1

        whole_probability = bag_probabilities(output.whole_plane_logits).float()
        whole_score = whole_probability.cpu().numpy()
        conditional_region_probability = bag_logits.sigmoid()
        region_probability = (
            (conditional_region_probability * whole_probability[:, None])
            .float()
            .cpu()
            .numpy()
        )
        region_target_np = bt.region_target.cpu().numpy()
        region_valid_np = bt.region_hard_valid.cpu().numpy()
        vertebra_target_np = bt.vertebra_target.cpu().numpy()
        study_ids = _batch_strings(batch, "study_id")
        levels = _batch_strings(batch, "level")
        folds = _batch_folds(batch)
        for i in range(len(study_ids)):
            record: dict[str, Any] = {
                "study_id": study_ids[i],
                "level": levels[i],
                "fold": folds[i],
                "vertebra_target": int(vertebra_target_np[i]),
                "vertebra_score": float(whole_score[i]),
            }
            for column_index, column in enumerate(REGION_COLUMNS):
                record[f"{column}_target"] = float(region_target_np[i, column_index])
                record[f"{column}_target_valid"] = bool(
                    region_valid_np[i, column_index]
                )
                record[f"{column}_score"] = float("nan")
                record[f"{column}_conditional_score"] = float("nan")
            for position, region_index in enumerate(active_regions):
                column = REGION_COLUMNS[region_index]
                record[f"{column}_score"] = float(region_probability[i, position])
                record[f"{column}_conditional_score"] = float(
                    conditional_region_probability[i, position]
                )
            records.append(record)
        progress.set_postfix(whole_bce=f"{total_whole_loss / batch_count:.4f}")

    if batch_count == 0:
        raise ValueError("評価対象loaderが空です")
    if was_training:
        model.train()

    whole_loss_mean = total_whole_loss / batch_count
    region_losses = compute_conditional_region_losses(
        torch.cat(region_logit_chunks),
        torch.cat(region_target_chunks),
        torch.cat(region_valid_chunks),
        torch.cat(vertebra_target_chunks),
    )
    predictions = pd.DataFrame(records)
    metrics: dict[str, float] = {
        "whole": whole_loss_mean,
        "region_loss": float(region_losses.bce),
        "region_centered_loss": float(region_losses.centered),
        "whole_auroc": safe_auroc(
            predictions["vertebra_target"].to_numpy(),
            predictions["vertebra_score"].to_numpy(),
        ),
        "whole_ap": safe_average_precision(
            predictions["vertebra_target"].to_numpy(),
            predictions["vertebra_score"].to_numpy(),
        ),
    }
    region_aps: list[float] = []
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        valid = predictions[f"{column}_target_valid"].to_numpy()
        if valid.any():
            region_ap = safe_average_precision(
                predictions.loc[valid, f"{column}_target"].to_numpy(),
                predictions.loc[valid, f"{column}_score"].to_numpy(),
            )
            metrics[f"{column}_ap"] = region_ap
            metrics[f"{column}_auroc"] = safe_auroc(
                predictions.loc[valid, f"{column}_target"].to_numpy(),
                predictions.loc[valid, f"{column}_score"].to_numpy(),
            )
            region_aps.append(region_ap)
    if not region_aps:
        raise ValueError(
            f"{progress_description}: active_regionsのいずれにも有効セルがありません"
        )
    metrics["region_macro_ap"] = float(np.mean(region_aps))
    metrics.update(_conditional_region_metrics(predictions, active_regions))
    return metrics, predictions


def _conditional_region_metrics(
    predictions: pd.DataFrame, active_regions: tuple[int, ...]
) -> dict[str, float]:
    """人手GT×椎体陽性cellだけでconditionalなregion AP/AUROCを集計する。

    region headは`vertebra_target == 1`のbagでしか学習していないため、局在性能は
    その条件下でのみ意味を持つ。`{column}_score`はwhole確率を掛けたmarginalであり、
    母集団の大半を占めるwhole-negative cellとwhole headのdriftに支配されるので、
    ここではwhole確率を含まない`{column}_conditional_score`で評価する。

    有効cellや陽性が無いregionは集計から外し、macroが空ならNaNを返す。人手GTの
    陽性は1 foldあたり数十個しかなく、退化は正常に起こりうるため例外にはしない。
    """
    metrics: dict[str, float] = {}
    positive = predictions["vertebra_target"].to_numpy(dtype=bool)
    aps: list[float] = []
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        valid = predictions[f"{column}_target_valid"].to_numpy(dtype=bool) & positive
        if not valid.any():
            continue
        targets = predictions.loc[valid, f"{column}_target"].to_numpy()
        scores = predictions.loc[valid, f"{column}_conditional_score"].to_numpy()
        ap = safe_average_precision(targets, scores)
        metrics[f"{column}_cond_ap"] = ap
        metrics[f"{column}_cond_auroc"] = safe_auroc(targets, scores)
        metrics[f"{column}_cond_n_positive"] = float(targets.sum())
        if np.isfinite(ap):
            aps.append(ap)
    metrics["region_cond_macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def _hard_region_metrics(
    predictions: pd.DataFrame, active_regions: tuple[int, ...]
) -> dict[str, float]:
    """hard-valid cellだけでend-to-end region AP/AUROCを集計する。"""
    metrics: dict[str, float] = {}
    aps: list[float] = []
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        valid = predictions[f"{column}_target_valid"].to_numpy(dtype=bool)
        if not valid.any():
            continue
        targets = predictions.loc[valid, f"{column}_target"].to_numpy()
        scores = predictions.loc[valid, f"{column}_score"].to_numpy()
        ap = safe_average_precision(targets, scores)
        metrics[f"{column}_ap"] = ap
        metrics[f"{column}_auroc"] = safe_auroc(targets, scores)
        aps.append(ap)
    if not aps:
        raise ValueError("hard-validなregion cellがありません")
    metrics["region_macro_ap"] = float(np.mean(aps))
    metrics.update(_conditional_region_metrics(predictions, active_regions))
    return metrics


@torch.no_grad()
def _collect_diagnostic_arrays(
    model: RegionBranchModel,
    diagnostic_loader: DataLoader[Any],
    active_regions: tuple[int, ...],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """固定diagnostic subsetのwhole/region bag logitとpseudo targetをまとめて返す。"""
    was_training = model.training
    model.eval()
    active = list(active_regions)
    region_chunks: list[np.ndarray] = []
    whole_chunks: list[np.ndarray] = []
    pseudo_target_chunks: list[np.ndarray] = []
    pseudo_valid_chunks: list[np.ndarray] = []
    for batch in diagnostic_loader:
        bt = batch_tensors(batch, device)
        output = model(bt.inputs, bt.region_mask, need_whole=True, need_region=True)
        if (
            output.whole_plane_logits is None
            or output.region_plane_logits is None
            or output.region_plane_valid is None
        ):
            raise RuntimeError("診断にはwhole/region両方のlogitが必要です")
        bag_logits, _ = region_bag_logits(
            output.region_plane_logits, output.region_plane_valid
        )
        whole_probability = bag_probabilities(output.whole_plane_logits).clamp(
            PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        region_chunks.append(bag_logits.float().cpu().numpy())
        whole_chunks.append(torch.logit(whole_probability).float().cpu().numpy())
        pseudo_target_chunks.append(bt.region_target[:, active].float().cpu().numpy())
        pseudo_valid_chunks.append(bt.region_pseudo_valid[:, active].cpu().numpy())
    if was_training:
        model.train()
    return (
        np.concatenate(region_chunks, axis=0),
        np.concatenate(whole_chunks, axis=0),
        np.concatenate(pseudo_target_chunks, axis=0),
        np.concatenate(pseudo_valid_chunks, axis=0),
    )


def _set_loaders_epoch(loaders: Any, epoch: int) -> None:
    """natural samplerへepochを伝える。"""
    natural_sampler = loaders.natural.sampler
    if not isinstance(natural_sampler, EpochShuffleSampler):
        raise TypeError("natural loaderにはEpochShuffleSamplerが必要です")
    natural_sampler.set_epoch(epoch)


def _update_early_stopping(
    current_metric: float, best_metric: float, bad_epochs: int
) -> tuple[float, int, bool]:
    """val unified centered region lossでearly stoppingを更新する（minimize）。"""
    if not np.isfinite(current_metric):
        raise FloatingPointError("val region_centered_lossが非有限値です")
    if current_metric < best_metric:
        return current_metric, 0, True
    return best_metric, bad_epochs + 1, False


def _resume_state(
    model: RegionBranchModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_path: Path,
    device: torch.device,
    config: dict[str, Any],
    resume: bool,
) -> tuple[
    int, int, int, float, dict[str, float], int, float, dict[str, float], float, int
]:
    """必要時に最後のcheckpointを復元し、学習再開状態を返す。"""
    if not resume:
        return (
            1,
            0,
            0,
            float("inf"),
            {"region_centered_loss": float("inf")},
            0,
            float("inf"),
            {"whole": float("inf")},
            float("inf"),
            0,
        )
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
        int(checkpoint["best_region_epoch"]),
        float(checkpoint["best_region_loss"]),
        {
            key: float(value)
            for key, value in checkpoint["best_region_val_metrics"].items()
        },
        int(checkpoint["best_whole_epoch"]),
        float(checkpoint["best_whole_loss"]),
        {
            key: float(value)
            for key, value in checkpoint["best_whole_val_metrics"].items()
        },
        float(checkpoint["early_stopping_best_region_loss"]),
        int(checkpoint["no_improvement"]),
    )


def _save_checkpoint(
    path: Path,
    model: RegionBranchModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_region_epoch: int,
    best_region_loss: float,
    best_region_val_metrics: dict[str, float],
    best_whole_epoch: int,
    best_whole_loss: float,
    best_whole_val_metrics: dict[str, float],
    early_stopping_best_region_loss: float,
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
            "best_region_epoch": best_region_epoch,
            "best_region_loss": best_region_loss,
            "best_region_val_metrics": best_region_val_metrics,
            "best_whole_epoch": best_whole_epoch,
            "best_whole_loss": best_whole_loss,
            "best_whole_val_metrics": best_whole_val_metrics,
            "early_stopping_best_region_loss": early_stopping_best_region_loss,
            "no_improvement": no_improvement,
            "checkpoint_role": checkpoint_role,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """蓄積した行をCSVへ安全に書き直す。"""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """再開時に既存CSVを読み戻す。"""
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _append_log(path: Path, row: dict[str, Any]) -> None:
    """主要epoch情報をコンソールとローカルログへ出力する。"""
    line = (
        f"epoch={row['epoch']} "
        f"train_total={row['train_total_loss']:.6f} "
        f"train_whole={row['train_whole_loss']:.6f} "
        f"train_region={row['train_region_loss']:.6f} "
        f"train_region_centered={row['train_region_centered_loss']:.6f} "
        f"val_whole={row['val_whole']:.6f} "
        f"val_region_centered={row['val_region_centered_loss']:.6f} "
        f"val_region_macro_ap={row['val_region_macro_ap']:.6f} "
        f"val_region_cond_macro_ap={row['val_region_cond_macro_ap']:.6f} "
        f"val_whole_auroc={row['val_whole_auroc']:.6f} "
        f"pretrained_lr={row['pretrained_lr']:.3e} region_lr={row['region_lr']:.3e} "
        f"seconds={row['epoch_seconds']:.2f} "
        f"is_best_region={row['is_best_region']} "
        f"is_best_whole={row['is_best_whole']} "
        f"bad_epochs={row['early_stopping_bad_epochs']} "
        f"collapse_alarm={row['collapse_alarm']}"
    )
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """予測CSVを一時ファイル経由で置換する。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


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
