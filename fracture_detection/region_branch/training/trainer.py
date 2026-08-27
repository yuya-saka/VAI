"""region_branchのfold単位の学習・検証・checkpoint制御。

1 stepはnatural batch（whole path、mixupあり）と補助region batch
（human+negative+pseudo、mixupなし）を順にforward/backwardし、勾配を蓄積してから
1回だけoptimizer更新する。目的関数は
`L = L_whole + lambda_k*(L_exact + alpha_k*L_rank)`のまま変えない。
checkpoint選択とearly stoppingはval `L_whole + lambda_k*L_exact`の最小化で行う
（`L_rank`はvalで構造的に計算不能なため）。
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
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.region_branch.data_pipeline.loaders import OuterFoldLoaders
from fracture_detection.region_branch.data_pipeline.sampling import (
    batch_tensors,
    concatenate_batches,
    set_source_loader_epoch,
)
from fracture_detection.region_branch.modeling.losses import (
    ExactLossTerms,
    combine_exact_terms,
    compute_exact_loss_terms,
    region_bag_logits,
    region_rank_loss,
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

    best_epoch: int
    best_val_metrics: dict[str, float]
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
    """1 outer foldを学習し、best checkpointでouterを一度だけ推論する。

    natural stream / 3ソース補助loader・inner/outer/diagnostic評価loaderは
    すべて呼び出し側（CLI）が構築して渡す。ここではmanifestを読まない
    （Baseline 0のtrain_foldと同じ責務分離。合成データでのunit testを可能にする）。
    """
    training = config["training"]
    region = config["region"]
    active_regions = tuple(int(value) for value in region["active_regions"])
    max_epochs = int(training["max_epochs"])
    min_epoch = int(training["min_epoch"])
    patience = int(training["early_stopping_patience"])
    pos_weight = float(training["pos_weight"])
    lambda_value = float(calibration.lambda_)
    alpha_value = float(calibration.alpha)

    print(
        f"[outer {outer_fold}] 学習を初期化しています: device={device}, "
        f"active_regions={active_regions}, alpha={alpha_value:.6f}, lambda={lambda_value:.6f}, "
        f"steps/epoch={loaders.steps_per_epoch}, "
        f"human_pool={len(loaders.pools.human):,}, negative_pool={len(loaders.pools.negative):,}, "
        f"pseudo_pool={len(loaders.pools.pseudo):,}",
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
    best_path = fold_dir / "best_model.pt"
    last_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "history.csv"
    diagnostics_path = fold_dir / "diagnostics.csv"
    log_path = fold_dir / "training.log"

    (
        start_epoch,
        global_step,
        best_epoch,
        best_val_metrics,
        early_stopping_best_total,
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
                outer_fold,
                lambda_value,
                alpha_value,
                f"outer{outer_fold} epoch{epoch}/{max_epochs} 学習",
            )
            val_metrics, _ = evaluate(
                model,
                inner_loader,
                device,
                pos_weight,
                lambda_value,
                active_regions,
                f"outer{outer_fold} epoch{epoch}/{max_epochs} val検証",
            )
            stopped_epoch = epoch
            scheduler.step()

            region_array, whole_array, teacher_array = _collect_diagnostic_arrays(
                model, diagnostic_loader, active_regions, device
            )
            diagnostic_record = compute_diagnostics(
                epoch, region_array, whole_array, teacher_array, active_regions
            )
            collapse_alarm = collapse_monitor.update(diagnostic_record)
            diagnostic_rows.append(diagnostic_record.as_row())
            _write_rows(diagnostics_path, diagnostic_rows)

            eligible = epoch >= min_epoch
            current_total = val_metrics["total"]
            checkpoint_improved = (
                eligible
                and np.isfinite(current_total)
                and current_total < best_val_metrics["total"]
            )
            early_stopping_improved = False
            if eligible:
                (
                    early_stopping_best_total,
                    no_improvement,
                    early_stopping_improved,
                ) = _update_early_stopping(
                    current_total, early_stopping_best_total, no_improvement
                )
            if checkpoint_improved:
                best_epoch = epoch
                best_val_metrics = val_metrics
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    global_step,
                    best_epoch,
                    best_val_metrics,
                    early_stopping_best_total,
                    no_improvement,
                    checkpoint_role="best_val_total",
                )
            _save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                global_step,
                best_epoch,
                best_val_metrics,
                early_stopping_best_total,
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
                "is_best": checkpoint_improved,
                "early_stopping_improved": early_stopping_improved,
                "early_stopping_best_total": early_stopping_best_total,
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

    if best_epoch < min_epoch or not best_path.is_file():
        raise RuntimeError("min_epoch以降に有効なbest checkpointを保存できませんでした")
    outer_prediction_path = fold_dir / "outer_predictions.csv"
    if outer_prediction_path.exists():
        raise RuntimeError("outer予測が既に存在するため再推論を拒否しました")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_role") != "best_val_total":
        raise ValueError(
            f"best checkpoint roleが不正です: {checkpoint.get('checkpoint_role')}"
        )
    model.load_state_dict(checkpoint["model"])
    print(
        f"[outer {outer_fold}] outerを1回だけ推論しています(val total最小のcheckpoint)",
        flush=True,
    )
    outer_metrics, outer_predictions = evaluate(
        model,
        outer_loader,
        device,
        pos_weight,
        lambda_value,
        active_regions,
        f"outer{outer_fold} outer推論",
    )
    _atomic_write_csv(outer_predictions, outer_prediction_path)
    (fold_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "val_fold": int(config["runtime"]["inner_fold"]),
                "active_regions": list(active_regions),
                "alpha": alpha_value,
                "lambda": lambda_value,
                "stopped_epoch": stopped_epoch,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "outer_metrics": outer_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return FoldTrainingResult(
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
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
    outer_fold: int,
    lambda_value: float,
    alpha_value: float,
    progress_description: str,
) -> tuple[dict[str, float], int, float, float]:
    """1 epoch学習し、平均損失群・勾配ノルム・最後の学習率を返す。"""
    model.train()
    sums: dict[str, float] = {
        "whole_loss": 0.0,
        "exact_loss": 0.0,
        "human_loss": 0.0,
        "negative_loss": 0.0,
        "rank_loss": 0.0,
        "total_loss": 0.0,
        "grad_norm": 0.0,
        "clip_fraction": 0.0,
        "mixup_fraction": 0.0,
    }
    batch_count = 0
    pretrained_lr, region_lr = optimizer_learning_rates(optimizer)
    progress = tqdm(
        zip(
            loaders.natural,
            loaders.human,
            loaders.negative,
            loaders.pseudo,
            strict=True,
        ),
        total=loaders.steps_per_epoch,
        desc=progress_description,
        leave=False,
        dynamic_ncols=True,
    )
    for natural_batch, human_batch, negative_batch, pseudo_batch in progress:
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
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            whole_output = model(
                whole_inputs, nbt.region_mask, need_whole=True, need_region=False
            )
            if whole_output.whole_plane_logits is None:
                raise RuntimeError("whole_plane_logitsが計算されませんでした")
            if use_mixup:
                l_whole = mixup_lambda * broadcast_bce_loss(
                    whole_output.whole_plane_logits, targets_a, pos_weight
                ) + (1.0 - mixup_lambda) * broadcast_bce_loss(
                    whole_output.whole_plane_logits, targets_b, pos_weight
                )
            else:
                l_whole = broadcast_bce_loss(
                    whole_output.whole_plane_logits, nbt.vertebra_target, pos_weight
                )
        _backward_finite_loss(l_whole, "whole loss")

        aux_batch = concatenate_batches([human_batch, negative_batch, pseudo_batch])
        abt = batch_tensors(aux_batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            region_output = model(
                abt.inputs, abt.region_mask, need_whole=False, need_region=True
            )
            if (
                region_output.region_plane_logits is None
                or region_output.region_plane_valid is None
            ):
                raise RuntimeError("region_plane_logitsが計算されませんでした")
        bag_logits, cell_valid = region_bag_logits(
            region_output.region_plane_logits, region_output.region_plane_valid
        )
        exact_terms = compute_exact_loss_terms(
            bag_logits,
            abt.region_targets[:, active_regions],
            abt.region_target_valid[:, active_regions],
            cell_valid,
            abt.vertebra_target,
        )
        l_exact, l_h, l_n = combine_exact_terms(exact_terms, bag_logits)

        teacher_outer_fold = torch.full(
            (bag_logits.shape[0],), outer_fold, dtype=torch.int64, device=device
        )
        generator = torch.Generator().manual_seed(
            _rank_pair_seed(outer_fold, global_step)
        )
        l_rank, _ = region_rank_loss(
            bag_logits,
            abt.region_scores[:, active_regions],
            abt.vertebra_target,
            teacher_outer_fold,
            loaders.temperatures[list(active_regions)],
            generator,
        )

        weighted_region_loss = lambda_value * (l_exact + alpha_value * l_rank)
        _backward_finite_loss(weighted_region_loss, "weighted region loss")
        total = l_whole.detach() + weighted_region_loss.detach()
        gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("学習gradientが非有限値です")
        if float(gradient_norm) > gradient_clip_norm:
            sums["clip_fraction"] += 1.0
        optimizer.step()

        sums["whole_loss"] += float(l_whole.detach())
        sums["exact_loss"] += float(l_exact.detach())
        sums["human_loss"] += float(l_h.detach())
        sums["negative_loss"] += float(l_n.detach())
        sums["rank_loss"] += float(l_rank.detach())
        sums["total_loss"] += float(total.detach())
        sums["grad_norm"] += float(gradient_norm)
        if use_mixup:
            sums["mixup_fraction"] += 1.0
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


def _rank_pair_seed(outer_fold: int, step: int) -> int:
    """stepごとに変わるが再現可能なranking pair生成seed。"""
    return (outer_fold * 1_000_003 + step) % (2**31 - 1)


@torch.no_grad()
def evaluate(
    model: RegionBranchModel,
    loader: DataLoader[Any],
    device: torch.device,
    pos_weight: float,
    lambda_value: float,
    active_regions: tuple[int, ...],
    progress_description: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """foldを1回通し、whole/exact損失・簡易AUROC/APと個票予測を返す。"""
    was_training = model.training
    model.eval()
    total_whole_loss = 0.0
    batch_count = 0
    n_active = len(active_regions)
    terms = ExactLossTerms(
        human_weighted_loss=torch.zeros(n_active, device=device),
        human_weight=torch.zeros(n_active, device=device),
        negative_loss=torch.zeros(n_active, device=device),
        negative_count=torch.zeros(n_active, device=device),
    )
    records: list[dict[str, Any]] = []
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
        batch_terms = compute_exact_loss_terms(
            bag_logits,
            bt.region_targets[:, active_regions],
            bt.region_target_valid[:, active_regions],
            cell_valid,
            bt.vertebra_target,
        )
        terms = terms + batch_terms
        total_whole_loss += float(whole_loss)
        batch_count += 1

        whole_score = bag_probabilities(output.whole_plane_logits).float().cpu().numpy()
        region_probability = bag_logits.sigmoid().float().cpu().numpy()
        region_target_np = bt.region_targets.cpu().numpy()
        region_valid_np = bt.region_target_valid.cpu().numpy()
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
            for position, region_index in enumerate(active_regions):
                column = REGION_COLUMNS[region_index]
                record[f"{column}_score"] = float(region_probability[i, position])
            records.append(record)
        progress.set_postfix(whole_bce=f"{total_whole_loss / batch_count:.4f}")

    if batch_count == 0:
        raise ValueError("評価対象loaderが空です")
    if was_training:
        model.train()

    l_exact, l_h, l_n = combine_exact_terms(terms, terms.human_weighted_loss)
    whole_loss_mean = total_whole_loss / batch_count
    predictions = pd.DataFrame(records)
    metrics = {
        "whole_loss": whole_loss_mean,
        "exact_loss": float(l_exact),
        "human_loss": float(l_h),
        "negative_loss": float(l_n),
        "total": whole_loss_mean + lambda_value * float(l_exact),
        "whole_auroc": safe_auroc(
            predictions["vertebra_target"].to_numpy(),
            predictions["vertebra_score"].to_numpy(),
        ),
        "whole_ap": safe_average_precision(
            predictions["vertebra_target"].to_numpy(),
            predictions["vertebra_score"].to_numpy(),
        ),
    }
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        valid = predictions[f"{column}_target_valid"].to_numpy()
        if valid.any():
            metrics[f"{column}_ap"] = safe_average_precision(
                predictions.loc[valid, f"{column}_target"].to_numpy(),
                predictions.loc[valid, f"{column}_score"].to_numpy(),
            )
            metrics[f"{column}_auroc"] = safe_auroc(
                predictions.loc[valid, f"{column}_target"].to_numpy(),
                predictions.loc[valid, f"{column}_score"].to_numpy(),
            )
    return metrics, predictions


@torch.no_grad()
def _collect_diagnostic_arrays(
    model: RegionBranchModel,
    diagnostic_loader: DataLoader[Any],
    active_regions: tuple[int, ...],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """固定diagnostic subsetのwhole/region bag logitと教師scoreをまとめて返す。"""
    was_training = model.training
    model.eval()
    region_chunks: list[np.ndarray] = []
    whole_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
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
        score_chunks.append(bt.region_scores[:, active_regions].float().cpu().numpy())
    if was_training:
        model.train()
    return (
        np.concatenate(region_chunks, axis=0),
        np.concatenate(whole_chunks, axis=0),
        np.concatenate(score_chunks, axis=0),
    )


def _set_loaders_epoch(loaders: Any, epoch: int) -> None:
    """natural samplerと3ソースqueueへepochを伝える。"""
    natural_sampler = loaders.natural.sampler
    if not isinstance(natural_sampler, EpochShuffleSampler):
        raise TypeError("natural loaderにはEpochShuffleSamplerが必要です")
    natural_sampler.set_epoch(epoch)
    for loader in (loaders.human, loaders.negative, loaders.pseudo):
        set_source_loader_epoch(loader, epoch)


def _update_early_stopping(
    current_total: float, best_total: float, bad_epochs: int
) -> tuple[float, int, bool]:
    """val totalに基づくearly stopping状態を更新する。"""
    if not np.isfinite(current_total):
        raise FloatingPointError("val totalが非有限値です")
    if current_total < best_total:
        return current_total, 0, True
    return best_total, bad_epochs + 1, False


def _resume_state(
    model: RegionBranchModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_path: Path,
    device: torch.device,
    config: dict[str, Any],
    resume: bool,
) -> tuple[int, int, int, dict[str, float], float, int]:
    """必要時に最後のcheckpointを復元し、学習再開状態を返す。"""
    default_metrics = {"total": float("inf")}
    if not resume:
        return 1, 0, 0, default_metrics, float("inf"), 0
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
        {key: float(value) for key, value in checkpoint["best_val_metrics"].items()},
        float(checkpoint["early_stopping_best_total"]),
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
    best_epoch: int,
    best_val_metrics: dict[str, float],
    early_stopping_best_total: float,
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
            "best_val_metrics": best_val_metrics,
            "early_stopping_best_total": early_stopping_best_total,
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
        f"train_exact={row['train_exact_loss']:.6f} "
        f"train_rank={row['train_rank_loss']:.6f} "
        f"val_total={row['val_total']:.6f} "
        f"val_whole_auroc={row['val_whole_auroc']:.6f} "
        f"pretrained_lr={row['pretrained_lr']:.3e} region_lr={row['region_lr']:.3e} "
        f"seconds={row['epoch_seconds']:.2f} "
        f"is_best={row['is_best']} "
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
