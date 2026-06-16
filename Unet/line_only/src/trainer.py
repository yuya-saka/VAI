"""1 fold の学習制御と実験オーケストレーション。"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import losses as line_losses
from .data_utils import (
    create_data_loaders,
    create_model_optimizer_scheduler,
    prepare_datasets_and_splits,
)
from .evaluation import evaluate, peak_dist
from .example_writer import save_examples
from .experiment import (
    build_epoch_log,
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    print_line_summary,
    resolve_fold_paths,
    update_best_summary,
)
from .inference import predict_lines_and_eval_test
from .model import VERTEBRA_TO_IDX

tempfile.tempdir = "/tmp"

TrainingStats = dict[str, float]

__all__ = [
    "evaluate",
    "peak_dist",
    "predict_lines_and_eval_test",
    "run_training_loop",
    "save_examples",
    "train_one_fold",
]


def _vertebra_indices(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """バッチの椎体名をモデル入力用インデックスへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0) for vertebra in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[dict[str, Any]],
    device: torch.device,
    image_size: int,
    grad_clip: float,
    use_line_loss: bool,
    lambda_angle: float,
    lambda_rho: float,
    confidence_gate_low: float,
    confidence_gate_high: float,
    warmup_weight: float,
) -> TrainingStats:
    """1 epoch の学習を実行し、平均損失を返す。"""
    model.train()
    loss_sum = 0.0
    angle_sum = 0.0
    rho_sum = 0.0
    gate_sum = 0.0
    step_count = 0

    for batch in train_loader:
        images = batch["image"].to(device).float()
        vertebra_indices = _vertebra_indices(batch, device)
        gt_heatmaps = batch["heatmaps"].to(device).float()
        gt_params = batch.get("line_params_gt")
        pred_heatmaps = torch.sigmoid(model(images, vertebra_indices))
        mse_loss = F.mse_loss(pred_heatmaps, gt_heatmaps, reduction="mean")
        loss = mse_loss

        if use_line_loss and gt_params is not None:
            line_loss = line_losses.compute_line_loss(
                pred_heatmaps,
                gt_params.to(device).float(),
                image_size,
                lambda_angle=lambda_angle,
                lambda_rho=lambda_rho,
                use_line_loss=True,
                confidence_gate_low=confidence_gate_low,
                confidence_gate_high=confidence_gate_high,
            )
            loss = mse_loss + warmup_weight * line_loss["total"]
            angle_sum += line_loss["angle"].item()
            rho_sum += line_loss["rho"].item()
            gate_sum += line_loss["gate_ratio"].item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += loss.item()
        step_count += 1

    denominator = max(1, step_count)
    return {
        "loss": loss_sum / denominator,
        "angle": angle_sum / denominator,
        "rho": rho_sum / denominator,
        "gate": gate_sum / denominator,
    }


def run_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    train_loader: Iterable[dict[str, Any]],
    val_loader: Iterable[dict[str, Any]],
    device: torch.device,
    cfg: dict[str, Any],
    best_path: Path,
    wandb_enabled: bool = False,
    _wandb: Any | None = None,
) -> None:
    """早期停止付きの学習ループを実行する。"""
    training_config = cfg.get("training", {})
    evaluation_config = cfg.get("evaluation", {})
    loss_config = cfg.get("loss", {})

    epochs = int(training_config.get("epochs", 20))
    early_stopping_patience = int(training_config.get("early_stopping_patience", 20))
    grad_clip = float(training_config.get("grad_clip", 1.0))
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    heatmap_threshold = float(evaluation_config.get("heatmap_threshold", 0.2))
    use_line_loss = bool(loss_config.get("use_line_loss", False))
    lambda_angle = float(
        loss_config.get("lambda_angle", loss_config.get("lambda_theta", 1.0))
    )
    lambda_rho = float(loss_config.get("lambda_rho", 1.0))
    warmup_epochs = int(loss_config.get("warmup_epochs", 10))
    warmup_start_epoch = int(loss_config.get("warmup_start_epoch", 0))
    warmup_mode = str(loss_config.get("warmup_mode", "linear"))
    confidence_gate_low = float(loss_config.get("confidence_gate_low", 0.3))
    confidence_gate_high = float(loss_config.get("confidence_gate_high", 0.6))

    best_peak_dist = float("inf")
    best_val_loss = float("inf")
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        warmup_weight = line_losses.get_warmup_weight(
            epoch,
            warmup_epochs,
            warmup_mode,
            warmup_start_epoch,
        )
        train_stats = _train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            image_size,
            grad_clip,
            use_line_loss,
            lambda_angle,
            lambda_rho,
            confidence_gate_low,
            confidence_gate_high,
            warmup_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            image_size,
            heatmap_threshold,
        )
        scheduler.step(val_metrics["val_loss_mse"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            build_epoch_log(
                epoch,
                epochs,
                learning_rate,
                train_stats,
                val_metrics,
                use_line_loss,
                warmup_weight,
                time.time() - start_time,
            )
        )

        if wandb_enabled and _wandb is not None:
            log_wandb_epoch(
                _wandb,
                epoch,
                learning_rate,
                train_stats,
                val_metrics,
                use_line_loss,
                warmup_weight,
            )

        # ベストエポック選択: peak_dist_mean（ピーク位置精度）で判定
        if val_metrics["peak_dist_mean"] < best_peak_dist - 1e-8:
            best_peak_dist = val_metrics["peak_dist_mean"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val": val_metrics},
                best_path,
            )
            print(
                f"  [SAVE] best -> {best_path} "
                f"(peak_dist={best_peak_dist:.4f}px, "
                f"val_mse={val_metrics['val_loss_mse']:.6f})"
            )
            if wandb_enabled and _wandb is not None:
                update_best_summary(_wandb, epoch, best_peak_dist, val_metrics)

        # Early stopping: val_loss_mse で判定
        if val_metrics["val_loss_mse"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["val_loss_mse"]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(
                    "[EARLY STOP] "
                    f"no improvement for {early_stopping_patience} epochs. "
                    f"best_val_loss={best_val_loss:.6f}, "
                    f"best_peak_dist={best_peak_dist:.4f}px"
                )
                break


def train_one_fold(cfg: dict[str, Any]) -> dict[str, Any]:
    """1 fold の学習・テスト・成果物保存を実行する。"""
    data_config = cfg.get("data", {})
    training_config = cfg.get("training", {})
    evaluation_config = cfg.get("evaluation", {})
    test_fold = int(data_config.get("test_fold", 0))
    heatmap_threshold = float(evaluation_config.get("heatmap_threshold", 0.2))

    (
        train_samples,
        val_samples,
        test_samples,
        root_dir,
        group,
        image_size,
        sigma,
        seed,
    ) = prepare_datasets_and_splits(cfg)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_samples,
        val_samples,
        test_samples,
        root_dir,
        group,
        image_size,
        sigma,
        seed,
        cfg,
    )

    gpu_id = int(training_config.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    wandb_enabled, wandb_module = initialize_wandb(cfg, test_fold)
    model, optimizer, scheduler = create_model_optimizer_scheduler(cfg, device)
    best_path, visualization_dir, line_output_dir = resolve_fold_paths(
        cfg,
        test_fold,
    )

    run_training_loop(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        cfg,
        best_path,
        wandb_enabled=wandb_enabled,
        _wandb=wandb_module,
    )

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print(
            "[WARNING] No best checkpoint saved "
            "(no improvement during training). Using current model state."
        )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        image_size,
        heatmap_threshold,
    )
    print(
        f"[TEST] fold={test_fold}  "
        f"mse={test_metrics['val_loss_mse']:.6f}  "
        f"peak={test_metrics['peak_dist_mean']:.2f}px"
    )
    line_summary = predict_lines_and_eval_test(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        device=device,
        dataset_root=root_dir,
        out_dir=line_output_dir,
    )
    print_line_summary(line_summary)

    print("[INFO] saving example overlays ...")
    save_examples(
        model,
        val_loader,
        device,
        visualization_dir / f"fold{test_fold}" / "val",
        n_save=16,
        tag="VAL",
    )
    save_examples(
        model,
        test_loader,
        device,
        visualization_dir / f"fold{test_fold}" / "test",
        n_save=16,
        tag="TEST",
    )
    print(f"[INFO] saved to {visualization_dir}/")

    if wandb_enabled and wandb_module is not None:
        finish_wandb(wandb_module, test_metrics, line_summary)

    return {
        "test_mse": test_metrics["val_loss_mse"],
        "test_peak_dist_mean": test_metrics["peak_dist_mean"],
        "line_perpendicular_dist_px_mean": line_summary["perpendicular_dist_px_mean"],
        "line_angle_error_deg_mean": line_summary["angle_error_deg_mean"],
        "line_rho_error_px_mean": line_summary["rho_error_px_mean"],
        "per_vertebra": test_metrics.get("per_vertebra", {}),
    }
