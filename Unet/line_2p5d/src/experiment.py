"""Weights & Biasesによる2.5D実験ログ管理。"""

from __future__ import annotations

import math
from typing import Any


def _get_wandb() -> Any | None:
    """wandbを遅延importする。"""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def initialize_wandb(
    config: dict[str, Any],
    fold: int,
    wandb_module: Any | None = None,
) -> tuple[bool, Any | None]:
    """設定からfold単位のwandb runを初期化する。"""
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return False, None
    module = wandb_module if wandb_module is not None else _get_wandb()
    if module is None:
        print(
            "[WARNING] wandb.enabled=trueですがwandbをimportできないため、"
            "ログをスキップします。",
            flush=True,
        )
        return False, None
    experiment = config["experiment"]
    default_project = f"unet-{experiment['phase']}-{experiment['name']}"
    module.init(
        project=wandb_config.get("project") or default_project,
        name=wandb_config.get("run_name") or f"fold{fold}",
        config=config,
        reinit=True,
    )
    return True, module


def log_wandb_epoch(
    wandb_module: Any,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """epoch指標のうち有限なscalar値をwandbへ送る。"""
    values = {
        key: float(value)
        for key, value in metrics.items()
        if key != "epoch"
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    }
    wandb_module.log(values, step=epoch)


def update_best_summary(
    wandb_module: Any,
    epoch: int,
    selection_metric: str,
    selection_value: float,
    validation_metrics: dict[str, Any],
) -> None:
    """best checkpointのepochと主要検証指標をsummaryへ保存する。"""
    summary = wandb_module.run.summary
    summary["best_epoch"] = epoch
    summary["best_selection_metric"] = selection_metric
    summary["best_selection_value"] = selection_value
    for key, value in validation_metrics.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            summary[f"best_{key}"] = float(value)


def finish_wandb(
    wandb_module: Any,
    test_metrics: dict[str, Any],
    line_summary: dict[str, Any],
) -> None:
    """test指標と出力件数をsummaryへ保存し、runを終了する。"""
    summary = wandb_module.run.summary
    for key, value in test_metrics.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metric_name = key.removeprefix("val_")
            summary[f"test_{metric_name}"] = float(value)
    summary["test_output_samples"] = int(line_summary["n_samples"])
    summary["line_extend_ratio"] = float(line_summary["line_extend_ratio"])
    wandb_module.finish()
