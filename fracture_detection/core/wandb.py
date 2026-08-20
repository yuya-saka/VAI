"""Shared W&B lifecycle and history synchronization."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def _load_wandb() -> Any | None:
    """Load W&B only when experiment tracking is used."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def initialize_wandb(
    config: dict[str, Any],
    outer_fold: int,
    fold_dir: Path,
    *,
    force: bool = False,
) -> Any | None:
    """Start or resume one fold-specific W&B run."""
    wandb_config = config.get("wandb", {})
    if not force and not wandb_config.get("enabled", True):
        return None
    wandb_module = _load_wandb()
    if wandb_module is None:
        warnings.warn("wandbがimportできないため追跡を無効化します", stacklevel=2)
        return None

    experiment = config["experiment"]
    project = (
        wandb_config.get("project")
        or f"fracture-{experiment['phase']}-{experiment['name']}"
    )
    configured_name = wandb_config.get("run_name")
    base_name = configured_name or experiment["name"]
    run_name = f"{base_name}-outer{outer_fold}"
    run_id = _resolve_run_id(wandb_module, fold_dir)
    try:
        return wandb_module.init(
            project=project,
            name=run_name,
            group=experiment["name"],
            job_type=str(config["arm"]["name"]),
            config=config,
            id=run_id,
            resume="allow",
            dir=str(fold_dir),
        )
    except Exception as error:
        warnings.warn(f"W&B初期化に失敗しました: {error}", stacklevel=2)
        return None


def log_wandb_epoch(run: Any | None, row: dict[str, Any]) -> None:
    """Log one complete shared-trainer history row."""
    if run is None:
        return
    epoch = int(row["epoch"])
    run.log(_python_scalars(row), step=epoch)
    if bool(row.get("is_best_val_auroc")):
        run.summary["best_epoch"] = epoch
        run.summary["best_val_auroc"] = row["val_auroc"]
    if bool(row.get("is_best_val_prauc")):
        run.summary["best_val_prauc_epoch"] = epoch
        run.summary["best_val_prauc"] = row["val_average_precision"]


def sync_history(run: Any, history: pd.DataFrame) -> int:
    """Upload all locally persisted epochs to a W&B run."""
    for row in history.to_dict(orient="records"):
        log_wandb_epoch(run, row)
    synced_epochs = len(history)
    run.summary["synced_epochs"] = synced_epochs
    return synced_epochs


def finish_wandb(run: Any | None) -> None:
    """Finish a W&B run when one was initialized."""
    if run is not None:
        run.finish()


def _resolve_run_id(wandb_module: Any, fold_dir: Path) -> str:
    path = fold_dir / "wandb_run_id.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    run_id = str(wandb_module.util.generate_id())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{run_id}\n", encoding="utf-8")
    temporary.replace(path)
    return run_id


def _python_scalars(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }
