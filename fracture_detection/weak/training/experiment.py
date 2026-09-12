"""Output paths and W&B experiment management for the weak-label model.

Adapted from ``baseline0/training/experiment.py`` (same path-resolution and
atomic-config-save pattern), pointed at this package's own output tree
instead of baseline0's, and with logging fields renamed for GT-passes instead
of epochs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

WEAK_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_root(config: dict[str, Any]) -> Path:
    """Resolve the experiment root from the required experiment.phase/name."""
    experiment = config.get("experiment", {})
    phase = experiment.get("phase")
    name = experiment.get("name")
    if not isinstance(phase, str) or not isinstance(name, str) or not phase or not name:
        raise ValueError("experiment.phaseとexperiment.nameは必須です")
    if (
        any(part in {".", ".."} for part in (phase, name))
        or "/" in phase
        or "/" in name
    ):
        raise ValueError("experiment.phase/nameにpath区切りは使えません")
    return WEAK_DIR / "outputs" / phase / name


def resolve_fold_dir(config: dict[str, Any], fold: int) -> Path:
    """Create and return the local artifact directory for one outer fold."""
    if fold not in range(5):
        raise ValueError(f"foldが不正です: {fold}")
    fold_dir = resolve_experiment_root(config) / f"outer{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def save_effective_config(config: dict[str, Any]) -> Path:
    """Save the CLI-resolved effective config at the experiment root."""
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "config.yaml"
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なる実効configがすでに存在します: {output_path}")
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def save_fold_effective_config(config: dict[str, Any], fold_dir: Path) -> Path:
    """Save the fold-specific effective config in its artifact directory."""
    output_path = fold_dir / "effective_config.yaml"
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なるfold実効configがすでに存在します: {output_path}")
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def _get_wandb() -> Any | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def initialize_wandb(config: dict[str, Any], fold: int) -> Any | None:
    """Start a fold-scoped W&B run only when configured and importable."""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None
    wandb_module = _get_wandb()
    if wandb_module is None:
        warnings.warn("wandb.enabled=trueですがwandbをimportできません", stacklevel=2)
        return None

    experiment = config["experiment"]
    project = (
        wandb_config.get("project")
        or f"weak-{experiment['phase']}-{experiment['name']}"
    )
    run_name = wandb_config.get("run_name") or f"fold{fold}"
    try:
        wandb_module.init(project=project, name=run_name, config=config, reinit=True)
    except Exception as error:  # W&B通信失敗時もローカル学習は継続する。
        warnings.warn(f"W&B初期化に失敗しました: {error}", stacklevel=2)
        return None
    return wandb_module


def log_wandb_pass(
    wandb_module: Any,
    gt_pass: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    transferred_lr: float,
    new_lr: float,
    elapsed_seconds: float,
) -> None:
    """Log one GT-pass's train/validation metrics and learning rates."""
    payload = {
        "gt_pass": gt_pass,
        "transferred_lr": transferred_lr,
        "new_lr": new_lr,
        "gt_pass_seconds": elapsed_seconds,
    }
    payload.update({f"train_{key}": value for key, value in train_metrics.items()})
    payload.update({f"val_{key}": value for key, value in validation_metrics.items()})
    wandb_module.log(payload, step=gt_pass)


def update_best_summary(
    wandb_module: Any, gt_pass: int, validation_metrics: dict[str, float]
) -> None:
    """Record the best-checkpoint summary in W&B."""
    wandb_module.run.summary["best_gt_pass"] = gt_pass
    for key, value in validation_metrics.items():
        wandb_module.run.summary[f"best_val_{key}"] = value


def finish_wandb(
    wandb_module: Any | None,
    gt_pass: int,
    validation_metrics: dict[str, float] | None,
    train_rows: int,
    validation_rows: int,
) -> None:
    """Save the final summary and close the W&B run."""
    if wandb_module is None:
        return
    wandb_module.run.summary["stopped_gt_pass"] = gt_pass
    if validation_metrics is not None:
        for key, value in validation_metrics.items():
            wandb_module.run.summary[f"final_val_{key}"] = value
    wandb_module.run.summary["train_rows"] = train_rows
    wandb_module.run.summary["validation_rows"] = validation_rows
    wandb_module.finish()
