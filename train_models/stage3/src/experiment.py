"""Stage3 output paths and resume safeguards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_output_base(config: dict[str, Any], root: Path) -> Path:
    experiment = config.get("experiment", {})
    return (
        root
        / "train_models"
        / "stage3"
        / "outputs"
        / str(experiment.get("phase", "default"))
        / str(experiment.get("name", "default"))
    )


def resolve_fold_paths(
    config: dict[str, Any], fold: int, root: Path
) -> tuple[Path, Path, Path]:
    fold_dir = resolve_output_base(config, root) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir / "best_model.pt", fold_dir / "latest.pt", fold_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def reject_unresumed_reuse(output_dir: Path, resume: bool) -> None:
    if resume:
        return
    if list(output_dir.glob("fold*/training.jsonl")):
        raise RuntimeError(
            f"output directory already has training history: {output_dir} "
            "(pass --resume or change experiment.name)"
        )


def prune_training_jsonl(path: Path, start_epoch: int) -> None:
    if not path.exists():
        return
    kept: list[str] = []
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line and int(json.loads(line).get("epoch", -1)) < start_epoch:
                kept.append(line)
    with path.open("w", encoding="utf-8") as file:
        for line in kept:
            file.write(line + "\n")


_RESUME_ENV_ONLY_PATHS = frozenset(
    {
        "training.gpu_id",
        "training.gpu_ids",
        "training.n_gpu",
        "training.num_workers",
        "training.persistent_workers",
        "training.prefetch_factor",
        "data.start_fold",
        "data.end_fold",
        "data.stage_to_local",
        "data.stage_root",
        "data.stage_workers",
    }
)


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(_flatten(value, f"{path}."))
        else:
            result[path] = value
    return result


def validate_resume_config(
    saved_config: dict[str, Any], current_config: dict[str, Any], next_epoch: int
) -> None:
    def relevant(config: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in _flatten(config).items()
            if key not in _RESUME_ENV_ONLY_PATHS and not key.startswith("wandb.")
        }

    saved = relevant(saved_config)
    current = relevant(current_config)
    saved.pop("training.epochs", None)
    current_epochs = current.pop("training.epochs", None)
    mismatches = [
        f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError("resume config mismatch:\n" + "\n".join(mismatches))
    if current_epochs is not None and int(current_epochs) < next_epoch:
        raise ValueError("training.epochs is below the resumed next epoch")


def _get_wandb() -> Any | None:
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def initialize_wandb(config: dict[str, Any], fold: int) -> tuple[bool, Any | None]:
    """Initialize one rank-0 W&B run for a fold."""
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return False, None
    client = _get_wandb()
    if client is None:
        print("[WARNING] wandb is enabled but not installed", flush=True)
        return False, None
    experiment = config.get("experiment", {})
    phase = str(experiment.get("phase", "stage3"))
    experiment_name = str(experiment["name"])
    run = client.init(
        project=wandb_config.get("project") or f"{phase}-{experiment_name}",
        name=wandb_config.get("run_name") or f"fold{fold}",
        config=config,
        reinit="finish_previous",
    )
    return True, run


def log_wandb_epoch(
    run: Any,
    epoch: int,
    train_stats: dict[str, float],
    valid_metrics: dict[str, Any],
    learning_rates: list[float],
    patience: int,
    best_auroc: float,
) -> None:
    """Log Stage3 optimization, classification, and control metrics."""
    at_05 = valid_metrics.get("at_05", {})
    payload = {
        "epoch": epoch,
        "train/loss": train_stats["loss"],
        "train/bag_loss": train_stats["bag_loss"],
        "train/negative_instance_loss": train_stats["negative_instance_loss"],
        "valid/loss": valid_metrics.get("loss", float("nan")),
        "valid/auroc": valid_metrics.get("auroc", float("nan")),
        "valid/auprc": valid_metrics.get("auprc", float("nan")),
        "valid/f1_05": at_05.get("f1", float("nan")),
        "valid/precision_05": at_05.get("precision", float("nan")),
        "valid/recall_05": at_05.get("recall", float("nan")),
        "training/patience": patience,
        "training/best_auroc": best_auroc,
    }
    for index, learning_rate in enumerate(learning_rates):
        payload[f"learning_rate/group_{index}"] = learning_rate
    run.log(payload, step=epoch)


def finish_wandb(run: Any, metrics: dict[str, Any]) -> None:
    """Store fold-best metrics and close the run."""
    run.summary["best/auroc"] = metrics.get("auroc", float("nan"))
    run.summary["best/auprc"] = metrics.get("auprc", float("nan"))
    run.summary["best/f1_05"] = metrics.get("at_05", {}).get("f1", float("nan"))
    run.finish()
