"""Stage4 output paths and immutable-run safeguards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train_models.stage3.src.experiment import (
    finish_wandb,
    initialize_wandb,
    reject_unresumed_reuse,
    validate_resume_config,
)

__all__ = [
    "append_jsonl",
    "finish_wandb",
    "initialize_wandb",
    "reject_unresumed_reuse",
    "resolve_fold_paths",
    "resolve_output_base",
    "validate_resume_config",
]


def resolve_output_base(config: dict[str, Any], root: Path) -> Path:
    """Resolve an arm/seed-specific Stage4 output directory."""
    experiment = config.get("experiment", {})
    seed = int(config.get("data", {}).get("random_seed", 42))
    return (
        root
        / "train_models"
        / "stage4"
        / "outputs"
        / str(experiment.get("name", "default"))
        / f"seed{seed}"
    )


def resolve_fold_paths(
    config: dict[str, Any],
    fold: int,
    root: Path,
) -> tuple[Path, Path, Path]:
    """Return final/latest checkpoint paths and create the fold directory."""
    fold_dir = resolve_output_base(config, root) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir / "final_model.pt", fold_dir / "latest.pt", fold_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one structured training record."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")
