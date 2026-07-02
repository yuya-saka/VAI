"""Output paths and lightweight experiment logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_output_base(config: dict[str, Any], root: Path) -> Path:
    """Resolve a Stage2-only output directory."""
    experiment = config.get("experiment", {})
    phase = str(experiment.get("phase", "default"))
    name = str(experiment.get("name", "default"))
    return root / "train_models" / "stage2" / "outputs" / phase / name


def resolve_fold_paths(
    config: dict[str, Any], fold: int, root: Path
) -> tuple[Path, Path, Path]:
    """Create and return the best-model, resume-state, and fold directory paths."""
    fold_dir = resolve_output_base(config, root) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir / "best_model.pt", fold_dir / "latest.pt", fold_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one structured training record."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def reject_unresumed_reuse(output_dir: Path, resume: bool) -> None:
    """Raise if a fresh (non-resume) run would append to existing training history."""
    if resume:
        return
    if list(output_dir.glob("fold*/training.jsonl")):
        raise RuntimeError(
            f"output directory already has training history: {output_dir} "
            "(pass --resume to continue it, or change experiment.name)"
        )


def prune_training_jsonl(path: Path, start_epoch: int) -> None:
    """Drop rows for epochs at or after ``start_epoch`` before a resume re-runs them."""
    if not path.exists():
        return
    kept_lines: list[str] = []
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if int(json.loads(line).get("epoch", -1)) < start_epoch:
                kept_lines.append(line)
    with path.open("w", encoding="utf-8") as file:
        for line in kept_lines:
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
    }
)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dotted-path keys."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def validate_resume_config(
    saved_config: dict[str, Any],
    current_config: dict[str, Any],
    next_epoch: int,
) -> None:
    """Reject resuming with a config that changed anything but environment/schedule knobs."""
    saved_flat = {
        key: value
        for key, value in _flatten_config(saved_config).items()
        if key not in _RESUME_ENV_ONLY_PATHS and not key.startswith("wandb.")
    }
    current_flat = {
        key: value
        for key, value in _flatten_config(current_config).items()
        if key not in _RESUME_ENV_ONLY_PATHS and not key.startswith("wandb.")
    }
    saved_flat.pop("training.epochs", None)
    current_epochs = current_flat.pop("training.epochs", None)

    mismatches = [
        f"{key}: saved={saved_flat.get(key)!r} current={current_flat.get(key)!r}"
        for key in sorted(set(saved_flat) | set(current_flat))
        if saved_flat.get(key) != current_flat.get(key)
    ]
    if mismatches:
        raise ValueError(
            "resume config mismatch against the checkpoint's saved config "
            "(use an identical config, or start a new experiment.name):\n"
            + "\n".join(mismatches)
        )
    if current_epochs is not None and int(current_epochs) < next_epoch:
        raise ValueError(
            f"training.epochs={current_epochs} is below the resumed "
            f"next_epoch={next_epoch}; increase epochs or start a new experiment.name"
        )
