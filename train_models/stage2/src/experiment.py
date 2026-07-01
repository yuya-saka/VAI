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
) -> tuple[Path, Path]:
    """Create and return the checkpoint and fold directory paths."""
    fold_dir = resolve_output_base(config, root) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir / "best_model.pt", fold_dir


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one structured training record."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")
