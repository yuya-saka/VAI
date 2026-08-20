"""Upload existing shared-trainer history files to W&B without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from fracture_detection.core.wandb import (
    finish_wandb,
    initialize_wandb,
    sync_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既存の学習履歴をW&Bへ同期")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def sync_experiment(experiment_dir: Path) -> int:
    """Synchronize every completed epoch found under an experiment directory."""
    synced = 0
    fold_dirs = sorted(experiment_dir.glob("outer[0-4]"))
    if not fold_dirs:
        raise FileNotFoundError(f"outer foldがありません: {experiment_dir}")
    for fold_dir in fold_dirs:
        history_path = fold_dir / "history.csv"
        config_path = fold_dir / "effective_config.yaml"
        if not history_path.is_file() or not config_path.is_file():
            continue
        config = _load_config(config_path)
        outer_fold = int(fold_dir.name.removeprefix("outer"))
        run = initialize_wandb(config, outer_fold, fold_dir, force=True)
        if run is None:
            raise RuntimeError(f"W&B runを開始できませんでした: {fold_dir}")
        try:
            count = sync_history(run, pd.read_csv(history_path))
        finally:
            finish_wandb(run)
        synced += count
        print(f"[outer {outer_fold}] {count} epochs synced", flush=True)
    return synced


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config最上位がmappingではありません: {path}")
    return config


def main() -> None:
    args = parse_args()
    synced = sync_experiment(args.experiment_dir)
    print(f"total: {synced} epochs synced", flush=True)


if __name__ == "__main__":
    main()
