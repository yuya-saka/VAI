"""出力path、ローカルログ、任意W&B連携。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def experiment_dir(config: dict[str, Any]) -> Path:
    """実験のroot出力ディレクトリを返す。"""
    experiment_config = config.get("experiment", {})
    root = Path(
        experiment_config.get(
            "output_dir",
            "Unet/outputs/line_surface_3d",
        )
    )
    name = str(experiment_config.get("name", "default"))
    return root / name


def fold_paths(
    config: dict[str, Any],
    fold: int,
) -> dict[str, Path]:
    """fold固有の成果物pathを作成する。"""
    root = experiment_dir(config)
    paths = {
        "root": root,
        "checkpoint": root / "checkpoints" / f"best_fold{fold}.pt",
        "metrics": root / "metrics" / f"fold{fold}.jsonl",
        "manifest": root / "manifests" / f"fold{fold}.json",
        "test_metrics": root / "metrics" / f"test_fold{fold}.json",
        "prediction": root / "predictions" / f"fold{fold}",
        "visualization": root / "visualizations" / f"fold{fold}",
    }
    for path in paths.values():
        directory = path if path.suffix == "" else path.parent
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_effective_config(config: dict[str, Any]) -> Path:
    """CLI上書き後のconfigを保存する。"""
    output_path = experiment_dir(config) / "config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def append_epoch_metrics(path: Path, values: dict[str, Any]) -> None:
    """epoch指標をJSON Linesへ追記する。"""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(values, ensure_ascii=False) + "\n")


def save_json(path: Path, values: Any) -> None:
    """JSONを親directory作成込みで保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def initialize_wandb(
    config: dict[str, Any],
    fold: int,
) -> Any | None:
    """有効時だけW&B runを初期化する。"""
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandbがないためローカルログのみ使用します")
        return None
    experiment_name = str(config.get("experiment", {}).get("name", "default"))
    wandb.init(
        project=wandb_config.get("project") or "vai-line-surface-3d",
        name=wandb_config.get("run_name") or f"{experiment_name}-fold{fold}",
        config=config,
        reinit="finish_previous",
    )
    return wandb
