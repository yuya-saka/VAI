"""Baseline 1の出力パスとW&B実験管理。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

BASELINE1_DIR = Path(__file__).resolve().parent


def resolve_experiment_root(config: dict[str, Any]) -> Path:
    """必須のphase/nameから実験ルートを解決する。"""
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
    return BASELINE1_DIR / "outputs" / phase / name


def resolve_fold_dir(config: dict[str, Any], fold: int) -> Path:
    """foldのローカル成果物ディレクトリを作成して返す。"""
    if fold not in range(5):
        raise ValueError(f"foldが不正です: {fold}")
    fold_dir = resolve_experiment_root(config) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def save_effective_config(config: dict[str, Any]) -> Path:
    """CLI反映後の実効設定を実験ルートへ保存する。"""
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "config.yaml"
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なる実効configがすでに存在します: {output_path}")
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def save_fold_effective_config(config: dict[str, Any], fold_dir: Path) -> Path:
    """fold固有のCLI反映設定を成果物ディレクトリへ保存する。"""
    output_path = fold_dir / "effective_config.yaml"
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なるfold実効configがすでに存在します: {output_path}")
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def _get_wandb() -> Any | None:
    """必要な場合だけW&Bを読み込む。"""
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def initialize_wandb(config: dict[str, Any], fold: int) -> Any | None:
    """有効時だけfold単位のW&B実行を開始する。"""
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
        or f"fracture-{experiment['phase']}-{experiment['name']}"
    )
    run_name = wandb_config.get("run_name") or f"fold{fold}"
    try:
        wandb_module.init(project=project, name=run_name, config=config, reinit=True)
    except Exception as error:  # W&B通信失敗時もローカル学習は継続する。
        warnings.warn(f"W&B初期化に失敗しました: {error}", stacklevel=2)
        return None
    return wandb_module


def log_wandb_epoch(
    wandb_module: Any,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    backbone_lr: float,
    head_lr: float,
    elapsed_seconds: float,
) -> None:
    """1 epoch分のBaseline 1指標をW&Bへ記録する。"""
    wandb_module.log(
        {
            "epoch": epoch,
            "train_bce": train_metrics["loss"],
            "train_grad_norm": train_metrics["grad_norm"],
            "train_gradient_clip_fraction": train_metrics["clip_fraction"],
            "val_bce": validation_metrics["loss"],
            "val_auroc": validation_metrics["auroc"],
            "val_average_precision": validation_metrics["average_precision"],
            "val_negative_score_mean": validation_metrics["negative_score_mean"],
            "val_positive_score_mean": validation_metrics["positive_score_mean"],
            "val_score_gap": validation_metrics["score_gap"],
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "epoch_seconds": elapsed_seconds,
        },
        step=epoch,
    )


def update_best_summary(
    wandb_module: Any, epoch: int, validation_metrics: dict[str, float]
) -> None:
    """最良checkpointの指標をW&B summaryへ保存する。"""
    wandb_module.run.summary["best_epoch"] = epoch
    wandb_module.run.summary["best_val_auroc"] = validation_metrics["auroc"]
    wandb_module.run.summary["best_val_average_precision"] = validation_metrics[
        "average_precision"
    ]
    wandb_module.run.summary["best_val_bce"] = validation_metrics["loss"]


def finish_wandb(
    wandb_module: Any | None,
    epoch: int,
    validation_metrics: dict[str, float] | None,
    train_rows: int,
    validation_rows: int,
) -> None:
    """最終summaryを保存してW&B実行を閉じる。"""
    if wandb_module is None:
        return
    wandb_module.run.summary["stopped_epoch"] = epoch
    if validation_metrics is not None:
        wandb_module.run.summary["final_val_auroc"] = validation_metrics["auroc"]
        wandb_module.run.summary["final_val_average_precision"] = validation_metrics[
            "average_precision"
        ]
    wandb_module.run.summary["train_rows"] = train_rows
    wandb_module.run.summary["validation_rows"] = validation_rows
    wandb_module.finish()
