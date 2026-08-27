"""region_branchの出力パス解決と軽量W&Bヘルパ。"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from fracture_detection.region_branch.data_pipeline.constants import CALIBRATION_DIR

REGION_BRANCH_DIR = Path(__file__).resolve().parents[1]


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
    return REGION_BRANCH_DIR / "outputs" / phase / name


def resolve_calibration_path(config: dict[str, Any], outer_fold: int) -> Path:
    """校正結果JSONのfold別パスを解決する。

    `experiment.phase`/`name`とは独立し、configで指定したcalibration version配下へ置く。
    校正は統合4領域model・単一領域4modelの計5 configすべてで共有する。
    """
    if outer_fold not in range(5):
        raise ValueError(f"outer foldが不正です: {outer_fold}")
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("config.calibrationはmappingである必要があります")
    version = calibration.get("version")
    if not isinstance(version, str) or re.fullmatch(r"v[1-9][0-9]*", version) is None:
        raise ValueError("calibration.versionはv1, v2, ...の形式が必要です")
    return CALIBRATION_DIR / version / f"outer{outer_fold}" / "calibration.json"


def resolve_fold_dir(config: dict[str, Any], fold: int) -> Path:
    """foldのローカル成果物ディレクトリを作成して返す。"""
    if fold not in range(5):
        raise ValueError(f"foldが不正です: {fold}")
    fold_dir = resolve_experiment_root(config) / f"outer{fold}"
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
        or f"region-branch-{experiment['phase']}-{experiment['name']}"
    )
    run_name = wandb_config.get("run_name") or f"fold{fold}"
    try:
        wandb_module.init(project=project, name=run_name, config=config, reinit=True)
    except Exception as error:  # W&B通信失敗時もローカル学習は継続する。
        warnings.warn(f"W&B初期化に失敗しました: {error}", stacklevel=2)
        return None
    return wandb_module


def log_wandb_epoch(wandb_module: Any, epoch: int, row: dict[str, Any]) -> None:
    """1 epoch分の履歴行をそのままW&Bへ記録する。"""
    wandb_module.log(row, step=epoch)


def finish_wandb(wandb_module: Any | None, stopped_epoch: int) -> None:
    """W&B実行を閉じる。"""
    if wandb_module is None:
        return
    wandb_module.run.summary["stopped_epoch"] = stopped_epoch
    wandb_module.finish()
