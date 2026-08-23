"""mtl_type2専用の実験成果物パス。

`core.experiment`はpackageを`{"baseline0","mtl","proposed"}`に限定して
おり、正式パイプラインが凍結している`core/`は変更しない方針のため、
同じ役割を持つ最小限の実装をこのprojectだけで独立に持つ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from fracture_detection.common.constants import REPO_ROOT


def resolve_experiment_root(config: dict[str, Any]) -> Path:
    """phase/nameから成果物rootを返す。"""
    phase = str(config["experiment"]["phase"])
    name = str(config["experiment"]["name"])
    return REPO_ROOT / "fracture_detection" / "mtl_type2" / "outputs" / phase / name


def resolve_fold_dir(config: dict[str, Any], outer_fold: int) -> Path:
    """outer fold固有ディレクトリを作成する。"""
    if outer_fold not in range(5):
        raise ValueError("outer foldは0から4が必要です")
    path = resolve_experiment_root(config) / f"outer{outer_fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_effective_config(config: dict[str, Any], path: Path) -> Path:
    """異なる内容での上書きを拒否してYAMLを保存する。"""
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なる実効configが既に存在します: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path
