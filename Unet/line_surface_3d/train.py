#!/usr/bin/env python
"""line_surface_3dをfold単位で学習するCLI。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
UNET_DIR = PROJECT_DIR.parent
if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from line_surface_3d.src.data_utils import load_config, set_seed  # noqa: E402
from line_surface_3d.src.experiment import (  # noqa: E402
    experiment_dir,
    save_effective_config,
)
from line_surface_3d.src.trainer import train_one_fold  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="3D line surface学習")
    parser.add_argument(
        "--config",
        default="Unet/line_surface_3d/config/plane.yaml",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=4)
    return parser.parse_args()


def _mean_metrics(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """fold間で有限な数値指標を平均する。"""
    keys = sorted({key for result in results.values() for key in result})
    averages: dict[str, float] = {}
    for key in keys:
        values = []
        for result in results.values():
            value = result.get(key)
            if not isinstance(value, int | float):
                continue
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                values.append(numeric_value)
        if values:
            averages[key] = sum(values) / len(values)
    return averages


def main() -> None:
    """指定範囲のfoldを逐次実行する。"""
    args = parse_args()
    base_config = load_config(args.config)
    if args.gpu_id is not None:
        base_config["training"]["gpu_id"] = args.gpu_id
    save_effective_config(base_config)
    results: dict[str, dict[str, Any]] = {}
    for fold in range(args.start_fold, args.end_fold + 1):
        config: dict[str, Any] = copy.deepcopy(base_config)
        config["data"]["test_fold"] = fold
        set_seed(int(config["data"].get("random_seed", 42)))
        results[f"fold{fold}"] = train_one_fold(config)
    summary = {
        "per_fold": results,
        "average": _mean_metrics(results),
    }
    output_path = experiment_dir(base_config) / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()
