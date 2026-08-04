#!/usr/bin/env python
"""2.5D局所整合性モデルの5-fold学習CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
UNET_DIR = PROJECT_DIR.parent
if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from line_2p5d.src.data_utils import load_config  # noqa: E402
from line_2p5d.src.trainer import train_one_fold  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="2.5D局所整合性線検出の学習")
    parser.add_argument(
        "--config",
        default="Unet/line_2p5d/config/geometry.yaml",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--start_fold", type=int, default=None)
    parser.add_argument("--end_fold", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """指定fold範囲を順番に学習する。"""
    args = parse_args()
    config = load_config(args.config)
    if args.gpu_id is not None:
        config["training"]["gpu_id"] = args.gpu_id
    n_folds = int(config["data"].get("n_folds", 5))
    folds_config = config.get("folds", {})
    start_fold = (
        args.start_fold
        if args.start_fold is not None
        else int(folds_config.get("start", 0))
    )
    end_fold = (
        args.end_fold
        if args.end_fold is not None
        else int(folds_config.get("end", n_folds - 1))
    )
    if not 0 <= start_fold <= end_fold < n_folds:
        raise ValueError("fold範囲が不正です")
    results: dict[str, dict] = {}
    for fold in range(start_fold, end_fold + 1):
        fold_config = {
            **config,
            "data": {**config["data"], "test_fold": fold},
        }
        results[f"fold_{fold}"] = train_one_fold(fold_config)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
