"""統合4領域モデルと単一領域4モデルの領域別OOF比較。

4本ensembleはprimary endpointにしない。統合modelの各領域scoreを、対応する
単一領域modelのscoreと領域ごとに1:1で比較する。
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from fracture_detection.baseline0.evaluation.metrics import (
    safe_auroc,
    safe_average_precision,
)
from fracture_detection.region_branch.cli.evaluate import collect_oof_predictions
from fracture_detection.region_branch.config.schema import load_config
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.region_branch.training.experiment import resolve_experiment_root

DEFAULT_ARM_CONFIGS = {
    1: Path("fracture_detection/region_branch/config/region_branch_r1.yaml"),
    2: Path("fracture_detection/region_branch/config/region_branch_r2.yaml"),
    3: Path("fracture_detection/region_branch/config/region_branch_r3.yaml"),
    4: Path("fracture_detection/region_branch/config/region_branch_r4.yaml"),
}
DEFAULT_INTEGRATED_CONFIG = Path(
    "fracture_detection/region_branch/config/region_branch_all.yaml"
)
Metric = Callable[[np.ndarray, np.ndarray], float]


def paired_cluster_bootstrap_difference(
    targets: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    metric: Metric,
    n_bootstrap: int = 1000,
    seed: int = 20260807,
) -> tuple[float, float]:
    """study単位で復元抽出し、`metric(a)-metric(b)`の95%区間を返す。"""
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        value_a = metric(targets[indices], scores_a[indices])
        value_b = metric(targets[indices], scores_b[indices])
        if np.isfinite(value_a) and np.isfinite(value_b):
            values.append(value_a - value_b)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def compare_region(
    integrated_pooled: pd.DataFrame,
    single_pooled: pd.DataFrame,
    region_index: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """1領域について統合modelと単一modelのAP/AUROC差を比較する。"""
    column = REGION_COLUMNS[region_index]
    target_column = f"{column}_target"
    valid_column = f"{column}_target_valid"
    score_column = f"{column}_score"
    merged = integrated_pooled[
        ["study_id", "level", target_column, valid_column, score_column]
    ].merge(
        single_pooled[["study_id", "level", score_column]],
        on=["study_id", "level"],
        suffixes=("_integrated", "_single"),
        validate="one_to_one",
    )
    valid = merged[valid_column].astype(bool)
    subset = merged.loc[valid]
    if subset.empty:
        raise ValueError(f"{column}に有効セルがありません")
    targets = subset[target_column].to_numpy()
    integrated_scores = subset[f"{score_column}_integrated"].to_numpy()
    single_scores = subset[f"{score_column}_single"].to_numpy()
    groups = subset["study_id"].to_numpy()

    ap_ci = paired_cluster_bootstrap_difference(
        targets,
        integrated_scores,
        single_scores,
        groups,
        safe_average_precision,
        n_bootstrap,
    )
    auroc_ci = paired_cluster_bootstrap_difference(
        targets, integrated_scores, single_scores, groups, safe_auroc, n_bootstrap
    )
    return {
        "region": column,
        "n": int(len(subset)),
        "positives": int(targets.sum()),
        "integrated_ap": safe_average_precision(targets, integrated_scores),
        "single_ap": safe_average_precision(targets, single_scores),
        "ap_difference_ci": ap_ci,
        "integrated_auroc": safe_auroc(targets, integrated_scores),
        "single_auroc": safe_auroc(targets, single_scores),
        "auroc_difference_ci": auroc_ci,
    }


def compare_arms(
    integrated_config_path: Path,
    single_config_paths: dict[int, Path],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """統合modelと単一4modelの領域別AP/AUROC差をすべて比較する。"""
    integrated_config = load_config(integrated_config_path)
    integrated_active = tuple(
        int(v) for v in integrated_config["region"]["active_regions"]
    )
    if set(integrated_active) != set(range(len(REGION_COLUMNS))):
        raise ValueError(f"integrated configは4領域全てが必要です: {integrated_active}")
    integrated_pooled = collect_oof_predictions(integrated_config)

    results: list[dict[str, Any]] = []
    for region_index_1based, config_path in sorted(single_config_paths.items()):
        region_index = region_index_1based - 1
        single_config = load_config(config_path)
        single_active = tuple(int(v) for v in single_config["region"]["active_regions"])
        if single_active != (region_index,):
            raise ValueError(
                f"{config_path}のactive_regionsが{REGION_COLUMNS[region_index]}用では"
                f"ありません: {single_active}"
            )
        single_pooled = collect_oof_predictions(single_config)
        results.append(
            compare_region(integrated_pooled, single_pooled, region_index, n_bootstrap)
        )
    return results


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="統合4領域modelと単一領域4modelの比較")
    parser.add_argument(
        "--integrated-config", type=Path, default=DEFAULT_INTEGRATED_CONFIG
    )
    parser.add_argument("--r1-config", type=Path, default=DEFAULT_ARM_CONFIGS[1])
    parser.add_argument("--r2-config", type=Path, default=DEFAULT_ARM_CONFIGS[2])
    parser.add_argument("--r3-config", type=Path, default=DEFAULT_ARM_CONFIGS[3])
    parser.add_argument("--r4-config", type=Path, default=DEFAULT_ARM_CONFIGS[4])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    single_config_paths = {
        1: args.r1_config,
        2: args.r2_config,
        3: args.r3_config,
        4: args.r4_config,
    }
    results = compare_arms(
        args.integrated_config, single_config_paths, args.n_bootstrap
    )
    integrated_config = load_config(args.integrated_config)
    output_root = resolve_experiment_root(integrated_config)
    output_path = output_root / "compare_arms.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in results:
        print(
            f"{row['region']}: integrated_AP={row['integrated_ap']:.6f} "
            f"single_AP={row['single_ap']:.6f} diff_CI={row['ap_difference_ci']}",
            flush=True,
        )
    print(f"比較結果を保存しました: {output_path}", flush=True)


if __name__ == "__main__":
    main()
