"""cross-fitted level-only領域AP床と設計段階MDEを生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from fracture_detection.common.constants import (
    INPUT_MANIFEST_CSV,
    REGION_COLUMNS,
)
from fracture_detection.common.metrics import safe_average_precision
from fracture_detection.common.power import (
    cluster_bootstrap_standard_error,
    paired_normal_mde,
)
from fracture_detection.common.splits import resolve_nested_folds

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "level_floor_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "level_floor_metrics.json"
POWER_JSON = OUTPUT_DIR / "region_floor_power.json"
EXPECTED_ANNOTATED_ROWS = 268
MDE_CORRELATIONS = (0.5, 0.7, 0.8, 0.9)


def build_cross_fitted_level_floor(manifest: pd.DataFrame) -> pd.DataFrame:
    """3 training foldsのJeffreys平滑化率をouter annotated bagへ割り当てる。"""
    required = {
        "study_id",
        "level",
        "fold",
        "vertebra_target",
        "has_region_target",
        *REGION_COLUMNS,
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    annotated = manifest[manifest["has_region_target"].astype(bool)].copy()
    if len(annotated) != EXPECTED_ANNOTATED_ROWS:
        raise ValueError(f"領域アノテーションは268 bagが必要です: {len(annotated)}")
    if not annotated["vertebra_target"].eq(1).all():
        raise ValueError("領域アノテーション済みbagは全て骨折陽性である必要があります")

    frames: list[pd.DataFrame] = []
    for outer_fold in range(5):
        assignment = resolve_nested_folds(outer_fold)
        training = annotated[annotated["fold"].isin(assignment.train_folds)]
        outer = annotated[annotated["fold"].eq(outer_fold)].copy()
        counts = training.groupby("level", observed=True)[list(REGION_COLUMNS)].agg(
            ["sum", "count"]
        )
        for region in REGION_COLUMNS:
            positive = counts[(region, "sum")]
            total = counts[(region, "count")]
            rates = (positive + 0.5) / (total + 1.0)
            outer[f"{region}_floor_score"] = outer["level"].map(rates)
        score_columns = [f"{region}_floor_score" for region in REGION_COLUMNS]
        if outer[score_columns].isna().any().any():
            raise ValueError(f"outer fold={outer_fold}に未定義のlevel床があります")
        outer["inner_fold"] = assignment.inner_fold
        outer["train_folds"] = ",".join(map(str, assignment.train_folds))
        frames.append(outer)

    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != EXPECTED_ANNOTATED_ROWS:
        raise ValueError("cross-fitted床の行数が268と一致しません")
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError("cross-fitted床に重複bagがあります")
    return predictions.sort_values(["study_id", "level"]).reset_index(drop=True)


def evaluate_level_floor(predictions: pd.DataFrame) -> dict[str, Any]:
    """領域別APと床周辺のpatient-cluster MDEを返す。"""
    groups = predictions["study_id"].astype(str).to_numpy()
    metrics: dict[str, Any] = {
        "population": "268 annotated fracture-positive vertebrae",
        "fit": "outerごとに3 training folds、Jeffreys (x+0.5)/(n+1)",
        "tie_handling": "sklearn.metrics.average_precision_score groups equal thresholds",
        "scikit_learn_version": sklearn.__version__,
        "regions": {},
    }
    power: dict[str, Any] = {
        "method": "patient-cluster bootstrap SE around cross-fitted floor; two-sided normal approximation",
        "power": 0.80,
        "unadjusted_alpha": 0.05,
        "holm_worst_case_alpha": 0.0125,
        "bootstrap_iterations": 10_000,
        "regions": {},
    }
    for region_index, region in enumerate(REGION_COLUMNS):
        targets = predictions[region].to_numpy(dtype=np.float64)
        scores = predictions[f"{region}_floor_score"].to_numpy(dtype=np.float64)
        average_precision = safe_average_precision(targets, scores)
        standard_error = cluster_bootstrap_standard_error(
            targets,
            scores,
            groups,
            safe_average_precision,
            n_bootstrap=10_000,
            seed=20260807 + region_index,
        )
        metrics["regions"][region] = {
            "n": int(len(targets)),
            "positives": int(targets.sum()),
            "average_precision": average_precision,
        }
        power["regions"][region] = {
            "bootstrap_standard_error": standard_error,
            "mde_alpha_0.05": {
                str(correlation): paired_normal_mde(standard_error, correlation)
                for correlation in MDE_CORRELATIONS
            },
            "mde_holm_worst_case_alpha_0.0125": {
                str(correlation): paired_normal_mde(
                    standard_error, correlation, alpha=0.0125
                )
                for correlation in MDE_CORRELATIONS
            },
        }
    return {"metrics": metrics, "power": power}


def write_level_floor_outputs(
    predictions: pd.DataFrame,
    results: dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """床予測、指標、MDEを再現可能な成果物として保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_CSV.name
    metrics_path = output_dir / METRICS_JSON.name
    power_path = output_dir / POWER_JSON.name
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(
        json.dumps(results["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    power_path.write_text(
        json.dumps(results["power"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return predictions_path, metrics_path, power_path


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="cross-fitted level-only床を生成")
    parser.add_argument("--manifest", type=Path, default=INPUT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """凍結manifestから床とMDE成果物を生成する。"""
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"study_id": str, "level": str})
    predictions = build_cross_fitted_level_floor(manifest)
    results = evaluate_level_floor(predictions)
    paths = write_level_floor_outputs(predictions, results, args.output_dir)
    print(f"level-only床を書き込みました: {', '.join(map(str, paths))}")


if __name__ == "__main__":
    main()
