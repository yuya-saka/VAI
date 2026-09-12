"""Validate and pool nested outer predictions into one OOF evaluation.

Region-based metrics (conditional localization, region detection) are
computed on ``region_observed_all``-filtered rows only -- a bag with any
unobserved region is dropped and its count reported, per
``fracture_detection/REGION_MIL_DESIGN.md`` section 8. Whole detection uses
every OOF row, since ``p_whole`` is well-defined regardless of per-region
observability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.weak.config.schema import load_config
from fracture_detection.weak.evaluation.metrics import (
    evaluate_conditional_localization,
    evaluate_region_detection,
    evaluate_whole_detection,
)
from fracture_detection.weak.training.experiment import resolve_experiment_root
from fracture_detection.weak.training.trainer import CHECKPOINT_ROLE

REQUIRED_PREDICTION_COLUMNS = {
    "study_id",
    "level",
    "fold",
    "bag_group",
    "vertebra_target",
    "region_1",
    "region_2",
    "region_3",
    "region_4",
    "q_1",
    "q_2",
    "q_3",
    "q_4",
    "p_whole",
    "region_observed_all",
}


def validate_outer_prediction_frame(
    predictions: pd.DataFrame, expected: pd.DataFrame, outer_fold: int
) -> pd.DataFrame:
    """Validate one outer fold's predictions against the frozen manifest."""
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(
            f"outer={outer_fold}の予測に必要な列がありません: {sorted(missing)}"
        )
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError(f"outer={outer_fold}の予測に重複IDがあります")
    if not predictions["fold"].eq(outer_fold).all():
        raise ValueError(f"outer={outer_fold}の予測に別foldが混在しています")
    for column in ("q_1", "q_2", "q_3", "q_4", "p_whole"):
        if (
            not np.isfinite(predictions[column]).all()
            or not predictions[column].between(0.0, 1.0).all()
        ):
            raise ValueError(f"outer={outer_fold}の{column}が有限な[0,1]ではありません")

    expected_keys = expected[["study_id", "level", "vertebra_target"]].copy()
    merged = expected_keys.merge(
        predictions[["study_id", "level", "vertebra_target"]],
        on=["study_id", "level"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_expected", "_prediction"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"outer={outer_fold}の予測IDがmanifestと一致しません")
    if (
        not merged["vertebra_target_expected"]
        .eq(merged["vertebra_target_prediction"])
        .all()
    ):
        raise ValueError(f"outer={outer_fold}のtargetがmanifestと一致しません")
    return predictions.sort_values(["study_id", "level"]).reset_index(drop=True)


def collect_oof_predictions(config: dict[str, Any]) -> pd.DataFrame:
    """Validate and pool all 5 outer folds' predictions and checkpoint contracts."""
    manifest = load_manifest()
    output_root = resolve_experiment_root(config)
    frames: list[pd.DataFrame] = []
    for outer_fold in range(5):
        fold_dir = output_root / f"outer{outer_fold}"
        prediction_path = fold_dir / "outer_predictions.csv"
        checkpoint_path = fold_dir / "best_region.pt"
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"outer={outer_fold}の予測またはbest checkpointがありません"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_role") != CHECKPOINT_ROLE:
            raise ValueError(
                f"outer={outer_fold}のcheckpoint roleが不正です: "
                f"{checkpoint.get('checkpoint_role')}"
            )
        runtime = checkpoint.get("config", {}).get("runtime", {})
        assignment = resolve_nested_folds(outer_fold)
        expected_runtime = {
            "outer_fold": outer_fold,
            "inner_fold": assignment.inner_fold,
            "train_folds": list(assignment.train_folds),
        }
        if runtime != expected_runtime:
            raise ValueError(
                f"outer={outer_fold}のcheckpoint nested設定が不正です: {runtime}"
            )
        expected = manifest[manifest["fold"].eq(outer_fold)]
        predictions = pd.read_csv(
            prediction_path, dtype={"study_id": str, "level": str}
        )
        frames.append(
            validate_outer_prediction_frame(predictions, expected, outer_fold)
        )
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.duplicated(["study_id", "level"]).any() or len(pooled) != len(manifest):
        raise ValueError("pooled OOFの行数またはID一意性が不正です")
    return pooled.sort_values(["study_id", "level"]).reset_index(drop=True)


def evaluate_oof(config: dict[str, Any], n_bootstrap: int) -> dict[str, Any]:
    """Save the validated pooled OOF predictions and compute OOF metrics."""
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    pooled = collect_oof_predictions(config)
    pooled.to_csv(output_root / "oof_predictions.csv", index=False)

    observed = pooled[pooled["region_observed_all"]]
    summary = {
        "n_dropped_unobserved": int((~pooled["region_observed_all"]).sum()),
        "conditional_localization": evaluate_conditional_localization(
            observed, n_bootstrap=n_bootstrap
        ),
        "region_detection": evaluate_region_detection(
            observed, n_bootstrap=n_bootstrap
        ),
        "whole_detection": evaluate_whole_detection(pooled, n_bootstrap=n_bootstrap),
    }
    (output_root / "oof_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="weak region-MILのnested pooled OOF評価"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("fracture_detection/weak/config/weak.yaml")
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = evaluate_oof(load_config(args.config), args.n_bootstrap)
    localization = summary["conditional_localization"]
    whole = summary["whole_detection"]
    print(
        f"条件付き局在 macro AP={localization['macro_average_precision']:.6f} "
        f"(n={localization['n']}) | "
        f"whole AP={whole['average_precision']:.6f} AUROC={whole['auroc']:.6f} "
        f"(n={whole['n']}) | dropped_unobserved={summary['n_dropped_unobserved']}"
    )


if __name__ == "__main__":
    main()
