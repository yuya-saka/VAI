"""region_branchのnested outer予測を検証・集計するCLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.region_branch.config.schema import load_config
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS
from fracture_detection.region_branch.evaluation.metrics import (
    evaluate_region_branch_prediction_frame,
)
from fracture_detection.region_branch.training.experiment import resolve_experiment_root

PREDICTION_COLUMNS = {
    "study_id",
    "level",
    "fold",
    "vertebra_target",
    "vertebra_score",
}


def validate_outer_prediction_frame(
    predictions: pd.DataFrame, expected: pd.DataFrame, outer_fold: int
) -> pd.DataFrame:
    """1 outer foldの予測が凍結manifestと一致するか検証する。"""
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(
            f"outer={outer_fold}の予測に必要な列がありません: {sorted(missing)}"
        )
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError(f"outer={outer_fold}の予測に重複IDがあります")
    if not predictions["fold"].eq(outer_fold).all():
        raise ValueError(f"outer={outer_fold}の予測に別foldが混在しています")
    if (
        not np.isfinite(predictions["vertebra_score"]).all()
        or not predictions["vertebra_score"].between(0.0, 1.0).all()
    ):
        raise ValueError(
            f"outer={outer_fold}のvertebra_scoreが有限な[0,1]ではありません"
        )

    expected_keys = expected[["study_id", "level", "vertebra_target"]].copy()
    merged = expected_keys.merge(
        predictions,
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
        raise ValueError(f"outer={outer_fold}のvertebra_targetがmanifestと一致しません")
    return predictions


def collect_oof_predictions(
    config: dict[str, Any],
    prediction_filename: str = "outer_predictions.csv",
) -> pd.DataFrame:
    """5 outer foldの予測とcheckpoint契約を検証してpoolする。"""
    manifest = load_manifest()
    output_root = resolve_experiment_root(config)
    frames: list[pd.DataFrame] = []
    for outer_fold in range(5):
        fold_dir = output_root / f"outer{outer_fold}"
        prediction_path = fold_dir / prediction_filename
        if not prediction_path.is_file():
            raise FileNotFoundError(f"outer={outer_fold}の予測がありません")
        assignment = resolve_nested_folds(outer_fold)
        expected_runtime = {
            "outer_fold": outer_fold,
            "inner_fold": assignment.inner_fold,
            "train_folds": list(assignment.train_folds),
        }
        for checkpoint_name, expected_role in (
            ("best_region.pt", "best_region"),
            ("best_whole.pt", "best_whole"),
        ):
            checkpoint_path = fold_dir / checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"outer={outer_fold}の{checkpoint_name}がありません"
                )
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint.get("checkpoint_role") != expected_role:
                raise ValueError(
                    f"outer={outer_fold}の{checkpoint_name} roleが不正です: "
                    f"{checkpoint.get('checkpoint_role')}"
                )
            runtime = checkpoint.get("config", {}).get("runtime", {})
            if runtime != expected_runtime:
                raise ValueError(
                    f"outer={outer_fold}の{checkpoint_name} nested設定が不正です: {runtime}"
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


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """torch importを遅延させ、checkpointのconfig/roleだけを読む。"""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpointの形式が不正です: {path}")
    return checkpoint


def evaluate_oof(config: dict[str, Any], n_bootstrap: int) -> dict[str, Any]:
    """検証済みOOFと患者cluster bootstrap指標を保存する。"""
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    active_regions = tuple(int(value) for value in config["region"]["active_regions"])

    pooled = collect_oof_predictions(config)
    pooled.to_csv(output_root / "oof_predictions.csv", index=False)

    metrics = evaluate_region_branch_prediction_frame(
        pooled, active_regions, n_bootstrap=n_bootstrap
    )
    metrics["per_outer"] = _summarize_per_outer(pooled, active_regions)
    (output_root / "oof_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"OOF評価を保存しました: {output_root}", flush=True)
    print(
        f"whole: AUROC={metrics['vertebra']['auroc']:.6f} AP={metrics['vertebra']['average_precision']:.6f}",
        flush=True,
    )
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        region_metrics = metrics["regions"][column]
        print(
            f"{column}: n={region_metrics['n']} AUROC={region_metrics['auroc']:.6f} "
            f"AP={region_metrics['average_precision']:.6f}",
            flush=True,
        )
    return metrics


def _summarize_per_outer(
    pooled: pd.DataFrame, active_regions: tuple[int, ...]
) -> list[dict[str, Any]]:
    """outer foldごとのwhole AUROC/APを軽量に記録する。"""
    from fracture_detection.baseline0.evaluation.metrics import (
        safe_auroc,
        safe_average_precision,
    )

    rows: list[dict[str, Any]] = []
    for outer_fold in sorted(pooled["fold"].unique()):
        subset = pooled[pooled["fold"] == outer_fold]
        rows.append(
            {
                "outer_fold": int(outer_fold),
                "n": int(len(subset)),
                "whole_auroc": safe_auroc(
                    subset["vertebra_target"].to_numpy(),
                    subset["vertebra_score"].to_numpy(),
                ),
                "whole_ap": safe_average_precision(
                    subset["vertebra_target"].to_numpy(),
                    subset["vertebra_score"].to_numpy(),
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="region_branchのOOF評価")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/region_branch/config/region_branch_all.yaml"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    config = load_config(args.config)
    evaluate_oof(config, args.n_bootstrap)


if __name__ == "__main__":
    main()
