"""Baseline 0のnested outer予測を検証・集計するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fracture_detection.baseline0.config.schema import load_config
from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.baseline0.evaluation.metrics import (
    binary_decision_metrics,
    binary_metrics,
    evaluate_vertebra_prediction_frame,
)
from fracture_detection.baseline0.training.experiment import resolve_experiment_root


def validate_outer_prediction_frame(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    """1 outer foldの予測が凍結manifestと一致するか検証する。"""
    required = {
        "study_id",
        "level",
        "fold",
        "vertebra_target",
        "vertebra_score",
        "decision_threshold",
        "vertebra_prediction",
    }
    missing = required - set(predictions.columns)
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
        raise ValueError(f"outer={outer_fold}のscoreが有限な[0,1]ではありません")
    if (
        not np.isfinite(predictions["decision_threshold"]).all()
        or not predictions["decision_threshold"].between(0.0, 1.0).all()
    ):
        raise ValueError(f"outer={outer_fold}の閾値が有限な[0,1]ではありません")
    if predictions["decision_threshold"].nunique(dropna=False) != 1:
        raise ValueError(f"outer={outer_fold}内で閾値が一定ではありません")
    if not predictions["vertebra_prediction"].isin([0, 1]).all():
        raise ValueError(f"outer={outer_fold}の二値予測が0/1ではありません")
    threshold = float(predictions["decision_threshold"].iloc[0])
    expected_predictions = predictions["vertebra_score"].ge(threshold).astype(int)
    if not predictions["vertebra_prediction"].eq(expected_predictions).all():
        raise ValueError(f"outer={outer_fold}の閾値適用結果が一致しません")

    expected_keys = expected[["study_id", "level", "vertebra_target"]].copy()
    merged = expected_keys.merge(
        predictions[list(required)],
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


def collect_oof_predictions(
    config: dict[str, Any],
    prediction_filename: str = "outer_predictions.csv",
    checkpoint_filename: str = "best_model.pt",
    expected_checkpoint_role: str = "best_val_auroc",
) -> pd.DataFrame:
    """5 outer foldの予測とcheckpoint契約を検証してpoolする。"""
    manifest = load_manifest()
    output_root = resolve_experiment_root(config)
    frames: list[pd.DataFrame] = []
    for outer_fold in range(5):
        fold_dir = output_root / f"outer{outer_fold}"
        prediction_path = fold_dir / prediction_filename
        checkpoint_path = fold_dir / checkpoint_filename
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"outer={outer_fold}の予測またはbest checkpointがありません"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_role") != expected_checkpoint_role:
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
    """検証済みOOFと患者cluster bootstrap指標を保存する。"""
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = {
        "auroc_checkpoint": {
            "prediction_filename": "outer_predictions.csv",
            "checkpoint_filename": "best_model.pt",
            "checkpoint_role": "best_val_auroc",
            "oof_prediction_filename": "oof_predictions.csv",
            "oof_metrics_filename": "oof_metrics.json",
        },
        "prauc_checkpoint": {
            "prediction_filename": "outer_predictions_prauc_checkpoint.csv",
            "checkpoint_filename": "best_val_prauc_model.pt",
            "checkpoint_role": "best_val_prauc",
            "oof_prediction_filename": "oof_predictions_prauc_checkpoint.csv",
            "oof_metrics_filename": "oof_metrics_prauc_checkpoint.json",
        },
    }
    summary: dict[str, Any] = {
        "primary_checkpoint": "best_val_auroc",
        "threshold_selection": "maximum_val_f1",
        "threshold_tie_break": "highest_threshold",
    }
    for variant_name, filenames in variants.items():
        pooled = collect_oof_predictions(
            config,
            prediction_filename=filenames["prediction_filename"],
            checkpoint_filename=filenames["checkpoint_filename"],
            expected_checkpoint_role=filenames["checkpoint_role"],
        )
        pooled.to_csv(output_root / filenames["oof_prediction_filename"], index=False)
        metrics = evaluate_vertebra_prediction_frame(
            pooled,
            n_bootstrap=n_bootstrap,
        )
        (output_root / filenames["oof_metrics_filename"]).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[variant_name] = {
            "pooled": metrics,
            "per_outer": _summarize_per_outer(pooled),
        }
    (output_root / "all_outer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _summarize_per_outer(pooled: pd.DataFrame) -> list[dict[str, Any]]:
    """各outerの閾値なし・固定閾値指標をまとめる。"""
    summaries: list[dict[str, Any]] = []
    for outer_fold, frame in pooled.groupby("fold", sort=True):
        targets = frame["vertebra_target"].to_numpy()
        threshold_free = binary_metrics(targets, frame["vertebra_score"].to_numpy())
        decisions = binary_decision_metrics(
            targets,
            frame["vertebra_prediction"].to_numpy(),
        )
        summaries.append(
            {
                "outer_fold": int(outer_fold),
                "decision_threshold": float(frame["decision_threshold"].iloc[0]),
                **threshold_free,
                **decisions,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Baseline 0のnested pooled OOF評価")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/baseline0/config/baseline0.yaml"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    summary = evaluate_oof(load_config(args.config), args.n_bootstrap)
    auroc_checkpoint = summary["auroc_checkpoint"]["pooled"]
    prauc_checkpoint = summary["prauc_checkpoint"]["pooled"]
    print(
        f"AUROC-best OOF: AUROC={auroc_checkpoint['auroc']:.6f} "
        f"PR-AUC={auroc_checkpoint['average_precision']:.6f} "
        f"precision={auroc_checkpoint['precision']:.6f} "
        f"recall={auroc_checkpoint['recall']:.6f} "
        f"F1={auroc_checkpoint['f1']:.6f} n={auroc_checkpoint['n']}\n"
        f"PR-AUC-best OOF: AUROC={prauc_checkpoint['auroc']:.6f} "
        f"PR-AUC={prauc_checkpoint['average_precision']:.6f} "
        f"precision={prauc_checkpoint['precision']:.6f} "
        f"recall={prauc_checkpoint['recall']:.6f} "
        f"F1={prauc_checkpoint['f1']:.6f} n={prauc_checkpoint['n']}"
    )


if __name__ == "__main__":
    main()
