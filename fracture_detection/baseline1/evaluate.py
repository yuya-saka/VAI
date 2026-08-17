"""Baseline 1の5分割OOF予測を検証・集計するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fracture_detection.baseline1.config import load_config
from fracture_detection.baseline1.dataset import load_mode_manifest
from fracture_detection.baseline1.experiment import resolve_experiment_root
from fracture_detection.common.metrics import evaluate_vertebra_prediction_frame


def validate_fold_prediction_frame(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """1 foldのOOF予測が固定済み検証マニフェストと一致するか検証する。"""
    required = {"study_id", "level", "fold", "vertebra_target", "vertebra_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"fold={fold}の予測に必要な列がありません: {sorted(missing)}")
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError(f"fold={fold}の予測に重複IDがあります")
    if not predictions["fold"].eq(fold).all():
        raise ValueError(f"fold={fold}の予測に別foldが混在しています")
    if (
        not np.isfinite(predictions["vertebra_score"]).all()
        or not predictions["vertebra_score"].between(0.0, 1.0).all()
    ):
        raise ValueError(f"fold={fold}のscoreが有限な[0,1]ではありません")

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
        raise ValueError(f"fold={fold}の予測IDがvalidation manifestと一致しません")
    if (
        not merged["vertebra_target_expected"]
        .eq(merged["vertebra_target_prediction"])
        .all()
    ):
        raise ValueError(f"fold={fold}の予測targetがvalidation manifestと一致しません")
    return predictions.sort_values(["study_id", "level"]).reset_index(drop=True)


def collect_oof_predictions(config: dict[str, Any]) -> pd.DataFrame:
    """5 foldの予測・checkpoint契約を検証し、統合したOOF表を返す。"""
    manifest = load_mode_manifest(config["data"]["mode"])
    output_root = resolve_experiment_root(config)
    frames: list[pd.DataFrame] = []
    for fold in range(5):
        fold_dir = output_root / f"fold{fold}"
        prediction_path = fold_dir / "val_predictions.csv"
        checkpoint_path = fold_dir / "best_model.pt"
        if not prediction_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"fold={fold}の予測またはbest checkpointがありません"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_fold = checkpoint.get("config", {}).get("runtime", {}).get("fold")
        if checkpoint_fold != fold:
            raise ValueError(
                f"fold={fold}のcheckpoint設定と出力directoryが一致しません"
            )
        expected = manifest[manifest["fold"].eq(fold)]
        predictions = pd.read_csv(
            prediction_path, dtype={"study_id": str, "level": str}
        )
        frames.append(validate_fold_prediction_frame(predictions, expected, fold))
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.duplicated(["study_id", "level"]).any() or len(pooled) != len(manifest):
        raise ValueError("poolしたOOF予測の行数またはID一意性が不正です")
    return pooled.sort_values(["study_id", "level"]).reset_index(drop=True)


def evaluate_oof(config: dict[str, Any], n_bootstrap: int) -> dict[str, Any]:
    """検証済みOOFを保存し、患者単位bootstrap指標を返す。"""
    pooled = collect_oof_predictions(config)
    output_root = resolve_experiment_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_root / "oof_predictions.csv", index=False)
    metrics = evaluate_vertebra_prediction_frame(pooled, n_bootstrap=n_bootstrap)
    per_fold = (
        pooled.groupby("fold", sort=True)
        .agg(rows=("study_id", "size"), positives=("vertebra_target", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {"pooled": metrics, "per_fold": per_fold}
    (output_root / "oof_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "all_folds_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Baseline 1のpooled OOF評価")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fracture_detection/baseline1/config/matched_b0.yaml"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLIのエントリポイント。"""
    args = parse_args()
    summary = evaluate_oof(load_config(args.config), args.n_bootstrap)
    pooled = summary["pooled"]
    print(
        f"OOF AUROC={pooled['auroc']:.6f} "
        f"AP={pooled['average_precision']:.6f} n={pooled['n']}"
    )


if __name__ == "__main__":
    main()
