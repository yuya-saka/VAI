"""region_branchの領域別評価指標（validity mask付き）。

region sigmoidはsource-balanced samplingとranking supervisionで学習されており、
population prevalenceを表す較正済み確率ではない。scoreとしてAP/AUROCだけで
評価する（`.claude/docs/research/20260825-region-loss-balancing.md`の方針）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fracture_detection.baseline0.evaluation.metrics import (
    binary_metrics,
    evaluate_vertebra_prediction_frame,
)
from fracture_detection.region_branch.data_pipeline.constants import REGION_COLUMNS


def region_metrics_masked(
    predictions: pd.DataFrame,
    active_regions: tuple[int, ...],
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """`region_r_target_valid`が真のセルだけで領域別AP/AUROCと患者bootstrap CIを返す。"""
    if "study_id" not in predictions.columns:
        raise ValueError("予測表にstudy_id列がありません")

    result: dict[str, Any] = {}
    for region_index in active_regions:
        column = REGION_COLUMNS[region_index]
        target_column = f"{column}_target"
        valid_column = f"{column}_target_valid"
        score_column = f"{column}_score"
        required_columns = {target_column, valid_column, score_column}
        missing_columns = required_columns - set(predictions.columns)
        if missing_columns:
            raise ValueError(
                f"{column}に必要な列がありません: {sorted(missing_columns)}"
            )
        valid = predictions[valid_column].astype(bool)
        if not valid.any():
            raise ValueError(f"{column}に有効セルがありません")
        subset = predictions.loc[valid]
        result[column] = binary_metrics(
            subset[target_column].to_numpy(),
            subset[score_column].to_numpy(),
            groups=subset["study_id"].to_numpy(),
            n_bootstrap=n_bootstrap,
        )
    return result


def evaluate_region_branch_prediction_frame(
    predictions: pd.DataFrame,
    active_regions: tuple[int, ...],
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """標準列を持つOOF予測表からwhole・領域別指標をまとめて返す。"""
    vertebra = evaluate_vertebra_prediction_frame(predictions, n_bootstrap=n_bootstrap)
    regions = region_metrics_masked(
        predictions, active_regions, n_bootstrap=n_bootstrap
    )
    return {"vertebra": vertebra, "regions": regions}
