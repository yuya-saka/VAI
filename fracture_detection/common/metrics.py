"""患者単位の再標本化に対応した共通評価指標。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from fracture_detection.common.constants import N_REGIONS, REGION_COLUMNS

Metric = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def _as_arrays(
    targets: NDArray[Any], scores: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    target_array = np.asarray(targets, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if target_array.shape != score_array.shape:
        raise ValueError("targetとscoreの形状が一致していません")
    if not np.isfinite(target_array).all() or not np.isfinite(score_array).all():
        raise ValueError("targetまたはscoreに非有限値があります")
    return target_array, score_array


def safe_auroc(targets: NDArray[Any], scores: NDArray[Any]) -> float:
    """2クラスが揃わない場合はNaNを返すAUROC。"""
    target_array, score_array = _as_arrays(targets, scores)
    if np.unique(target_array).size < 2:
        return float("nan")
    return float(roc_auc_score(target_array, score_array))


def safe_average_precision(targets: NDArray[Any], scores: NDArray[Any]) -> float:
    """陽性がない場合はNaNを返すAverage Precision。"""
    target_array, score_array = _as_arrays(targets, scores)
    if target_array.sum() == 0:
        return float("nan")
    return float(average_precision_score(target_array, score_array))


def cluster_bootstrap_interval(
    targets: NDArray[Any],
    scores: NDArray[Any],
    groups: NDArray[Any],
    metric: Metric,
    n_bootstrap: int = 1000,
    seed: int = 20260807,
) -> tuple[float, float]:
    """studyを単位に復元抽出し、指標の95%区間を返す。"""
    target_array, score_array = _as_arrays(targets, scores)
    group_array = np.asarray(groups)
    if len(group_array) != len(target_array):
        raise ValueError("group数がtarget数と一致していません")
    if n_bootstrap < 1:
        raise ValueError("bootstrap回数は1以上である必要があります")

    unique_groups = np.unique(group_array)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(group_array == group) for group in sampled_groups]
        )
        value = metric(target_array[indices], score_array[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def binary_metrics(
    targets: NDArray[Any],
    scores: NDArray[Any],
    groups: NDArray[Any] | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """二値分類の件数、陽性率、AUROC、APを返す。"""
    target_array, score_array = _as_arrays(targets, scores)
    result: dict[str, Any] = {
        "n": int(len(target_array)),
        "positives": int(target_array.sum()),
        "prevalence": float(target_array.mean()),
        "auroc": safe_auroc(target_array, score_array),
        "average_precision": safe_average_precision(target_array, score_array),
    }
    if groups is None:
        return result
    result["auroc_ci"] = cluster_bootstrap_interval(
        target_array, score_array, groups, safe_auroc, n_bootstrap=n_bootstrap
    )
    result["average_precision_ci"] = cluster_bootstrap_interval(
        target_array,
        score_array,
        groups,
        safe_average_precision,
        n_bootstrap=n_bootstrap,
    )
    return result


def region_metrics(
    targets: NDArray[Any],
    scores: NDArray[Any],
    groups: NDArray[Any] | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """アノテーション済み母集団の領域別APを計算する。"""
    target_array, score_array = _as_arrays(targets, scores)
    if target_array.ndim != 2 or target_array.shape[1] != N_REGIONS:
        raise ValueError(f"領域配列は(N, {N_REGIONS})形状である必要があります")

    result: dict[str, Any] = {}
    average_precisions: list[float] = []
    for index, column in enumerate(REGION_COLUMNS):
        metrics = binary_metrics(
            target_array[:, index],
            score_array[:, index],
            groups=groups,
            n_bootstrap=n_bootstrap,
        )
        result[column] = metrics
        average_precisions.append(float(metrics["average_precision"]))
    result["macro_average_precision"] = float(np.nanmean(average_precisions))
    return result


def side_balanced_accuracy(targets: NDArray[Any], scores: NDArray[Any]) -> float:
    """R2/R3の片側だけが陽性の標本で左右balanced accuracyを計算する。"""
    target_array, score_array = _as_arrays(targets, scores)
    if target_array.ndim != 2 or target_array.shape[1] != N_REGIONS:
        raise ValueError(f"領域配列は(N, {N_REGIONS})形状である必要があります")
    discordant = target_array[:, 1].astype(bool) ^ target_array[:, 2].astype(bool)
    if not discordant.any():
        return float("nan")
    side_targets = target_array[discordant, 2].astype(int)
    if np.unique(side_targets).size < 2:
        return float("nan")
    side_predictions = (score_array[discordant, 2] > score_array[discordant, 1]).astype(
        int
    )
    return float(balanced_accuracy_score(side_targets, side_predictions))


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """標準列を持つOOF予測表から椎体・領域・左右指標をまとめて返す。"""
    required = {
        "study_id",
        "level",
        "vertebra_target",
        "vertebra_score",
        "has_region_target",
    }
    for column in REGION_COLUMNS:
        required.update({f"{column}_target", f"{column}_score"})
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"予測表に必要な列がありません: {sorted(missing)}")
    vertebra = evaluate_vertebra_prediction_frame(
        predictions,
        n_bootstrap=n_bootstrap,
    )
    annotated = predictions[predictions["has_region_target"].astype(bool)]
    region_targets = annotated[
        [f"{column}_target" for column in REGION_COLUMNS]
    ].to_numpy()
    region_scores = annotated[
        [f"{column}_score" for column in REGION_COLUMNS]
    ].to_numpy()
    regions = region_metrics(
        region_targets,
        region_scores,
        groups=annotated["study_id"].to_numpy(),
        n_bootstrap=n_bootstrap,
    )
    side_accuracy = side_balanced_accuracy(region_targets, region_scores)
    return {
        "vertebra": vertebra,
        "regions": regions,
        "side_balanced_accuracy": side_accuracy,
    }


def evaluate_vertebra_prediction_frame(
    predictions: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """椎体スコアだけを持つOOF予測表から標準二値指標を計算する。"""
    required = {"study_id", "level", "vertebra_target", "vertebra_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"予測表に必要な列がありません: {sorted(missing)}")
    if predictions.duplicated(["study_id", "level"]).any():
        raise ValueError("予測表に重複したstudy_id・levelがあります")
    return binary_metrics(
        predictions["vertebra_target"].to_numpy(),
        predictions["vertebra_score"].to_numpy(),
        groups=predictions["study_id"].to_numpy(),
        n_bootstrap=n_bootstrap,
    )
