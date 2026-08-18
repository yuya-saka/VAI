"""領域APの設計段階MDEを患者クラスタbootstrapから近似する。"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Metric = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def cluster_bootstrap_standard_error(
    targets: NDArray[np.float64],
    scores: NDArray[np.float64],
    groups: NDArray[np.str_],
    metric: Metric,
    n_bootstrap: int,
    seed: int,
) -> float:
    """患者単位bootstrap標準偏差を返す。"""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrapは2以上である必要があります")
    if not (len(targets) == len(scores) == len(groups)):
        raise ValueError("targets、scores、groupsの長さが一致しません")
    unique_groups = np.unique(groups)
    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([group_indices[index] for index in sampled])
        value = metric(targets[indices], scores[indices])
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def paired_normal_mde(
    single_model_standard_error: float,
    correlation: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """等分散2モデルのpaired差に対する両側normal近似MDEを返す。"""
    if (
        not math.isfinite(single_model_standard_error)
        or single_model_standard_error < 0
    ):
        raise ValueError("standard errorは0以上の有限値である必要があります")
    if not 0 <= correlation < 1:
        raise ValueError("correlationは0以上1未満である必要があります")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alphaとpowerは0から1の範囲内である必要があります")
    difference_se = single_model_standard_error * math.sqrt(2.0 * (1.0 - correlation))
    critical = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    return critical * difference_se
