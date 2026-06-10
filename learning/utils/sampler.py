"""
不均衡対策: WeightedRandomSampler で bag を ~50:50 にサンプリングする。

loss 側に追加 class weight は重ねない（二重補正禁止）。
"""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    bag ラベルリストから WeightedRandomSampler を生成する。

    陽性・陰性を ~50:50 でサンプリングする。
    epoch あたりのサンプル数 = len(labels)（replacement=True）。

    Args:
        labels: bag ラベルのリスト（0 or 1）

    Returns:
        WeightedRandomSampler
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        # 片方しかない場合は均一サンプリング
        weights = [1.0] * len(labels)
    else:
        weight_pos = 1.0 / n_pos
        weight_neg = 1.0 / n_neg
        weights = [weight_pos if y == 1 else weight_neg for y in labels]

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True,
    )
