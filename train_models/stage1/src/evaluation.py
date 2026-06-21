"""
1 epoch 分の validation metrics 計算。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.metrics import compute_level_metrics, compute_oof_metrics


def compute_epoch_metrics(
    labels: NDArray,
    probs: NDArray,
    study_uids: NDArray,
    vertebrae: NDArray,
) -> dict:
    """
    validation epoch の評価指標を計算する。

    Args:
        labels: shape [N] の 0/1 ground-truth ラベル
        probs: shape [N] の予測確率 [0, 1]
        study_uids: shape [N] の study UID 文字列
        vertebrae: shape [N] の椎体レベル文字列 ('C1'〜'C7')

    Returns:
        {
            auroc, auprc, prevalence,
            at_05: {precision, recall, f1},
            at_opt: {threshold, precision, recall, f1},
            per_vertebra: {C1: {...}, ..., C7: {...}},
        }
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    study_uids = np.asarray(study_uids)
    vertebrae = np.asarray(vertebrae)

    overall = compute_oof_metrics(labels, probs, groups=study_uids)
    per_vertebra = compute_level_metrics(labels, probs, vertebrae)

    return {**overall, "per_vertebra": per_vertebra}
