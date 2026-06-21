"""
椎体骨折分類の評価指標。

learning/utils/metrics.py から移植・拡張。
compute_level_metrics に AUROC / AUPRC を追加。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def find_optimal_threshold(y_true: NDArray, y_prob: NDArray) -> float:
    """
    F1 を最大化するしきい値を求める。

    陽性例が 0 または全て陽性の場合は 0.5 を返す。
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precision[:-1] + recall[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def _prf_at_threshold(y_true: NDArray, y_prob: NDArray, threshold: float) -> dict:
    """指定しきい値で Precision / Recall / F1 を計算する。"""
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def _safe_auroc(y_true: NDArray, y_prob: NDArray) -> float:
    """陽性例が 0 または全陽性の場合は nan を返す AUROC。"""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: NDArray, y_prob: NDArray) -> float:
    """陽性例が 0 または全陽性の場合は nan を返す AUPRC。"""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _bootstrap_ci(
    y_true: NDArray,
    y_prob: NDArray,
    groups: NDArray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    患者クラスタ bootstrap で AUROC / AUPRC の 95% CI を計算する。

    患者単位でリサンプリングし、bag を追随させる。
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(groups)
    aurocs, auprcs = [], []

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in sampled])
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))

    if not aurocs:
        return {
            "auroc_lo": float("nan"),
            "auroc_hi": float("nan"),
            "auprc_lo": float("nan"),
            "auprc_hi": float("nan"),
        }

    return {
        "auroc_lo": float(np.percentile(aurocs, 2.5)),
        "auroc_hi": float(np.percentile(aurocs, 97.5)),
        "auprc_lo": float(np.percentile(auprcs, 2.5)),
        "auprc_hi": float(np.percentile(auprcs, 97.5)),
    }


def compute_oof_metrics(
    y_true: NDArray,
    y_prob: NDArray,
    groups: NDArray | None = None,
) -> dict:
    """
    OOF pooled の主評価指標を計算する。

    Args:
        y_true: shape [N] の 0/1 ラベル
        y_prob: shape [N] の予測確率 [0, 1]
        groups: shape [N] の患者 ID 文字列（bootstrap CI 用、省略可）

    Returns:
        {auroc, auprc, prevalence, at_05, at_opt, bootstrap_ci (optional)}
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    prevalence = float(y_true.mean())
    auroc = _safe_auroc(y_true, y_prob)
    auprc = _safe_auprc(y_true, y_prob)
    at_05 = _prf_at_threshold(y_true, y_prob, threshold=0.5)
    opt_thresh = find_optimal_threshold(y_true, y_prob)
    at_opt = {"threshold": opt_thresh, **_prf_at_threshold(y_true, y_prob, opt_thresh)}

    result: dict = {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": prevalence,
        "at_05": at_05,
        "at_opt": at_opt,
    }

    if groups is not None:
        result["bootstrap_ci"] = _bootstrap_ci(
            y_true, y_prob, np.asarray(groups)
        )

    return result


def compute_level_metrics(
    y_true: NDArray,
    y_prob: NDArray,
    levels: NDArray,
) -> dict[str, dict]:
    """
    椎体レベル別の評価指標を計算する。

    learning/utils/metrics.py から移植し、AUROC / AUPRC を追加。

    Args:
        y_true: shape [N] の 0/1 ラベル
        y_prob: shape [N] の予測確率
        levels: shape [N] の椎体レベル文字列 ('C1'〜'C7')

    Returns:
        レベル名をキー、{n_pos, n_total, auroc, auprc, precision, recall, f1} を値とする dict
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    levels = np.asarray(levels)

    result: dict[str, dict] = {}

    for lv in VERTEBRAE:
        mask = levels == lv
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        prf = _prf_at_threshold(yt, yp, threshold=0.5)
        result[lv] = {
            "n_pos": int(yt.sum()),
            "n_total": int(mask.sum()),
            "auroc": _safe_auroc(yt, yp),
            "auprc": _safe_auprc(yt, yp),
            **prf,
        }

    return result
