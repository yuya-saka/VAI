"""
RSNA頸椎骨折分類の損失関数。

RSNA 2022 1位解法 (cell 17, 19) に準拠した
BCE陽性重み付き損失とmixup。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_bce = nn.BCEWithLogitsLoss(reduction="none")


def criterion(
    logits: Tensor,
    targets: Tensor,
    positive_weight: float = 2.0,
    activated: bool = False,
) -> Tensor:
    """
    陽性重み付きBCE損失を計算する。

    陽性ラベル(>0)のサンプルに positive_weight 倍の重みを付与し、
    重みで正規化した平均を返す。

    Args:
        logits: 予測値（activated=False のときlogit、True のときsigmoid後確率）
        targets: 正解ラベル (0/1)
        positive_weight: 陽性サンプルの損失重み
        activated: True のとき BCELoss を使用（logit でない場合）

    Returns:
        スカラー損失値
    """
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    if activated:
        losses = nn.BCELoss(reduction="none")(logits_flat, targets_flat)
    else:
        losses = _bce(logits_flat, targets_flat)

    pos_mask = targets_flat > 0
    losses = losses.clone()
    losses[pos_mask] *= positive_weight

    norm = torch.ones_like(logits_flat)
    norm[pos_mask] *= positive_weight

    return losses.sum() / norm.sum()


def mixup(
    inputs: Tensor,
    targets: Tensor,
    clip: tuple[float, float] = (0.0, 1.0),
) -> tuple[Tensor, Tensor, Tensor, float]:
    """
    Batch-level mixupを適用する。

    Args:
        inputs: 入力テンソル (bs, ...)
        targets: ラベルテンソル (bs, ...)
        clip: mixup係数λのクリップ範囲

    Returns:
        (mixed_inputs, original_targets, shuffled_targets, lam)
    """
    indices = torch.randperm(inputs.size(0))
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]

    lam = float(np.random.uniform(clip[0], clip[1]))
    mixed_inputs = inputs * lam + shuffled_inputs * (1.0 - lam)

    return mixed_inputs, targets, shuffled_targets, lam
