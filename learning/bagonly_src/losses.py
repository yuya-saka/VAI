"""
バッグ単位 mean pooling BCE 損失。

top-k 選択・center loss なし。
bag 内全 slice の logit を平均して BCE を計算する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def bag_mean_loss(logits: Tensor, label: float) -> tuple[Tensor, Tensor]:
    """
    bag 内全 slice の logit を mean pooling して BCE を計算する。

    Args:
        logits: shape [t] の per-slice logit
        label: bag ラベル（0 or 1）

    Returns:
        (loss, bag_logit)  bag_logit は推論時の確率計算用 shape [1]
    """
    bag_logit = logits.mean().unsqueeze(0)  # [1]
    target = torch.tensor([label], device=logits.device)
    loss = F.binary_cross_entropy_with_logits(bag_logit, target)
    return loss, bag_logit


def batch_bag_mean_loss(
    logits_list: list[Tensor],
    labels: Tensor,
) -> tuple[Tensor, dict]:
    """
    バッチ全体の bag mean loss を計算する。

    Args:
        logits_list: bag ごとの logit list。各要素は shape [t_i]
        labels: shape [B] の bag ラベル (0/1)

    Returns:
        (total_loss, {'bag_mean': float})
    """
    losses: list[Tensor] = []
    for logits, label in zip(logits_list, labels.tolist()):
        loss, _ = bag_mean_loss(logits, label)
        losses.append(loss)
    total = torch.stack(losses).mean()
    return total, {"bag_mean": total.item()}
