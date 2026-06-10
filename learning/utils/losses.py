"""
DMIL（Deep Multiple Instance Learning）損失 + center loss の実装。

参考: Hibi et al. 2023 IJCARS WSAD
重要: BCEWithLogitsLoss に logit を直接渡すこと（sigmoid二重適用厳禁）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def select_topk(logits: Tensor, mode: str = "capped", alpha: float | None = None) -> int:
    """
    bag内のslice数 t から top-k を決定する。

    Args:
        logits: shape [t] のlogit tensor
        mode: 'capped' = clip(round(0.10*t), 3, 8)、'ratio' = ceil(t/alpha)
        alpha: ratio モード用パラメータ（mode='ratio' 時は必須）

    Returns:
        選択する上位slice数 k
    """
    t = logits.shape[0]
    if mode == "capped":
        return int(min(max(round(0.10 * t), 3), 8))
    elif mode == "ratio":
        if alpha is None:
            raise ValueError("alpha は ratio モードで必須です")
        return int(math.ceil(t / alpha))
    else:
        raise ValueError(f"未知の mode: {mode}")


def dmil_loss(logits: Tensor, label: float, k: int) -> Tensor:
    """
    1 bag 分の DMIL 損失を計算する。

    top-k の logit に BCEWithLogitsLoss を適用する。
    sigmoid を事前に適用してはならない（二重適用バグ）。

    Args:
        logits: shape [t] の per-slice logit
        label: bag ラベル（0 or 1）
        k: top-k の k

    Returns:
        スカラー損失
    """
    topk_indices = torch.topk(logits, k=k).indices
    topk_logits = logits[topk_indices]
    target = torch.full_like(topk_logits, fill_value=label)
    return F.binary_cross_entropy_with_logits(topk_logits, target)


def center_loss(logits: Tensor, label: float) -> Tensor:
    """
    1 bag 分の center loss を計算する（陰性bag専用）。

    陰性bag(y=0)の slice スコアの分散を縮小する。
    mean は detach して勾配を止める（自己 anchor への縮小のみ）。

    Args:
        logits: shape [t] の per-slice logit
        label: bag ラベル（0 or 1）

    Returns:
        スカラー損失（陽性bagなら0）
    """
    if label != 0:
        return torch.tensor(0.0, device=logits.device)
    scores = torch.sigmoid(logits)
    mu = scores.detach().mean()
    return ((scores - mu) ** 2).mean()


def dmil_center_loss(
    logits_list: list[Tensor],
    labels: Tensor,
    beta: float = 5.0,
    beta_warmup: float = 1.0,
    topk_mode: str = "capped",
    alpha: float | None = None,
) -> tuple[Tensor, dict]:
    """
    バッチ全体の DMIL + center loss を計算する。

    Args:
        logits_list: bag ごとの logit list。各要素は shape [t_i]
        labels: shape [B] の bag ラベル (0/1)
        beta: center loss の重み係数（warmup前の目標値）
        beta_warmup: warmup スケール（0.0→1.0、実効 beta = beta * beta_warmup）
        topk_mode: 'capped' または 'ratio'
        alpha: ratio モード用パラメータ

    Returns:
        (total_loss, {'dmil': float, 'center': float})
    """
    dmil_losses: list[Tensor] = []
    center_losses: list[Tensor] = []
    n_neg = 0

    for logits, label in zip(logits_list, labels.tolist()):
        k = select_topk(logits, mode=topk_mode, alpha=alpha)
        dmil_losses.append(dmil_loss(logits, label, k))
        closs = center_loss(logits, label)
        center_losses.append(closs)
        if label == 0:
            n_neg += 1

    dmil_mean = torch.stack(dmil_losses).mean()
    center_sum = torch.stack(center_losses).sum()
    effective_beta = beta * beta_warmup
    total = dmil_mean + effective_beta * center_sum / max(1, n_neg)

    return total, {
        "dmil": dmil_mean.item(),
        "center": (center_sum / max(1, n_neg)).item(),
    }
