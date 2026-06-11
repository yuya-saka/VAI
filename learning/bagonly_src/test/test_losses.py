"""
bag_mean_loss / batch_bag_mean_loss のユニットテスト。
"""

from __future__ import annotations

import torch
import pytest

from learning.bagonly_src.losses import bag_mean_loss, batch_bag_mean_loss


def test_bag_mean_loss_positive_low_loss() -> None:
    # 全スライスが高スコア（陽性）→ 陽性ラベルで低損失
    logits = torch.tensor([3.0, 3.0, 3.0, 3.0])
    loss, bag_logit = bag_mean_loss(logits, label=1.0)
    assert loss.item() < 0.1
    assert bag_logit.shape == (1,)


def test_bag_mean_loss_negative_low_loss() -> None:
    # 全スライスが低スコア（陰性）→ 陰性ラベルで低損失
    logits = torch.tensor([-3.0, -3.0, -3.0, -3.0])
    loss, bag_logit = bag_mean_loss(logits, label=0.0)
    assert loss.item() < 0.1


def test_bag_mean_loss_mismatch_high_loss() -> None:
    # 全スライスが高スコアなのに陰性ラベル → 高損失
    logits = torch.tensor([3.0, 3.0, 3.0])
    loss_wrong, _ = bag_mean_loss(logits, label=0.0)
    loss_correct, _ = bag_mean_loss(logits, label=1.0)
    assert loss_wrong.item() > loss_correct.item()


def test_bag_mean_loss_bag_logit_is_mean() -> None:
    # bag_logit が全スライスの平均になっているか
    logits = torch.tensor([1.0, 2.0, 3.0])
    _, bag_logit = bag_mean_loss(logits, label=1.0)
    assert abs(bag_logit.item() - 2.0) < 1e-5


def test_batch_bag_mean_loss_shape() -> None:
    # バッチ処理で loss が scalar tensor になっているか
    logits_list = [torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0])]
    labels = torch.tensor([1.0, 0.0])
    total_loss, breakdown = batch_bag_mean_loss(logits_list, labels)
    assert total_loss.shape == ()
    assert "bag_mean" in breakdown


def test_batch_bag_mean_loss_is_mean_of_bag_losses() -> None:
    # バッチ損失が各 bag 損失の平均になっているか
    logits_list = [torch.tensor([2.0, 2.0]), torch.tensor([-2.0, -2.0])]
    labels = torch.tensor([1.0, 0.0])
    total, _ = batch_bag_mean_loss(logits_list, labels)

    loss0, _ = bag_mean_loss(logits_list[0], 1.0)
    loss1, _ = bag_mean_loss(logits_list[1], 0.0)
    expected = (loss0 + loss1) / 2
    assert abs(total.item() - expected.item()) < 1e-5


def test_bag_mean_loss_single_slice() -> None:
    # スライス数 1 でも動作するか
    logits = torch.tensor([0.5])
    loss, bag_logit = bag_mean_loss(logits, label=1.0)
    assert loss.shape == ()
    assert bag_logit.shape == (1,)
