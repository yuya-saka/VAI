"""seg_only 損失関数の単体テスト"""

import torch

from ..utils.losses import build_class_weights, compute_seg_only_loss


def _dummy_batch(batch_size: int = 4, h: int = 64, w: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """テスト用ダミーバッチを作成"""
    seg_logits = torch.randn(batch_size, 5, h, w)
    gt_mask = torch.randint(0, 5, (batch_size, h, w), dtype=torch.long)
    return seg_logits, gt_mask


def test_compute_seg_only_loss_keys():
    """必要なキーが返されることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    losses = compute_seg_only_loss(seg_logits, gt_mask)
    expected_keys = {'total', 'ce_loss', 'dice_fg_loss', 'raw_dice_fg_loss'}
    assert expected_keys.issubset(losses.keys())
    print(f'test_compute_seg_only_loss_keys: {list(losses.keys())}')


def test_compute_seg_only_loss_finite():
    """損失値が有限であることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    losses = compute_seg_only_loss(seg_logits, gt_mask)
    assert torch.isfinite(losses['total']).item()
    assert torch.isfinite(losses['ce_loss']).item()
    assert torch.isfinite(losses['dice_fg_loss']).item()
    print(f'test_compute_seg_only_loss_finite: total={losses["total"].item():.4f}')


def test_compute_seg_only_loss_backward():
    """逆伝播時に NaN 勾配が出ないことを確認"""
    seg_logits, gt_mask = _dummy_batch()
    seg_logits.requires_grad_(True)
    losses = compute_seg_only_loss(seg_logits, gt_mask)
    losses['total'].backward()
    assert seg_logits.grad is not None
    assert torch.isfinite(seg_logits.grad).all()
    print('test_compute_seg_only_loss_backward: gradients finite')


def test_compute_seg_only_loss_with_class_weights():
    """クラス重みありでも損失が有限であることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    class_weights = build_class_weights(background_weight=0.3)
    losses = compute_seg_only_loss(seg_logits, gt_mask, class_weights=class_weights)
    assert torch.isfinite(losses['total']).item()
    print('test_compute_seg_only_loss_with_class_weights: ok')


def test_compute_seg_only_loss_gamma_scaling():
    """gamma_dice が dice_fg_loss のスケールに影響することを確認"""
    seg_logits, gt_mask = _dummy_batch()
    loss_low = compute_seg_only_loss(seg_logits, gt_mask, gamma_dice=0.1)
    loss_high = compute_seg_only_loss(seg_logits, gt_mask, gamma_dice=1.0)
    assert loss_high['dice_fg_loss'].item() > loss_low['dice_fg_loss'].item()
    print(f'test_compute_seg_only_loss_gamma_scaling: low={loss_low["dice_fg_loss"].item():.4f} high={loss_high["dice_fg_loss"].item():.4f}')


def test_build_class_weights_shape():
    """build_class_weights が (5,) テンソルを返すことを確認"""
    weights = build_class_weights(background_weight=0.3)
    assert weights.shape == (5,)
    assert abs(weights[0].item() - 0.3) < 1e-6
    assert abs(weights[1].item() - 1.0) < 1e-6
    print(f'test_build_class_weights_shape: {weights.tolist()}')
