"""seg_only 損失関数の単体テスト"""

import torch

from ..utils.losses import (
    boundary_band_dice_loss,
    build_class_weights,
    compute_seg_only_loss,
    make_internal_boundary_band,
)


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


def test_make_internal_boundary_band_shape():
    """内部境界バンドの形状と dtype が期待どおりであることを確認"""
    gt_mask = torch.randint(0, 5, (4, 64, 64), dtype=torch.long)
    band = make_internal_boundary_band(gt_mask, radius=1)
    assert band.shape == (4, 64, 64)
    assert band.dtype == torch.float32


def test_make_internal_boundary_band_no_fg():
    """前景が存在しない場合に内部境界バンドがゼロになることを確認"""
    gt_mask = torch.zeros(2, 32, 32, dtype=torch.long)  # all bg
    band = make_internal_boundary_band(gt_mask)
    assert band.sum().item() == 0.0


def test_make_internal_boundary_band_uniform_fg():
    """前景が単一クラスの場合は内部境界なし"""
    gt_mask = torch.ones(2, 32, 32, dtype=torch.long)  # all class 1
    band = make_internal_boundary_band(gt_mask)
    assert band.sum().item() == 0.0


def test_make_internal_boundary_band_detects_fg_fg():
    """前景クラス同士の境界のみを内部境界として検出できることを確認"""
    gt_mask = torch.ones(1, 32, 32, dtype=torch.long)
    gt_mask[:, :, 16:] = 2  # left half=1, right half=2
    band = make_internal_boundary_band(gt_mask, radius=0)
    assert band[0, :, 15].sum() > 0 and band[0, :, 16].sum() > 0
    assert band[0, :, 0].sum() == 0  # far from boundary


def test_make_internal_boundary_band_ignores_bg_fg():
    """背景と前景の境界は内部境界として扱わないことを確認"""
    gt_mask = torch.zeros(1, 32, 32, dtype=torch.long)
    gt_mask[:, :, 16:] = 1  # left=bg, right=fg
    band = make_internal_boundary_band(gt_mask)
    assert band.sum().item() == 0.0  # bg-fg is NOT internal boundary


def test_boundary_band_dice_loss_finite():
    """boundary band dice loss が有限値を返すことを確認"""
    seg_logits, gt_mask = _dummy_batch()
    band = make_internal_boundary_band(gt_mask)
    loss = boundary_band_dice_loss(seg_logits, gt_mask, band)
    assert torch.isfinite(loss)


def test_boundary_band_dice_loss_zero_band():
    """境界バンドがゼロのとき boundary band dice loss が 0 になることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    band = torch.zeros(4, 64, 64)
    loss = boundary_band_dice_loss(seg_logits, gt_mask, band)
    assert loss.item() == 0.0


def test_compute_seg_only_loss_boundary_keys():
    """境界損失キーが返り値に含まれ有限であることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    losses = compute_seg_only_loss(seg_logits, gt_mask, alpha_boundary=2.0, lambda_bd=0.2)
    assert 'boundary_dice_loss' in losses
    assert torch.isfinite(losses['boundary_dice_loss'])


def test_compute_seg_only_loss_boundary_backward():
    """境界損失込みの total で逆伝播して勾配が有限であることを確認"""
    seg_logits, gt_mask = _dummy_batch()
    seg_logits.requires_grad_(True)
    losses = compute_seg_only_loss(seg_logits, gt_mask, alpha_boundary=2.0, lambda_bd=0.2)
    losses['total'].backward()
    assert seg_logits.grad is not None
    assert torch.isfinite(seg_logits.grad).all()


def test_compute_seg_only_loss_backward_compat():
    """後方互換性の確認"""
    seg_logits, gt_mask = _dummy_batch()
    losses = compute_seg_only_loss(seg_logits, gt_mask, alpha_boundary=0.0, lambda_bd=0.0)
    assert losses['boundary_dice_loss'].item() == 0.0
