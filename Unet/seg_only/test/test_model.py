"""SegOnlyUNet のモデル単体テスト"""

import torch

from ..src.model import SegOnlyUNet


def test_seg_only_unet_output_shape():
    """出力テンソル形状が期待通りであることを確認"""
    model = SegOnlyUNet()
    x = torch.randn(2, 2, 64, 64)
    outputs = model(x)
    assert 'seg_logits' in outputs
    assert 'line_heatmaps' not in outputs
    assert outputs['seg_logits'].shape == (2, 5, 64, 64)
    print(f'test_seg_only_unet_output_shape: seg={outputs["seg_logits"].shape}')


def test_seg_only_unet_param_count():
    """パラメータ数が multitask より少ないことを確認（line_decoder なし）"""
    model = SegOnlyUNet()
    total_params = sum(p.numel() for p in model.parameters())
    # multitask は ~1.58M なので seg_only は ~1.0M 前後（line_decoder 分少ない）
    assert total_params < 1_580_000
    print(f'test_seg_only_unet_param_count: params={total_params}')


def test_seg_only_unet_backward_no_nan():
    """逆伝播時に勾配が NaN を含まないことを確認"""
    model = SegOnlyUNet()
    x = torch.randn(2, 2, 64, 64)
    outputs = model(x)
    loss = outputs['seg_logits'].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    print(f'test_seg_only_unet_backward_no_nan: {len(grads)} tensors finite')


def test_seg_only_unet_batch_size_1():
    """バッチサイズ1でも forward が正常に動作することを確認"""
    model = SegOnlyUNet()
    x = torch.randn(1, 2, 64, 64)
    outputs = model(x)
    assert outputs['seg_logits'].shape == (1, 5, 64, 64)
    print('test_seg_only_unet_batch_size_1: ok')


def test_seg_only_unet_deterministic():
    """evalモードでは同一入力に対して同一出力になることを確認"""
    model = SegOnlyUNet()
    model.eval()
    x = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1['seg_logits'], out2['seg_logits'], atol=1e-7, rtol=0.0)
    print('test_seg_only_unet_deterministic: ok')


def test_seg_only_unet_vertebra_conditioning_output_shape():
    """椎体条件入力ありでも出力形状が変わらないことを確認"""
    model = SegOnlyUNet(num_vertebra=7)
    x = torch.randn(2, 2, 64, 64)
    v_idx = torch.tensor([0, 3], dtype=torch.long)
    outputs = model(x, v_idx)
    assert outputs['seg_logits'].shape == (2, 5, 64, 64)


def test_seg_only_unet_vertebra_conditioning_disabled():
    """椎体条件無効時は従来どおり forward できることを確認"""
    model = SegOnlyUNet(num_vertebra=0)
    x = torch.randn(2, 2, 64, 64)
    outputs = model(x)
    assert outputs['seg_logits'].shape == (2, 5, 64, 64)


def test_seg_only_unet_vertebra_conditioning_identity_init():
    """恒等初期化直後は条件有無で出力が一致することを確認"""
    model = SegOnlyUNet(num_vertebra=7)
    model.eval()
    x = torch.randn(2, 2, 64, 64)
    v_idx = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        out_cond = model(x, v_idx)
        out_uncond = model(x)
    assert torch.allclose(out_cond['seg_logits'], out_uncond['seg_logits'], atol=1e-5)
