"""seg_sdf ResUNet のモデル単体テスト"""

import torch

from ..src.model import ResUNet


def test_resunet_output_shape():
    """出力テンソル形状が期待通りであることを確認"""
    test_name = "test_resunet_output_shape"

    # 準備
    model = ResUNet()
    x = torch.randn(2, 2, 64, 64)

    # 実行
    outputs = model(x)

    # 検証
    assert outputs["seg_logits"].shape == (2, 5, 64, 64)
    assert outputs["sdf_field"].shape == (2, 4, 64, 64)
    print(f"✓ {test_name}: seg={outputs['seg_logits'].shape}, sdf={outputs['sdf_field'].shape}")


def test_resunet_param_count():
    """2つのインスタンスでパラメータ数が一致し、0より大きいことを確認"""
    test_name = "test_resunet_param_count"

    # 準備
    model_a = ResUNet()
    model_b = ResUNet()

    # 実行
    total_params_a = sum(param.numel() for param in model_a.parameters())
    total_params_b = sum(param.numel() for param in model_b.parameters())

    # 検証
    assert total_params_a > 0
    assert total_params_a == total_params_b
    print(f"✓ {test_name}: params_a={total_params_a}, params_b={total_params_b}")


def test_resunet_backward_no_nan():
    """逆伝播時に勾配が NaN を含まないことを確認"""
    test_name = "test_resunet_backward_no_nan"

    # 準備
    model = ResUNet()
    x = torch.randn(2, 2, 64, 64)

    # 実行
    outputs = model(x)
    loss = outputs["seg_logits"].mean() + outputs["sdf_field"].mean()
    loss.backward()

    # 検証
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    print(f"✓ {test_name}: all_gradients_finite={len(grads)} tensors")


def test_resunet_batch_size_1():
    """バッチサイズ1でも forward が正常に動作することを確認"""
    test_name = "test_resunet_batch_size_1"

    # 準備
    model = ResUNet()
    x = torch.randn(1, 2, 64, 64)

    # 実行
    outputs = model(x)

    # 検証
    assert outputs["seg_logits"].shape == (1, 5, 64, 64)
    assert outputs["sdf_field"].shape == (1, 4, 64, 64)
    print(f"✓ {test_name}: batch_size=1 forward ok")


def test_resunet_deterministic():
    """evalモードでは同一入力に対して同一出力になることを確認"""
    test_name = "test_resunet_deterministic"

    # 準備
    model = ResUNet()
    model.eval()
    x = torch.randn(2, 2, 64, 64)

    # 実行
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    # 検証
    assert torch.allclose(out1["seg_logits"], out2["seg_logits"], atol=1e-7, rtol=0.0)
    assert torch.allclose(out1["sdf_field"], out2["sdf_field"], atol=1e-7, rtol=0.0)
    print(f"✓ {test_name}: deterministic outputs in eval mode")


def test_resunet_vertebra_conditioning_output_shape():
    """椎体条件入力ありでも出力形状が変わらないことを確認"""
    model = ResUNet(num_vertebra=7)
    x = torch.randn(2, 2, 64, 64)
    v_idx = torch.tensor([0, 3], dtype=torch.long)
    outputs = model(x, v_idx)
    assert outputs['seg_logits'].shape == (2, 5, 64, 64)
    assert outputs['sdf_field'].shape == (2, 4, 64, 64)


def test_resunet_vertebra_conditioning_disabled():
    """椎体条件無効時は従来どおり forward できることを確認"""
    model = ResUNet(num_vertebra=0)
    x = torch.randn(2, 2, 64, 64)
    outputs = model(x)
    assert outputs['seg_logits'].shape == (2, 5, 64, 64)


def test_resunet_vertebra_conditioning_identity_init():
    """恒等初期化直後は条件有無で出力が一致することを確認"""
    model = ResUNet(num_vertebra=7)
    model.eval()
    x = torch.randn(2, 2, 64, 64)
    v_idx = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        out_cond = model(x, v_idx)
        out_uncond = model(x)
    assert torch.allclose(out_cond['seg_logits'], out_uncond['seg_logits'], atol=1e-5)
    assert torch.allclose(out_cond['sdf_field'], out_uncond['sdf_field'], atol=1e-5)


def test_resunet_sdf_field_finite():
    """sdf_field の出力がすべて有限値であることを確認"""
    model = ResUNet()
    x = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        outputs = model(x)
    assert torch.isfinite(outputs["sdf_field"]).all()
