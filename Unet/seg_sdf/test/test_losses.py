"""Seg+SDF 損失関数の単体テスト"""

import torch

from ..utils.losses import compute_sdf_seg_loss, extract_gt_line_params


def _create_dummy_batch(batch_size: int = 4, height: int = 64, width: int = 64) -> tuple[torch.Tensor, ...]:
    """損失テスト用のダミーバッチを作成する"""
    seg_logits = torch.randn(batch_size, 5, height, width)
    sdf_field = torch.randn(batch_size, 4, height, width)
    gt_mask = torch.randint(0, 5, (batch_size, height, width), dtype=torch.long)
    gt_sdf = torch.rand(batch_size, 4, height, width, dtype=torch.float32) * 2.0 - 1.0
    return seg_logits, sdf_field, gt_mask, gt_sdf


def test_compute_sdf_seg_loss_all_labeled():
    """全サンプルにセグラベルがある場合の損失計算を確認する"""
    # 準備
    lambda_sdf = 3.0
    seg_logits, sdf_field, gt_mask, gt_sdf = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.ones(4, dtype=torch.bool)

    # 実行
    losses = compute_sdf_seg_loss(
        seg_logits=seg_logits,
        sdf_field=sdf_field,
        gt_mask=gt_mask,
        gt_sdf=gt_sdf,
        has_seg_label=has_seg_label,
        lambda_sdf=lambda_sdf,
    )

    # 検証
    expected_keys = {"total", "raw_seg_loss", "raw_sdf_loss", "weighted_sdf_loss"}
    assert set(losses.keys()) == expected_keys
    expected_total = losses["raw_seg_loss"].item() + lambda_sdf * losses["raw_sdf_loss"].item()
    assert abs(losses["total"].item() - expected_total) < 1e-6


def test_compute_sdf_seg_loss_none_labeled():
    """セグラベルが1件も無い場合の損失計算を確認する"""
    # 準備
    lambda_sdf = 3.0
    seg_logits, sdf_field, gt_mask, gt_sdf = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.zeros(4, dtype=torch.bool)

    # 実行
    losses = compute_sdf_seg_loss(
        seg_logits=seg_logits,
        sdf_field=sdf_field,
        gt_mask=gt_mask,
        gt_sdf=gt_sdf,
        has_seg_label=has_seg_label,
        lambda_sdf=lambda_sdf,
    )

    # 検証
    assert losses["raw_seg_loss"].item() == 0.0
    assert torch.isclose(
        losses["weighted_sdf_loss"],
        lambda_sdf * losses["raw_sdf_loss"],
        atol=1e-6,
    )


def test_compute_sdf_seg_loss_partial():
    """部分教師あり（混在）でも損失が有限値で計算できることを確認する"""
    # 準備
    seg_logits, sdf_field, gt_mask, gt_sdf = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.tensor([True, False, True, False], dtype=torch.bool)

    # 実行
    losses = compute_sdf_seg_loss(
        seg_logits=seg_logits,
        sdf_field=sdf_field,
        gt_mask=gt_mask,
        gt_sdf=gt_sdf,
        has_seg_label=has_seg_label,
        lambda_sdf=3.0,
    )

    # 検証
    assert torch.isfinite(losses["total"]).item()
    assert torch.isfinite(losses["raw_seg_loss"]).item()
    assert torch.isfinite(losses["raw_sdf_loss"]).item()
    assert losses["raw_seg_loss"].item() > 0.0


def test_compute_sdf_seg_loss_backward():
    """逆伝播時に NaN 勾配が発生しないことを確認する"""
    # 準備
    seg_logits, sdf_field, gt_mask, gt_sdf = _create_dummy_batch(batch_size=4)
    seg_logits.requires_grad_(True)
    sdf_field.requires_grad_(True)
    has_seg_label = torch.tensor([True, False, True, False], dtype=torch.bool)

    # 実行
    losses = compute_sdf_seg_loss(
        seg_logits=seg_logits,
        sdf_field=sdf_field,
        gt_mask=gt_mask,
        gt_sdf=gt_sdf,
        has_seg_label=has_seg_label,
        lambda_sdf=3.0,
    )
    losses["total"].backward()

    # 検証
    assert seg_logits.grad is not None
    assert sdf_field.grad is not None
    assert torch.isfinite(seg_logits.grad).all()
    assert torch.isfinite(sdf_field.grad).all()


def test_compute_sdf_seg_loss_lambda_scaling():
    """weighted_sdf_loss が lambda_sdf 倍になっていることを確認する"""
    # 準備
    lambda_sdf = 5.0
    seg_logits, sdf_field, gt_mask, gt_sdf = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.ones(4, dtype=torch.bool)

    # 実行
    losses = compute_sdf_seg_loss(
        seg_logits=seg_logits,
        sdf_field=sdf_field,
        gt_mask=gt_mask,
        gt_sdf=gt_sdf,
        has_seg_label=has_seg_label,
        lambda_sdf=lambda_sdf,
    )

    # 検証
    assert torch.isclose(losses["weighted_sdf_loss"], lambda_sdf * losses["raw_sdf_loss"], atol=1e-6)


def test_extract_gt_line_params_horizontal():
    """水平線のGT抽出で phi≈π/2, rho≈0 になることを確認する"""
    import math

    # 準備
    points = [[0, 32], [63, 32]]

    # 実行
    phi, rho = extract_gt_line_params(points, image_size=64)

    # 検証
    assert abs(phi - (math.pi / 2)) < 1e-2
    assert abs(rho) < 1e-2
