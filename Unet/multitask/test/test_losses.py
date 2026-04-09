"""multitask 損失関数の単体テスト"""

import math

import torch

from ..utils.losses import (
    compute_multitask_loss,
    extract_gt_line_params,
    extract_pred_line_params_batch,
    get_warmup_weight,
)


def _create_dummy_batch(batch_size: int = 4, height: int = 64, width: int = 64) -> tuple[torch.Tensor, ...]:
    """損失テスト用のダミーバッチを作成"""
    seg_logits = torch.randn(batch_size, 5, height, width)
    line_heatmaps = torch.randn(batch_size, 4, height, width)
    gt_mask = torch.randint(0, 5, (batch_size, height, width), dtype=torch.long)
    gt_heatmap = torch.rand(batch_size, 4, height, width)
    return seg_logits, line_heatmaps, gt_mask, gt_heatmap


def test_compute_multitask_loss_all_labeled():
    """全サンプルにセグラベルがある場合の損失計算を確認"""
    test_name = "test_compute_multitask_loss_all_labeled"

    # Arrange
    alpha = 0.03
    seg_logits, line_heatmaps, gt_mask, gt_heatmap = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.ones(4, dtype=torch.bool)

    # Act
    losses = compute_multitask_loss(
        seg_logits=seg_logits,
        line_heatmaps=line_heatmaps,
        gt_mask=gt_mask,
        gt_heatmap=gt_heatmap,
        has_seg_label=has_seg_label,
        alpha=alpha,
    )

    # Assert
    expected_keys = {"total", "raw_line_loss", "raw_seg_loss", "weighted_seg_loss"}
    assert expected_keys.issubset(losses.keys())
    expected_total = losses["raw_line_loss"] + alpha * losses["raw_seg_loss"]
    assert torch.isclose(losses["total"].detach(), expected_total, atol=1e-6)
    print(f"✓ {test_name}: total≈line+alpha*seg verified")


def test_compute_multitask_loss_none_labeled():
    """セグラベルが1件も無い場合は line loss のみになることを確認"""
    test_name = "test_compute_multitask_loss_none_labeled"

    # Arrange
    seg_logits, line_heatmaps, gt_mask, gt_heatmap = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.zeros(4, dtype=torch.bool)

    # Act
    losses = compute_multitask_loss(
        seg_logits=seg_logits,
        line_heatmaps=line_heatmaps,
        gt_mask=gt_mask,
        gt_heatmap=gt_heatmap,
        has_seg_label=has_seg_label,
        alpha=0.03,
    )

    # Assert
    assert losses["raw_seg_loss"].item() == 0.0
    assert losses["weighted_seg_loss"].item() == 0.0
    assert torch.isclose(losses["total"].detach(), losses["raw_line_loss"], atol=1e-6)
    print(f"✓ {test_name}: total=line_loss when no seg labels")


def test_compute_multitask_loss_partial():
    """部分教師あり（混在）でも損失が有限値で計算できることを確認"""
    test_name = "test_compute_multitask_loss_partial"

    # Arrange
    seg_logits, line_heatmaps, gt_mask, gt_heatmap = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.tensor([True, False, True, False], dtype=torch.bool)

    # Act
    losses = compute_multitask_loss(
        seg_logits=seg_logits,
        line_heatmaps=line_heatmaps,
        gt_mask=gt_mask,
        gt_heatmap=gt_heatmap,
        has_seg_label=has_seg_label,
        alpha=0.03,
    )

    # Assert
    assert torch.isfinite(losses["total"]).item()
    assert torch.isfinite(losses["raw_line_loss"]).item()
    assert torch.isfinite(losses["raw_seg_loss"]).item()
    assert losses["raw_seg_loss"].item() > 0.0
    print(f"✓ {test_name}: partial supervision loss is finite")


def test_compute_multitask_loss_backward():
    """multitask loss で逆伝播したときに NaN 勾配が出ないことを確認"""
    test_name = "test_compute_multitask_loss_backward"

    # Arrange
    seg_logits, line_heatmaps, gt_mask, gt_heatmap = _create_dummy_batch(batch_size=4)
    seg_logits.requires_grad_(True)
    line_heatmaps.requires_grad_(True)
    has_seg_label = torch.tensor([True, False, True, False], dtype=torch.bool)

    # Act
    losses = compute_multitask_loss(
        seg_logits=seg_logits,
        line_heatmaps=line_heatmaps,
        gt_mask=gt_mask,
        gt_heatmap=gt_heatmap,
        has_seg_label=has_seg_label,
        alpha=0.03,
    )
    losses["total"].backward()

    # Assert
    assert seg_logits.grad is not None
    assert line_heatmaps.grad is not None
    assert torch.isfinite(seg_logits.grad).all()
    assert torch.isfinite(line_heatmaps.grad).all()
    print(f"✓ {test_name}: gradients are finite")


def test_compute_multitask_loss_alpha_scaling():
    """weighted_seg_loss が alpha * raw_seg_loss と一致することを確認"""
    test_name = "test_compute_multitask_loss_alpha_scaling"

    # Arrange
    alpha = 0.07
    seg_logits, line_heatmaps, gt_mask, gt_heatmap = _create_dummy_batch(batch_size=4)
    has_seg_label = torch.ones(4, dtype=torch.bool)

    # Act
    losses = compute_multitask_loss(
        seg_logits=seg_logits,
        line_heatmaps=line_heatmaps,
        gt_mask=gt_mask,
        gt_heatmap=gt_heatmap,
        has_seg_label=has_seg_label,
        alpha=alpha,
    )

    # Assert
    assert torch.isclose(
        losses["weighted_seg_loss"],
        alpha * losses["raw_seg_loss"],
        atol=1e-6,
    )
    print(f"✓ {test_name}: weighted_seg_loss=alpha*raw_seg_loss")


def test_get_warmup_weight_linear():
    """linear warmup の基本点（開始・中間・終了）を確認"""
    test_name = "test_get_warmup_weight_linear"

    # Arrange
    warmup_epochs = 10

    # Act
    w_start = get_warmup_weight(current_epoch=0, warmup_epochs=warmup_epochs, warmup_mode="linear")
    w_mid = get_warmup_weight(current_epoch=5, warmup_epochs=warmup_epochs, warmup_mode="linear")
    w_end = get_warmup_weight(current_epoch=10, warmup_epochs=warmup_epochs, warmup_mode="linear")

    # Assert
    assert w_start == 0.0
    assert w_mid == 0.5
    assert w_end == 1.0
    print(f"✓ {test_name}: start={w_start}, mid={w_mid}, end={w_end}")


def test_extract_gt_line_params_horizontal():
    """水平線のGT抽出で phi≈π/2, rho≈0 になることを確認"""
    test_name = "test_extract_gt_line_params_horizontal"

    # Arrange
    points = [[0, 32], [63, 32]]

    # Act
    phi, rho = extract_gt_line_params(points, image_size=64)

    # Assert
    assert abs(phi - (math.pi / 2)) < 1e-2
    assert abs(rho) < 1e-2
    print(f"✓ {test_name}: phi={phi:.4f}, rho={rho:.4f}")


def test_extract_pred_no_nan():
    """ランダムヒートマップでも予測抽出結果に NaN が出ないことを確認"""
    test_name = "test_extract_pred_no_nan"

    # Arrange
    heatmaps = torch.sigmoid(torch.randn(2, 4, 64, 64))

    # Act
    pred_params, confidence = extract_pred_line_params_batch(heatmaps, image_size=64)

    # Assert
    assert not torch.isnan(pred_params).any()
    assert not torch.isnan(confidence).any()
    print(f"✓ {test_name}: no NaN in pred params/confidence")
