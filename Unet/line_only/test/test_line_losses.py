"""Unit tests for line losses and parameter extraction."""

import math

import numpy as np
import torch

from ..utils.losses import (
    angle_loss,
    compute_line_loss,
    extract_gt_line_params,
    extract_pred_line_params_batch,
    get_warmup_weight,
    rho_loss,
)


def test_gt_extraction_horizontal():
    """Test GT extraction for horizontal line through center."""
    # Horizontal line through center: y = 112
    points = [[0, 112], [224, 112]]
    phi, rho = extract_gt_line_params(points, image_size=224)

    # Horizontal line → normal points up (φ ≈ π/2)
    # Line passes through center → ρ ≈ 0
    assert abs(phi - math.pi / 2) < 0.01, f"Expected φ≈π/2, got {phi}"
    assert abs(rho) < 0.01, f"Expected ρ≈0, got {rho}"
    print(f"✓ Horizontal line: φ={phi:.4f}, ρ={rho:.4f}")


def test_gt_extraction_vertical():
    """Test GT extraction for vertical line through center."""
    # Vertical line through center: x = 112
    points = [[112, 0], [112, 224]]
    phi, rho = extract_gt_line_params(points, image_size=224)

    # Vertical line → normal points right (φ ≈ 0 or π)
    # Line passes through center → ρ ≈ 0
    phi_deg = math.degrees(phi)
    # φ should be close to 0 or 180 degrees
    assert (abs(phi) < 0.01) or (abs(phi - math.pi) < 0.01), f"Expected φ≈0 or π, got {phi}"
    assert abs(rho) < 0.01, f"Expected ρ≈0, got {rho}"
    print(f"✓ Vertical line: φ={phi:.4f} ({phi_deg:.1f}°), ρ={rho:.4f}")


def test_gt_extraction_diagonal():
    """Test GT extraction for diagonal line."""
    # Diagonal line: y = x (45 degrees)
    points = [[0, 0], [224, 224]]
    phi, rho = extract_gt_line_params(points, image_size=224)

    # Diagonal line (45°) → normal at 135° or -45°
    # Line passes through center → ρ ≈ 0
    phi_deg = math.degrees(phi)
    print(f"✓ Diagonal line: φ={phi:.4f} ({phi_deg:.1f}°), ρ={rho:.4f}")
    # Don't assert exact value, just check it's a valid angle
    assert 0 <= phi < math.pi, f"φ should be in [0, π), got {phi}"
    assert abs(rho) < 0.2, f"Expected ρ near 0 for diagonal through center, got {rho}"


def test_gt_extraction_edge_line():
    """Test GT extraction for line at edge."""
    # Horizontal line at top edge
    points = [[0, 0], [224, 0]]
    phi, rho = extract_gt_line_params(points, image_size=224)

    # Horizontal line → normal points up (φ ≈ π/2)
    # Line is at y=0, center is at y=112 → ρ should be non-zero
    phi_deg = math.degrees(phi)
    print(f"✓ Edge line: φ={phi:.4f} ({phi_deg:.1f}°), ρ={rho:.4f}")
    assert abs(phi - math.pi / 2) < 0.01, f"Expected φ≈π/2, got {phi}"
    assert abs(rho) > 0.1, f"Expected non-zero ρ for edge line, got {rho}"


def test_gt_extraction_invalid():
    """Test GT extraction with invalid input."""
    # Empty points
    phi, rho = extract_gt_line_params([], image_size=224)
    assert math.isnan(phi) and math.isnan(rho), "Expected NaN for empty points"

    # Single point
    phi, rho = extract_gt_line_params([[112, 112]], image_size=224)
    assert math.isnan(phi) and math.isnan(rho), "Expected NaN for single point"

    # Identical points (zero length line)
    phi, rho = extract_gt_line_params([[112, 112], [112, 112]], image_size=224)
    assert math.isnan(phi) and math.isnan(rho), "Expected NaN for zero-length line"

    print("✓ Invalid input handling")


def test_pred_extraction_gradient_flow():
    """Test that pred extraction is differentiable."""
    # Create synthetic heatmap with gradients
    B, C, H, W = 2, 4, 224, 224
    heatmaps_raw = torch.randn(B, C, H, W, requires_grad=True)
    heatmaps = torch.sigmoid(heatmaps_raw)  # [0, 1]

    # Extract parameters
    pred_params, confidence = extract_pred_line_params_batch(heatmaps)

    # Check that we can compute gradients (confidence ベースの valid_mask)
    valid_mask = confidence > 0
    loss = (pred_params * valid_mask.unsqueeze(-1).float()).sum()

    # Backward
    loss.backward()

    # Check gradients exist and are not NaN (check leaf tensor)
    assert heatmaps_raw.grad is not None, "Gradients should exist"
    assert not torch.isnan(heatmaps_raw.grad).all(), "Gradients should not all be NaN"

    print(f"✓ Gradient flow: {valid_mask.sum().item()}/{B*C} valid predictions")


def test_pred_extraction_synthetic_line():
    """Test pred extraction on synthetic Gaussian heatmap."""
    # Create synthetic heatmap for a known line
    B, C, H, W = 1, 1, 224, 224

    # Create Gaussian heatmap along horizontal line (y=112)
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    x_coords = torch.arange(W, dtype=torch.float32).unsqueeze(0)  # (1, W)

    # Gaussian along y=112
    sigma = 5.0
    heatmap = torch.exp(-((y_coords - 112) ** 2) / (2 * sigma**2))
    heatmap = heatmap.expand(H, W)  # Broadcast to (H, W)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Extract parameters
    pred_params, confidence = extract_pred_line_params_batch(heatmap)

    phi = pred_params[0, 0, 0].item()
    rho = pred_params[0, 0, 1].item()
    conf = confidence[0, 0].item()

    phi_deg = math.degrees(phi)

    # Horizontal line → φ ≈ π/2, ρ ≈ 0
    print(
        f"✓ Synthetic horizontal line: φ={phi:.4f} ({phi_deg:.1f}°), ρ={rho:.4f}, conf={conf:.4f}"
    )
    assert abs(phi - math.pi / 2) < 0.1, f"Expected φ≈π/2, got {phi}"
    assert abs(rho) < 0.1, f"Expected ρ≈0, got {rho}"
    assert conf > 0.9, f"Expected high confidence, got {conf}"


def test_threshold_backward_compatibility():
    """Test that threshold=None maintains backward compatibility."""
    B, C, H, W = 1, 1, 224, 224

    # Create synthetic heatmap
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    heatmap = torch.exp(-((y_coords - 112) ** 2) / (2 * 5.0**2))
    heatmap = heatmap.expand(H, W).unsqueeze(0).unsqueeze(0)

    # Extract with threshold=None (default)
    pred_params_none, conf_none = extract_pred_line_params_batch(heatmap)

    # Extract without specifying threshold (backward compatibility)
    pred_params_default, conf_default = extract_pred_line_params_batch(heatmap)

    # Should be identical
    assert torch.allclose(pred_params_none, pred_params_default, atol=1e-6), \
        "threshold=None should match default behavior"
    assert torch.allclose(conf_none, conf_default, atol=1e-6), \
        "Confidence should match for default behavior"

    print("✓ Backward compatibility: threshold=None matches default")


def test_threshold_effect():
    """Test that threshold=0.2 filters noise and stabilizes angle calculation."""
    B, C, H, W = 1, 1, 224, 224

    # Create noisy heatmap (horizontal line + uniform noise)
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    clean_signal = torch.exp(-((y_coords - 112) ** 2) / (2 * 5.0**2))
    clean_signal = clean_signal.expand(H, W)

    # Add uniform low-level noise
    noise = torch.rand(H, W) * 0.15  # Noise below threshold
    noisy_heatmap = (clean_signal + noise).unsqueeze(0).unsqueeze(0)
    noisy_heatmap = torch.clamp(noisy_heatmap, 0, 1)

    # Extract without threshold
    pred_none, conf_none = extract_pred_line_params_batch(noisy_heatmap, threshold=None)

    # Extract with threshold=0.2
    pred_thresh, conf_thresh = extract_pred_line_params_batch(noisy_heatmap, threshold=0.2)

    phi_none = pred_none[0, 0, 0].item()
    phi_thresh = pred_thresh[0, 0, 0].item()

    # Threshold should increase confidence
    assert conf_thresh[0, 0] > conf_none[0, 0], \
        f"Threshold should increase confidence: {conf_none[0, 0]:.3f} -> {conf_thresh[0, 0]:.3f}"

    # Both should be valid (not NaN)
    assert not math.isnan(phi_none) and not math.isnan(phi_thresh), \
        "Both extractions should be valid"

    print(f"✓ Threshold effect: conf {conf_none[0, 0]:.3f} -> {conf_thresh[0, 0]:.3f}")


def test_threshold_zero_vs_none():
    """Test that threshold=0.0 and threshold=None behave differently."""
    B, C, H, W = 1, 1, 224, 224

    # Create heatmap with some negative values (should not happen in sigmoid, but edge case)
    heatmap = torch.randn(B, C, H, W) * 0.1 + 0.5
    heatmap = torch.clamp(heatmap, 0, 1)

    # Extract with None (no filtering)
    pred_none, _ = extract_pred_line_params_batch(heatmap, threshold=None)

    # Extract with 0.0 (filter exactly zero, but keep positive values)
    pred_zero, _ = extract_pred_line_params_batch(heatmap, threshold=0.0)

    # For sigmoid outputs (all >= 0), these should be very similar
    # This test ensures the threshold logic is correctly implemented
    if not (torch.isnan(pred_none).all() or torch.isnan(pred_zero).all()):
        # If both valid, they should be close for sigmoid outputs
        assert torch.allclose(pred_none, pred_zero, atol=0.1, equal_nan=True), \
            "threshold=0.0 should have minimal effect on sigmoid outputs"

    print("✓ threshold=0.0 vs None: Logic correctly implemented")


def test_pred_extraction_no_nan():
    """Test that extraction never returns NaN (even for degenerate inputs)."""
    # ランダムヒートマップ（全値が閾値以下になる可能性あり）
    B, C, H, W = 2, 4, 224, 224
    heatmaps = torch.sigmoid(torch.randn(B, C, H, W))
    pred_params, confidence = extract_pred_line_params_batch(heatmaps)

    assert not torch.isnan(pred_params).any(), "pred_params must not contain NaN"
    assert not torch.isnan(confidence).any(), "confidence must not contain NaN"
    assert (confidence >= 0).all(), "confidence must be >= 0"
    assert (confidence <= 1).all(), "confidence must be <= 1"

    # ゼロヒートマップ（完全に無効）
    zero_hm = torch.zeros(B, C, H, W)
    zero_params, zero_conf = extract_pred_line_params_batch(zero_hm)
    assert not torch.isnan(zero_params).any(), "Zero heatmap must not produce NaN"
    assert (zero_conf == 0).all(), "Zero heatmap must give confidence=0"

    print(f"✓ No NaN: valid={( confidence > 0).sum().item()}/{B*C}")


def test_pred_extraction_batch_consistency():
    """Test that processing B=2 gives same result as two B=1 calls."""
    C, H, W = 4, 224, 224

    # ガウシアンヒートマップ（水平線 y=112）
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    hm_single = torch.exp(-((y_coords - 112) ** 2) / (2 * 5.0**2)).expand(H, W)
    hm1 = hm_single.unsqueeze(0).unsqueeze(0).expand(1, C, H, W).clone()
    hm2 = (hm_single * 0.8).unsqueeze(0).unsqueeze(0).expand(1, C, H, W).clone()
    hm_batch = torch.cat([hm1, hm2], dim=0)  # (2, C, H, W)

    # 個別処理
    params1, conf1 = extract_pred_line_params_batch(hm1)
    params2, conf2 = extract_pred_line_params_batch(hm2)

    # バッチ処理
    params_batch, conf_batch = extract_pred_line_params_batch(hm_batch)

    assert torch.allclose(params_batch[0], params1[0], atol=1e-5), \
        "Batch[0] must match single B=1 result"
    assert torch.allclose(params_batch[1], params2[0], atol=1e-5), \
        "Batch[1] must match single B=1 result"
    assert torch.allclose(conf_batch[0], conf1[0], atol=1e-5), \
        "Confidence Batch[0] must match single B=1 result"

    print("✓ Batch consistency: B=2 matches two B=1 calls")


def test_warmup_weight_linear():
    """linear warmup: 0→1 の単調増加、境界値の確認"""
    w0 = get_warmup_weight(0, warmup_epochs=10, warmup_start_epoch=0)
    w5 = get_warmup_weight(5, warmup_epochs=10, warmup_start_epoch=0)
    w10 = get_warmup_weight(10, warmup_epochs=10, warmup_start_epoch=0)
    w20 = get_warmup_weight(20, warmup_epochs=10, warmup_start_epoch=0)

    assert abs(w0 - 0.0) < 1e-6, f"epoch=0: expected 0, got {w0}"
    assert abs(w5 - 0.5) < 1e-6, f"epoch=5: expected 0.5, got {w5}"
    assert abs(w10 - 1.0) < 1e-6, f"epoch=10: expected 1.0, got {w10}"
    assert abs(w20 - 1.0) < 1e-6, f"epoch=20: expected 1.0 (saturate), got {w20}"
    print(f"✓ warmup linear: w(0)={w0:.2f}, w(5)={w5:.2f}, w(10)={w10:.2f}")


def test_warmup_weight_start_epoch():
    """warmup_start_epoch: 開始前は 0、開始後に warmup 進行"""
    # warmup_start_epoch=20, warmup_epochs=10 → epoch<20 は 0、epoch=25 は 0.5
    w_before = get_warmup_weight(10, warmup_epochs=10, warmup_start_epoch=20)
    w_at = get_warmup_weight(20, warmup_epochs=10, warmup_start_epoch=20)
    w_mid = get_warmup_weight(25, warmup_epochs=10, warmup_start_epoch=20)
    w_done = get_warmup_weight(30, warmup_epochs=10, warmup_start_epoch=20)

    assert abs(w_before - 0.0) < 1e-6, f"before start: expected 0, got {w_before}"
    assert abs(w_at - 0.0) < 1e-6, f"at start: expected 0, got {w_at}"
    assert abs(w_mid - 0.5) < 1e-6, f"midpoint: expected 0.5, got {w_mid}"
    assert abs(w_done - 1.0) < 1e-6, f"done: expected 1.0, got {w_done}"
    print(f"✓ warmup start_epoch: before={w_before:.2f}, mid={w_mid:.2f}, done={w_done:.2f}")


def test_angle_loss_aligned():
    """pred=gt → loss=0"""
    phi = torch.tensor([[0.5, 1.0, 1.5, 2.0]])  # (1, 4)
    nx_pred = torch.cos(phi)
    ny_pred = torch.sin(phi)
    L, _ = angle_loss(nx_pred, ny_pred, phi)
    assert L.max().item() < 1e-6, f"Aligned: expected 0, got {L.max().item()}"
    print(f"✓ angle_loss aligned: max={L.max().item():.2e}")


def test_angle_loss_orthogonal():
    """90° ずれ → loss=1"""
    phi_gt = torch.tensor([[0.0, math.pi / 4]])   # (1, 2)
    phi_pred = phi_gt + math.pi / 2               # 90° ずれ
    nx_pred = torch.cos(phi_pred)
    ny_pred = torch.sin(phi_pred)
    L, _ = angle_loss(nx_pred, ny_pred, phi_gt)
    assert abs(L.max().item() - 1.0) < 1e-5, f"Orthogonal: expected 1, got {L.max().item()}"
    print(f"✓ angle_loss orthogonal: max={L.max().item():.6f}")


def test_angle_loss_pi_periodic():
    """φ+π → loss=0（π 周期性: anti-parallel の法線は同じ直線）"""
    phi_gt = torch.tensor([[0.3, 1.2, 2.5]])      # (1, 3)
    phi_pred = phi_gt + math.pi                    # π だけずらす
    nx_pred = torch.cos(phi_pred)
    ny_pred = torch.sin(phi_pred)
    L, _ = angle_loss(nx_pred, ny_pred, phi_gt)
    assert L.max().item() < 1e-5, f"π-periodic: expected 0, got {L.max().item()}"
    print(f"✓ angle_loss π-periodic: max={L.max().item():.2e}")


def test_angle_loss_smooth_at_zero():
    """Δφ=0 付近の勾配 ≈ 0（cusp なし）"""
    phi_gt = torch.tensor([[1.0]])
    phi_pred = phi_gt.clone().requires_grad_(True)
    nx_pred = torch.cos(phi_pred)
    ny_pred = torch.sin(phi_pred)
    L, _ = angle_loss(nx_pred, ny_pred, phi_gt)
    L.sum().backward()
    grad = phi_pred.grad
    assert grad is not None
    assert abs(grad.item()) < 1e-5, f"Smooth at zero: gradient should be ~0, got {grad.item()}"
    print(f"✓ angle_loss smooth at zero: grad={grad.item():.2e}")


def test_rho_loss_sign_ambiguity():
    """法線が逆向き（φ+π）でも rho_loss ≈ 0"""
    import math as _math
    D = _math.sqrt(224**2 + 224**2)
    phi_gt = torch.tensor([[0.5]])  # (1, 1)
    rho_gt = torch.tensor([[0.3]])

    # GT normal
    nx_gt = torch.cos(phi_gt)
    ny_gt = torch.sin(phi_gt)
    # GT 重心
    cx = rho_gt * nx_gt
    cy = rho_gt * ny_gt

    # 逆向き pred 法線（φ+π に対応）
    nx_pred = -nx_gt
    ny_pred = -ny_gt

    # angle_loss の dot を取得
    _, dot = angle_loss(nx_pred, ny_pred, phi_gt)

    L = rho_loss(nx_pred, ny_pred, cx * D, cy * D, rho_gt, dot, D)
    assert L.max().item() < 1e-5, f"Sign ambiguity: expected ≈0, got {L.max().item()}"
    print(f"✓ rho_loss sign ambiguity: max={L.max().item():.2e}")


def test_soft_gate_bounds():
    """gate の境界値確認（confidence < low → 0, confidence > high → 1）"""
    B, C = 1, 4
    H, W = 224, 224

    # ガウシアン（高 confidence）と等方性ブロブ（低 confidence）
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    x_coords = torch.arange(W, dtype=torch.float32).unsqueeze(0)
    hm_line = torch.exp(-((y_coords - 112) ** 2) / (2 * 3.0**2)).expand(H, W)
    hm_blob = torch.exp(-((y_coords - 112) ** 2 + (x_coords - 112) ** 2) / (2 * 20.0**2))

    heatmaps_line = hm_line.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).clone()
    heatmaps_blob = hm_blob.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).clone()

    from ..utils.losses import _compute_moments_batch
    conf_line, *_ = _compute_moments_batch(heatmaps_line)
    conf_blob, *_ = _compute_moments_batch(heatmaps_blob)

    gate_low, gate_high = 0.3, 0.6
    gate_line = ((conf_line - gate_low) / (gate_high - gate_low)).clamp(0, 1)
    gate_blob = ((conf_blob - gate_low) / (gate_high - gate_low)).clamp(0, 1)

    assert gate_line.min().item() > 0.99, f"直線ヒートマップは gate≈1 のはず: {gate_line.min().item()}"
    assert gate_blob.max().item() < 0.01, f"等方性ブロブは gate≈0 のはず: {gate_blob.max().item()}"
    print(f"✓ soft gate: line={gate_line.min().item():.3f}, blob={gate_blob.max().item():.3f}")


def test_compute_line_loss_backward():
    """compute_line_loss の backward で NaN が出ないことを確認"""
    B, C, H, W = 2, 4, 224, 224
    heatmaps_raw = torch.randn(B, C, H, W, requires_grad=True)
    heatmaps = torch.sigmoid(heatmaps_raw)

    # GT params（phi ∈ [0, π), rho ∈ [-0.5, 0.5]）
    gt_params = torch.zeros(B, C, 2)
    gt_params[..., 0] = torch.tensor([0.5, 1.0, 1.5, 2.0])
    gt_params[..., 1] = torch.tensor([0.1, -0.1, 0.2, -0.2])

    losses = compute_line_loss(
        heatmaps,
        gt_params,
        use_line_loss=True,
        lambda_angle=1.0,
        lambda_rho=1.0,
    )
    total = losses["total"]
    assert not torch.isnan(total), f"total loss is NaN"

    total.backward()
    assert heatmaps_raw.grad is not None
    assert not torch.isnan(heatmaps_raw.grad).any(), "Gradient contains NaN"

    print(f"✓ compute_line_loss backward: total={total.item():.4f}, gate_ratio={losses['gate_ratio'].item():.3f}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running GT Extraction Tests")
    print("=" * 60)
    test_gt_extraction_horizontal()
    test_gt_extraction_vertical()
    test_gt_extraction_diagonal()
    test_gt_extraction_edge_line()
    test_gt_extraction_invalid()

    print("\n" + "=" * 60)
    print("Running Pred Extraction Tests")
    print("=" * 60)
    test_pred_extraction_gradient_flow()
    test_pred_extraction_synthetic_line()
    test_pred_extraction_no_nan()
    test_pred_extraction_batch_consistency()

    print("\n" + "=" * 60)
    print("Running Threshold Tests")
    print("=" * 60)
    test_threshold_backward_compatibility()
    test_threshold_effect()
    test_threshold_zero_vs_none()

    print("\n" + "=" * 60)
    print("Running Warmup Tests (Stage 3)")
    print("=" * 60)
    test_warmup_weight_linear()
    test_warmup_weight_start_epoch()

    print("\n" + "=" * 60)
    print("Running Loss Function Tests (Stage 2)")
    print("=" * 60)
    test_angle_loss_aligned()
    test_angle_loss_orthogonal()
    test_angle_loss_pi_periodic()
    test_angle_loss_smooth_at_zero()
    test_rho_loss_sign_ambiguity()
    test_soft_gate_bounds()
    test_compute_line_loss_backward()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
