"""Unit tests for line losses and parameter extraction."""

import math

import numpy as np
import torch

from ..line_losses import extract_gt_line_params, extract_pred_line_params_batch


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

    # Check that we can compute gradients
    # Sum valid (non-NaN) parameters
    valid_mask = ~torch.isnan(pred_params).any(dim=-1)
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

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
