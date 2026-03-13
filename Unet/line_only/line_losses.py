"""Line losses and parameter extraction for geometric line constraints."""

import math

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# GT Line Parameter Extraction
# -------------------------
def extract_gt_line_params(polyline_points, image_size=224):
    """
    Extract (φ, ρ) from GT polyline annotation.

    Args:
        polyline_points: List of [x, y] points defining the line (at least 2)
        image_size: Image dimension (assumes square)

    Returns:
        (phi_rad, rho_normalized) or (nan, nan) if invalid

    Coordinate system:
        - Origin: Image center (image_size/2, image_size/2)
        - x-axis: Column direction (left to right)
        - y-axis: Row direction (top to bottom)
        - φ: Angle of normal vector [0, π)
        - ρ: Signed distance from origin, normalized by diagonal D
    """
    if polyline_points is None or len(polyline_points) < 2:
        return float("nan"), float("nan")

    # Get endpoints
    p1 = np.array(polyline_points[0], dtype=np.float64)
    p2 = np.array(polyline_points[-1], dtype=np.float64)

    # Convert to center coordinates
    center = image_size / 2.0
    p1_c = p1 - center
    p2_c = p2 - center

    # Line direction vector
    direction = p2_c - p1_c
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-6:
        return float("nan"), float("nan")
    direction = direction / norm_dir

    # Normal vector (90° CCW rotation)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Ensure φ in [0, π)
    if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
        normal = -normal

    # Extract φ and ρ
    phi = np.arctan2(normal[1], normal[0])
    midpoint = (p1_c + p2_c) / 2.0
    rho = np.dot(normal, midpoint)

    # Normalize ρ
    D = np.sqrt(image_size**2 + image_size**2)
    rho_norm = rho / D

    return float(phi), float(rho_norm)


# -------------------------
# Pred Line Parameter Extraction (Codex-Optimized)
# -------------------------
def extract_pred_line_params_batch(heatmaps, image_size=224, min_mass=1e-6):
    """
    Extract (φ, ρ) from predicted heatmaps using moments method.

    Key improvements from Codex review:
    - Use MAXIMUM eigenvalue eigenvector for line direction
    - Float64 precision for covariance
    - Regularization for stability
    - Confidence-based masking

    Args:
        heatmaps: (B, 4, H, W) predicted heatmaps (after sigmoid)
        image_size: Image dimension
        min_mass: Minimum heatmap mass threshold

    Returns:
        pred_params: (B, 4, 2) tensor of (phi_rad, rho_normalized)
        confidence: (B, 4) tensor of eigenvalue ratio
        Invalid lines marked with NaN
    """
    B, C, H, W = heatmaps.shape
    device = heatmaps.device

    # Coordinate grids (center origin)
    y_grid = torch.arange(H, device=device, dtype=torch.float32) - H / 2.0
    x_grid = torch.arange(W, device=device, dtype=torch.float32) - W / 2.0
    Y, X = torch.meshgrid(y_grid, x_grid, indexing="ij")

    D = math.sqrt(image_size**2 + image_size**2)
    output = torch.zeros(B, C, 2, device=device)
    confidence = torch.zeros(B, C, device=device)

    for b in range(B):
        for c in range(C):
            hm = heatmaps[b, c]
            M00 = hm.sum()

            # Guard: Skip if low mass
            if M00 < min_mass:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # Weighted centroid
            cx = (hm * X).sum() / M00
            cy = (hm * Y).sum() / M00

            # Covariance (FLOAT64 for stability)
            dx = (X - cx).double()
            dy = (Y - cy).double()
            hm_d = hm.double()

            mu20 = (hm_d * dx * dx).sum() / M00
            mu02 = (hm_d * dy * dy).sum() / M00
            mu11 = (hm_d * dx * dy).sum() / M00

            # Regularization
            eps_reg = 1e-6
            mu20 = mu20 + eps_reg
            mu02 = mu02 + eps_reg

            # Eigenvalues
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11 * mu11
            discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
            sqrt_disc = torch.sqrt(discriminant)

            lambda1 = (trace + sqrt_disc) / 2  # Larger (line direction)
            lambda2 = (trace - sqrt_disc) / 2  # Smaller

            # Confidence: eigenvalue ratio
            if lambda1 > 1e-8:
                conf = 1.0 - lambda2 / lambda1
                confidence[b, c] = conf  # Keep as tensor for gradient flow
            else:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # Line direction: eigenvector of MAXIMUM eigenvalue
            if abs(mu11) > 1e-8:
                dir_x = mu11
                dir_y = lambda1 - mu20
            else:
                # Axis-aligned (use tensor for consistency)
                if mu20 > mu02:
                    dir_x = torch.tensor(1.0, dtype=mu20.dtype, device=device)
                    dir_y = torch.tensor(0.0, dtype=mu20.dtype, device=device)
                else:
                    dir_x = torch.tensor(0.0, dtype=mu20.dtype, device=device)
                    dir_y = torch.tensor(1.0, dtype=mu20.dtype, device=device)

            # Normalize direction
            dir_norm = torch.sqrt(dir_x * dir_x + dir_y * dir_y)
            dir_x = dir_x / (dir_norm + 1e-10)
            dir_y = dir_y / (dir_norm + 1e-10)

            # Normal: 90° CCW rotation
            nx = -dir_y
            ny = dir_x

            # Ensure φ in [0, π)
            if ny < 0 or (ny == 0 and nx < 0):
                nx, ny = -nx, -ny

            # Compute φ and ρ
            phi = torch.atan2(ny, nx)
            rho = nx * cx + ny * cy
            rho_norm = rho / D

            output[b, c, 0] = phi.float()
            output[b, c, 1] = rho_norm.float()

    return output, confidence


# -------------------------
# Loss Functions (Codex-Optimized)
# -------------------------
def angle_loss(pred_params, gt_params, confidence, valid_mask):
    """
    Angle loss: 1 - |n_pred · n_gt|

    More efficient than 1 - |cos(φ_pred - φ_gt)| and avoids atan2 gradients.

    Args:
        pred_params: (B, 4, 2) predicted (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        confidence: (B, 4) confidence weights
        valid_mask: (B, 4) boolean mask

    Returns:
        Scalar loss
    """
    pred_phi = pred_params[..., 0]
    gt_phi = gt_params[..., 0]

    # Compute normal vectors
    pred_nx = torch.cos(pred_phi)
    pred_ny = torch.sin(pred_phi)
    gt_nx = torch.cos(gt_phi)
    gt_ny = torch.sin(gt_phi)

    # Inner product
    dot = pred_nx * gt_nx + pred_ny * gt_ny

    # Loss: 1 - |dot|
    loss = 1.0 - torch.abs(dot)

    # Weight by confidence and validity
    weights = confidence * valid_mask.float()
    weighted_loss = loss * weights

    return weighted_loss.sum() / (weights.sum() + 1e-8)


def rho_loss(pred_params, gt_params, confidence, valid_mask):
    """
    Rho loss: min(|ρ_p - ρ_g|, |ρ_p + ρ_g|)

    Sign-invariant to handle (φ, ρ) ≡ (φ+π, -ρ) ambiguity.
    Uses smooth min for differentiability.

    Args:
        pred_params: (B, 4, 2) predicted (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        confidence: (B, 4) confidence weights
        valid_mask: (B, 4) boolean mask

    Returns:
        Scalar loss
    """
    pred_rho = pred_params[..., 1]
    gt_rho = gt_params[..., 1]

    # Two possible errors
    err1 = torch.abs(pred_rho - gt_rho)
    err2 = torch.abs(pred_rho + gt_rho)

    # Smooth minimum (differentiable)
    alpha = 10.0
    exp1 = torch.exp(-alpha * err1)
    exp2 = torch.exp(-alpha * err2)
    loss = (err1 * exp1 + err2 * exp2) / (exp1 + exp2 + 1e-8)

    # SmoothL1
    loss = F.smooth_l1_loss(loss, torch.zeros_like(loss), reduction="none")

    # Weight by confidence and validity
    weights = confidence * valid_mask.float()
    weighted_loss = loss * weights

    return weighted_loss.sum() / (weights.sum() + 1e-8)


def compute_line_loss(
    pred_heatmaps,
    gt_line_params,
    image_size=224,
    lambda_theta=0.1,
    lambda_rho=0.05,
    use_angle=False,
    use_rho=False,
    min_confidence=0.3,
):
    """
    Combined line loss with confidence-based masking.

    Args:
        pred_heatmaps: (B, 4, H, W) predicted heatmaps after sigmoid
        gt_line_params: (B, 4, 2) GT (phi, rho)
        image_size: Image size
        lambda_theta: Weight for angle loss
        lambda_rho: Weight for rho loss
        use_angle: Enable angle loss
        use_rho: Enable rho loss
        min_confidence: Minimum eigenvalue ratio for loss computation

    Returns:
        Dictionary with keys:
            'total': combined line loss
            'angle': angle loss component (or 0)
            'rho': rho loss component (or 0)
    """
    pred_params, confidence = extract_pred_line_params_batch(pred_heatmaps, image_size)

    # Valid mask
    gt_valid = ~torch.isnan(gt_line_params).any(dim=-1)
    pred_valid = ~torch.isnan(pred_params).any(dim=-1)
    conf_valid = confidence > min_confidence
    valid_mask = gt_valid & pred_valid & conf_valid

    losses = {}
    total = torch.tensor(0.0, device=pred_heatmaps.device)

    if use_angle:
        loss_a = angle_loss(pred_params, gt_line_params, confidence, valid_mask)
        losses["angle"] = loss_a
        total = total + lambda_theta * loss_a
    else:
        losses["angle"] = torch.tensor(0.0, device=pred_heatmaps.device)

    if use_rho:
        loss_r = rho_loss(pred_params, gt_line_params, confidence, valid_mask)
        losses["rho"] = loss_r
        total = total + lambda_rho * loss_r
    else:
        losses["rho"] = torch.tensor(0.0, device=pred_heatmaps.device)

    losses["total"] = total
    return losses


# -------------------------
# Warmup Schedule
# -------------------------
def get_warmup_weight(current_epoch, warmup_epochs, warmup_mode="linear"):
    """
    Compute warmup weight w(t) for line losses.

    Args:
        current_epoch: Current epoch (1-indexed)
        warmup_epochs: Number of epochs to warmup
        warmup_mode: 'linear', 'cosine', or 'step'

    Returns:
        Weight in [0, 1]
    """
    if current_epoch > warmup_epochs or warmup_epochs == 0:
        return 1.0

    progress = current_epoch / warmup_epochs

    if warmup_mode == "linear":
        return progress
    elif warmup_mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        return progress
