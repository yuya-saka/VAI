"""Evaluation metrics for line detection."""

import math

import torch


def compute_angle_error(pred_params, gt_params, valid_mask):
    """
    Angle error in degrees using arccos.

    Args:
        pred_params: (B, 4, 2) predicted (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        valid_mask: (B, 4) boolean mask for valid samples

    Returns:
        Mean angle error in degrees
    """
    pred_phi = pred_params[..., 0]
    gt_phi = gt_params[..., 0]

    # Compute normals and dot product
    pred_nx = torch.cos(pred_phi)
    pred_ny = torch.sin(pred_phi)
    gt_nx = torch.cos(gt_phi)
    gt_ny = torch.sin(gt_phi)
    dot = pred_nx * gt_nx + pred_ny * gt_ny

    # Clamp and arccos
    dot_clamped = torch.clamp(torch.abs(dot), 0.0, 1.0)
    angle_error_rad = torch.acos(dot_clamped)
    angle_error_deg = torch.rad2deg(angle_error_rad)

    # Average over valid samples
    if valid_mask is not None:
        angle_error_deg = angle_error_deg * valid_mask.float()
        return float(angle_error_deg.sum() / (valid_mask.sum() + 1e-8))
    return float(angle_error_deg.mean())


def compute_rho_error(pred_params, gt_params, image_size, valid_mask):
    """
    Rho error in pixels (sign-invariant).

    Args:
        pred_params: (B, 4, 2) predicted (phi, rho) - rho is normalized
        gt_params: (B, 4, 2) GT (phi, rho) - rho is normalized
        image_size: Image size
        valid_mask: (B, 4) boolean mask

    Returns:
        Mean rho error in pixels
    """
    pred_rho = pred_params[..., 1]
    gt_rho = gt_params[..., 1]

    # Min of two possible errors (sign-invariant)
    err1 = torch.abs(pred_rho - gt_rho)
    err2 = torch.abs(pred_rho + gt_rho)
    err = torch.minimum(err1, err2)

    # Convert to pixels
    D = math.sqrt(image_size**2 + image_size**2)
    err_px = err * D

    # Average over valid samples
    if valid_mask is not None:
        err_px = err_px * valid_mask.float()
        return float(err_px.sum() / (valid_mask.sum() + 1e-8))
    return float(err_px.mean())


def compute_perpendicular_distance(
    gt_polyline_points, pred_phi, pred_rho, image_size, num_samples=20
):
    """
    Average perpendicular distance from GT segment points to pred line.

    Args:
        gt_polyline_points: (N, 2) array of GT line points
        pred_phi: Predicted angle (radians)
        pred_rho: Predicted rho (normalized)
        image_size: Image size
        num_samples: Number of points to sample on GT segment

    Returns:
        Mean perpendicular distance in pixels
    """
    import numpy as np

    if gt_polyline_points is None or len(gt_polyline_points) < 2:
        return float("nan")

    gt_pts = np.array(gt_polyline_points, dtype=np.float64)

    # Denormalize rho
    D = math.sqrt(image_size**2 + image_size**2)
    rho_px = pred_rho * D

    # Normal vector
    nx = math.cos(pred_phi)
    ny = math.sin(pred_phi)

    # Sample points uniformly on GT polyline
    total_length = 0.0
    lengths = []
    for i in range(len(gt_pts) - 1):
        seg_len = np.linalg.norm(gt_pts[i + 1] - gt_pts[i])
        lengths.append(seg_len)
        total_length += seg_len

    if total_length < 1e-6:
        return float("nan")

    # Sample uniformly
    sample_distances = np.linspace(0, total_length, num_samples)
    sampled_points = []

    cumulative = 0.0
    for dist in sample_distances:
        # Find which segment this distance falls on
        current_cumulative = 0.0
        for i, seg_len in enumerate(lengths):
            if current_cumulative + seg_len >= dist:
                # Interpolate on this segment
                t = (dist - current_cumulative) / seg_len if seg_len > 0 else 0.0
                pt = (1 - t) * gt_pts[i] + t * gt_pts[i + 1]
                sampled_points.append(pt)
                break
            current_cumulative += seg_len

    if not sampled_points:
        return float("nan")

    sampled_points = np.array(sampled_points)

    # Convert to image-center coordinates
    center = image_size / 2.0
    sampled_points_centered = sampled_points - center

    # Distance from each point to line: |n^T * p - rho|
    distances = np.abs(
        sampled_points_centered[:, 0] * nx + sampled_points_centered[:, 1] * ny - rho_px
    )

    return float(distances.mean())
