"""評価指標: セグメンテーション + SDF"""

import math

import torch


CLASS_NAMES = ["bg", "body", "right", "left", "posterior"]


def compute_seg_metrics(
    seg_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int = 5,
    eps: float = 1e-6,
) -> dict:
    """
    セグメンテーション評価指標を計算する

    引数:
        seg_logits: セグメンテーションロジット (B, C, H, W)
        gt_mask: GTマスク (B, H, W) int64
        num_classes: クラス数（デフォルト5）
        eps: ゼロ除算防止

    戻り値:
        miou, dice, fg_miou, fg_mdice などを含む辞書
    """
    pred_class = seg_logits.argmax(dim=1)  # (B, H, W)
    ious = []
    dices = []
    for c in range(num_classes):
        pred_c = pred_class == c
        gt_c = gt_mask == c
        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        iou = (intersection + eps) / (union + eps)
        ious.append(iou.item())
        dice_denom = pred_c.sum().float() + gt_c.sum().float()
        dice = (2.0 * intersection + eps) / (dice_denom + eps)
        dices.append(dice.item())
    # class 0 = bg を除いた前景クラスのみ
    fg_ious = ious[1:]
    fg_dices = dices[1:]
    return {
        "miou": float(sum(ious) / len(ious)),
        "per_class_iou": ious,
        "per_class": {
            name: {"iou": ious[i], "dice": dices[i]}
            for i, name in enumerate(CLASS_NAMES)
        },
        "dice": float(sum(dices) / len(dices)),
        "fg_miou": float(sum(fg_ious) / len(fg_ious)),
        "fg_mdice": float(sum(fg_dices) / len(fg_dices)),
    }


def compute_sdf_mae(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
) -> float:
    """
    SDF 予測の平均絶対誤差を計算する

    引数:
        pred_sdf: 予測 SDF (B, 4, H, W)
        gt_sdf: GT SDF (B, 4, H, W) [-1, 1]

    戻り値:
        平均絶対誤差（スカラー float）
    """
    mae = torch.mean(torch.abs(pred_sdf - gt_sdf))
    return float(mae.item())


def compute_sdf_boundary_iou(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    threshold: float = 0.1,
    eps: float = 1e-6,
) -> float:
    """
    SDF ゼロ交差領域のIoUを計算する（境界近傍の一致度）

    ゼロ交差領域: |sdf| < threshold の画素を境界として扱う

    引数:
        pred_sdf: 予測 SDF (B, 4, H, W)
        gt_sdf: GT SDF (B, 4, H, W) [-1, 1]
        threshold: 境界と判定する SDF 絶対値の閾値
        eps: ゼロ除算防止

    戻り値:
        境界IoU（スカラー float）
    """
    pred_boundary = torch.abs(pred_sdf) < threshold
    gt_boundary = torch.abs(gt_sdf) < threshold

    intersection = (pred_boundary & gt_boundary).float().sum()
    union = (pred_boundary | gt_boundary).float().sum()
    return float((intersection + eps) / (union + eps))


def compute_angle_error(pred_params, gt_params, valid_mask):
    """
    arccos を用いた角度誤差（度数法）

    引数:
        pred_params: (B, 4, 2) 予測された (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        valid_mask: (B, 4) 有効なサンプルのブール値マスク

    戻り値:
        平均角度誤差（度）
    """
    pred_phi = pred_params[..., 0]
    gt_phi = gt_params[..., 0]

    pred_nx = torch.cos(pred_phi)
    pred_ny = torch.sin(pred_phi)
    gt_nx = torch.cos(gt_phi)
    gt_ny = torch.sin(gt_phi)
    dot = pred_nx * gt_nx + pred_ny * gt_ny

    dot_clamped = torch.clamp(torch.abs(dot), 0.0, 1.0)
    angle_error_rad = torch.acos(dot_clamped)
    angle_error_deg = torch.rad2deg(angle_error_rad)

    if valid_mask is not None:
        angle_error_deg = angle_error_deg * valid_mask.float()
        return float(angle_error_deg.sum() / (valid_mask.sum() + 1e-8))
    return float(angle_error_deg.mean())


def compute_rho_error(pred_params, gt_params, image_size, valid_mask):
    """
    Rho 誤差（ピクセル単位、符号不変）

    引数:
        pred_params: (B, 4, 2) 予測された (phi, rho) - rho は正規化済み
        gt_params: (B, 4, 2) GT (phi, rho) - rho は正規化済み
        image_size: 画像サイズ
        valid_mask: (B, 4) ブール値マスク

    戻り値:
        平均 rho 誤差（ピクセル）
    """
    pred_rho = pred_params[..., 1]
    gt_rho = gt_params[..., 1]

    err1 = torch.abs(pred_rho - gt_rho)
    err2 = torch.abs(pred_rho + gt_rho)
    err = torch.minimum(err1, err2)

    D = math.sqrt(image_size**2 + image_size**2)
    err_px = err * D

    if valid_mask is not None:
        err_px = err_px * valid_mask.float()
        return float(err_px.sum() / (valid_mask.sum() + 1e-8))
    return float(err_px.mean())


def compute_perpendicular_distance(
    gt_polyline_points, pred_phi, pred_rho, image_size, num_samples=20
):
    """
    GT線分上の点から予測直線への垂直距離の平均

    引数:
        gt_polyline_points: (N, 2) GT線の点の配列
        pred_phi: 予測角度（ラジアン）
        pred_rho: 予測 rho（正規化済み）
        image_size: 画像サイズ
        num_samples: GT線分上にサンプリングする点の数

    戻り値:
        平均垂直距離（ピクセル）
    """
    import numpy as np

    if gt_polyline_points is None or len(gt_polyline_points) < 2:
        return float("nan")

    gt_pts = np.array(gt_polyline_points, dtype=np.float64)

    D = math.sqrt(image_size**2 + image_size**2)
    rho_px = pred_rho * D

    nx = math.cos(pred_phi)
    ny = math.sin(pred_phi)

    total_length = 0.0
    lengths = []
    for i in range(len(gt_pts) - 1):
        seg_len = np.linalg.norm(gt_pts[i + 1] - gt_pts[i])
        lengths.append(seg_len)
        total_length += seg_len

    if total_length < 1e-6:
        return float("nan")

    sample_distances = np.linspace(0, total_length, num_samples)
    sampled_points = []

    for dist in sample_distances:
        current_cumulative = 0.0
        for i, seg_len in enumerate(lengths):
            if current_cumulative + seg_len >= dist:
                t = (dist - current_cumulative) / seg_len if seg_len > 0 else 0.0
                pt = (1 - t) * gt_pts[i] + t * gt_pts[i + 1]
                sampled_points.append(pt)
                break
            current_cumulative += seg_len

    if not sampled_points:
        return float("nan")

    sampled_points = np.array(sampled_points)

    center = image_size / 2.0
    sampled_points_centered = sampled_points - center

    distances = np.abs(
        sampled_points_centered[:, 0] * nx + sampled_points_centered[:, 1] * ny - rho_px
    )

    return float(distances.mean())
