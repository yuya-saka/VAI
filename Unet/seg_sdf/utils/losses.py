"""損失関数: セグメンテーション + Truncated SDF 回帰"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# GT 直線パラメータ抽出（multitask/utils/losses.py と同一実装）
# -------------------------
def extract_gt_line_params(polyline_points, image_size=224):
    """
    GT折れ線アノテーションから (φ, ρ) を抽出（PCA法）

    引数:
        polyline_points: 直線を定義する [x, y] 点のリスト（最低2点）
                        入力は画像座標系 (x=col, y=row, Y下向き)
        image_size: 画像の次元（正方形を仮定）

    戻り値:
        (phi_rad, rho_normalized) または無効な場合は (nan, nan)
    """
    if polyline_points is None or len(polyline_points) < 2:
        return float("nan"), float("nan")

    center = image_size / 2.0
    pts = np.array(polyline_points, dtype=np.float64)

    # 画像座標 → 数学座標変換（Y軸上向き）
    pm = np.column_stack([pts[:, 0] - center, -(pts[:, 1] - center)])
    cen = pm.mean(axis=0)

    xc = pm - cen
    cov = (xc.T @ xc) / max(1, len(pts))

    if cov.max() < 1e-10:
        return float("nan"), float("nan")

    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, np.argmax(evals)]

    nx, ny = -d[1], d[0]

    # φ を [0, π) に制限
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny

    phi = np.arctan2(ny, nx)
    rho = nx * cen[0] + ny * cen[1]

    D = np.sqrt(image_size**2 + image_size**2)
    return float(phi), float(rho / D)


# -------------------------
# Seg+SDF 損失計算
# -------------------------
SDF_SMOOTH_L1_BETA = 0.1  # SmoothL1 の beta パラメータ（小さい値でL1に近い挙動）


def compute_sdf_seg_loss(
    seg_logits: torch.Tensor,
    sdf_field: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_sdf: torch.Tensor,
    has_seg_label: torch.Tensor,
    lambda_sdf: float = 3.0,
) -> dict[str, torch.Tensor]:
    """
    Seg+SDF 多タスク損失を計算する（部分教師対応）

    L = L_seg + lambda_sdf * L_sdf

    L_seg: CrossEntropy(seg_logits, gt_mask) - has_seg_label==True のサンプルのみ
    L_sdf: SmoothL1(sdf_field, gt_sdf, beta=0.1) - 全サンプル

    引数:
        seg_logits: セグメンテーションロジット (B, 5, H, W)
        sdf_field: SDF予測フィールド (B, 4, H, W) - 線形出力（活性化なし）
        gt_mask: GTセグメンテーションマスク (B, H, W) int64
        gt_sdf: GT truncated SDF (B, 4, H, W) float [-1, 1]
        has_seg_label: セグGTありフラグ (B,) bool
        lambda_sdf: SDF損失の重み（デフォルト 3.0）

    戻り値:
        {
            'total': 総損失,
            'raw_seg_loss': CE損失（detach済み、ラベルなし時は 0.0）,
            'raw_sdf_loss': SmoothL1損失（detach済み）,
            'weighted_sdf_loss': lambda_sdf * SDF損失（detach済み）,
        }
    """
    # SDF損失: 全サンプルに適用（教師信号は常にある）
    sdf_loss = F.smooth_l1_loss(sdf_field, gt_sdf, beta=SDF_SMOOTH_L1_BETA, reduction="mean")

    if has_seg_label.any():
        seg_loss = F.cross_entropy(
            seg_logits[has_seg_label],
            gt_mask[has_seg_label],
            reduction="mean",
        )
        total = seg_loss + lambda_sdf * sdf_loss
    else:
        seg_loss = torch.tensor(0.0, device=seg_logits.device)
        total = lambda_sdf * sdf_loss

    return {
        "total": total,
        "raw_seg_loss": seg_loss.detach(),
        "raw_sdf_loss": sdf_loss.detach(),
        "weighted_sdf_loss": (lambda_sdf * sdf_loss).detach(),
    }
