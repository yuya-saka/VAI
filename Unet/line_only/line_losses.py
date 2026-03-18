"""直線の幾何学的制約のための損失関数とパラメータ抽出"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# GT 直線パラメータ抽出
# -------------------------
def extract_gt_line_params(polyline_points, image_size=224):
    """
    GT折れ線アノテーションから (φ, ρ) を抽出

    引数:
        polyline_points: 直線を定義する [x, y] 点のリスト（最低2点）
        image_size: 画像の次元（正方形を仮定）

    戻り値:
        (phi_rad, rho_normalized) または無効な場合は (nan, nan)

    座標系:
        - 原点: 画像中心 (image_size/2, image_size/2)
        - x軸: 列方向（左から右）
        - y軸: 行方向（上から下）
        - φ: 法線ベクトルの角度 [0, π)
        - ρ: 原点からの符号付き距離、対角線Dで正規化
    """
    if polyline_points is None or len(polyline_points) < 2:
        return float("nan"), float("nan")

    # 端点を取得
    p1 = np.array(polyline_points[0], dtype=np.float64)
    p2 = np.array(polyline_points[-1], dtype=np.float64)

    # 中心座標系に変換
    center = image_size / 2.0
    p1_c = p1 - center
    p2_c = p2 - center

    # 直線の方向ベクトル
    direction = p2_c - p1_c
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-6:
        return float("nan"), float("nan")
    direction = direction / norm_dir

    # 法線ベクトル（90度反時計回りに回転）
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # φ を [0, π) に制限
    if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
        normal = -normal

    # φ と ρ を抽出
    phi = np.arctan2(normal[1], normal[0])
    midpoint = (p1_c + p2_c) / 2.0
    rho = np.dot(normal, midpoint)

    # ρ を正規化
    D = np.sqrt(image_size**2 + image_size**2)
    rho_norm = rho / D

    return float(phi), float(rho_norm)


# -------------------------
# 予測直線パラメータ抽出（Codex最適化版）
# -------------------------
def extract_pred_line_params_batch(heatmaps, image_size=224, min_mass=1e-6):
    """
    モーメント法を用いて予測ヒートマップから (φ, ρ) を抽出

    Codexレビューによる主要改善点:
    - 直線方向には最大固有値の固有ベクトルを使用
    - 共分散行列にはFloat64精度
    - 安定性のための正則化
    - 信頼度ベースのマスキング

    引数:
        heatmaps: (B, 4, H, W) 予測ヒートマップ（sigmoid後）
        image_size: 画像の次元
        min_mass: ヒートマップ質量の最小閾値

    戻り値:
        pred_params: (B, 4, 2) (phi_rad, rho_normalized) のテンソル
        confidence: (B, 4) 固有値比のテンソル
        無効な直線はNaNでマーク
    """
    B, C, H, W = heatmaps.shape
    device = heatmaps.device

    # 座標グリッド（中心原点）
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

            # ガード: 質量が小さい場合はスキップ
            if M00 < min_mass:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # 重み付き重心
            cx = (hm * X).sum() / M00
            cy = (hm * Y).sum() / M00

            # 共分散（安定性のためFLOAT64）
            dx = (X - cx).double()
            dy = (Y - cy).double()
            hm_d = hm.double()

            mu20 = (hm_d * dx * dx).sum() / M00
            mu02 = (hm_d * dy * dy).sum() / M00
            mu11 = (hm_d * dx * dy).sum() / M00

            # 正則化
            eps_reg = 1e-6
            mu20 = mu20 + eps_reg
            mu02 = mu02 + eps_reg

            # 固有値
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11 * mu11
            discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
            sqrt_disc = torch.sqrt(discriminant)

            lambda1 = (trace + sqrt_disc) / 2  # 大きい方（直線方向）
            lambda2 = (trace - sqrt_disc) / 2  # 小さい方

            # 信頼度: 固有値比
            if lambda1 > 1e-8:
                conf = 1.0 - lambda2 / lambda1
                confidence[b, c] = conf  # 勾配フローのためテンソルのまま
            else:
                output[b, c] = float("nan")
                confidence[b, c] = 0.0
                continue

            # 直線方向: 最大固有値の固有ベクトル
            if abs(mu11) > 1e-8:
                dir_x = mu11
                dir_y = lambda1 - mu20
            else:
                # 軸に平行（一貫性のためテンソルを使用）
                if mu20 > mu02:
                    dir_x = torch.tensor(1.0, dtype=mu20.dtype, device=device)
                    dir_y = torch.tensor(0.0, dtype=mu20.dtype, device=device)
                else:
                    dir_x = torch.tensor(0.0, dtype=mu20.dtype, device=device)
                    dir_y = torch.tensor(1.0, dtype=mu20.dtype, device=device)

            # 方向ベクトルを正規化
            dir_norm = torch.sqrt(dir_x * dir_x + dir_y * dir_y)
            dir_x = dir_x / (dir_norm + 1e-10)
            dir_y = dir_y / (dir_norm + 1e-10)

            # 法線: 90度反時計回りに回転
            nx = -dir_y
            ny = dir_x

            # φ を [0, π) に制限
            if ny < 0 or (ny == 0 and nx < 0):
                nx, ny = -nx, -ny

            # φ と ρ を計算
            phi = torch.atan2(ny, nx)
            rho = nx * cx + ny * cy
            rho_norm = rho / D

            output[b, c, 0] = phi.float()
            output[b, c, 1] = rho_norm.float()

    return output, confidence


# -------------------------
# 損失関数（Codex最適化版）
# -------------------------
def angle_loss(pred_params, gt_params, confidence, valid_mask):
    """
    角度損失: 1 - |n_pred · n_gt|

    1 - |cos(φ_pred - φ_gt)| より効率的で、atan2の勾配を回避

    引数:
        pred_params: (B, 4, 2) 予測された (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        confidence: (B, 4) 信頼度の重み
        valid_mask: (B, 4) ブール値マスク

    戻り値:
        スカラー損失
    """
    # 全サンプル無効時のガード（NaN/無限大防止）
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_params.device, requires_grad=True)

    # 有効なエントリのみを抽出（NaN処理の前に実行）
    valid_pred_phi = pred_params[..., 0][valid_mask]
    valid_gt_phi = gt_params[..., 0][valid_mask]
    valid_conf = confidence[valid_mask].detach()  # 勾配フロー防止

    # 法線ベクトルを計算（有効なエントリのみ）
    pred_nx = torch.cos(valid_pred_phi)
    pred_ny = torch.sin(valid_pred_phi)
    gt_nx = torch.cos(valid_gt_phi)
    gt_ny = torch.sin(valid_gt_phi)

    # 内積
    dot = pred_nx * gt_nx + pred_ny * gt_ny

    # 損失: 1 - |dot|
    loss = 1.0 - torch.abs(dot)

    # 信頼度で重み付け
    weighted_loss = (loss * valid_conf).sum() / (valid_conf.sum() + 1e-8)

    # MSEスケール（~0.01）に合わせるためのスケール係数
    return weighted_loss * 0.01


def rho_loss(pred_params, gt_params, confidence, valid_mask):
    """
    Rho損失: min(|ρ_p - ρ_g|, |ρ_p + ρ_g|)

    (φ, ρ) ≡ (φ+π, -ρ) の曖昧性を扱うための符号不変
    微分可能性のためにsmooth minを使用

    引数:
        pred_params: (B, 4, 2) 予測された (phi, rho)
        gt_params: (B, 4, 2) GT (phi, rho)
        confidence: (B, 4) 信頼度の重み
        valid_mask: (B, 4) ブール値マスク

    戻り値:
        スカラー損失
    """
    # 全サンプル無効時のガード（NaN/無限大防止）
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_params.device, requires_grad=True)

    # 有効なエントリのみを抽出（NaN処理の前に実行）
    valid_pred_rho = pred_params[..., 1][valid_mask]
    valid_gt_rho = gt_params[..., 1][valid_mask]
    valid_conf = confidence[valid_mask].detach()  # 勾配フロー防止

    # 2つの可能性のある誤差（有効なエントリのみ）
    err1 = torch.abs(valid_pred_rho - valid_gt_rho)
    err2 = torch.abs(valid_pred_rho + valid_gt_rho)

    # smooth最小値（微分可能）
    alpha = 10.0
    exp1 = torch.exp(-alpha * err1)
    exp2 = torch.exp(-alpha * err2)
    loss = (err1 * exp1 + err2 * exp2) / (exp1 + exp2 + 1e-8)

    # 信頼度で重み付け
    weighted_loss = (loss * valid_conf).sum() / (valid_conf.sum() + 1e-8)

    # MSEスケール（~0.01）に合わせるためのスケール係数
    return weighted_loss * 0.01


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
    信頼度ベースのマスキングを用いた統合直線損失

    引数:
        pred_heatmaps: (B, 4, H, W) sigmoid後の予測ヒートマップ
        gt_line_params: (B, 4, 2) GT (phi, rho)
        image_size: 画像サイズ
        lambda_theta: 角度損失の重み
        lambda_rho: rho損失の重み
        use_angle: 角度損失を有効化
        use_rho: rho損失を有効化
        min_confidence: 損失計算のための最小固有値比

    戻り値:
        以下のキーを持つ辞書:
            'total': 統合直線損失
            'angle': 角度損失成分（または0）
            'rho': rho損失成分（または0）
    """
    pred_params, confidence = extract_pred_line_params_batch(pred_heatmaps, image_size)

    # 有効マスク
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
# Warmupスケジュール
# -------------------------
def get_warmup_weight(current_epoch, warmup_epochs, warmup_mode="linear"):
    """
    直線損失のためのwarmup重み w(t) を計算

    引数:
        current_epoch: 現在のエポック（1から始まる）
        warmup_epochs: warmupするエポック数
        warmup_mode: 'linear', 'cosine', または 'step'

    戻り値:
        [0, 1] の重み
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
