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
    GT折れ線アノテーションから (φ, ρ) を抽出（PCA法）

    引数:
        polyline_points: 直線を定義する [x, y] 点のリスト（最低2点）
                        入力は画像座標系 (x=col, y=row, Y下向き)
        image_size: 画像の次元（正方形を仮定）

    戻り値:
        (phi_rad, rho_normalized) または無効な場合は (nan, nan)
        出力は数学座標系 (Y上向き)

    PCA法を使う理由:
        V字型ポリライン（全データの57.3%）では先頭・末尾点が同じ端に集まるため
        端点法だと角度が大きくずれる。PCAは全点を使うため正確な主軸を返す。
    """
    if polyline_points is None or len(polyline_points) < 2:
        return float("nan"), float("nan")

    center = image_size / 2.0
    pts = np.array(polyline_points, dtype=np.float64)

    # image coords → math coords 変換（Y軸上向き）
    pm = np.column_stack([pts[:, 0] - center, -(pts[:, 1] - center)])
    cen = pm.mean(axis=0)

    # PCAで主軸方向を取得
    xc = pm - cen
    cov = (xc.T @ xc) / max(1, len(pts))

    # 全点が同一（ゼロ分散）の場合は無効
    if cov.max() < 1e-10:
        return float("nan"), float("nan")

    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, np.argmax(evals)]

    # 法線ベクトル（90度反時計回り）
    nx, ny = -d[1], d[0]

    # φ を [0, π) に制限
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny

    # φ と ρ を計算
    phi = np.arctan2(ny, nx)
    rho = nx * cen[0] + ny * cen[1]

    # ρ を正規化
    D = np.sqrt(image_size**2 + image_size**2)

    return float(phi), float(rho / D)


# -------------------------
# 内部ヘルパー: モーメント計算（phi 正規化なし）
# -------------------------
def _compute_moments_batch(heatmaps, min_mass=1e-6, threshold=None):
    """
    ヒートマップから損失計算・抽出に使う中間変数を一括計算

    phi [0,π) 正規化は行わない（損失パスで atan2/sign-flip を回避するため）

    引数:
        heatmaps: (B, C, H, W) sigmoid後の予測ヒートマップ
        min_mass: 有効判定の最小質量閾値
        threshold: 評価時ノイズ抑制閾値（訓練時は None）

    戻り値（すべて (B, C) 形状）:
        confidence: 異方性比 (lam1-lam2)/(lam1+lam2+eps)、無効時は 0
        nx, ny:     法線ベクトル（正規化済み、phi 正規化なし）
        cx, cy:     重み付き重心
        valid:      有効フラグ (bool)
    """
    B, C, H, W = heatmaps.shape
    device = heatmaps.device
    N = B * C

    # 座標グリッド（中心原点、数学座標系: Y上向き）
    y_grid = -(torch.arange(H, device=device, dtype=torch.float32) - H / 2.0)
    x_grid = torch.arange(W, device=device, dtype=torch.float32) - W / 2.0
    Y, X = torch.meshgrid(y_grid, x_grid, indexing="ij")  # (H, W)

    # (B, C, H, W) → (N, H, W)
    hm = heatmaps.reshape(N, H, W)

    # 評価時ノイズ抑制（勾配は通さない）
    if threshold is not None:
        hm = torch.where(hm >= threshold, hm, torch.zeros_like(hm))

    # 質量: (N,)
    hm_flat = hm.reshape(N, -1)        # (N, H*W)
    X_flat = X.reshape(-1)             # (H*W,)
    Y_flat = Y.reshape(-1)             # (H*W,)
    M = hm_flat.sum(dim=-1)            # (N,)
    M_safe = M.clamp(min=min_mass)

    # 重み付き重心: (N,)
    cx = (hm_flat * X_flat).sum(dim=-1) / M_safe
    cy = (hm_flat * Y_flat).sum(dim=-1) / M_safe

    # 重み付き共分散: (N,)
    dx = X_flat.unsqueeze(0) - cx.unsqueeze(1)   # (N, H*W)
    dy = Y_flat.unsqueeze(0) - cy.unsqueeze(1)   # (N, H*W)
    sxx = (hm_flat * dx * dx).sum(dim=-1) / M_safe + 1e-6
    syy = (hm_flat * dy * dy).sum(dim=-1) / M_safe + 1e-6
    sxy = (hm_flat * dx * dy).sum(dim=-1) / M_safe

    # 固有値: hypot で数値安定化
    disc = torch.hypot(sxx - syy, 2.0 * sxy)
    trace = sxx + syy
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2

    # 有効マスク
    valid = (M >= min_mass) & (lam1 > 1e-8)  # (N,)

    # confidence = (lam1-lam2)/(lam1+lam2+eps): [0,1]
    conf = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    conf = torch.where(valid, conf, torch.zeros_like(conf))

    # 直線方向の固有ベクトル（縮退時フォールバック付き）
    dir_x_raw = sxy
    dir_y_raw = lam1 - sxx
    dir_norm = torch.hypot(dir_x_raw, dir_y_raw)
    degenerate = dir_norm < 1e-8
    dir_x_fb = torch.where(sxx >= syy, torch.ones_like(dir_x_raw), torch.zeros_like(dir_x_raw))
    dir_y_fb = torch.where(sxx >= syy, torch.zeros_like(dir_x_raw), torch.ones_like(dir_x_raw))
    dir_x = torch.where(degenerate, dir_x_fb, dir_x_raw / dir_norm.clamp(min=1e-8))
    dir_y = torch.where(degenerate, dir_y_fb, dir_y_raw / dir_norm.clamp(min=1e-8))

    # 法線: 90度反時計回り（phi 正規化なし）
    nx = -dir_y  # (N,)
    ny =  dir_x  # (N,)

    return (
        conf.view(B, C),
        nx.view(B, C),
        ny.view(B, C),
        cx.view(B, C),
        cy.view(B, C),
        valid.view(B, C),
    )


# -------------------------
# 予測直線パラメータ抽出（評価・デバッグ用）
# -------------------------
def extract_pred_line_params_batch(heatmaps, image_size=224, min_mass=1e-6, threshold=None):
    """
    モーメント法を用いて予測ヒートマップから (φ, ρ) を一括抽出

    評価・デバッグ用。訓練損失パスでは _compute_moments_batch を直接使うこと
    （atan2 / phi 正規化の勾配問題を回避するため）

    引数:
        heatmaps: (B, C, H, W) 予測ヒートマップ（sigmoid後）
        image_size: 画像の次元
        min_mass: ヒートマップ質量の最小閾値
        threshold: ヒートマップの閾値処理（評価時 0.2 推奨、訓練時 None）

    戻り値:
        pred_params: (B, C, 2) (phi_rad, rho_normalized)（NaN なし、無効時は 0）
        confidence: (B, C) 異方性比 [0,1]（無効時は 0）
    """
    B, C, _, _ = heatmaps.shape
    device = heatmaps.device
    D = math.sqrt(image_size**2 + image_size**2)

    conf, nx, ny, cx, cy, valid = _compute_moments_batch(heatmaps, min_mass, threshold)

    # (B*C,) に flatten して phi 正規化を一括適用
    nx_f = nx.reshape(-1)
    ny_f = ny.reshape(-1)
    cx_f = cx.reshape(-1)
    cy_f = cy.reshape(-1)
    valid_f = valid.reshape(-1)

    # φ∈[0,π) 正規化（評価互換性のため維持）
    flip = (ny_f < 0) | ((ny_f == 0) & (nx_f < 0))
    sgn = torch.where(flip, torch.full_like(nx_f, -1.0), torch.ones_like(nx_f))
    nx_n = nx_f * sgn
    ny_n = ny_f * sgn

    phi = torch.atan2(ny_n, nx_n)
    rho = (nx_n * cx_f + ny_n * cy_f) / D

    # 無効エントリはゼロ（NaN なし）
    phi = torch.where(valid_f, phi, torch.zeros_like(phi))
    rho = torch.where(valid_f, rho, torch.zeros_like(rho))

    output = torch.stack([phi, rho], dim=-1).view(B, C, 2)
    return output, conf


# -------------------------
# 損失関数（再設計版）
# -------------------------
def angle_loss(nx_pred, ny_pred, phi_gt):
    """
    角度損失（要素ごと）: L = 1 - (n_pred · n_gt)²

    π 周期・全域滑らか（|dot| と違い dot=0 で cusp なし）
    atan2 / sign-flip を使わないため勾配が安定

    引数:
        nx_pred, ny_pred: (B, C) 予測法線ベクトル（phi 正規化なし）
        phi_gt:           (B, C) GT 角度 [rad]

    戻り値:
        L_ang: (B, C) 要素ごとの損失 [0, 1]
        dot:   (B, C) 内積（rho_loss の符号整合に利用）
    """
    nx_gt = torch.cos(phi_gt)
    ny_gt = torch.sin(phi_gt)
    dot = nx_pred * nx_gt + ny_pred * ny_gt   # (B, C)
    L_ang = 1.0 - dot.pow(2)                  # [0, 1], π 周期
    return L_ang, dot


def rho_loss(nx_pred, ny_pred, cx, cy, rho_gt, dot, D):
    """
    ρ 損失（要素ごと）: detach 符号整合 + SmoothL1

    dot の符号で予測法線を GT 方向に揃えてから ρ を計算。
    smooth-min 廃止: φ∈[0,π) 正規化後の曖昧性はこれで解決。

    引数:
        nx_pred, ny_pred: (B, C) 予測法線ベクトル（phi 正規化なし）
        cx, cy:           (B, C) 重み付き重心
        rho_gt:           (B, C) GT 正規化済み ρ
        dot:              (B, C) angle_loss の内積（detach してから使う）
        D:                float 正規化定数 (= √2·image_size)

    戻り値:
        L_rho: (B, C) 要素ごとの損失
    """
    # dot の符号で法線方向を GT に整合（勾配は通さない）
    sgn = torch.where(
        dot.detach() >= 0,
        torch.ones_like(dot),
        -torch.ones_like(dot),
    )
    nx_a = sgn * nx_pred
    ny_a = sgn * ny_pred
    rho_pred = (nx_a * cx + ny_a * cy) / D
    return F.smooth_l1_loss(rho_pred, rho_gt, reduction="none")  # (B, C)


def compute_line_loss(
    pred_heatmaps,
    gt_line_params,
    image_size=224,
    lambda_angle=1.0,
    lambda_rho=1.0,
    use_line_loss=False,
    confidence_gate_low=0.3,
    confidence_gate_high=0.6,
):
    """
    ソフト信頼度ゲート付き統合直線損失

    引数:
        pred_heatmaps:        (B, C, H, W) sigmoid後の予測ヒートマップ
        gt_line_params:       (B, C, 2) GT (phi, rho)（無効時は NaN）
        image_size:           画像サイズ
        lambda_angle:         角度損失の重み（L_ang ∈ [0,1] なのでスケール調整不要）
        lambda_rho:           ρ 損失の重み
        use_line_loss:        直線損失を有効にするか
        confidence_gate_low:  gate が 0 になる信頼度下限
        confidence_gate_high: gate が 1 になる信頼度上限

    戻り値:
        dict:
            'total':      統合直線損失
            'angle':      角度損失成分（または 0）
            'rho':        ρ 損失成分（または 0）
            'gate_ratio': gate が有効なサンプルの割合
    """
    device = pred_heatmaps.device
    D = math.sqrt(image_size**2 + image_size**2)
    zero = torch.tensor(0.0, device=device)

    if not use_line_loss:
        return {"total": zero, "angle": zero, "rho": zero, "gate_ratio": zero}

    conf, nx, ny, cx, cy, pred_valid = _compute_moments_batch(pred_heatmaps)

    # GT 有効マスク（NaN チェック）
    gt_valid = ~torch.isnan(gt_line_params).any(dim=-1)  # (B, C)

    # ソフト信頼度ゲート
    gate_range = confidence_gate_high - confidence_gate_low
    gate = ((conf - confidence_gate_low) / gate_range).clamp(0.0, 1.0).detach()
    gate = gate * (gt_valid & pred_valid).float()  # 無効エントリはゼロ

    gate_sum = gate.sum().clamp(min=1.0)
    gate_ratio = gate.sum() / (gt_valid.float().sum().clamp(min=1.0))

    phi_gt = gt_line_params[..., 0]   # (B, C)
    rho_gt = gt_line_params[..., 1]   # (B, C)

    L_ang, dot = angle_loss(nx, ny, phi_gt)        # (B, C)
    L_rho = rho_loss(nx, ny, cx, cy, rho_gt, dot, D)  # (B, C)

    loss_a = (gate * L_ang).sum() / gate_sum
    loss_r = (gate * L_rho).sum() / gate_sum

    total = lambda_angle * loss_a + lambda_rho * loss_r

    return {
        "total": total,
        "angle": loss_a,
        "rho": loss_r,
        "gate_ratio": gate_ratio.detach(),
    }


# -------------------------
# Warmup スケジュール
# -------------------------
def get_warmup_weight(current_epoch, warmup_epochs, warmup_mode="linear", warmup_start_epoch=0):
    """
    直線損失のための warmup 重み w(t) を計算

    引数:
        current_epoch:       現在のエポック（1 から始まる）
        warmup_epochs:       warmup するエポック数
        warmup_mode:         'linear' または 'cosine'
        warmup_start_epoch:  warmup を開始するエポック（デフォルト 0 で旧動作と互換）

    戻り値:
        [0, 1] の重み
    """
    # 開始前はゼロ
    if current_epoch < warmup_start_epoch:
        return 0.0

    # 開始後の経過エポック
    elapsed = current_epoch - warmup_start_epoch

    if elapsed >= warmup_epochs or warmup_epochs == 0:
        return 1.0

    progress = elapsed / warmup_epochs

    if warmup_mode == "linear":
        return progress
    elif warmup_mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        return progress
