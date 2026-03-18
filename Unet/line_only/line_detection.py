# Unet/line-only/line_detection.py
# Consolidated from line_detection_moments.py

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import line_losses, line_metrics


# -------------------------
# ヘルパー関数
# -------------------------
def polyline_length(pts: list[list[float]] | None) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    d = pts[1:] - pts[:-1]
    return float(np.sqrt((d**2).sum(axis=1)).sum())


def centroid_dist_px(c_pred, c_gt) -> float:
    return float(math.sqrt((c_pred[0] - c_gt[0]) ** 2 + (c_pred[1] - c_gt[1]) ** 2))


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """直線なので180度周期で差を取る（向きは無視）"""
    d = abs(a_deg - b_deg) % 180.0
    return float(min(d, 180.0 - d))


def _clip_pt(x, y, W, H):
    x = float(np.clip(x, 0, W - 1))
    y = float(np.clip(y, 0, H - 1))
    return [x, y]


# -------------------------
# モーメントベースの直線検出
# -------------------------
def detect_line_moments(
    hm: np.ndarray,
    length_px: float | None = None,
    extend_ratio: float = 1.10,
    min_mass: float = 1e-6,
    clip: bool = True,
    top_p_fallback: float = 0.02,
) -> dict[str, Any] | None:
    """
    ヒートマップから直線を検出（モーメント法）

    内部計算は数学座標系（Y上向き）で実施し、
    出力の endpoints のみ画像座標系（Y下向き）に変換

    引数:
        hm: (H,W) float [0,1] ガウス分布様のヒートマップ

    戻り値:
      {
        "centroid": [xbar, ybar],  # 重心座標（画像座標系）
        "angle_rad": theta,         # 角度（ラジアン、数学座標系）
        "angle_deg": theta_deg,     # 角度（度、数学座標系）
        "dir": [dx, dy],            # 方向ベクトル（数学座標系）
        "endpoints": [[x1,y1],[x2,y2]], # 端点座標（画像座標系、描画用）
        "length": L                 # 直線の長さ
      }
    """
    if hm is None:
        return None

    hm = hm.astype(np.float64)
    H, W = hm.shape

    M00 = hm.sum()
    if M00 < min_mass:
        return None

    # 数学座標系（Y上向き、中心原点）で計算
    y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
    x_grid = np.arange(W, dtype=np.float64) - W / 2.0
    X, Y = np.meshgrid(x_grid, y_grid)

    # 1次モーメント → 重心（数学座標系）
    xbar = (hm * X).sum() / (M00 + 1e-12)
    ybar = (hm * Y).sum() / (M00 + 1e-12)

    # 2次中心モーメント（M00で正規化）
    dx = X - xbar
    dy = Y - ybar
    mu20 = (hm * (dx**2)).sum() / (M00 + 1e-12)
    mu02 = (hm * (dy**2)).sum() / (M00 + 1e-12)
    mu11 = (hm * (dx * dy)).sum() / (M00 + 1e-12)

    # 角度計算: theta = 0.5 * atan2(2*mu11, mu20 - mu02)
    theta = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
    theta_deg = float(np.degrees(theta) % 180.0)

    vx = float(math.cos(theta))
    vy = float(math.sin(theta))

    # 直線長さの決定
    if length_px is None or length_px <= 1e-6:
        # フォールバック: 上位p%の点を方向ベクトルに射影して推定
        thr = np.quantile(hm, 1.0 - float(top_p_fallback))
        yy, xx = np.where(hm >= thr)
        if len(xx) >= 10:
            # yy, xx は indices なので X[yy, xx], Y[yy, xx] で math coords を取得
            x_pts = X[yy, xx]
            y_pts = Y[yy, xx]
            t = (x_pts - xbar) * vx + (y_pts - ybar) * vy
            est = float(t.max() - t.min())
            length_px = max(est, 20.0)
        else:
            length_px = 40.0

    L = float(length_px * extend_ratio)

    # 端点を math coords で計算
    x1_math = xbar - 0.5 * L * vx
    y1_math = ybar - 0.5 * L * vy
    x2_math = xbar + 0.5 * L * vx
    y2_math = ybar + 0.5 * L * vy

    # math coords → image coords 変換（描画用）
    center_x = W / 2.0
    center_y = H / 2.0
    x1_img = x1_math + center_x
    y1_img = -y1_math + center_y  # Y軸反転
    x2_img = x2_math + center_x
    y2_img = -y2_math + center_y  # Y軸反転

    if clip:
        x1_img, y1_img = _clip_pt(x1_img, y1_img, W, H)
        x2_img, y2_img = _clip_pt(x2_img, y2_img, W, H)
    else:
        x1_img, y1_img = float(x1_img), float(y1_img)
        x2_img, y2_img = float(x2_img), float(y2_img)

    # centroid も image coords に変換（描画用）
    xbar_img = xbar + center_x
    ybar_img = -ybar + center_y

    return {
        "centroid": [float(xbar_img), float(ybar_img)],  # 画像座標系
        "angle_rad": float(theta),         # 数学座標系
        "angle_deg": float(theta_deg),     # 数学座標系
        "dir": [vx, vy],                   # 数学座標系
        "endpoints": [[x1_img, y1_img], [x2_img, y2_img]],  # 画像座標系
        "length": float(L),
        "M00": float(M00),
        "mu20": float(mu20),
        "mu02": float(mu02),
        "mu11": float(mu11),
    }


def draw_line_overlay(ct01: np.ndarray, lines4: dict[str, Any], save_path: Path):
    """
    直線検出結果をCT画像に重ねて描画

    引数:
        ct01: (H,W) float [0,1] CT画像
        lines4: {"line_1": {...}, ...} 直線検出結果
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    colors = {
        "line_1": (0, 255, 0),  # 緑
        "line_2": (0, 0, 255),  # 赤
        "line_3": (255, 0, 0),  # 青
        "line_4": (0, 255, 255),  # 黄
    }

    for k, v in lines4.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        (x1, y1), (x2, y2) = ep
        c = colors.get(k, (255, 255, 255))
        cv2.line(
            img,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img, (int(round(cx)), int(round(cy))), 2, c, -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def draw_line_comparison(
    ct01: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
):
    """
    GT線と予測線を横並びで比較表示

    引数:
        ct01: (H,W) float [0,1] CT画像
        pred_lines: {"line_1": {"endpoints": [[x1,y1],[x2,y2]], ...}, ...} 予測結果
        gt_lines: {"line_1": [[x,y], ...], ...} GT（lines.jsonの形式）
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)

    # GT画像（左）
    img_gt = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)
    # 予測画像（右）
    img_pred = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)

    colors = {
        "line_1": (0, 255, 0),  # 緑
        "line_2": (0, 0, 255),  # 赤
        "line_3": (255, 0, 0),  # 青
        "line_4": (0, 255, 255),  # 黄
    }

    # GT線を描画（折れ線）
    for k, pts in gt_lines.items():
        if pts is None or len(pts) < 2:
            continue
        c = colors.get(k, (255, 255, 255))
        pts_i32 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_gt, [pts_i32], isClosed=False, color=c, thickness=2)

    # 予測線を描画（直線）
    for k, v in pred_lines.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        c = colors.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = ep
        cv2.line(
            img_pred,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        # centroidにマーカー
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img_pred, (int(round(cx)), int(round(cy))), 3, c, -1)

    # ラベル追加
    cv2.putText(
        img_gt, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        img_pred, "Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # 横に結合
    combined = np.concatenate([img_gt, img_pred], axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


def draw_heatmap_with_lines(
    ct01: np.ndarray,
    heatmaps: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
    alpha: float = 0.6,
):
    """
    ヒートマップ・予測線・GT線の3つを横並びで表示

    引数:
        ct01: (H,W) float [0,1] CT画像
        heatmaps: (4,H,W) float [0,1] 予測ヒートマップ（4チャンネル）
        pred_lines: {"line_1": {"endpoints": [[x1,y1],[x2,y2]], ...}, ...} 予測線
        gt_lines: {"line_1": [[x,y], ...], ...} GT線（lines.jsonの形式）
        save_path: 保存先パス
        alpha: ヒートマップのブレンド係数
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    colors = {
        "line_1": (0, 255, 0),  # 緑
        "line_2": (0, 0, 255),  # 赤
        "line_3": (255, 0, 0),  # 青
        "line_4": (0, 255, 255),  # 黄
    }

    # 1. ヒートマップ画像（左）
    hm_merged = np.max(heatmaps, axis=0)  # 4チャンネルを最大値で統合
    hm_merged = np.clip(hm_merged, 0, 1)
    hm_u8 = (hm_merged * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    img_heatmap = cv2.addWeighted(base.copy(), 1 - alpha, heat_color, alpha, 0)
    cv2.putText(
        img_heatmap,
        "Heatmap",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 2. 予測線画像（中央）
    img_pred = base.copy()
    for k, v in pred_lines.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        c = colors.get(k, (255, 255, 255))
        (x1, y1), (x2, y2) = ep
        cv2.line(
            img_pred,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            c,
            2,
        )
        # centroidにマーカー
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img_pred, (int(round(cx)), int(round(cy))), 3, c, -1)

    cv2.putText(
        img_pred,
        "Pred Lines",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 3. GT線画像（右）
    img_gt = base.copy()
    for k, pts in gt_lines.items():
        if pts is None or len(pts) < 2:
            continue
        c = colors.get(k, (255, 255, 255))
        pts_i32 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_gt, [pts_i32], isClosed=False, color=c, thickness=2)

    cv2.putText(
        img_gt, "GT Lines", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # 3つを横に結合
    combined = np.concatenate([img_heatmap, img_pred, img_gt], axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


# -------------------------
# GT lines.jsonのキャッシュ（直線長さの参照元）
# -------------------------
class LinesJsonCache:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _load(self, sample: str, vertebra: str) -> dict[str, Any]:
        key = (sample, vertebra)
        if key in self._cache:
            return self._cache[key]
        lj = self.root_dir / sample / vertebra / "lines.json"
        if not lj.exists():
            self._cache[key] = {}
            return self._cache[key]
        try:
            data = json.loads(lj.read_text())
        except Exception:
            data = {}
        self._cache[key] = data
        return data

    def get_lines_for_slice(
        self, sample: str, vertebra: str, slice_idx: int
    ) -> dict[str, Any] | None:
        data = self._load(sample, vertebra)
        return data.get(str(int(slice_idx)), None)


def gt_centroid_angle_from_polyline(pts_xy):
    """
    GTの折れ線から重心と角度を計算（PCAベース）

    引数:
        pts_xy: [[x,y], ...] lines.jsonの点列

    戻り値:
        (centroid [x,y], angle_deg) または (None, None)
    """
    if pts_xy is None or len(pts_xy) < 2:
        return None, None

    pts = np.asarray(pts_xy, dtype=np.float64)  # (N,2) x,y
    c = pts.mean(axis=0)

    # PCA (TLS) で主軸方向
    xc = pts - c[None, :]
    C = (xc.T @ xc) / max(1, len(pts))  # (2,2)

    evals, evecs = np.linalg.eigh(C)
    d = evecs[:, np.argmax(evals)]
    d = d / (np.linalg.norm(d) + 1e-12)

    theta = math.atan2(d[1], d[0])  # y,x
    angle_deg = float(np.degrees(theta) % 180.0)
    return [float(c[0]), float(c[1])], angle_deg


# -------------------------
# メイン関数: テストデータに対する直線予測と評価
# -------------------------
@torch.no_grad()
def predict_lines_and_eval_test(
    cfg: dict[str, Any],
    model: nn.Module,
    test_loader,
    device: torch.device,
    dataset_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """
    テストデータに対する直線検出と評価（バリデーションと統一）

    処理内容:
        - GT: polylineから直接(φ, ρ)を抽出
        - 予測: ヒートマップから(φ, ρ)を抽出
        - 評価: line_metricsの関数を使用（バリデーションと同じ）
        - 描画: moments法で端点を計算（可視化用）
        - 保存: オーバーレイ画像(PNG) + 検出結果(JSON)

    戻り値:
        サマリー辞書（平均誤差、チャンネル別統計など）
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = float(cfg.get("evaluation", {}).get("line_extend_ratio", 1.10))
    hm_thr = float(
        cfg.get("evaluation", {}).get("heatmap_threshold", 0.15)
    )  # 参照用（強制閾値ではない）
    image_size = int(cfg.get("data", {}).get("image_size", 224))

    cache = LinesJsonCache(Path(dataset_root))

    # Line params-based metrics (統一評価)
    angle_errors = []
    rho_errors = []
    perp_dists = []

    per_ch = {
        1: {"angle": [], "rho": [], "perp": []},
        2: {"angle": [], "rho": [], "perp": []},
        3: {"angle": [], "rho": [], "perp": []},
        4: {"angle": [], "rho": [], "perp": []},
    }

    # 椎体ごとの統計用辞書
    per_vertebra = {
        v: {"angle": [], "rho": [], "perp": []}
        for v in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }
    saved = 0

    for batch in test_loader:
        x = batch["image"].to(device).float()
        pred = torch.sigmoid(model(x))

        # Extract pred line params for entire batch
        pred_params, confidence = line_losses.extract_pred_line_params_batch(
            pred, image_size
        )

        x_np = x.cpu().numpy()
        pr_np = pred.cpu().numpy()
        pred_params_np = pred_params.cpu().numpy()

        B = pr_np.shape[0]
        for i in range(B):
            sample = batch["sample"][i]
            vertebra = batch["vertebra"][i]
            slice_idx = int(batch["slice_idx"][i])
            ct01 = x_np[i, 0]

            name = f"{sample}_{vertebra}_slice{slice_idx:03d}"
            gt_lines = cache.get_lines_for_slice(sample, vertebra, slice_idx) or {}

            pred_lines_out = {}
            metrics_out = {}

            for c in range(4):
                k = f"line_{c + 1}"

                # GT折れ線
                gt_pts = gt_lines.get(k, None)

                # GT line params from polyline (統一評価)
                gt_phi, gt_rho = line_losses.extract_gt_line_params(gt_pts, image_size)

                # Pred line params
                pred_phi = pred_params_np[i, c, 0]
                pred_rho = pred_params_np[i, c, 1]

                # GT線長（描画長さの参照）
                Lgt = polyline_length(gt_pts)
                if Lgt <= 1e-6:
                    Lgt = None

                # ヒートマップモーメントから予測（描画用）
                pred_info = detect_line_moments(
                    pr_np[i, c], length_px=Lgt, extend_ratio=ext
                )
                pred_lines_out[k] = pred_info

                # 統一評価メトリクス
                if not (np.isnan(gt_phi) or np.isnan(pred_phi)):
                    # Angle error
                    gt_params_single = torch.tensor([[gt_phi, gt_rho]], dtype=torch.float32)
                    pred_params_single = torch.tensor([[pred_phi, pred_rho]], dtype=torch.float32)
                    valid_mask = torch.tensor([True])

                    angle_err = line_metrics.compute_angle_error(
                        pred_params_single, gt_params_single, valid_mask
                    )
                    rho_err = line_metrics.compute_rho_error(
                        pred_params_single, gt_params_single, image_size, valid_mask
                    )
                    perp_dist = line_metrics.compute_perpendicular_distance(
                        gt_pts, pred_phi, pred_rho, image_size
                    )

                    angle_errors.append(angle_err)
                    rho_errors.append(rho_err)
                    perp_dists.append(perp_dist)

                    per_ch[c + 1]["angle"].append(angle_err)
                    per_ch[c + 1]["rho"].append(rho_err)
                    per_ch[c + 1]["perp"].append(perp_dist)

                    if vertebra in per_vertebra:
                        per_vertebra[vertebra]["angle"].append(angle_err)
                        per_vertebra[vertebra]["rho"].append(rho_err)
                        per_vertebra[vertebra]["perp"].append(perp_dist)

                    metrics_out[k] = {
                        "angle_error_deg": float(angle_err),
                        "rho_error_px": float(rho_err),
                        "perpendicular_dist_px": float(perp_dist),
                        "gt_phi": float(gt_phi),
                        "gt_rho": float(gt_rho),
                        "pred_phi": float(pred_phi),
                        "pred_rho": float(pred_rho),
                    }
                else:
                    metrics_out[k] = {
                        "angle_error_deg": None,
                        "rho_error_px": None,
                        "perpendicular_dist_px": None,
                    }

            # GT/予測の比較画像を保存（2パネル版）
            draw_line_comparison(
                ct01, pred_lines_out, gt_lines, out_dir / f"{name}_comparison.png"
            )

            # ヒートマップ・予測線・GT線の3パネル版を保存
            draw_heatmap_with_lines(
                ct01,
                pr_np[i],  # (4,H,W) ヒートマップ
                pred_lines_out,
                gt_lines,
                out_dir / f"{name}_heatmap_lines.png",
            )

            with open(out_dir / f"{name}_PRED_lines.json", "w") as f:
                json.dump(
                    {
                        "pred_lines": pred_lines_out,
                        "metrics": metrics_out,
                        "heatmap_threshold_ref": hm_thr,
                    },
                    f,
                    indent=2,
                )

            saved += 1

    def _mean(vals):
        v = [
            x
            for x in vals
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]
        return float(np.mean(v)) if len(v) else float("nan")

    summary = {
        "n_samples": int(saved),
        # 統一評価メトリクス
        "angle_error_deg_mean": _mean(angle_errors),
        "rho_error_px_mean": _mean(rho_errors),
        "perpendicular_dist_px_mean": _mean(perp_dists),
        "per_channel": {
            f"line_{k}": {
                "angle_error_deg_mean": _mean(v["angle"]),
                "rho_error_px_mean": _mean(v["rho"]),
                "perpendicular_dist_px_mean": _mean(v["perp"]),
                "n": int(len(v["angle"])),
            }
            for k, v in per_ch.items()
        },
        "per_vertebra": {
            v: {
                "angle_error_deg_mean": _mean(vals["angle"]) if vals["angle"] else None,
                "rho_error_px_mean": _mean(vals["rho"]) if vals["rho"] else None,
                "perpendicular_dist_px_mean": _mean(vals["perp"]) if vals["perp"] else None,
                "n": int(len(vals["angle"])),
            }
            for v, vals in per_vertebra.items()
        },
        "line_extend_ratio": float(ext),
        "heatmap_threshold_ref": float(hm_thr),
        "out_dir": str(out_dir),
    }
    return summary
