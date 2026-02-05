# Unet/line_detection_moments.py
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn as nn


# -------------------------
# helpers
# -------------------------
def polyline_length(pts: Optional[List[List[float]]]) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    d = pts[1:] - pts[:-1]
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())

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
# moment-based line detection
# -------------------------
def detect_line_moments(
    hm: np.ndarray,
    length_px: Optional[float] = None,
    extend_ratio: float = 1.10,
    min_mass: float = 1e-6,
    clip: bool = True,
    top_p_fallback: float = 0.02,
) -> Optional[Dict[str, Any]]:
    """
    hm: (H,W) float [0,1] (gaussian-like)
    return:
      {
        "centroid": [xbar, ybar],
        "angle_rad": theta,
        "angle_deg": theta_deg,
        "dir": [dx, dy],
        "endpoints": [[x1,y1],[x2,y2]],
        "length": L
      }
    """
    if hm is None:
        return None

    hm = hm.astype(np.float64)
    H, W = hm.shape

    M00 = hm.sum()
    if M00 < min_mass:
        return None

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    # 1st moments -> centroid
    M10 = (hm * X).sum()
    M01 = (hm * Y).sum()
    xbar = M10 / (M00 + 1e-12)
    ybar = M01 / (M00 + 1e-12)

    # 2nd central moments (normalized by M00)
    dx = X - xbar
    dy = Y - ybar
    mu20 = (hm * (dx ** 2)).sum() / (M00 + 1e-12)
    mu02 = (hm * (dy ** 2)).sum() / (M00 + 1e-12)
    mu11 = (hm * (dx * dy)).sum() / (M00 + 1e-12)

    # angle = 0.5 * atan2(2mu11, mu20 - mu02)
    theta = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
    theta_deg = float(np.degrees(theta) % 180.0)

    vx = float(math.cos(theta))
    vy = float(math.sin(theta))

    # choose length
    if length_px is None or length_px <= 1e-6:
        # fallback: estimate by projecting top-p points onto direction
        thr = np.quantile(hm, 1.0 - float(top_p_fallback))
        yy, xx = np.where(hm >= thr)
        if len(xx) >= 10:
            t = (xx - xbar) * vx + (yy - ybar) * vy
            est = float(t.max() - t.min())
            length_px = max(est, 20.0)
        else:
            length_px = 40.0

    L = float(length_px * extend_ratio)

    x1 = xbar - 0.5 * L * vx
    y1 = ybar - 0.5 * L * vy
    x2 = xbar + 0.5 * L * vx
    y2 = ybar + 0.5 * L * vy

    if clip:
        x1, y1 = _clip_pt(x1, y1, W, H)
        x2, y2 = _clip_pt(x2, y2, W, H)
    else:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    return {
        "centroid": [float(xbar), float(ybar)],
        "angle_rad": float(theta),
        "angle_deg": float(theta_deg),
        "dir": [vx, vy],
        "endpoints": [[x1, y1], [x2, y2]],
        "length": float(L),
        "M00": float(M00),
        "mu20": float(mu20),
        "mu02": float(mu02),
        "mu11": float(mu11),
    }


def draw_line_overlay(ct01: np.ndarray, lines4: Dict[str, Any], save_path: Path):
    """
    ct01: (H,W) float [0,1]
    lines4: {"line_1": {...}, ...}
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    colors = {
        "line_1": (0, 255, 0),
        "line_2": (0, 0, 255),
        "line_3": (255, 0, 0),
        "line_4": (0, 255, 255),
    }

    for k, v in lines4.items():
        if v is None:
            continue
        ep = v.get("endpoints")
        if ep is None or len(ep) != 2:
            continue
        (x1, y1), (x2, y2) = ep
        c = colors.get(k, (255, 255, 255))
        cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), c, 2)
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img, (int(round(cx)), int(round(cy))), 2, c, -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def draw_line_comparison(
    ct01: np.ndarray,
    pred_lines: Dict[str, Any],
    gt_lines: Dict[str, Any],
    save_path: Path
):
    """
    GT線と予測線を横並びで比較表示
    ct01: (H,W) float [0,1]
    pred_lines: {"line_1": {"endpoints": [[x1,y1],[x2,y2]], ...}, ...}
    gt_lines: {"line_1": [[x,y], ...], ...} (lines.jsonの形式)
    """
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)

    # GT画像（左）
    img_gt = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)
    # 予測画像（右）
    img_pred = cv2.cvtColor(ct_u8.copy(), cv2.COLOR_GRAY2BGR)

    colors = {
        "line_1": (0, 255, 0),    # 緑
        "line_2": (0, 0, 255),    # 赤
        "line_3": (255, 0, 0),    # 青
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
        cv2.line(img_pred, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), c, 2)
        # centroidにマーカー
        cx, cy = v.get("centroid", [None, None])
        if cx is not None:
            cv2.circle(img_pred, (int(round(cx)), int(round(cy))), 3, c, -1)

    # ラベル追加
    cv2.putText(img_gt, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_pred, "Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 横に結合
    combined = np.concatenate([img_gt, img_pred], axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


# -------------------------
# GT lines.json cache (length source)
# -------------------------
class LinesJsonCache:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _load(self, sample: str, vertebra: str) -> Dict[str, Any]:
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

    def get_lines_for_slice(self, sample: str, vertebra: str, slice_idx: int) -> Optional[Dict[str, Any]]:
        data = self._load(sample, vertebra)
        return data.get(str(int(slice_idx)), None)

def gt_centroid_angle_from_polyline(pts_xy):
    """
    pts_xy: [[x,y], ...] lines.json の点列
    return: (centroid [x,y], angle_deg) or (None, None)
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
# main callable: predict lines on TEST + evaluate
# -------------------------
@torch.no_grad()
def predict_lines_and_eval_test(
    cfg: Dict[str, Any],
    model: nn.Module,
    test_loader,
    device: torch.device,
    dataset_root: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    - test_loader の各サンプルに対して
      pred heatmap -> moments -> line endpoints
    - 評価：GTヒートマップから同じ moments で得た線との
      centroid距離(px)・角度差(deg)
    - 線長：lines.json のGT線長を基準に少し超える（extend_ratio）
    - 保存：overlay png + json

    return: summary dict
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = float(cfg.get("evaluation", {}).get("line_extend_ratio", 1.10))
    hm_thr = float(cfg.get("evaluation", {}).get("heatmap_threshold", 0.15))  # 参照用（強制閾値ではない）

    cache = LinesJsonCache(Path(dataset_root))

    all_cd, all_ad = [], []
    per_ch = {1: {"cd": [], "ad": []}, 2: {"cd": [], "ad": []}, 3: {"cd": [], "ad": []}, 4: {"cd": [], "ad": []}}
    saved = 0

    for batch in test_loader:
        x = batch["image"].to(device).float()
        gt = batch["heatmaps"].to(device).float()
        pred = torch.sigmoid(model(x))

        x_np = x.cpu().numpy()
        gt_np = gt.cpu().numpy()
        pr_np = pred.cpu().numpy()

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
                k = f"line_{c+1}"

                # GT polyline
                gt_pts = gt_lines.get(k, None)
                gt_centroid, gt_angle = gt_centroid_angle_from_polyline(gt_pts)

                #  GT length (draw length reference) 
                Lgt = polyline_length(gt_pts)
                if Lgt <= 1e-6:
                     Lgt = None

                # Pred from heatmap moments（あなたの指定手法）
                pred_info = detect_line_moments(pr_np[i, c], length_px=Lgt, extend_ratio=ext)

                pred_lines_out[k] = pred_info

                if pred_info is not None and gt_centroid is not None and gt_angle is not None:
                    cd = centroid_dist_px(pred_info["centroid"], gt_centroid)
                    ad = angle_diff_deg(pred_info["angle_deg"], gt_angle)
                    all_cd.append(cd); all_ad.append(ad)
                    per_ch[c+1]["cd"].append(cd); per_ch[c+1]["ad"].append(ad)

                    metrics_out[k] = {"centroid_dist_px": cd, "angle_diff_deg": ad}
                else:
                    metrics_out[k] = {"centroid_dist_px": None, "angle_diff_deg": None}


            # save: GT/予測の比較画像を保存
            draw_line_comparison(ct01, pred_lines_out, gt_lines, out_dir / f"{name}_comparison.png")
            with open(out_dir / f"{name}_PRED_lines.json", "w") as f:
                json.dump(
                    {"pred_lines": pred_lines_out, "metrics": metrics_out, "heatmap_threshold_ref": hm_thr},
                    f,
                    indent=2
                )

            saved += 1

    def _mean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if len(v) else float("nan")

    summary = {
        "n_samples": int(saved),
        "centroid_dist_px_mean": _mean(all_cd),
        "angle_diff_deg_mean": _mean(all_ad),
        "per_channel": {
            f"line_{k}": {
                "centroid_dist_px_mean": _mean(v["cd"]),
                "angle_diff_deg_mean": _mean(v["ad"]),
                "n": int(len(v["cd"])),
            }
            for k, v in per_ch.items()
        },
        "line_extend_ratio": float(ext),
        "heatmap_threshold_ref": float(hm_thr),
        "out_dir": str(out_dir),
    }
    return summary
