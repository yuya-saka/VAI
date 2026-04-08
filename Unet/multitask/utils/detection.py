"""モーメントベースの直線検出ロジック"""

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# -------------------------
# ヘルパー関数
# -------------------------
def polyline_length(pts: list[list[float]] | None) -> float:
    """折れ線の累積長を計算"""
    if pts is None or len(pts) < 2:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    d = pts[1:] - pts[:-1]
    return float(np.sqrt((d**2).sum(axis=1)).sum())


def line_extent(pts: list[list[float]] | None) -> float:
    """最遠点間距離（V字ポリラインでも正しい線長を返す）"""
    if pts is None or len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=np.float64)
    dists = np.sqrt(((arr[:, None] - arr[None, :]) ** 2).sum(axis=-1))
    return float(dists.max())


def centroid_dist_px(c_pred, c_gt) -> float:
    """2点間のユークリッド距離"""
    return float(math.sqrt((c_pred[0] - c_gt[0]) ** 2 + (c_pred[1] - c_gt[1]) ** 2))


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """直線なので180度周期で差を取る（向きは無視）"""
    d = abs(a_deg - b_deg) % 180.0
    return float(min(d, 180.0 - d))


def _clip_pt(x, y, W, H):
    """座標を画像境界内にクリップ"""
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
    threshold: float | None = 0.2,
) -> dict[str, Any] | None:
    """
    ヒートマップから直線を検出（モーメント法）

    内部計算は数学座標系（Y上向き）で実施し、
    出力の endpoints のみ画像座標系（Y下向き）に変換

    引数:
        hm: (H,W) float [0,1] ガウス分布様のヒートマップ
        threshold: しきい値。これ未満のピクセルをゼロにして
                   背景ノイズがモーメント計算に影響するのを防ぐ

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

    # しきい値適用: 背景ノイズを除去してモーメント計算の精度を向上
    if threshold is not None:
        hm = np.where(hm >= threshold, hm, 0.0)

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


# -------------------------
# GT lines.jsonのキャッシュ（直線長さの参照元）
# -------------------------
class LinesJsonCache:
    """GT折れ線アノテーションのキャッシュ"""

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
