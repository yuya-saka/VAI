"""SDF ゼロ交差ベースの直線検出、および汎用検出ユーティリティ"""

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
# SDF ゼロ交差ベースの直線検出
# -------------------------
def detect_line_from_sdf(
    sdf_ch: np.ndarray,
    image_size: int = 224,
    zero_threshold: float = 0.1,
    extend_ratio: float = 1.10,
    clip: bool = True,
) -> dict[str, Any] | None:
    """
    SDF チャンネルのゼロ交差領域から直線を検出する

    ゼロ交差領域（|sdf| < threshold）を二値マスクとして使い、
    モーメント法で直線の向きと重心を推定する。

    引数:
        sdf_ch: (H, W) float32 の SDF チャンネル（[-1, 1] 範囲）
        image_size: 画像サイズ
        zero_threshold: ゼロ交差と判定する閾値
        extend_ratio: 直線を両端から伸ばす比率
        clip: 端点を画像境界内にクリップするか

    戻り値:
        detect_line_moments と同形式の辞書、または None（検出失敗時）
    """
    H, W = sdf_ch.shape

    # ゼロ交差マスクを生成
    boundary_mask = (np.abs(sdf_ch) < zero_threshold).astype(np.float64)
    M00 = boundary_mask.sum()

    if M00 < 5.0:
        return None

    # 数学座標系で重心とモーメントを計算
    y_grid = -(np.arange(H, dtype=np.float64) - H / 2.0)
    x_grid = np.arange(W, dtype=np.float64) - W / 2.0
    X, Y = np.meshgrid(x_grid, y_grid)

    xbar = (boundary_mask * X).sum() / M00
    ybar = (boundary_mask * Y).sum() / M00

    dx = X - xbar
    dy = Y - ybar
    mu20 = (boundary_mask * (dx**2)).sum() / M00 + 1e-8
    mu02 = (boundary_mask * (dy**2)).sum() / M00 + 1e-8
    mu11 = (boundary_mask * (dx * dy)).sum() / M00

    theta = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
    theta_deg = float(np.degrees(theta) % 180.0)

    vx = float(math.cos(theta))
    vy = float(math.sin(theta))

    # 直線長さの推定（境界点の方向投影から）
    yy, xx = np.where(boundary_mask > 0)
    if len(xx) >= 2:
        x_pts = X[yy, xx]
        y_pts = Y[yy, xx]
        t = (x_pts - xbar) * vx + (y_pts - ybar) * vy
        length_px = max(float(t.max() - t.min()), 20.0)
    else:
        length_px = 40.0

    L = float(length_px * extend_ratio)

    # 端点（数学座標系 → 画像座標系に変換）
    center_x = W / 2.0
    center_y = H / 2.0
    x1_img = (xbar - 0.5 * L * vx) + center_x
    y1_img = -(ybar - 0.5 * L * vy) + center_y
    x2_img = (xbar + 0.5 * L * vx) + center_x
    y2_img = -(ybar + 0.5 * L * vy) + center_y

    if clip:
        x1_img, y1_img = _clip_pt(x1_img, y1_img, W, H)
        x2_img, y2_img = _clip_pt(x2_img, y2_img, W, H)

    # 重心を画像座標に変換
    xbar_img = xbar + center_x
    ybar_img = -ybar + center_y

    return {
        "centroid": [float(xbar_img), float(ybar_img)],
        "angle_rad": float(theta),
        "angle_deg": float(theta_deg),
        "dir": [vx, vy],
        "endpoints": [[x1_img, y1_img], [x2_img, y2_img]],
        "length": float(L),
        "M00": float(M00),
    }


# -------------------------
# GT lines.json のキャッシュ
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

    pts = np.asarray(pts_xy, dtype=np.float64)
    c = pts.mean(axis=0)

    xc = pts - c[None, :]
    C = (xc.T @ xc) / max(1, len(pts))

    evals, evecs = np.linalg.eigh(C)
    d = evecs[:, np.argmax(evals)]
    d = d / (np.linalg.norm(d) + 1e-12)

    theta = math.atan2(d[1], d[0])
    angle_deg = float(np.degrees(theta) % 180.0)
    return [float(c[0]), float(c[1])], angle_deg
