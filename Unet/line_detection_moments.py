# Unet/line_detection_moments.py

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


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

    引数:
        hm: (H,W) float [0,1] ガウス分布様のヒートマップ

    戻り値:
      {
        "centroid": [xbar, ybar],  # 重心座標
        "angle_rad": theta,         # 角度（ラジアン）
        "angle_deg": theta_deg,     # 角度（度）
        "dir": [dx, dy],            # 方向ベクトル
        "endpoints": [[x1,y1],[x2,y2]], # 端点座標
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

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    # 1次モーメント → 重心
    M10 = (hm * X).sum()
    M01 = (hm * Y).sum()
    xbar = M10 / (M00 + 1e-12)
    ybar = M01 / (M00 + 1e-12)

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
    テストデータに対する直線検出と評価

    処理内容:
        - test_loaderの各サンプルに対して予測ヒートマップからモーメント法で直線端点を計算
        - 評価: GTヒートマップから同じモーメント法で得た線との重心距離(px)と角度差(deg)
        - 線長: lines.jsonのGT線長を基準に少し延長（extend_ratio）
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

    cache = LinesJsonCache(Path(dataset_root))

    all_cd, all_ad = [], []
    per_ch = {
        1: {"cd": [], "ad": []},
        2: {"cd": [], "ad": []},
        3: {"cd": [], "ad": []},
        4: {"cd": [], "ad": []},
    }
    # 椎体ごとの統計用辞書を追加
    per_vertebra = {
        v: {"cd": [], "ad": []} for v in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    }
    saved = 0

    for batch in test_loader:
        x = batch["image"].to(device).float()
        pred = torch.sigmoid(model(x))

        x_np = x.cpu().numpy()
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
                k = f"line_{c + 1}"

                # GT折れ線
                gt_pts = gt_lines.get(k, None)
                gt_centroid, gt_angle = gt_centroid_angle_from_polyline(gt_pts)

                # GT線長（描画長さの参照）
                Lgt = polyline_length(gt_pts)
                if Lgt <= 1e-6:
                    Lgt = None

                # ヒートマップモーメントから予測
                pred_info = detect_line_moments(
                    pr_np[i, c], length_px=Lgt, extend_ratio=ext
                )

                pred_lines_out[k] = pred_info

                if (
                    pred_info is not None
                    and gt_centroid is not None
                    and gt_angle is not None
                ):
                    cd = centroid_dist_px(pred_info["centroid"], gt_centroid)
                    ad = angle_diff_deg(pred_info["angle_deg"], gt_angle)
                    all_cd.append(cd)
                    all_ad.append(ad)
                    per_ch[c + 1]["cd"].append(cd)
                    per_ch[c + 1]["ad"].append(ad)

                    # 椎体ごとの統計に追加
                    if vertebra in per_vertebra:
                        per_vertebra[vertebra]["cd"].append(cd)
                        per_vertebra[vertebra]["ad"].append(ad)

                    metrics_out[k] = {"centroid_dist_px": cd, "angle_diff_deg": ad}
                else:
                    metrics_out[k] = {"centroid_dist_px": None, "angle_diff_deg": None}

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
        "per_vertebra": {
            v: {
                "centroid_dist_px_mean": _mean(vals["cd"]) if vals["cd"] else None,
                "angle_diff_deg_mean": _mean(vals["ad"]) if vals["ad"] else None,
                "n": int(len(vals["cd"])),
            }
            for v, vals in per_vertebra.items()
        },
        "line_extend_ratio": float(ext),
        "heatmap_threshold_ref": float(hm_thr),
        "out_dir": str(out_dir),
    }
    return summary
