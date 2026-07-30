"""面を (角度, ヒートマップ重心) で表した場合の z 依存を実測する。

rho ではなく、GTヒートマップの重心が z 方向にどう動くかを測り、
定数 / 1次 / 2次 のどれで表すべきかを leave-one-slice-out で決める。
"""

import glob
import json
import os

import cv2
import numpy as np

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
S = 224
SIGMA = 3.5


def heatmap_from_polyline(pts_xy: list) -> np.ndarray:
    """dataset.py の _heatmap_from_polyline と同じ生成をする。"""
    pts = np.array(pts_xy, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, S - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, S - 1)
    mask = np.zeros((S, S), np.uint8)
    cv2.polylines(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.exp(-(dist**2) / (2.0 * SIGMA**2)).astype(np.float32)


def moments(hm: np.ndarray) -> tuple[float, float, float]:
    """重心 (cx, cy) と主軸角度 alpha[rad] を返す。"""
    ys, xs = np.mgrid[0:S, 0:S]
    m = hm.sum()
    cx = float((hm * xs).sum() / m)
    cy = float((hm * ys).sum() / m)
    dx, dy = xs - cx, ys - cy
    cxx = float((hm * dx * dx).sum() / m)
    cyy = float((hm * dy * dy).sum() / m)
    cxy = float((hm * dx * dy).sum() / m)
    alpha = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return cx, cy, alpha


def loo(z: np.ndarray, y: np.ndarray, deg: int) -> float:
    errs = []
    for i in range(len(z)):
        m = np.ones(len(z), bool)
        m[i] = False
        if m.sum() < deg + 2:
            continue
        c = np.polyfit(z[m] - z.mean(), y[m], deg)
        errs.append(np.polyval(c, z[i] - z.mean()) - y[i])
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan


def main() -> None:
    res = {d: {"cent": [], "ang": []} for d in (0, 1, 2)}
    drift_c, drift_a, span = [], [], []

    for lj in sorted(glob.glob("data/dataset/sample*/C*/lines.json")):
        d = json.loads(open(lj).read())
        vd = os.path.dirname(lj)
        zs, feats = [], []
        for k in sorted(d, key=int):
            e = d[k]
            if not all(e.get(x) and len(e[x]) >= 2 for x in LINE_KEYS):
                continue
            if not os.path.exists(f"{vd}/images/slice_{int(k):03d}.png"):
                continue
            row = []
            for key in LINE_KEYS:
                cx, cy, a = moments(heatmap_from_polyline(e[key]))
                row.append([cx, cy, np.cos(2 * a), np.sin(2 * a)])
            zs.append(int(k))
            feats.append(row)
        if len(zs) < 5:
            continue

        z = np.array(zs, float)
        f = np.array(feats)  # (T, 4, 4)
        span.append(z.max() - z.min())
        for li in range(4):
            cx, cy = f[:, li, 0], f[:, li, 1]
            # doubled-angle から連続な角度を復元
            ang = np.unwrap(np.arctan2(f[:, li, 3], f[:, li, 2])) / 2

            drift_c.append(np.hypot(cx.max() - cx.min(), cy.max() - cy.min()))
            drift_a.append(np.degrees(ang.max() - ang.min()))

            for deg in (0, 1, 2):
                ex, ey = loo(z, cx, deg), loo(z, cy, deg)
                ea = loo(z, ang, deg)
                if not np.isnan(ex):
                    res[deg]["cent"].append(np.hypot(ex, ey))
                    res[deg]["ang"].append(np.degrees(ea))

    print(f"椎体×線 {len(drift_c)} 件, スラブ z 幅 median {np.median(span):.0f} slice "
          f"= {np.median(span) * 0.4:.1f}mm\n")
    print("帯内での実際の動き（GT）")
    print(f"  重心の移動量: median {np.median(drift_c):6.2f} px  p90 {np.percentile(drift_c, 90):6.2f} px")
    print(f"  角度の変化量: median {np.median(drift_a):6.2f} deg p90 {np.percentile(drift_a, 90):6.2f} deg")

    print("\nleave-one-slice-out（低いほど良い）")
    print(f"{'モデル':16s} {'重心誤差 px (median/p90)':30s} {'角度誤差 deg (median/p90)'}")
    names = {0: "定数", 1: "1次", 2: "2次"}
    for deg in (0, 1, 2):
        c = np.array(res[deg]["cent"])
        a = np.array(res[deg]["ang"])
        print(f"{names[deg]:16s} {np.median(c):9.2f} / {np.percentile(c, 90):7.2f}"
              f"              {np.median(a):9.2f} / {np.percentile(a, 90):7.2f}")


if __name__ == "__main__":
    main()
