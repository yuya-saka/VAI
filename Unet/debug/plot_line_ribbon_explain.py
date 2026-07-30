"""リボン表現（角度 + ヒートマップ重心を z の1次で表す）の図解を作る。

注意: 3D表示は必ず等方スケール（1px = 1px = 1slice）にすること。
matplotlib の自動スケールに任せると、x の変動幅が小さい場合に横方向へ
極端に引き伸ばされ、ほぼ平行な線が大きくばらついて見える。
"""

import json
import sys

import cv2
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

S = 224
SIGMA = 3.5


def heatmap_from_polyline(pts_xy: list) -> np.ndarray:
    """dataset.py の _heatmap_from_polyline と同じ生成をする。"""
    p = np.array(pts_xy, np.float32)
    p[:, 0] = np.clip(p[:, 0], 0, S - 1)
    p[:, 1] = np.clip(p[:, 1], 0, S - 1)
    m = np.zeros((S, S), np.uint8)
    cv2.polylines(m, [p.astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
    return np.exp(-(d**2) / (2 * SIGMA**2)).astype(np.float32)


def moments(hm: np.ndarray) -> tuple[float, float, float]:
    """重心 (cx, cy) と主軸角 alpha[rad] を返す。"""
    ys, xs = np.mgrid[0:S, 0:S]
    m = hm.sum()
    cx = float((hm * xs).sum() / m)
    cy = float((hm * ys).sum() / m)
    dx, dy = xs - cx, ys - cy
    cxx = float((hm * dx * dx).sum() / m)
    cyy = float((hm * dy * dy).sum() / m)
    cxy = float((hm * dx * dy).sum() / m)
    return cx, cy, 0.5 * np.arctan2(2 * cxy, cxx - cyy)


def main() -> None:
    vert_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/sample3/C5"
    key = sys.argv[2] if len(sys.argv) > 2 else "line_1"
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ribbon.png"

    data = json.loads(open(f"{vert_dir}/lines.json").read())
    zs = sorted(
        int(k)
        for k in data
        if all(data[k].get(f"line_{i}") and len(data[k][f"line_{i}"]) >= 2 for i in range(1, 5))
    )

    z, cent, ang, segs = [], [], [], []
    for zz in zs:
        pts = np.array(data[str(zz)][key], float)
        cx, cy, a = moments(heatmap_from_polyline(pts))
        u = np.array([np.cos(a), np.sin(a)])
        t = (pts - np.array([cx, cy])) @ u
        segs.append((np.array([cx, cy]) + t.min() * u, np.array([cx, cy]) + t.max() * u))
        z.append(zz)
        cent.append([cx, cy])
        ang.append(a)

    z = np.array(z, float)
    cent = np.array(cent)
    ang = np.unwrap(np.array(ang) * 2) / 2
    zc = z - z.mean()

    fig = plt.figure(figsize=(16, 5.4))

    # --- ① 3D: 等方スケールで描く ---
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    pts_all = np.array([p for s in segs for p in s])
    cx_m, cy_m = pts_all[:, 0].mean(), pts_all[:, 1].mean()
    half = max(np.ptp(pts_all[:, 0]), np.ptp(pts_all[:, 1])) / 2 * 1.15
    for (p0, p1), zz in zip(segs, z):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [zz, zz], color="lightcoral", lw=2)
    ax.plot(cent[:, 0], cent[:, 1], z, color="blue", lw=2.4, marker="o", ms=3.5,
            label="重心の軌跡")
    ax.set_xlim(cx_m - half, cx_m + half)
    ax.set_ylim(cy_m - half, cy_m + half)
    ax.set_zlim(z.min(), z.max())
    # 1px = 1px = 1slice（0.4mm等方なので z も同スケール）
    ax.set_box_aspect((2 * half, 2 * half, np.ptp(z)))
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("z (slice)")
    ax.set_title("① 線分を積む＝リボン（等方スケール）\n青＝ヒートマップ重心の軌跡", fontsize=11)
    ax.legend(fontsize=9)
    ax.view_init(elev=20, azim=-70)

    # --- ② 角度 ---
    ax2 = fig.add_subplot(1, 3, 2)
    deg = np.degrees(ang)
    s_a, i_a = np.polyfit(zc, deg, 1)
    ax2.plot(deg, z, "o", color="crimson", label="実測")
    ax2.plot(s_a * zc + i_a, z, "k-", lw=1.6, label=f"1次  {s_a:+.2f}°/slice")
    ax2.set_xlim(deg.mean() - 45, deg.mean() + 45)   # ±45°の実スケールで表示
    ax2.set_xlabel("線の向き α (度)")
    ax2.set_ylabel("z (slice)")
    ax2.set_title(f"② 角度は z にほぼ比例\n全変動 {np.ptp(deg):.1f}°（±45°スケールで表示）", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # --- ③ 重心 ---
    ax3 = fig.add_subplot(1, 3, 3)
    for i, (lab, col) in enumerate([("重心 x", "tab:blue"), ("重心 y", "tab:green")]):
        s, c0 = np.polyfit(zc, cent[:, i], 1)
        ax3.plot(cent[:, i], z, "o", color=col, label=f"{lab}  ({s:+.2f}px/slice)")
        ax3.plot(s * zc + c0, z, "-", color=col, lw=1.5, alpha=0.6)
    ax3.set_xlabel("重心座標 (px)")
    ax3.set_ylabel("z (slice)")
    ax3.set_title("③ 重心も z にほぼ比例", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    name = "/".join(vert_dir.rstrip("/").split("/")[-2:])
    fig.suptitle(
        f"{name}  {key} — 面 = (角度, 重心) を z の1次で表す：1本あたり8係数", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}  角度 {deg.min():.1f}〜{deg.max():.1f}° (幅 {np.ptp(deg):.1f}°), "
          f"重心x幅 {np.ptp(cent[:, 0]):.1f}px, 重心y幅 {np.ptp(cent[:, 1]):.1f}px")


if __name__ == "__main__":
    main()
