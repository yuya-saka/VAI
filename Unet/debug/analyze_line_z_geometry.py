"""アノテーション線が z 方向にどう変化するかを実測する。

各椎体・各線について (phi, rho) を z の関数として取り出し、
線形フィットの傾き・残差を測る。面が平面近似できるか、
スライス面に直交か斜めかを判定するための基礎統計。
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/nfs1/home/yamamoto-hiroto/research/VAI")
from Unet.multitask.utils.losses import extract_gt_line_params  # noqa: E402

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
IMAGE_SIZE = 224
D_NORM = np.sqrt(2) * IMAGE_SIZE  # rho 正規化係数
VOXEL_MM = 0.4


def collect(root: str) -> dict:
    out = {}
    for lj in sorted(glob.glob(f"{root}/sample*/C*/lines.json")):
        vert = os.path.dirname(lj)
        data = json.loads(open(lj).read())
        rows = []
        for k in sorted(data, key=int):
            entry = data[k]
            if not all(
                entry.get(key) and len(entry[key]) >= 2 for key in LINE_KEYS
            ):
                continue
            params = [extract_gt_line_params(entry[key], IMAGE_SIZE) for key in LINE_KEYS]
            if any(np.isnan(p[0]) for p in params):
                continue
            rows.append((int(k), params))
        if len(rows) >= 3:
            out[vert] = rows
    return out


def unwrap_phi(phis: np.ndarray) -> np.ndarray:
    """phi は [0, pi) 周期なので pi 周期で unwrap する。"""
    return np.unwrap(phis * 2.0) / 2.0


def analyze(rows: list, label: str) -> dict:
    z = np.array([r[0] for r in rows], dtype=float)
    res = {"n": len(z), "span_mm": (z.max() - z.min()) * VOXEL_MM}
    for li, key in enumerate(LINE_KEYS):
        phi = unwrap_phi(np.array([r[1][li][0] for r in rows]))
        rho_px = np.array([r[1][li][1] for r in rows]) * D_NORM

        # 線形フィット
        A = np.vstack([z, np.ones_like(z)]).T
        (a_phi, _), *_ = np.linalg.lstsq(A, phi, rcond=None)
        (a_rho, _), *_ = np.linalg.lstsq(A, rho_px, rcond=None)
        phi_fit = np.linalg.lstsq(A, phi, rcond=None)[0]
        rho_fit = np.linalg.lstsq(A, rho_px, rcond=None)[0]
        phi_resid = phi - A @ phi_fit
        rho_resid = rho_px - A @ rho_fit

        res[key] = {
            # 全変動
            "phi_range_deg": float(np.degrees(phi.max() - phi.min())),
            "rho_range_px": float(rho_px.max() - rho_px.min()),
            # 線形トレンド（1スライス=0.4mm あたり）
            "dphi_dz_deg_per_slice": float(np.degrees(a_phi)),
            "drho_dz_px_per_slice": float(a_rho),
            # 面のスライス面からの傾き: rho が z とともに動く量 = 面の傾斜角
            "tilt_deg": float(np.degrees(np.arctan(abs(a_rho)))),
            # 線形からの残差（非平面性）
            "phi_resid_rms_deg": float(np.degrees(np.sqrt((phi_resid**2).mean()))),
            "rho_resid_rms_px": float(np.sqrt((rho_resid**2).mean())),
        }
    return res


def summarize(all_res: list, title: str) -> None:
    print(f"\n{'=' * 70}\n{title}  (椎体数 {len(all_res)})\n{'=' * 70}")
    spans = [r["span_mm"] for r in all_res]
    print(f"z span (mm): median {np.median(spans):.1f}  min {min(spans):.1f}  max {max(spans):.1f}")
    for key in LINE_KEYS:
        vals = {
            m: np.array([r[key][m] for r in all_res])
            for m in (
                "phi_range_deg",
                "rho_range_px",
                "tilt_deg",
                "phi_resid_rms_deg",
                "rho_resid_rms_px",
            )
        }
        print(f"\n{key}")
        for m, v in vals.items():
            print(
                f"  {m:22s} median {np.median(v):7.2f}  p90 {np.percentile(v, 90):7.2f}  max {v.max():7.2f}"
            )


if __name__ == "__main__":
    for root, title in (
        ("data/dataset", "手動アノテーション（薄いスラブ ~11 slice）"),
        ("data/dataset_zprop", "z伝播後（椎体全体）"),
    ):
        rows_by_vert = collect(root)
        res = [analyze(rows, v) for v, rows in rows_by_vert.items()]
        summarize(res, title)
