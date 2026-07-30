"""面のモデル次数選択と、4本の線の連動性を実測する。

1. アンカー帯（手動アノテーション ~11 slice）内で leave-one-slice-out を行い、
   定数 / 線形 / 2次 のどの z 依存モデルが最も汎化するかを比較する。
2. 4本の線の dphi/dz が椎体内で連動しているか（全体のねじれか、線ごと独立か）を測る。
3. junction（line_1∩line_2, line_3∩line_4）が z に対してどれだけ滑らかか。
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
D_NORM = np.sqrt(2) * IMAGE_SIZE


def load_vertebra_params(root: str) -> dict:
    """椎体ごとに (z, 4本の (phi, rho_px)) を返す。"""
    out = {}
    for lj in sorted(glob.glob(f"{root}/sample*/C*/lines.json")):
        data = json.loads(open(lj).read())
        zs, params = [], []
        for k in sorted(data, key=int):
            e = data[k]
            if not all(e.get(key) and len(e[key]) >= 2 for key in LINE_KEYS):
                continue
            p = [extract_gt_line_params(e[key], IMAGE_SIZE) for key in LINE_KEYS]
            if any(np.isnan(v[0]) for v in p):
                continue
            zs.append(int(k))
            params.append([[v[0], v[1] * D_NORM] for v in p])
        if len(zs) >= 5:
            out[os.path.dirname(lj)] = (np.array(zs, float), np.array(params))
    return out


def unwrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.unwrap(phi * 2.0) / 2.0


def loo_error(z: np.ndarray, y: np.ndarray, degree: int) -> float:
    """leave-one-out で degree 次多項式フィットの予測誤差 RMS を返す。"""
    errs = []
    for i in range(len(z)):
        mask = np.ones(len(z), bool)
        mask[i] = False
        if mask.sum() < degree + 2:
            continue
        coef = np.polyfit(z[mask] - z.mean(), y[mask], degree)
        pred = np.polyval(coef, z[i] - z.mean())
        errs.append(pred - y[i])
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan


def line_intersection(p1: tuple, p2: tuple) -> np.ndarray | None:
    """(phi, rho) 表現の2直線の交点を返す。"""
    a = np.array([[np.cos(p1[0]), np.sin(p1[0])], [np.cos(p2[0]), np.sin(p2[0])]])
    b = np.array([p1[1], p2[1]])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, b)


def main() -> None:
    verts = load_vertebra_params("data/dataset")
    print(f"解析対象椎体: {len(verts)}")

    # ---- 1. モデル次数の leave-one-out 比較 ----
    res = {d: {"phi": [], "rho": []} for d in (0, 1, 2)}
    for z, p in verts.values():
        for li in range(4):
            phi = unwrap_phi(p[:, li, 0])
            rho = p[:, li, 1]
            for d in (0, 1, 2):
                e_phi = loo_error(z, phi, d)
                e_rho = loo_error(z, rho, d)
                if not np.isnan(e_phi):
                    res[d]["phi"].append(np.degrees(e_phi))
                    res[d]["rho"].append(e_rho)

    print("\n=== アンカー帯内 leave-one-slice-out（低いほど良い）===")
    print(f"{'model':10s} {'phi err deg (median/p90)':30s} {'rho err px (median/p90)'}")
    names = {0: "定数(直交面)", 1: "線形(平面/斜め)", 2: "2次(曲面)"}
    for d in (0, 1, 2):
        ph = np.array(res[d]["phi"])
        rh = np.array(res[d]["rho"])
        print(
            f"{names[d]:10s} {np.median(ph):9.2f} / {np.percentile(ph, 90):7.2f}"
            f"          {np.median(rh):9.2f} / {np.percentile(rh, 90):7.2f}"
        )

    # ---- 2. 4本の線の連動性 ----
    slopes = []
    for z, p in verts.values():
        zc = z - z.mean()
        s = [np.polyfit(zc, unwrap_phi(p[:, li, 0]), 1)[0] for li in range(4)]
        slopes.append(np.degrees(s))
    slopes = np.array(slopes)  # (V, 4)
    print("\n=== dphi/dz の線間相関（椎体をまたいだ相関行列）===")
    corr = np.corrcoef(slopes.T)
    for i in range(4):
        print("  " + "  ".join(f"{corr[i, j]:6.2f}" for j in range(4)))
    common = slopes.mean(axis=1)
    resid = slopes - common[:, None]
    print(f"共通成分 std {common.std():.3f} deg/slice, 線ごと残差 std {resid.std():.3f} deg/slice")

    # ---- 3. junction の z 方向の滑らかさ ----
    jerr = {"linear": [], "quad": []}
    for z, p in verts.values():
        for pair in ((0, 1), (2, 3)):
            js = []
            zz = []
            for i in range(len(z)):
                pt = line_intersection(tuple(p[i, pair[0]]), tuple(p[i, pair[1]]))
                if pt is not None and np.abs(pt).max() < 3 * IMAGE_SIZE:
                    js.append(pt)
                    zz.append(z[i])
            if len(js) < 5:
                continue
            js = np.array(js)
            zz = np.array(zz, float)
            for name, d in (("linear", 1), ("quad", 2)):
                e = [loo_error(zz, js[:, k], d) for k in range(2)]
                jerr[name].append(np.sqrt(np.mean(np.square(e))))
    for name in ("linear", "quad"):
        v = np.array(jerr[name])
        print(
            f"\njunction LOO 予測誤差 ({name}): median {np.median(v):.2f} px, "
            f"p90 {np.percentile(v, 90):.2f} px  (n={len(v)})"
        )


if __name__ == "__main__":
    main()
